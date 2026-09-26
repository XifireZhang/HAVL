from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch.utils.data import default_collate

from .data import Replay


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NativeKuaiSimAdapter:
    """Single-item adapter around native KREnvironment_WholeSession_GPU.

    `done` is the simulator's temper-threshold terminal. A collection run boundary
    is not written as a terminal or truncation transition.
    """

    def __init__(self, config: dict, seed: int):
        self.config = config
        self.root = Path(config["root"])
        self.code = self.root / "code"
        sys.path.insert(0, str(self.code))
        from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU

        reset_rng(seed)
        arguments = Namespace(
            device=config.get("device", "cuda:0"),
            max_step_per_episode=20,
            initial_temper=float(config["initial_temper"]),
            uirm_log_path=str(Path(config["model_log"]).relative_to("code"))
            if str(config["model_log"]).startswith("code/") else config["model_log"],
            slate_size=int(config["slate_size"]),
            episode_batch_size=int(config["episode_batch_size"]),
            item_correlation=float(config["item_correlation"]),
            single_response=True,
        )
        started = time.perf_counter()
        with working_directory(self.code):
            self.env = KREnvironment_WholeSession_GPU(arguments)
        self.load_seconds = time.perf_counter() - started
        self.device = torch.device(arguments.device)
        self.initial_temper = arguments.initial_temper
        self.batch_size = arguments.episode_batch_size
        self.catalog_features = self.env.candidate_item_encoding.detach()
        self.encoded_to_candidate = torch.full(
            (int(self.env.candidate_iids.max()) + 1,), -1,
            dtype=torch.long, device=self.device,
        )
        self.encoded_to_candidate[self.env.candidate_iids] = torch.arange(
            self.env.n_candidate, device=self.device
        )
        self.user_splits: dict[str, np.ndarray] = {}
        self.row_splits: dict[str, np.ndarray] = {}
        self.current_split: str | None = None
        self._original_train_rows = self.env.reader.data["train"]
        self.encoded_to_raw_user = {
            int(encoded): raw for raw, encoded in self.env.reader.user_id_vocab.items()
        }

    @property
    def state_dim(self) -> int:
        return int(self.env.gt_state_dim + 1)

    @property
    def action_feature_dim(self) -> int:
        return int(self.catalog_features.shape[1])

    def metadata(self) -> dict:
        statistics = self.env.reader.get_statistics()
        return {
            "native_environment": type(self.env).__name__,
            "load_seconds": self.load_seconds,
            "raw_records": int(statistics["raw_data_size"]),
            "supported_users": int(statistics["n_user"]),
            "candidate_items": int(self.env.n_candidate),
            "action": "single item index over the full native candidate catalog",
            "reward": "sampled is_click for slate_size=1",
            "terminal": "native temper threshold current_temper < 1",
            "truncation": "none; stream boundary is not labeled terminal",
            "initial_state_sampling": "interaction-weighted rows within the selected user split",
        }

    def configure_user_splits(self, splits: dict[str, np.ndarray]) -> None:
        inverse = {encoded: raw for raw, encoded in self.env.reader.user_id_vocab.items()}
        for name, encoded_users in splits.items():
            self.user_splits[name] = np.asarray(encoded_users, dtype=np.int64)
            rows = [self.env.reader.user_history[inverse[int(user)]] for user in encoded_users]
            # Keep raw reader-row identifiers sorted so fixed user contexts can
            # be mapped back to split-local Dataset positions by searchsorted.
            self.row_splits[name] = np.sort(np.concatenate(rows).astype(np.int64))

    def use_split(self, name: str) -> None:
        if name not in self.row_splits:
            raise KeyError(name)
        self.env.reader.data["train"] = self.row_splits[name]
        self.env.reader.phase = "train"
        self.current_split = name

    def state(self, observation: dict) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.env.get_ground_truth_user_state(
                observation["user_profile"], observation["user_history"]
            ).squeeze(1)
            temper = (self.env.current_temper / self.initial_temper)[:, None]
            return torch.cat([encoded, temper], dim=1).detach()

    def uniform_user_context_rows(
        self,
        encoded_users: np.ndarray,
        contexts_per_user: int,
        seed: int,
    ) -> np.ndarray:
        """Select reader rows uniformly by user, then within user without outcome filtering."""
        generator = np.random.default_rng(seed)
        rows = []
        for encoded in np.asarray(encoded_users, dtype=np.int64):
            raw = self.encoded_to_raw_user[int(encoded)]
            candidates = np.asarray(self.env.reader.user_history[raw], dtype=np.int64)
            replace = len(candidates) < contexts_per_user
            rows.extend(
                generator.choice(candidates, size=contexts_per_user, replace=replace).tolist()
            )
        return np.asarray(rows, dtype=np.int64)

    def observation_from_rows(self, rows: np.ndarray) -> dict:
        """Build observations from raw reader-row identifiers in the active split."""
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) != self.batch_size:
            raise ValueError(f"probe observation requires exactly {self.batch_size} rows")
        if self.current_split is None:
            raise RuntimeError("use_split must be called before selecting fixed rows")
        split_rows = self.row_splits[self.current_split]
        positions = np.searchsorted(split_rows, rows)
        valid = positions < len(split_rows)
        if not np.all(valid) or not np.all(split_rows[positions] == rows):
            raise ValueError("requested reader row is not present in the active user split")
        samples = [self.env.reader[int(position)] for position in positions]
        return self.env.get_observation_from_batch(default_collate(samples))

    def install_probe_observation(self, observation: dict) -> None:
        """Install a fixed batch without constructing or advancing a DataLoader iterator."""
        self.env.current_observation = deepcopy(observation)
        self.env.current_temper = torch.full(
            (self.batch_size,), self.initial_temper, device=self.device
        )
        self.env.current_step = torch.zeros(self.batch_size, device=self.device)
        self.env.current_sum_reward = torch.zeros(self.batch_size, device=self.device)
        self.env.sample_batch = deepcopy(observation)
        self.env.current_sample_head_in_batch = 0
        self.env.env_history = {
            "step": [0.0],
            "leave": [],
            "temper": [],
            "coverage": [],
            "ILD": [],
        }

    def capture_probe_state(self) -> dict:
        """Capture every mutable field touched by a one-step environment transition.

        The DataLoader iterator is intentionally excluded. ``probe_one_step`` never
        consumes it, which makes restoration exact even when a sampled response
        would terminate a user.
        """
        names = (
            "current_observation",
            "current_temper",
            "current_step",
            "current_sum_reward",
            "sample_batch",
            "current_sample_head_in_batch",
            "env_history",
        )
        return {name: deepcopy(getattr(self.env, name)) for name in names}

    def restore_probe_state(self, snapshot: dict) -> None:
        for name, value in snapshot.items():
            setattr(self.env, name, deepcopy(value))

    def probe_one_step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample one response and successor without replacement-user side effects."""
        snapshot = self.capture_probe_state()
        try:
            with torch.no_grad():
                response = self.env.get_response({"action": action[:, None]})
                done = self.env.get_leave_signal(None, None, response)
                update = self.env.update_observation(
                    None,
                    action[:, None],
                    response["immediate_response"],
                    done,
                    update_current=False,
                )
                next_state = self.state(update["updated_observation"])
                reward = response["immediate_response"][:, 0, 0]
            return {
                "reward": reward.detach().clone(),
                "continued": (~done).float().detach().clone(),
                "next_state": next_state.detach().clone(),
                "immediate_response": response["immediate_response"].detach().clone(),
            }
        finally:
            self.restore_probe_state(snapshot)

    def probe_policy_rollout(
        self,
        observation: dict,
        first_action: torch.Tensor,
        action_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: int,
        discount: float,
    ) -> dict[str, torch.Tensor]:
        """Roll out a fixed policy from saved session states without user replacement.

        The supplied action is used at step zero and ``action_fn`` thereafter.
        A trajectory becomes absorbing when the native temper model terminates it;
        no DataLoader rows are consumed to replace terminated users.  The native
        environment state is restored after the call, while RNG advancement is
        intentionally retained so repeated calls are independent samples.
        """
        if len(first_action) != self.batch_size:
            raise ValueError("rollout requires one initial action per environment row")
        snapshot = self.capture_probe_state()
        try:
            self.install_probe_observation(observation)
            active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            discounted_return = torch.zeros(self.batch_size, device=self.device)
            rewards = torch.zeros(
                self.batch_size, int(horizon), device=self.device
            )
            survival = torch.zeros_like(rewards)
            first_next_state = None
            first_continued = None
            first_reward = None
            for step in range(int(horizon)):
                state = self.state(self.env.current_observation)
                action = first_action if step == 0 else action_fn(state)
                with torch.no_grad():
                    response = self.env.get_response({"action": action[:, None]})
                    done = self.env.get_leave_signal(None, None, response)
                    update = self.env.update_observation(
                        None,
                        action[:, None],
                        response["immediate_response"],
                        done,
                        update_current=True,
                    )
                    reward = response["immediate_response"][:, 0, 0] * active.float()
                    continued = active & (~done)
                    next_state = self.state(update["updated_observation"])
                rewards[:, step] = reward
                survival[:, step] = active.float()
                discounted_return += (float(discount) ** step) * reward
                if step == 0:
                    first_next_state = next_state.detach().clone()
                    first_continued = continued.float().detach().clone()
                    first_reward = reward.detach().clone()
                active = continued
                self.env.current_step += 1
                if not active.any():
                    break
            assert first_next_state is not None
            assert first_continued is not None
            assert first_reward is not None
            return {
                "discounted_return": discounted_return.detach().clone(),
                "reward_curve": rewards.detach().clone(),
                "survival_curve": survival.detach().clone(),
                "first_reward": first_reward,
                "first_continued": first_continued,
                "first_next_state": first_next_state,
                "final_continued": active.float().detach().clone(),
            }
        finally:
            self.restore_probe_state(snapshot)

    def _random_actions(self, observation: dict, generator: torch.Generator) -> torch.Tensor:
        actions = torch.randint(
            self.env.n_candidate, (self.batch_size,), generator=generator,
            device=self.device,
        )
        history = observation["user_history"]["history"]
        for _ in range(4):
            encoded = self.env.candidate_iids[actions]
            collision = (history == encoded[:, None]).any(dim=1)
            if not collision.any():
                break
            actions[collision] = torch.randint(
                self.env.n_candidate, (int(collision.sum()),), generator=generator,
                device=self.device,
            )
        return actions

    def random_actions(self, observation: dict, generator: torch.Generator) -> torch.Tensor:
        """Sample one catalog action per row, avoiding observed history when possible."""
        return self._random_actions(observation, generator)

    def forbidden_actions(self, observation: dict) -> list[torch.Tensor]:
        history = observation["user_history"]["history"]
        output = []
        for row in history:
            valid = row[(row > 0) & (row < len(self.encoded_to_candidate))]
            mapped = self.encoded_to_candidate[valid]
            output.append(torch.unique(mapped[mapped >= 0]))
        return output

    def reset(self, split: str, seed: int) -> dict:
        """Reset the native environment on a deterministic user split."""
        self.use_split(split)
        reset_rng(seed)
        return self.env.reset({"batch_size": self.batch_size, "empty_history": False})

    def collect(self, split: str, steps: int, seed: int, id_offset: int) -> Replay:
        self.use_split(split)
        reset_rng(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed + 991)
        observation = self.env.reset({"batch_size": self.batch_size, "empty_history": False})
        blocks: dict[str, list[np.ndarray]] = {
            name: [] for name in (
                "state", "action_feature", "action_index", "reward", "continued",
                "next_state", "user_id", "transition_id",
            )
        }
        for step in range(steps):
            state = self.state(observation)
            user_id = observation["user_profile"]["user_id"].detach().clone()
            action = self._random_actions(observation, generator)
            action_feature = self.catalog_features[action].detach()
            next_observation, response, _ = self.env.step({"action": action[:, None]})
            reward = response["immediate_response"][:, 0, 0]
            continued = ~response["done"]
            next_state = self.state(next_observation)
            identifiers = torch.arange(
                id_offset + step * self.batch_size,
                id_offset + (step + 1) * self.batch_size,
                device=self.device,
            )
            values = {
                "state": state,
                "action_feature": action_feature,
                "action_index": action,
                "reward": reward,
                "continued": continued,
                "next_state": next_state,
                "user_id": user_id,
                "transition_id": identifiers,
            }
            for name, value in values.items():
                blocks[name].append(value.detach().cpu().numpy())
            observation = next_observation
        arrays = {name: np.concatenate(value) for name, value in blocks.items()}
        return Replay(
            state=arrays["state"].astype(np.float32),
            action_feature=arrays["action_feature"].astype(np.float32),
            action_index=arrays["action_index"].astype(np.int32),
            reward=arrays["reward"].astype(np.float32),
            continued=arrays["continued"].astype(np.float32),
            next_state=arrays["next_state"].astype(np.float32),
            user_id=arrays["user_id"].astype(np.int32),
            transition_id=arrays["transition_id"].astype(np.int64),
        )

    def evaluate(
        self,
        split: str,
        steps: int,
        seed: int,
        discount: float,
        action_fn: Callable[[torch.Tensor, dict], torch.Tensor],
    ) -> dict:
        self.use_split(split)
        reset_rng(seed)
        observation = self.env.reset({"batch_size": self.batch_size, "empty_history": False})
        returns = torch.zeros(self.batch_size, device=self.device)
        lengths = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_users: list[int] = []
        step_rewards: list[float] = []
        for _ in range(steps):
            state = self.state(observation)
            users = observation["user_profile"]["user_id"].detach().clone()
            with torch.no_grad():
                action = action_fn(state, observation)
            next_observation, response, _ = self.env.step({"action": action[:, None]})
            reward = response["immediate_response"][:, 0, 0]
            returns += discount ** lengths * reward
            lengths += 1
            step_rewards.append(float(reward.mean()))
            done = response["done"]
            for value, length, user in zip(returns[done], lengths[done], users[done]):
                completed_returns.append(float(value))
                completed_lengths.append(int(length))
                completed_users.append(int(user))
            returns[done] = 0.0
            lengths[done] = 0
            observation = next_observation
        return {
            "completed_episode_count": len(completed_returns),
            "mean_discounted_return": float(np.mean(completed_returns)) if completed_returns else float("nan"),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float("nan"),
            "mean_step_reward": float(np.mean(step_rewards)),
            "episode_returns": completed_returns,
            "episode_lengths": completed_lengths,
            "episode_users": completed_users,
            "unfinished_streams_excluded": self.batch_size,
        }

    def evaluate_complete_context_batch(
        self,
        observation: dict,
        action_fn: Callable[[torch.Tensor, dict], torch.Tensor],
        horizon: int,
        discount: float,
    ) -> dict[str, torch.Tensor]:
        """Evaluate one fixed batch to native termination without user replacement.

        ``observation`` must contain exactly ``episode_batch_size`` rows.  Rewards
        after a row terminates are masked, so the returned undiscounted total is
        the complete-session metric rather than a fixed stream-window statistic.
        ``completed`` is explicit to prevent silently dropping long sessions if a
        future environment can exceed the declared horizon.
        """
        snapshot = self.capture_probe_state()
        try:
            self.install_probe_observation(observation)
            active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            completed = torch.zeros_like(active)
            total = torch.zeros(self.batch_size, device=self.device)
            discounted = torch.zeros_like(total)
            length = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            for step in range(int(horizon)):
                state = self.state(self.env.current_observation)
                with torch.no_grad():
                    action = action_fn(state, self.env.current_observation)
                    response = self.env.get_response({"action": action[:, None]})
                    done = self.env.get_leave_signal(None, None, response)
                    self.env.update_observation(
                        None,
                        action[:, None],
                        response["immediate_response"],
                        done,
                        update_current=True,
                    )
                    reward = response["immediate_response"][:, 0, 0] * active.float()
                total += reward
                discounted += (float(discount) ** step) * reward
                length += active.long()
                newly_done = active & done
                completed |= newly_done
                active &= ~done
                self.env.current_step += active.float()
                if not active.any():
                    break
            return {
                "total_reward": total.detach().clone(),
                "discounted_return": discounted.detach().clone(),
                "episode_length": length.detach().clone(),
                "completed": completed.detach().clone(),
            }
        finally:
            self.restore_probe_state(snapshot)
