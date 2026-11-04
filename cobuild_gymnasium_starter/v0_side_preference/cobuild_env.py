"""Cobuild v0: scheduling-only reinforcement-learning experiment.

This is NOT a physics simulation, and its human behavior is synthetic.
Actions: 0=wait, 1=send material to left, 2=send material to right.
Each worker has a pace and handover-side profile.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CobuildEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, n_parts=6, profile=None):
        super().__init__()
        self.n_parts = n_parts
        self.fixed_profile = profile
        self.action_space = spaces.Discrete(3)
        # [time_remaining, ready, delivery_in_transit, side, pace, parts_remaining]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self.reset()

    def _observation(self):
        return np.array([
            self.remaining / 6.0,
            float(self.material_ready),
            self.transit / 3.0,
            float(self.side),
            self.pace / 6.0,
            (self.n_parts - self.completed) / self.n_parts,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_profile is None:
            self.pace = int(self.np_random.choice([3, 5]))
            self.side = int(self.np_random.choice([0, 1])) # 0 left, 1 right
        else:
            self.pace = int(self.fixed_profile['pace'])
            self.side = int(self.fixed_profile['side'])
        self.remaining = self.pace
        self.completed = 0
        self.material_ready = False
        self.transit = 0
        self.time = 0
        self.idle = 0
        self.mistakes = 0
        return self._observation(), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action {action}")
        reward = -1.0  # minimize total time
        self.time += 1

        if action in (1, 2):
            if self.material_ready or self.transit:
                reward -= 1.0  # duplicate delivery attempt
            else:
                selected_side = action - 1
                self.transit = 1 if selected_side == self.side else 3
                if selected_side != self.side:
                    self.mistakes += 1
                    reward -= 1.0

        # A tick elapses. A delivery may arrive while work is in progress.
        if self.transit > 0:
            self.transit -= 1
            if self.transit == 0:
                self.material_ready = True

        if self.remaining > 0:
            self.remaining -= 1

        if self.remaining == 0:
            if self.material_ready:
                self.completed += 1
                self.material_ready = False
                if self.completed < self.n_parts:
                    self.remaining = self.pace
            else:
                self.idle += 1
                reward -= 1.0

        terminated = self.completed >= self.n_parts
        truncated = self.time >= 100 and not terminated
        if terminated:
            reward += 5.0
        info = {
            'steps': self.time,
            'worker_idle_steps': self.idle,
            'wrong_side_deliveries': self.mistakes,
            'completed_parts': self.completed,
            'profile': {'pace': self.pace, 'side': self.side},
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        return (f"tick={self.time} completed={self.completed}/{self.n_parts} "
                f"work_remaining={self.remaining} material_ready={self.material_ready} "
                f"transit={self.transit} idle={self.idle}")
