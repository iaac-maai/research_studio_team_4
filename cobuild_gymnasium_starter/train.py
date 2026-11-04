"""Train the Cobuild PPO agent(s) with action masking (sb3-contrib MaskablePPO).

    python train.py                         # trains both variants (profile + no-profile ablation)
    python train.py --variant profile       # only the Cobuild agent that observes the worker profile
    python train.py --timesteps 500000      # quicker, less converged run

Every episode draws a NEW random worker (profiles.sample_population), so the policy must learn how
body measurements and preferences map to good decisions, not memorise named personas.
Action masking removes commands that make no sense in the current state (e.g. "hold the board" when
no board is staged), which lets the agent discover rarely useful actions such as holding the board.
"""
import argparse
import time

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from cobuild_env import CobuildDrywallEnv
from strategies import MODELS_DIR


class ProgressPrinter(BaseCallback):
    """One readable line every `every` steps."""

    def __init__(self, every):
        super().__init__()
        self.every, self.next_print, self.t0 = every, every, time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next_print:
            self.next_print += self.every
            eps = list(self.model.ep_info_buffer)
            if eps:
                print(f'  {self.num_timesteps:>9,d} steps | mean episode reward {np.mean([e["r"] for e in eps]):7.1f}'
                      f' | mean episode length {np.mean([e["l"] for e in eps]):6.1f} ticks'
                      f' | {time.time() - self.t0:5.0f} s', flush=True)
        return True


def train(variant, timesteps, seed):
    observe_profile = variant == 'profile'
    venv = make_vec_env(CobuildDrywallEnv, n_envs=8, seed=seed, env_kwargs={'observe_profile': observe_profile})
    model = MaskablePPO('MlpPolicy', venv, seed=seed, verbose=0, device='cpu',
                        n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4,
                        gamma=0.995, gae_lambda=0.95, ent_coef=0.01,
                        policy_kwargs={'net_arch': [128, 128]})
    print(f'\nTraining MaskablePPO ({variant}) for {timesteps:,d} steps, seed {seed}', flush=True)
    model.learn(total_timesteps=timesteps, callback=ProgressPrinter(max(timesteps // 15, 1)))
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f'cobuild_ppo_{variant}'
    model.save(str(path))
    print(f'Saved {path}.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=['profile', 'no_profile', 'both'], default='both')
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for v in (['profile', 'no_profile'] if args.variant == 'both' else [args.variant]):
        train(v, args.timesteps, args.seed)
