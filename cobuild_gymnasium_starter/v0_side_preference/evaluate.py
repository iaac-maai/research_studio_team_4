"""Compare one learned policy and two transparent rule-based baselines.
All synthetic times are timesteps, NOT real construction minutes.
"""
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from cobuild_env import CobuildEnv
from profiles import profile_list
from baselines import STRATEGIES

MODEL = Path(__file__).with_name('cobuild_ppo.zip')
if not MODEL.exists():
    raise SystemExit('Run python train.py first')
model = PPO.load(str(MODEL))
profiles = profile_list()


def trial(profile, strategy):
    env = CobuildEnv(profile=profile)
    obs, _ = env.reset(seed=42)
    rule = STRATEGIES.get(strategy)
    while True:
        if rule is not None:
            a = rule(env, profile)
        else:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
        obs, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            return info


if __name__ == '__main__':
    print('Strategy          Mean steps   Mean idle steps   Wrong-side deliveries')
    for name in ['reactive', 'fixed_proactive', 'profile_rule', 'PPO_learned']:
        runs = [trial(p, name) for p in profiles]
        print(f'{name:18s}{np.mean([r["steps"] for r in runs]):10.1f}'
              f'{np.mean([r["worker_idle_steps"] for r in runs]):18.1f}'
              f'{np.mean([r["wrong_side_deliveries"] for r in runs]):23.1f}')
        for p, r in zip(profiles, runs):
            print(f'  worker pace={p["pace"]}, side={"left" if p["side"] == 0 else "right"}: '
                  f'{r["steps"]} steps, {r["worker_idle_steps"]} idle')
