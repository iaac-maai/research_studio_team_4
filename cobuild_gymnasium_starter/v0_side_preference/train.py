"""Train a single policy across different simulated worker profiles."""
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from cobuild_env import CobuildEnv

check_env(CobuildEnv(), warn=True)
model = PPO('MlpPolicy', CobuildEnv(), verbose=1, seed=42,
            n_steps=256, batch_size=64, learning_rate=0.0003)
model.learn(total_timesteps=30_000)
output = Path(__file__).with_name('cobuild_ppo')
model.save(str(output))
print(f'\nSaved trained policy to {output}.zip')
