"""The assistance strategies we compare.

    A  reactive               - brings material only once the worker is waiting for it
    B  proactive, generic     - anticipates the procedure, tuned for the AVERAGE worker
    C  Cobuild rule           - the same planner as B, given the INDIVIDUAL worker profile
    C  Cobuild PPO            - learned policy (PPO) that observes the worker profile
    ablation: PPO no profile  - same PPO training, but the profile in the observation is
                                replaced by the generic worker (does learning alone explain gains?)

B and the Cobuild rule run *the same code*; the only difference is which profile
they are given. So B vs C-rule isolates the value of the profile itself.

The rule-based strategies read the environment's task state (current step,
ticks elapsed, robot state) - the same information the PPO agent observes -
but NOT the hidden random variability of the worker. They plan with expected
durations computed from the profile they were given.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import config as C
import ergonomics as E
from cobuild_env import SUPPORT, WAIT
from profiles import GENERIC

MODELS_DIR = Path(__file__).with_name('models')
SAFETY_MARGIN_TICKS = 2  # leave this early to absorb variability in the worker's pace


def deliver(pose_idx):
    return 1 + pose_idx


def reactive_action(env, profile=GENERIC):
    """A: wait until the worker stops and asks, then fetch the item (generic pose)."""
    if env.robot_state != 'free' or env.done:
        return WAIT
    nxt = env.next_item
    if nxt is not None and env.worker_state == 'waiting' and env.expected_until_needed(profile) == 0:
        return deliver(E.best_pose(profile, nxt[1]))
    return WAIT


def planner_action(env, profile):
    """B / C-rule: proactive just-in-time planner using `profile`.

    1. Decide whether to hold the current board during its lift (support_is_worth_it).
       If yes, commit as soon as the worker starts cutting, and fetch the screws after the lift.
    2. Otherwise leave to fetch the next item when the expected time until the worker
       needs it is about to drop below the fetch time (+ a safety margin).
    3. Always hand over at the pose that is cheapest for `profile` (side, height, distance).
    """
    if env.robot_state != 'free' or env.done:
        return WAIT
    b = env.board
    nxt = env.next_item
    plan_support = E.support_is_worth_it(profile, b)
    if plan_support and env.step_idx <= C.STEPS.index('lift'):
        if env.support_available():
            return SUPPORT
        if nxt == (b, 'screws'):
            return WAIT  # keep the robot free to hold the board; screws come after the lift
    if nxt is None:
        return WAIT
    pose = E.best_pose(profile, nxt[1])
    lead = C.FETCH_TICKS[nxt[1]] + E.robot_approach_ticks(E.POSES[pose])
    until = env.expected_until_needed(profile, supported_lift=False)
    if until <= lead + SAFETY_MARGIN_TICKS:
        return deliver(pose)
    return WAIT


@dataclass
class Strategy:
    key: str
    label: str
    uses_profile: bool
    observe_profile: bool  # passed to CobuildDrywallEnv (only matters for PPO)
    act: Callable  # (env, obs) -> action


def _ppo(name):
    path = MODELS_DIR / f'{name}.zip'
    if not path.exists():
        raise SystemExit(f'No trained model at {path}. Run: python train.py --variant {name.replace("cobuild_ppo_", "")}')
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device='cpu')
    return lambda env, obs: int(model.predict(obs, deterministic=True)[0])


STRATEGY_KEYS = ['A_reactive', 'B_proactive_generic', 'C_cobuild_rule', 'C_cobuild_ppo', 'ablation_ppo_no_profile']
LABELS = {
    'A_reactive': 'A · Reactive',
    'B_proactive_generic': 'B · Proactive (average worker)',
    'C_cobuild_rule': 'C · Cobuild rule (profile)',
    'C_cobuild_ppo': 'C · Cobuild PPO (profile)',
    'ablation_ppo_no_profile': 'PPO without profile (ablation)',
}


def make_strategy(key):
    if key == 'A_reactive':
        return Strategy(key, LABELS[key], False, True, lambda env, obs: reactive_action(env))
    if key == 'B_proactive_generic':
        return Strategy(key, LABELS[key], False, True, lambda env, obs: planner_action(env, GENERIC))
    if key == 'C_cobuild_rule':
        return Strategy(key, LABELS[key], True, True, lambda env, obs: planner_action(env, env.profile))
    if key == 'C_cobuild_ppo':
        return Strategy(key, LABELS[key], True, True, _ppo('cobuild_ppo_profile'))
    if key == 'ablation_ppo_no_profile':
        return Strategy(key, LABELS[key], False, False, _ppo('cobuild_ppo_no_profile'))
    raise KeyError(key)
