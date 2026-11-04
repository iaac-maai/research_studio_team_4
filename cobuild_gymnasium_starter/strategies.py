"""The assistance strategies we compare.

    0  no robot               - the worker does everything alone (reference for "how much work is saved")
    A  reactive               - the worker asks, the robot reacts; generic handover pose
    B  proactive, generic     - just-in-time planner tuned for the AVERAGE worker
    C  Cobuild rule           - the SAME planner as B, given the INDIVIDUAL worker profile
    C  Cobuild PPO            - learned policy that observes the worker profile
    ablation: PPO no profile  - the same PPO training with the profile replaced by the generic worker

B and the Cobuild rule run the same code; the only difference is which profile they are given, so
B vs C-rule isolates the value of the profile. PPO vs PPO-no-profile does the same for the learner.
The rule-based planners see the task state (current phase, ticks elapsed, robot state), the same
information PPO observes, but NOT the hidden random variability of the worker.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import config as C
import ergonomics as E
from cobuild_env import BRING, LIFT_HOLD, STEADY, TOOL0, WAIT
from profiles import GENERIC

MODELS_DIR = Path(__file__).with_name('models')
SAFETY_MARGIN_TICKS = 3   # start this much earlier than strictly needed, to absorb variability


def reactive_action(env):
    """A: acts only when the worker has asked. Always presents tools at the generic pose."""
    if env.robot_state != 'free' or env.done or env.worker_state != 'requesting':
        return WAIT
    if env.request in ('cut_tools', 'gun'):
        return TOOL0 + E.best_tool_pose(GENERIC)
    if env.request in ('board_cut', 'board_wall'):
        return BRING
    if env.request == 'lift_hold':
        return LIFT_HOLD
    return WAIT


def planner_action(env, P):
    """B / C-rule: proactive just-in-time planner that believes the worker looks like P.

    1. Hold the board as soon as it is staged, if holding pays off for P (ergonomics.hold_is_worth_it).
    2. Bring the next board so that it arrives (plus alignment time) just before it is needed.
    3. Hand the next tool so that it arrives just before the worker needs it, at the pose that is
       cheapest for P (side, height, distance).
    4. Steady the board at the cut station whenever there is nothing more urgent.
    """
    if env.robot_state != 'free' or env.done:
        return WAIT
    b = env.board
    if E.hold_is_worth_it(P, b) and env.can_lift_hold():
        return LIFT_HOLD
    tgt = env.bring_target()
    if tgt is not None:
        i, dest = tgt
        r = env.time_to(P, lambda bb, n: bb == i and n == ('cut' if dest == 'cutstation' else 'place'))
        lead = C.BRING_TICKS[(env.boards[i]['loc'], dest)] + SAFETY_MARGIN_TICKS
        if dest == 'staged' and E.hold_is_worth_it(P, i):
            lead += C.LIFT_HOLD_SETUP_TICKS
        if r is not None and r[0] <= lead:
            return BRING
    if env.next_tool() is not None:
        pose = E.best_tool_pose(P)
        r = env.time_to(P, lambda bb, n: n == 'measure' or n.startswith('reload'))
        lead = C.FETCH_TOOL_TICKS + E.approach_ticks(pose) + SAFETY_MARGIN_TICKS
        if r is not None and r[0] <= lead:
            return TOOL0 + pose
    if env.can_steady():
        return STEADY
    return WAIT


@dataclass
class Strategy:
    key: str
    label: str
    observe_profile: bool = True
    robot_present: bool = True
    asks: bool = False
    act: Callable = None  # (env, obs) -> action

    def env_kwargs(self):
        return dict(observe_profile=self.observe_profile, robot_present=self.robot_present, ask_for_help=self.asks)


def _ppo(variant):
    path = MODELS_DIR / f'cobuild_ppo_{variant}.zip'
    if not path.exists():
        raise SystemExit(f'No trained model at {path}. Run: python train.py --variant {variant}')
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(path), device='cpu')
    return lambda env, obs: int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])


STRATEGY_KEYS = ['no_robot', 'A_reactive', 'B_proactive_generic', 'C_cobuild_rule', 'C_cobuild_ppo',
                 'ablation_ppo_no_profile']
LABELS = {
    'no_robot': '0 · No robot',
    'A_reactive': 'A · Reactive',
    'B_proactive_generic': 'B · Proactive (average worker)',
    'C_cobuild_rule': 'C · Cobuild rule (profile)',
    'C_cobuild_ppo': 'C · Cobuild PPO (profile)',
    'ablation_ppo_no_profile': 'PPO without profile (ablation)',
}


def make_strategy(key):
    label = LABELS[key]
    if key == 'no_robot':
        return Strategy(key, label, robot_present=False, act=lambda env, obs: WAIT)
    if key == 'A_reactive':
        return Strategy(key, label, asks=True, act=lambda env, obs: reactive_action(env))
    if key == 'B_proactive_generic':
        return Strategy(key, label, act=lambda env, obs: planner_action(env, GENERIC))
    if key == 'C_cobuild_rule':
        return Strategy(key, label, act=lambda env, obs: planner_action(env, env.profile))
    if key == 'C_cobuild_ppo':
        return Strategy(key, label, act=_ppo('profile'))
    if key == 'ablation_ppo_no_profile':
        return Strategy(key, label, observe_profile=False, act=_ppo('no_profile'))
    raise KeyError(key)
