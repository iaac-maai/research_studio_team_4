"""Tests for the Cobuild drywall simulation (v2).   Run:  python -m pytest -q tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import config as C
import ergonomics as E
from cobuild_env import BRING, LIFT_HOLD, N_ACTIONS, OBS_DIM, STEADY, TOOL0, WAIT, CobuildDrywallEnv
from profiles import GENERIC, PERSONAS
from strategies import make_strategy, planner_action, reactive_action


def rollout(profile, seed, strategy_key):
    s = make_strategy(strategy_key)
    env = CobuildDrywallEnv(profile=profile, **s.env_kwargs())
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, te, tr, info = env.step(s.act(env, obs))
        if te or tr:
            return info, te


def test_gymnasium_and_sb3_api():
    check_env(CobuildDrywallEnv(), warn=True)
    check_env(CobuildDrywallEnv(observe_profile=False), warn=True)


def test_observations_and_masks_under_random_valid_actions():
    env = CobuildDrywallEnv()
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    assert obs.shape == (OBS_DIM,)
    for _ in range(3000):
        mask = env.action_masks()
        assert mask[WAIT]
        obs, _, te, tr, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert env.observation_space.contains(obs)
        if te or tr:
            assert info['unnecessary_commands'] == 0       # masked actions are never invalid
            obs, _ = env.reset()


def test_same_seed_same_worker_variability():
    """Common random numbers: the worker's variability does not depend on the strategy."""
    a, b = CobuildDrywallEnv(PERSONAS['short_left']), CobuildDrywallEnv(PERSONAS['short_left'])
    a.reset(seed=7), b.reset(seed=7)
    assert np.array_equal(a.noise, b.noise)
    a.reset(seed=8)
    assert not np.array_equal(a.noise, b.noise)


@pytest.mark.parametrize('name', ['generic_average', *PERSONAS])
@pytest.mark.parametrize('key', ['no_robot', 'A_reactive', 'B_proactive_generic', 'C_cobuild_rule'])
def test_every_strategy_finishes_the_wall(name, key):
    p = GENERIC if name == 'generic_average' else PERSONAS[name]
    info, done = rollout(p, 3, key)
    assert done and info['boards_installed'] == C.N_BOARDS
    assert info['unnecessary_commands'] == 0 and info['ignored_commands'] == 0


def test_always_wait_with_a_robot_is_just_the_worker_alone():
    """A robot that never acts must leave the worker doing everything alone, and no_robot equals it."""
    env = CobuildDrywallEnv(profile=GENERIC)
    env.reset(seed=1)
    while True:
        _, _, te, tr, info = env.step(WAIT)
        if te or tr:
            break
    alone, _ = rollout(GENERIC, 1, 'no_robot')
    assert info['minutes'] == alone['minutes'] and info['strain'] == alone['strain']


def test_profile_planner_equals_B_for_the_average_worker():
    """B and the Cobuild rule are the same code; with the generic worker they must coincide."""
    for seed in range(5):
        b, _ = rollout(GENERIC, seed, 'B_proactive_generic')
        c, _ = rollout(GENERIC, seed, 'C_cobuild_rule')
        assert b == c


def test_robot_reduces_heavy_handling_and_strain():
    for name in ('short_left', 'low_capacity_right', 'generic_average'):
        p = GENERIC if name == 'generic_average' else PERSONAS[name]
        alone, _ = rollout(p, 5, 'no_robot')
        robot, _ = rollout(p, 5, 'B_proactive_generic')
        assert robot['strain'] < alone['strain'] and robot['heavy_ticks'] < alone['heavy_ticks']


def test_reactive_robot_makes_the_worker_wait():
    info, _ = rollout(GENERIC, 0, 'A_reactive')
    assert info['worker_wait_ticks'] > 20 and info['interruptions'] > 8


def test_calibration_against_published_field_data():
    """The average worker alone: ~4.8 min/board (Yuan 2016: 100 pieces per 8 h) and activity shares close to
    Yuan's Table 1 (cut/measure 19.8 %, lift+carry 19.1 %, screw 15.9 %, reload 19.1 %, other 21.4 %)."""
    mins, acc = [], {}
    for seed in range(60):
        info, _ = rollout(GENERIC, seed, 'no_robot')
        mins.append(info['minutes'] / C.N_BOARDS)
        for k, v in info['activity_ticks'].items():
            acc[k] = acc.get(k, 0) + v
    assert 4.3 < np.mean(mins) < 5.6
    tot = sum(acc.values())
    share = {k: 100 * v / tot for k, v in acc.items()}
    assert abs(share['measure'] + share['cut'] - 19.8) < 5
    assert abs(share['haul'] - 19.1) < 5
    assert abs(share['screw'] - 15.9) < 5
    assert abs(share['reload'] - 19.1) < 5
    assert abs(share['prep'] - 21.4) < 6


def test_holding_pays_off_only_for_some_workers():
    weak, strong = PERSONAS['low_capacity_short_left'], PERSONAS['expert_tall_right']
    assert all(E.hold_is_worth_it(weak, b) for b in range(C.N_BOARDS))
    assert not any(E.hold_is_worth_it(strong, b) for b in range(C.N_BOARDS))


def test_best_tool_pose_matches_the_body():
    short_left = PERSONAS['short_left']
    side, h, _ = E.POSES[E.best_tool_pose(short_left)]
    assert C.HANDOVER_SIDES[side] == 'left' and h == min(C.HANDOVER_HEIGHTS_M)
    assert E.POSES[E.best_tool_pose(PERSONAS['expert_tall_right'])][1] == max(C.HANDOVER_HEIGHTS_M)


def test_invalid_commands_are_counted():
    env = CobuildDrywallEnv(GENERIC)
    env.reset(seed=0)
    env.step(LIFT_HOLD)            # nothing is staged yet
    env.step(STEADY)               # no board at the cut station
    assert env.unnecessary == 2


def test_the_robot_does_one_thing_at_a_time():
    env = CobuildDrywallEnv(GENERIC)
    env.reset(seed=0)
    env.step(BRING)
    env.step(BRING)                # still busy bringing the first board
    assert env.ignored == 1 and env.brought == 1
