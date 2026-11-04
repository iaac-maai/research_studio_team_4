"""Sanity tests for the Cobuild drywall simulation.   Run:  python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

import config as C
import ergonomics as E
from cobuild_env import N_ACTIONS, SUPPORT, WAIT, CobuildDrywallEnv
from profiles import GENERIC, PERSONAS
from strategies import planner_action, reactive_action


def rollout(profile, seed, policy):
    env = CobuildDrywallEnv(profile=profile)
    env.reset(seed=seed)
    while True:
        _, _, terminated, truncated, info = env.step(policy(env))
        if terminated or truncated:
            return info, terminated


def test_gymnasium_and_sb3_api():
    check_env(CobuildDrywallEnv(), warn=True)
    check_env(CobuildDrywallEnv(observe_profile=False), warn=True)


def test_observations_stay_in_bounds_under_random_actions():
    env = CobuildDrywallEnv()
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        obs, _, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_same_seed_same_worker_variability():
    """Common random numbers: the worker's variability does not depend on the strategy."""
    a, b = CobuildDrywallEnv(PERSONAS['short_left']), CobuildDrywallEnv(PERSONAS['short_left'])
    a.reset(seed=7), b.reset(seed=7)
    assert np.array_equal(a.noise, b.noise)
    a.reset(seed=8)
    assert not np.array_equal(a.noise, b.noise)


@pytest.mark.parametrize('name', list(PERSONAS))
def test_rule_strategies_finish_the_wall(name):
    p = PERSONAS[name]
    for policy in (lambda e: reactive_action(e), lambda e: planner_action(e, GENERIC), lambda e: planner_action(e, p)):
        info, done = rollout(p, 3, policy)
        assert done and info['boards_installed'] == C.N_BOARDS
        assert info['unnecessary_deliveries'] == 0 and info['ignored_commands'] == 0


def test_always_wait_never_finishes():
    info, done = rollout(GENERIC, 0, lambda e: WAIT)
    assert not done and info['ticks'] == C.MAX_TICKS


def test_reactive_waits_for_every_item():
    info, _ = rollout(GENERIC, 0, reactive_action)
    assert info['interruptions'] == 2 * C.N_BOARDS


def test_profile_planner_equals_B_for_the_average_worker():
    """B and the Cobuild rule are the same code; with the generic worker they must coincide."""
    for seed in range(5):
        b, _ = rollout(GENERIC, seed, lambda e: planner_action(e, GENERIC))
        c, _ = rollout(GENERIC, seed, lambda e: planner_action(e, e.profile))
        assert b == c


def test_best_pose_matches_the_body():
    assert C.HANDOVER_SIDES[E.POSES[E.best_pose(PERSONAS['short_left'], 'board')][0]] == 'left'
    assert E.POSES[E.best_pose(PERSONAS['short_left'], 'board')][1] == min(C.HANDOVER_HEIGHTS_M)
    assert E.POSES[E.best_pose(PERSONAS['expert_tall_right'], 'board')][1] == max(C.HANDOVER_HEIGHTS_M)


def test_support_pays_off_only_for_some_workers():
    assert E.support_is_worth_it(PERSONAS['low_capacity_short_left'], 0)       # upper row, weak, short
    assert not E.support_is_worth_it(PERSONAS['expert_tall_right'], 0)
    assert not E.support_is_worth_it(PERSONAS['low_capacity_short_left'], 3)   # lower row


def test_invalid_support_is_counted_as_unnecessary():
    env = CobuildDrywallEnv(GENERIC)
    env.reset(seed=0)
    env.step(SUPPORT)  # worker is measuring, nothing to hold yet
    assert env.unnecessary == 1
