"""The floor-plan layer (geometry.py + the safety rule in CobuildDrywallEnv)."""
import numpy as np

import geometry as G
from cobuild_env import OBS_DIM, CobuildDrywallEnv
from profiles import PERSONAS
from strategies import make_strategy


def run(strategy_key, profile, seed):
    st = make_strategy(strategy_key)
    env = CobuildDrywallEnv(profile=profile, **st.env_kwargs())
    obs, info = env.reset(seed=seed)
    trace = []
    while True:
        obs, r, term, trunc, info = env.step(st.act(env, obs))
        trace.append((env.worker_pos.copy(), env.robot_pos.copy(), env.robot_state))
        if term or trunc:
            return env, info, trace


def test_no_robot_means_no_safety_events():
    for P in PERSONAS.values():
        _, info, _ = run('no_robot', P, 1)
        assert info['safety_stops'] == 0 and info['contacts'] == 0


def test_observation_includes_the_floor_plan():
    env = CobuildDrywallEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and np.all((obs >= 0) & (obs <= 1))


def test_spots_are_clear_of_obstacles_and_far_from_the_worker():
    for b in range(4):
        for spot in (G.spot_hold(b), G.spot_for_board('staged'), G.spot_for_board('cutstation')):
            assert np.allclose(spot, G.push_out(spot, G.ROBOT_RADIUS + 0.1))
        assert np.linalg.norm(G.spot_hold(b) - G.station('wall_right', b)) > G.ZONE      # worker works from the right


def test_planners_keep_the_robot_away_from_the_worker():
    """Planners may be stopped a few times, but bodies must (almost) never touch: separation >= ZONE."""
    for key in ('B_proactive_generic', 'C_cobuild_rule'):
        contacts = 0
        for name in ('short_left', 'expert_tall_right', 'low_capacity_right'):
            for seed in range(5):
                _, info, trace = run(key, PERSONAS[name], seed)
                contacts += info['contacts']
                if info['contacts'] == 0:
                    assert all(np.linalg.norm(w - r) >= G.ZONE - 1e-6 for w, r, _ in trace)
        assert contacts <= 20      # rare corner: a left-side handover next to the pallet (low_capacity_right)


def test_a_blocked_robot_stops_and_is_penalised():
    """Put the worker on the robot's route: the safety rule must stop someone and charge the penalty."""
    env = CobuildDrywallEnv(profile=PERSONAS['short_left'])
    env.reset(seed=0)
    env.robot_pos = np.array([1.0, -0.9])          # parked robot right next to the worker's wall station
    env.robot_state = 'holding'
    before = env.safety_stops
    _, reward, *_ = env.step(0)
    assert env.safety_stops == before + 1
    assert reward < -3
