"""The 3D choreography must not draw things through each other (see audit_3d.py)."""
import pytest

pytest.importorskip('mujoco')

import audit_3d  # noqa: E402
from profiles import PERSONAS  # noqa: E402

# Largest tolerated overlap (metres) per kind of contact: brief grazes of a few centimetres between drawn shapes.
SOFT = {'worker-robot': 0.15, 'worker-board': 0.10, 'robot-board': 0.07, 'board-board': 0.06,
        'floor': 0.03, 'robot-static': 0.08, 'board-static': 0.05}


@pytest.mark.parametrize('strategy', ['no_robot', 'B_proactive_generic', 'C_cobuild_rule'])
def test_nothing_goes_through_the_floor_or_the_set(strategy):
    _, stats = audit_3d.run(strategy, PERSONAS['short_left'])
    worst = audit_3d.summarise(stats)
    for kind in ('floor', 'robot-static', 'board-static'):
        assert worst.get(kind, (0, 0.0))[1] < SOFT[kind], f'{strategy}: {kind} overlap {worst[kind]}'


@pytest.mark.parametrize('strategy', ['B_proactive_generic', 'C_cobuild_rule'])
def test_worker_robot_and_boards_stay_clear(strategy):
    _, stats = audit_3d.run(strategy, PERSONAS['low_capacity_short_left'])
    for kind, (frames, depth) in audit_3d.summarise(stats).items():
        assert depth < SOFT.get(kind, 0.0), f'{strategy}: {kind} overlaps by {depth:.2f} m in {frames} frames'


@pytest.mark.parametrize('strategy', ['B_proactive_generic', 'C_cobuild_rule', 'no_robot'])
def test_video_never_shows_robot_in_worker_or_boards_moving_alone(strategy):
    """The two rules of audit_video.py, frame by frame, on a whole wall."""
    import audit_video
    events = audit_video.audit(strategy, 'expert_tall_right')
    assert not events, events[:5]
