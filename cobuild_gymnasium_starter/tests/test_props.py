"""Loose objects in the 3D: a released tool obeys gravity and contact; nothing floats or teleports."""
import numpy as np
import pytest

pytest.importorskip('mujoco')

import geometry as G  # noqa: E402
import props  # noqa: E402
import visualize_3d as V  # noqa: E402
from profiles import PERSONAS  # noqa: E402


def test_a_released_tool_falls_and_rests_on_the_cart():
    w = props.PropWorld(['K0', 'K1'])
    w.release('K0', [G.CART_XY[0] + 0.3, G.CART_XY[1] - 0.1, 0.9], [1, 0, 0, 0])
    heights = []
    for _ in range(40):
        w.step()
        heights.append(w.pose('K0')[0][2])
    assert heights[0] < 0.9 and heights[1] < heights[0]            # it falls
    assert abs(heights[-1] - (0.12 + props.TOOL_BOX[2])) < 0.01     # and stops on the cart top
    assert w.resting('K0')


def test_a_tool_nobody_released_does_not_move():
    w = props.PropWorld(['K0', 'K1'])
    for _ in range(10):
        w.step()
    assert w.pose('K1')[0][2] == -9 and not w.active


def test_no_prop_teleports_and_tools_end_up_on_the_cart():
    view = V.View('C_cobuild_rule', PERSONAS['short_left'], 1000, 64, 48)
    ck, last, worst = view.checker, {}, 0.0
    props_ = ['board0_g', 'board1_g', 'board2_g', 'board3_g', 'tK0_g', 'tK1_g', 'tG0_g', 'tG1_g']
    while not view.done:
        a, b = view.step()
        for f in range(1, 9):
            view.compose(a, b, f / 8)
            for p in props_:
                pos = view.scene.data.geom_xpos[ck.id[p]].copy()
                if pos[2] < -4:
                    last.pop(p, None)
                    continue
                if p in last:
                    worst = max(worst, float(np.linalg.norm(pos - last[p])))
                last[p] = pos
    assert worst < 0.9                                             # < ~1.2 m/s of simulated time
    assert view.freed == {'K0', 'K1'}
    for n in view.freed:
        assert abs(view.world.pose(n)[0][2] - 0.15) < 0.02
