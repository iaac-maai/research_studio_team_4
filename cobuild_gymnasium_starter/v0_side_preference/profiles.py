"""Named synthetic worker profiles shared by training, evaluation, and visualization.

pace: ticks of installation work per drywall component (lower = faster worker)
side: preferred material handover side, 0 = left, 1 = right

These are the only two traits CobuildEnv actually models. The research
proposal also mentions comfortable reaching distance; it is not wired into
the environment yet (see README "Next steps"), so it is intentionally left
out here rather than added as an unused, decorative field.
"""

PROFILES = {
    'fast_left': {'pace': 3, 'side': 0},
    'fast_right': {'pace': 3, 'side': 1},
    'slow_left': {'pace': 5, 'side': 0},
    'slow_right': {'pace': 5, 'side': 1},
}


def profile_list():
    return list(PROFILES.values())
