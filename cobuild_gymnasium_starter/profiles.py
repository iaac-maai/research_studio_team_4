"""Worker profiles: preferences (pace, handedness) + physical characteristics.

Traits and what they change (details in docs/MODEL.md):
    height_m          comfortable tool-handover height, and whether the upper row is within overhead reach
    reach_m           how far a tool can be presented without the worker stepping or leaning
    lift_capacity_kg  load ratio of a 34 kg board: speed and strain of lifting, carrying, placing, holding
    pace              multiplier on every task duration (1.0 average, 0.8 = 20 % faster)
    handedness        side on which the worker prefers to take tools (screw gun)
    fatigue_rate      each strain point makes later steps this fraction slower

All personas are SYNTHETIC. The published drywall field studies (docs/RESEARCH.md §6) include almost
only right-handed men, so short, left-handed or low-capacity personas are hypotheses, not data.
"""
from dataclasses import dataclass, asdict, replace

import numpy as np

import config as C


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    height_m: float
    reach_m: float
    lift_capacity_kg: float
    pace: float
    handedness: str
    fatigue_rate: float
    description: str = ''

    @property
    def side(self):
        return C.HANDOVER_SIDES.index(self.handedness)

    @property
    def comfort_handover_m(self):
        return C.COMFORT_HANDOVER_RATIO * self.height_m

    @property
    def overhead_reach_m(self):
        return C.OVERHEAD_REACH_RATIO * self.height_m

    def as_dict(self):
        return asdict(self)


# The "average worker" that a non-personalised robot is designed for.
GENERIC = WorkerProfile('generic_average', 1.75, 0.665, 34.0, 1.0, 'right', 0.0008,
                        'Population-average worker; what robots A and B are tuned for.')

PERSONAS = {p.name: p for p in [
    WorkerProfile('expert_tall_right', 1.88, 0.72, 45.0, 0.80, 'right', 0.0004,
                  'Experienced, tall and strong, close to what a generic robot expects but faster.'),
    WorkerProfile('expert_average_left', 1.75, 0.66, 38.0, 0.85, 'left', 0.0006,
                  'Experienced, average body, left-handed.'),
    WorkerProfile('novice_average_right', 1.74, 0.65, 34.0, 1.30, 'right', 0.0010,
                  'Apprentice: average body, slower.'),
    WorkerProfile('short_left', 1.58, 0.56, 28.0, 1.00, 'left', 0.0010,
                  'Short stature and reach, left-handed. The upper row is hard to reach.'),
    WorkerProfile('short_strong_right', 1.62, 0.58, 42.0, 0.90, 'right', 0.0006,
                  'Short but strong and fast: height matters, weight does not.'),
    WorkerProfile('tall_novice_left', 1.93, 0.75, 40.0, 1.25, 'left', 0.0008,
                  'Tall, long reach, left-handed apprentice.'),
    WorkerProfile('low_capacity_right', 1.70, 0.63, 22.0, 0.95, 'right', 0.0016,
                  'Reduced lifting capacity, tires quickly (e.g. returning from a back injury).'),
    WorkerProfile('low_capacity_short_left', 1.60, 0.57, 24.0, 1.10, 'left', 0.0014,
                  'Short, left-handed, reduced lifting capacity: the hardest case for a generic robot.'),
]}


def sample_population(rng, name='sampled'):
    """Random synthetic worker (training and the 'unseen workers' evaluation). Handedness 50/50 so the
    learner sees both sides equally often; reach is correlated with stature."""
    height = float(np.clip(rng.normal(1.75, 0.08), 1.55, 1.95))
    reach = float(np.clip(0.38 * height + rng.normal(0.0, 0.03), 0.52, 0.78))
    capacity = float(np.clip(rng.normal(34.0, 7.0), 20.0, 50.0))
    pace = float(rng.uniform(0.80, 1.30))
    handedness = 'left' if rng.random() < 0.5 else 'right'
    fatigue = float(rng.uniform(0.0004, 0.0016))
    return WorkerProfile(name, round(height, 3), round(reach, 3), round(capacity, 1),
                         round(pace, 3), handedness, round(fatigue, 5))


def get_profile(name):
    return GENERIC if name == 'generic_average' else PERSONAS[name]
