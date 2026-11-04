"""Worker profiles: who the robot is assisting.

A profile mixes work preferences (pace, preferred handover side) with
PHYSICAL characteristics (stature, comfortable reach, lifting capacity,
fatigue susceptibility). ergonomics.py turns these traits into time and
strain for each task step and handover pose.

All personas below are SYNTHETIC. They are meant to span the range of
workers a crew might have, not to represent real people. Traits are kept
neutral on purpose (no age or gender labels): the robot only needs body
measurements and preferences, which is also what a real Cobuild profile
would store.
"""
from dataclasses import dataclass, asdict, replace

import numpy as np

import config as C


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    height_m: float          # stature
    reach_m: float           # comfortable forward reach (shoulder to grip), without stepping
    lift_capacity_kg: float  # load the worker can lift and hold overhead comfortably
    pace: float              # multiplier on task durations: 1.0 = average, 0.8 = 20 % faster
    handedness: str          # 'left' or 'right': preferred side to receive material
    fatigue_rate: float      # each strain point makes later steps this much slower
    description: str = ''

    # Derived body measurements (anthropometric ratios, see config.py)
    @property
    def side(self):
        return C.HANDOVER_SIDES.index(self.handedness)

    @property
    def comfort_handover_m(self):
        return C.COMFORT_HANDOVER_RATIO * self.height_m

    @property
    def overhead_reach_m(self):
        return C.OVERHEAD_REACH_RATIO * self.height_m

    @property
    def load_ratio(self):
        return C.BOARD_WEIGHT_KG / self.lift_capacity_kg

    def as_dict(self):
        return asdict(self)


# The "average worker" a non-personalised robot is designed for: population medians
# of the distribution in sample_population(), right-handed (the majority).
GENERIC = WorkerProfile('generic_average', height_m=1.75, reach_m=0.665, lift_capacity_kg=34.0,
                        pace=1.0, handedness='right', fatigue_rate=0.004,
                        description='Population-average worker; what robots A and B are tuned for.')

PERSONAS = {p.name: p for p in [
    WorkerProfile('expert_tall_right', 1.88, 0.72, 45.0, 0.80, 'right', 0.002,
                  'Experienced, tall and strong. Close to what a generic robot expects, but faster.'),
    WorkerProfile('expert_average_left', 1.75, 0.66, 38.0, 0.85, 'left', 0.003,
                  'Experienced, average body, left-handed.'),
    WorkerProfile('novice_average_right', 1.74, 0.65, 34.0, 1.30, 'right', 0.005,
                  'Apprentice: average body, slower.'),
    WorkerProfile('short_left', 1.58, 0.56, 28.0, 1.00, 'left', 0.005,
                  'Short stature, short reach, left-handed. Overhead work is hard.'),
    WorkerProfile('short_strong_right', 1.62, 0.58, 42.0, 0.90, 'right', 0.003,
                  'Short but strong and fast: height matters, weight does not.'),
    WorkerProfile('tall_novice_left', 1.93, 0.75, 40.0, 1.25, 'left', 0.004,
                  'Tall, long reach, left-handed apprentice.'),
    WorkerProfile('low_capacity_right', 1.70, 0.63, 22.0, 0.95, 'right', 0.008,
                  'Reduced lifting capacity and tires quickly (e.g. returning from a back injury).'),
    WorkerProfile('low_capacity_short_left', 1.60, 0.57, 24.0, 1.10, 'left', 0.007,
                  'Short, left-handed, reduced lifting capacity: the hardest case for a generic robot.'),
]}


def sample_population(rng, name='sampled'):
    """Draw a random worker from a synthetic population (used for training and for
    the 'unseen workers' evaluation). Handedness is 50/50 so that the learner sees
    both sides equally often; reach is correlated with stature."""
    height = float(np.clip(rng.normal(1.75, 0.08), 1.55, 1.95))
    reach = float(np.clip(0.38 * height + rng.normal(0.0, 0.03), 0.52, 0.78))
    capacity = float(np.clip(rng.normal(34.0, 7.0), 20.0, 50.0))
    pace = float(rng.uniform(0.80, 1.30))
    handedness = 'left' if rng.random() < 0.5 else 'right'
    fatigue = float(rng.uniform(0.002, 0.008))
    return WorkerProfile(name, round(height, 3), round(reach, 3), round(capacity, 1),
                         round(pace, 3), handedness, round(fatigue, 4))


def get_profile(name):
    if name == 'generic_average':
        return GENERIC
    return PERSONAS[name]


def renamed(profile, name):
    return replace(profile, name=name)
