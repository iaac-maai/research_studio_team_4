"""CobuildDrywallEnv: a robot assistant supplying a drywall installer.

The worker covers a wall with N_BOARDS boards. For every board they:
    measure -> cut (needs a BOARD) -> lift & position -> fasten (needs SCREWS)

Every tick (≈ 15 s) the robot chooses one of 14 actions:
    0        wait
    1..12    fetch the next item and hand it over at pose k-1 (side x height x distance)
    13       hold the board against the wall during the next lift ("support")

The robot carries one item at a time and must hold it until the worker takes
it (a direct handover). The worker only takes an item when they start the
step that needs it. Durations depend on the worker's PHYSICAL profile
(ergonomics.py), on random variability, and on accumulated fatigue.

This is a scheduling model with simple ergonomics, NOT a physics simulation.
See docs/MODEL.md for a full explanation.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
import ergonomics as E
from profiles import GENERIC, WorkerProfile, sample_population

WAIT = 0
SUPPORT = 1 + len(E.POSES)
N_ACTIONS = SUPPORT + 1
ROBOT_STATES = ('free', 'fetching', 'holding', 'positioning', 'supporting')
ITEM_SEQUENCE = [(b, item) for b in range(C.N_BOARDS) for item in ('board', 'screws')]
ITEM_STEP = {'board': 'cut', 'screws': 'fasten'}


def action_label(action):
    if action == WAIT:
        return 'wait'
    if action == SUPPORT:
        return 'hold board during lift'
    return 'deliver @ ' + E.pose_label(action - 1)


class CobuildDrywallEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(self, profile=None, observe_profile=True, render_mode=None):
        """profile: a WorkerProfile, or None to draw a new random worker at every reset.
        observe_profile: if False the robot always 'sees' the generic average worker
        (used to train the no-profile ablation with the exact same algorithm)."""
        super().__init__()
        self.fixed_profile = profile
        self.observe_profile = observe_profile
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(27,), dtype=np.float32)
        self.profile = GENERIC
        self._reset_state()

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        profile = options.get('profile', self.fixed_profile)
        self.profile = profile if isinstance(profile, WorkerProfile) else sample_population(self.np_random)
        self._reset_state()
        # Worker variability is drawn ONCE per episode, before any robot action, so that
        # every strategy evaluated with the same seed faces exactly the same worker
        # ("common random numbers" -> a fair, paired comparison).
        self.noise = self.np_random.lognormal(0.0, C.DURATION_NOISE_SIGMA, size=(C.N_BOARDS, len(C.STEPS)))
        self._try_start_step()
        return self._observation(), self._info()

    def _reset_state(self):
        self.obs_profile = self.profile if self.observe_profile else GENERIC
        self.noise = np.ones((C.N_BOARDS, len(C.STEPS)))
        # worker
        self.board, self.step_idx = 0, 0
        self.worker_state = 'waiting'
        self.remaining, self.elapsed = 0, 0
        self.lift_supported = False
        # robot
        self.robot_state, self.robot_ticks = 'free', 0
        self.robot_item, self.robot_pose = None, None
        self.next_item_idx = 0
        # metrics
        self.time = 0
        self.wait_ticks = 0
        self.interruptions = 0
        self.strain = 0.0
        self.unnecessary = 0
        self.ignored = 0
        self.exposure = 0
        self.supports = 0
        self.handovers = 0
        self.wrong_side = 0
        self.awkward = 0
        self.last_action = WAIT
        self.last_event = 'start'
        self.done = False

    # ----------------------------------------------------------- properties
    @property
    def step_name(self):
        return C.STEPS[self.step_idx] if not self.done else 'done'

    @property
    def fatigue_multiplier(self):
        return 1.0 + C.FATIGUE_SCALE * self.profile.fatigue_rate * self.strain

    @property
    def next_item(self):
        """(board, item) the robot would fetch next, or None when everything was delivered."""
        return ITEM_SEQUENCE[self.next_item_idx] if self.next_item_idx < len(ITEM_SEQUENCE) else None

    # ------------------------------------------------------------- dynamics
    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f'Unknown action {action}')
        if self.done:
            raise RuntimeError('Episode is over, call reset()')
        self.last_action = action
        self.time += 1
        reward = C.R_TICK
        strain_before = self.strain

        reward += self._apply_robot_command(action)
        self._advance_robot()
        reward += self._advance_worker()
        if self.robot_state == 'holding':  # still hovering in the worker's space, item not taken
            self.exposure += 1
            reward += C.R_EXPOSURE
        reward += C.R_STRAIN * (self.strain - strain_before)

        terminated = self.done
        truncated = self.time >= C.MAX_TICKS and not terminated
        if terminated:
            reward += C.R_COMPLETE
        return self._observation(), float(reward), terminated, truncated, self._info()

    def _apply_robot_command(self, action):
        if action == WAIT:
            return 0.0
        if self.robot_state != 'free':
            self.ignored += 1
            return C.R_IGNORED
        if action == SUPPORT:
            if self.support_available():
                self.robot_state, self.robot_ticks = 'positioning', C.SUPPORT_SETUP_TICKS
                self.supports += 1
                self.last_event = f'robot moves to hold board {self.board + 1}'
                return 0.0
            self.unnecessary += 1
            return C.R_UNNECESSARY
        if self.next_item is None:
            self.unnecessary += 1
            return C.R_UNNECESSARY
        pose = E.POSES[action - 1]
        self.robot_item, self.robot_pose = self.next_item, action - 1
        self.next_item_idx += 1
        self.robot_state = 'fetching'
        self.robot_ticks = C.FETCH_TICKS[self.robot_item[1]] + E.robot_approach_ticks(pose)
        self.last_event = f'robot fetches {self.robot_item[1]} {self.robot_item[0] + 1}'
        return 0.0

    def support_available(self):
        return (self.robot_state == 'free' and self.step_name == 'cut'
                and self.worker_state == 'working')

    def _advance_robot(self):
        if self.robot_state == 'fetching':
            self.robot_ticks -= 1
            if self.robot_ticks == 0:
                self.robot_state = 'holding'
                self.last_event = f'robot holds {self.robot_item[1]} at {E.pose_label(self.robot_pose)}'
        elif self.robot_state == 'positioning':
            self.robot_ticks -= 1
            if self.robot_ticks == 0:
                self.robot_state = 'supporting'

    def _advance_worker(self):
        if self.worker_state == 'waiting':
            self.wait_ticks += 1
            self._try_start_step()
            return C.R_WORKER_WAIT
        self.remaining -= 1
        self.elapsed += 1
        if self.remaining > 0:
            return 0.0
        # step finished
        if self.step_name == 'lift' and self.lift_supported:
            self.robot_state = 'free'  # release the board, robot is free again
        self.step_idx += 1
        if self.step_idx == len(C.STEPS):
            self.step_idx = 0
            self.board += 1
            if self.board == C.N_BOARDS:
                self.done = True
                self.worker_state = 'done'
                self.last_event = 'wall finished'
                return 0.0
        if not self._try_start_step():
            self.interruptions += 1
        return 0.0

    def _try_start_step(self):
        """Start the current step if its requirements are met; otherwise the worker waits."""
        step, b, p = self.step_name, self.board, self.profile
        extra_ticks, extra_strain = 0, 0.0
        need = C.NEEDS[step]
        if need is not None:
            if not (self.robot_state == 'holding' and self.robot_item == (b, need)):
                self.worker_state = 'waiting'
                self.last_event = f'worker waits for {need}'
                return False
            extra_ticks, extra_strain, d = E.handover_cost(p, E.POSES[self.robot_pose], need)
            self.handovers += 1
            self.wrong_side += d['wrong_side']
            self.awkward += int(E.is_awkward(extra_ticks, extra_strain))
            self.robot_state, self.robot_item = 'free', None
        supported = False
        if step == 'lift':
            if self.robot_state == 'positioning':
                self.worker_state = 'waiting'
                self.last_event = 'worker waits for the robot to hold the board'
                return False
            supported = self.robot_state == 'supporting'
        base_ticks, base_strain = E.step_base(p, b, step, supported)
        duration = base_ticks * self.noise[b, self.step_idx] * self.fatigue_multiplier
        self.remaining = max(1, int(round(duration))) + extra_ticks
        self.elapsed = 0
        self.lift_supported = supported
        self.strain += base_strain + extra_strain
        self.worker_state = 'working'
        self.last_event = f'worker starts {step} (board {b + 1})' + (' with robot support' if supported else '')
        return True

    # ------------------------------------------------- expectations (no noise)
    def expected_remaining(self, profile, supported_lift=None):
        """Expected ticks left in the current step, using `profile` (what the robot believes)."""
        if self.done:
            return 0.0
        if self.worker_state == 'waiting':
            return self.expected_duration(profile, self.board, self.step_name, supported_lift)
        if self.step_name == 'lift':
            supported_lift = self.lift_supported
        return max(0.0, self.expected_duration(profile, self.board, self.step_name, supported_lift) - self.elapsed)

    def expected_duration(self, profile, board, step, supported_lift=None):
        if step == 'lift' and supported_lift is None:
            supported_lift = self.robot_state in ('positioning', 'supporting') and board == self.board
        return E.step_base(profile, board, step, bool(supported_lift))[0]

    def expected_until_needed(self, profile, item=None, supported_lift=None):
        """Expected ticks until the worker starts the step that needs `item` (default: next item)."""
        item = item or self.next_item
        if item is None or self.done:
            return None
        target_board, target_step = item[0], C.STEPS.index(ITEM_STEP[item[1]])
        if (self.board, self.step_idx) == (target_board, target_step):
            return 0.0  # needed now (worker is about to start / waiting)
        t = self.expected_remaining(profile, supported_lift)
        b, s = self.board, self.step_idx + 1
        while True:
            if s == len(C.STEPS):
                b, s = b + 1, 0
            if (b, s) == (target_board, target_step):
                return t
            t += self.expected_duration(profile, b, C.STEPS[s], supported_lift if b == self.board else False)
            s += 1

    # --------------------------------------------------------- observation
    def _observation(self):
        P = self.obs_profile
        step_onehot = [float(self.step_idx == i and not self.done) for i in range(len(C.STEPS))]
        robot_onehot = [float(self.robot_state == s) for s in ROBOT_STATES]
        nxt = self.next_item
        until = self.expected_until_needed(P)
        lift_board = min(self.board if self.step_idx <= C.STEPS.index('lift') else self.board + 1, C.N_BOARDS - 1)
        lift_ticks, lift_strain = E.lift_unsupported(P, lift_board)
        obs = [
            self.board / C.N_BOARDS,
            *step_onehot,
            float(self.worker_state == 'waiting'),
            float(E.is_upper(min(self.board, C.N_BOARDS - 1))),
            self.elapsed / 20.0,
            self.expected_remaining(P) / 20.0,
            1.0 if until is None else until / 40.0,
            *robot_onehot,
            self.robot_ticks / 5.0,
            float(nxt is not None and nxt[1] == 'board'),
            float(nxt is not None and nxt[1] == 'screws'),
            float(nxt is not None and nxt[0] == self.board),
            # the worker profile (or the generic one in the no-profile ablation)
            (P.height_m - 1.5) / 0.5,
            (P.reach_m - 0.5) / 0.3,
            (P.lift_capacity_kg - 15.0) / 40.0,
            (P.pace - 0.7) / 0.7,
            float(P.side),
            P.fatigue_rate / 0.01,
            # what the profile implies for the next lift
            lift_ticks * P.pace / 15.0,
            lift_strain / 10.0,
        ]
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _info(self):
        return {
            'ticks': self.time,
            'minutes': self.time * C.TICK_SECONDS / 60.0,
            'worker_wait_ticks': self.wait_ticks,
            'interruptions': self.interruptions,
            'strain': round(self.strain, 3),
            'unnecessary_deliveries': self.unnecessary,
            'ignored_commands': self.ignored,
            'robot_exposure_ticks': self.exposure,
            'supported_lifts': self.supports,
            'handovers': self.handovers,
            'wrong_side_handovers': self.wrong_side,
            'awkward_handovers': self.awkward,
            'boards_installed': self.board,
            'final_fatigue': round(self.fatigue_multiplier, 3),
            'profile': self.profile.name,
        }

    def render(self):
        return (f't={self.time:3d} board={min(self.board + 1, C.N_BOARDS)}/{C.N_BOARDS} '
                f'step={self.step_name:7s} worker={self.worker_state:8s} rem={self.remaining:2d} '
                f'robot={self.robot_state:10s} action={action_label(self.last_action):28s} '
                f'wait={self.wait_ticks} strain={self.strain:.1f} | {self.last_event}')
