"""CobuildDrywallEnv (v2): a robot arm assists a drywall installer covering a wall with 4 boards.

Worker, per board (whole boards skip measure and cut):
    prep -> [stool] -> [measure -> cut] -> (get the board to the wall) -> place -> screw_a -> reload_a
         -> screw_b -> reload_b
Heavy phases (carry, place, hold during the first screws) cost strain that depends on the worker's body.

Robot (ONE arm, one job at a time). Every tick it chooses one of 16 actions:
    0        wait
    1        BRING a board: stack -> cut station, stack -> wall staging, or cut station -> wall staging
    2        STEADY the board at the cut station while the worker measures and cuts
    3        LIFT_HOLD: lift the staged board onto the studs, align it, hold it during the first screws
    4..15    HAND a tool (cut tools, or a pre-loaded screw gun) at one of 12 poses (side x height x distance)

When the worker needs something the robot is not providing, the worker does it alone (slower and heavier).
Reactive mode (ask_for_help=True): the worker first asks for help (1 tick) and the robot only acts then.

This is a scheduling model with simple ergonomics, NOT a physics simulation. See docs/MODEL.md and
docs/RESEARCH.md for what is calibrated to published data and what is an assumption.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
import ergonomics as E
from profiles import GENERIC, WorkerProfile, sample_population

WAIT, BRING, STEADY, LIFT_HOLD, TOOL0 = 0, 1, 2, 3, 4
N_ACTIONS = TOOL0 + len(E.POSES)
ROBOT_STATES = ('free', 'bringing', 'positioning_steady', 'steadying', 'positioning_hold', 'holding',
                'fetching_tool', 'tool_ready')
LOCS = ('stack', 'cutstation', 'staged', 'held', 'wall')
PHASE_IDS = {n: i for i, n in enumerate(['prep', 'stool', 'tools_fetch', 'measure', 'cut', 'haul_stack_wall',
                                          'haul_stack_cut', 'haul_cut_wall', 'place', 'screw_a', 'reload_a',
                                          'screw_b', 'reload_b'])}
OBS_DIM = 43


def plan(b, P):
    """Phase names of board b for a worker with profile P (hauls are inserted on the fly)."""
    names = ['prep']
    if E.needs_stool(P, b):
        names.append('stool')
    if E.needs_cut(b):
        names += ['measure', 'cut']
    names += ['place', 'screw_a', 'reload_a', 'screw_b']
    if b < C.N_BOARDS - 1:
        names.append('reload_b')
    return names


def action_label(a):
    if a == WAIT:
        return 'wait'
    if a == BRING:
        return 'bring board'
    if a == STEADY:
        return 'steady board at cut station'
    if a == LIFT_HOLD:
        return 'lift & hold board'
    return 'hand tool @ ' + E.pose_label(a - TOOL0)


class CobuildDrywallEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(self, profile=None, observe_profile=True, robot_present=True, ask_for_help=False,
                 render_mode=None):
        """profile: a WorkerProfile, or None for a new random worker at every reset.
        observe_profile: False -> the agent 'sees' the generic average worker (no-profile ablation).
        robot_present: False -> the worker does everything alone (reference).
        ask_for_help: True -> reactive robot: the worker asks first and the robot only reacts."""
        super().__init__()
        self.fixed_profile = profile
        self.observe_profile = observe_profile
        self.robot_present = robot_present
        self.ask_for_help = ask_for_help
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.profile = GENERIC
        self._reset_state()

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        profile = options.get('profile', self.fixed_profile)
        self.profile = profile if isinstance(profile, WorkerProfile) else sample_population(self.np_random)
        self._reset_state()
        # The worker's variability is drawn once per episode, before any robot action: every strategy
        # evaluated with the same seed faces exactly the same worker ("common random numbers").
        self.noise = self.np_random.lognormal(0.0, C.DURATION_NOISE_SIGMA, size=(C.N_BOARDS, len(PHASE_IDS)))
        self._begin_board(0)
        if not self._try_start():
            self.interruptions += 1
        return self._observation(), self._info()

    def _reset_state(self):
        self.obs_profile = self.profile if self.observe_profile else GENERIC
        self.noise = np.ones((C.N_BOARDS, len(PHASE_IDS)))
        self.boards = [{'loc': 'stack', 'cut_done': not E.needs_cut(b)} for b in range(C.N_BOARDS)]
        self.board = 0
        self.queue = []
        self.phase, self.phase_mode = None, None
        self.remaining, self.elapsed, self.phase_total = 0, 0, 1
        self.worker_state = 'waiting'
        self.request = None
        self.asked = set()
        self.have_tool = None
        self.arrival = None   # (board, tick) of the last board the robot delivered to the wall staging
        self.robot_state, self.robot_ticks, self.robot_total = 'free', 0, 1
        self.robot_item, self.robot_pose, self.job = None, None, None
        self.time = self.wait_ticks = self.interruptions = 0
        self.strain = 0.0
        self.unnecessary = self.ignored = self.exposure = 0
        self.holds = self.brought = self.steadied = self.handovers = self.wrong_side = self.awkward = 0
        self.heavy_ticks = self.overhead_ticks = 0
        self.activity_ticks = {c: 0 for c in E.CATEGORIES}
        self.last_action, self.last_event = WAIT, 'start'
        self.done = False

    # -------------------------------------------------------------- properties
    @property
    def fatigue_multiplier(self):
        return 1.0 + C.FATIGUE_SCALE * self.profile.fatigue_rate * self.strain

    @property
    def cur(self):
        return self.boards[min(self.board, C.N_BOARDS - 1)]

    @property
    def activity(self):
        """Category the worker is in right now (used by the visualiser)."""
        if self.done:
            return 'done'
        if self.worker_state != 'working':
            return 'waiting'
        return E.CATEGORY[self.phase]

    # ----------------------------------------------------------------- stepping
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
        reward += self._robot_command(action)
        self._advance_robot()
        reward += self._advance_worker()
        if self.robot_state == 'tool_ready':
            self.exposure += 1
            reward += C.R_EXPOSURE
        reward += C.R_STRAIN * (self.strain - strain_before)
        terminated = self.done
        truncated = self.time >= C.MAX_TICKS and not terminated
        if terminated:
            reward += C.R_COMPLETE
        return self._observation(), float(reward), terminated, truncated, self._info()

    # ---------------------------------------------------------- what the robot can do now
    def bring_target(self):
        """(board index, destination) the robot could bring right now, or None."""
        for i in range(self.board, min(self.board + 2, C.N_BOARDS)):
            bd = self.boards[i]
            if bd['loc'] == 'stack':
                dest = 'cutstation' if E.needs_cut(i) else 'staged'
            elif bd['loc'] == 'cutstation' and bd['cut_done']:
                dest = 'staged'
            else:
                continue
            others = [o['loc'] for j, o in enumerate(self.boards) if j != i]
            if dest == 'cutstation' and 'cutstation' in others:
                continue
            return i, dest
        return None

    def can_steady(self):
        return (self.robot_state == 'free' and not self.done and E.needs_cut(self.board)
                and self.cur['loc'] == 'cutstation' and not self.cur['cut_done']
                and not (self.worker_state == 'working' and self.phase == 'cut'))

    def can_lift_hold(self):
        return (self.robot_state == 'free' and not self.done and self.cur['loc'] == 'staged'
                and 'place' in self.queue)

    def next_tool(self):
        """Tool the worker will need next ('cut_tools' | 'gun'), or None."""
        names = list(self.queue) + (plan(self.board + 1, self.profile) if self.board + 1 < C.N_BOARDS else [])
        for n in names:
            if n == 'measure' and self.have_tool != 'cut_tools':
                return 'cut_tools'
            if n.startswith('reload'):
                return 'gun'
        return None

    def action_masks(self):
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[WAIT] = True
        if self.robot_state != 'free' or self.done:
            return m
        m[BRING] = self.bring_target() is not None
        m[STEADY] = self.can_steady()
        m[LIFT_HOLD] = self.can_lift_hold()
        m[TOOL0:] = self.next_tool() is not None
        return m

    # ---------------------------------------------------------------- robot side
    def _robot_command(self, action):
        if action == WAIT:
            return 0.0
        if self.robot_state != 'free' or not self.robot_present:
            self.ignored += 1
            return C.R_IGNORED
        if action == BRING:
            tgt = self.bring_target()
            if tgt is None:
                self.unnecessary += 1
                return C.R_UNNECESSARY
            i, dest = tgt
            self.job = {'board': i, 'dest': dest}
            self.robot_state, self.robot_ticks = 'bringing', C.BRING_TICKS[(self.boards[i]['loc'], dest)]
            self.robot_total = self.robot_ticks
            self.brought += 1
            self.last_event = f'robot brings board {i + 1} to the {dest}'
        elif action == STEADY:
            if not self.can_steady():
                self.unnecessary += 1
                return C.R_UNNECESSARY
            self.robot_state, self.robot_ticks = 'positioning_steady', C.STEADY_SETUP_TICKS
            self.robot_total = self.robot_ticks
            self.job = {'board': self.board}
            self.steadied += 1
            self.last_event = 'robot steadies the board at the cut station'
        elif action == LIFT_HOLD:
            if not self.can_lift_hold():
                self.unnecessary += 1
                return C.R_UNNECESSARY
            self.robot_state, self.robot_ticks = 'positioning_hold', C.LIFT_HOLD_SETUP_TICKS
            self.robot_total = self.robot_ticks
            self.job = {'board': self.board}
            self.holds += 1
            self.last_event = f'robot lifts board {self.board + 1} onto the studs'
        else:
            item = self.next_tool()
            if item is None:
                self.unnecessary += 1
                return C.R_UNNECESSARY
            self.robot_item, self.robot_pose = item, action - TOOL0
            self.robot_state = 'fetching_tool'
            self.robot_ticks = C.FETCH_TOOL_TICKS + E.approach_ticks(self.robot_pose)
            self.robot_total = self.robot_ticks
            self.last_event = f'robot fetches the {item.replace("_", " ")}'
        return 0.0

    def _advance_robot(self):
        s = self.robot_state
        if s in ('bringing', 'positioning_steady', 'positioning_hold', 'fetching_tool'):
            self.robot_ticks -= 1
            if self.robot_ticks > 0:
                return
            if s == 'bringing':
                self.boards[self.job['board']]['loc'] = self.job['dest']
                if self.job['dest'] == 'staged':
                    self.arrival = (self.job['board'], self.time)
                self.robot_state, self.job = 'free', None
            elif s == 'positioning_steady':
                self.robot_state = 'steadying'
            elif s == 'positioning_hold':
                self.boards[self.job['board']]['loc'] = 'held'
                self.robot_state = 'holding'
            else:
                self.robot_state = 'tool_ready'
                self.last_event = f'robot presents the {self.robot_item.replace("_", " ")} at {E.pose_label(self.robot_pose)}'

    def _release_robot(self):
        self.robot_state, self.job, self.robot_item, self.robot_pose = 'free', None, None, None

    # --------------------------------------------------------------- worker side
    def _begin_board(self, b):
        self.board = b
        self.queue = plan(b, self.profile)
        self.asked.clear()

    def _advance_worker(self):
        if self.worker_state in ('waiting', 'requesting'):
            self.wait_ticks += 1
            self.activity_ticks['waiting'] += 1
            self._try_start()
            return C.R_WORKER_WAIT
        self.activity_ticks[E.CATEGORY[self.phase]] += 1
        self.heavy_ticks += E.is_heavy(self.phase, self.phase_mode)
        self.overhead_ticks += E.is_overhead(self.phase, self.board)
        self.remaining -= 1
        self.elapsed += 1
        if self.remaining > 0:
            return 0.0
        self._finish_phase()
        if not self._try_start():
            self.interruptions += 1
        return 0.0

    def _finish_phase(self):
        name, bd = self.phase, self.boards[self.board]
        if name == 'haul_stack_cut':
            bd['loc'] = 'cutstation'
        elif name.startswith('haul'):
            bd['loc'] = 'staged'
        elif name == 'tools_fetch':
            self.have_tool = 'cut_tools'
        elif name == 'cut':
            bd['cut_done'] = True
            if self.robot_state == 'steadying':
                # the robot already has the board in its grasp: it carries it to the wall
                self.job = {'board': self.board, 'dest': 'staged'}
                self.robot_state, self.robot_ticks = 'bringing', C.BRING_TICKS[('cutstation', 'staged')]
                self.robot_total = self.robot_ticks
                self.brought += 1
                self.last_event = 'robot carries the cut board to the wall'
            elif self.robot_state == 'positioning_steady':
                self._release_robot()
        elif name == 'place' and self.phase_mode != 'aligned':
            bd['loc'] = 'wall'
        if name == C.HOLD_RELEASE_AFTER and bd['loc'] == 'held':
            bd['loc'] = 'wall'
            self._release_robot()
        if name == 'screw_b':
            bd['loc'] = 'wall'
        self.phase = None
        if not self.queue:  # last phase of the board
            self.board += 1
            if self.board >= C.N_BOARDS:
                self.done, self.worker_state = True, 'done'
                self.last_event = 'wall finished'
            else:
                self._begin_board(self.board)

    def _try_start(self):
        """Start the next phase of the worker. Returns False if the worker has to wait."""
        while True:
            if self.done:
                return True
            res = self._resolve(self.queue[0])
            if res[0] == 'insert':
                self.queue.insert(0, res[1])
                continue
            if res[0] == 'wait':
                self.worker_state, self.request = 'waiting', None
                return False
            if res[0] == 'ask':
                self.worker_state, self.request = 'requesting', res[1]
                return False
            _, ticks, strain, mode = res
            name = self.queue.pop(0)
            self.phase, self.phase_mode = name, mode
            self.remaining, self.elapsed, self.worker_state, self.request = ticks, 0, 'working', None
            self.phase_total = ticks
            self.strain += strain
            self.asked.clear()
            self.last_event = f'worker: {name.replace("_", " ")} (board {self.board + 1})' + (
                ' with the robot' if mode in ('aligned', 'swap', 'handed', 'steady') else '')
            return True

    def _ok(self, name, mode='alone', extra_t=0.0, extra_s=0.0):
        key = 'reload' if name.startswith('reload') else name
        t, s = E.phase_cost(self.profile, self.board, key, 'free' if mode == 'handed' else mode)
        t = t * self.noise[self.board, PHASE_IDS[name]] * self.fatigue_multiplier
        return 'ok', max(1, int(round(t + extra_t))), s + extra_s, mode

    def _can_ask(self, key):
        if self.robot_present and self.ask_for_help and key not in self.asked:
            self.asked.add(key)
            return True
        return False

    def _resolve_tool(self, item, name):
        if self.robot_state == 'tool_ready' and self.robot_item == item:
            x_t, x_s, d = E.handover_cost(self.profile, self.robot_pose)
            self.handovers += 1
            self.wrong_side += d['wrong_side']
            self.awkward += int(E.is_awkward(x_t, x_s))
            self._release_robot()
            return self._ok(name, 'swap' if name.startswith('reload') else 'handed', x_t, x_s)
        if self.robot_state == 'fetching_tool' and self.robot_item == item:
            return ('wait',)
        if self._can_ask(item):
            return ('ask', item)
        if item == 'cut_tools':
            if self.have_tool != 'cut_tools':
                return ('insert', 'tools_fetch')
            self.have_tool = None
            if self.robot_state == 'tool_ready' and self.robot_item == 'cut_tools':
                self._release_robot()
        return self._ok(name, 'alone')

    def _resolve(self, name):
        bd, b = self.boards[self.board], self.board
        if name in ('prep', 'stool', 'screw_b', 'tools_fetch') or name.startswith('haul'):
            return self._ok(name)
        if name == 'screw_a':
            return self._ok(name, 'free' if bd['loc'] == 'held' else 'tack')
        if name == 'measure':
            return self._resolve_tool('cut_tools', name)
        if name.startswith('reload'):
            return self._resolve_tool('gun', name)
        brought_here = self.robot_state == 'bringing' and self.job['board'] == b
        if name == 'cut':
            if bd['loc'] == 'stack':
                if brought_here and self.job['dest'] == 'cutstation':
                    return ('wait',)
                if self._can_ask('board_cut'):
                    return ('ask', 'board_cut')
                return ('insert', 'haul_stack_cut')
            if self.robot_state == 'positioning_steady':
                return ('wait',)
            return self._ok(name, 'steady' if self.robot_state == 'steadying' else 'plain')
        if name == 'place':
            if bd['loc'] == 'held':
                return self._ok(name, 'aligned')
            if bd['loc'] in ('stack', 'cutstation'):
                if brought_here and self.job['dest'] == 'staged':
                    return ('wait',)
                if self._can_ask('board_wall'):
                    return ('ask', 'board_wall')
                return ('insert', 'haul_stack_wall' if bd['loc'] == 'stack' else 'haul_cut_wall')
            if self.robot_state == 'positioning_hold':
                return ('wait',)
            if self.robot_present and self.arrival == (b, self.time):
                return ('wait',)   # reaction tick: the board has just been delivered, the robot may hold it
            if self._can_ask('lift_hold'):
                return ('ask', 'lift_hold')
            return self._ok(name, 'alone')
        raise KeyError(name)

    # ----------------------------------------------- expectations (what a planner may know)
    def _exp_duration(self, P, b, name, active=False):
        key = 'reload' if name.startswith('reload') else name
        mode = 'alone'
        if key == 'screw_a':
            mode = 'free' if (b == self.board and self.boards[b]['loc'] == 'held') else 'tack'
        elif key == 'place' and b == self.board and self.boards[b]['loc'] == 'held':
            mode = 'aligned'
        elif key == 'cut':
            mode = 'plain'
        if key.startswith('haul'):
            return E.phase_cost(P, b, key, 'alone')[0]
        return E.phase_cost(P, b, key, mode)[0]

    def time_to(self, P, pred):
        """Expected ticks until the worker STARTS the first phase satisfying pred(board, name),
        planning with the profile P. Returns (ticks, board, name) or None."""
        if self.done:
            return None
        t = 0.0
        if self.worker_state == 'working':
            t = max(0.0, self._exp_duration(P, self.board, self.phase) - self.elapsed)
        for b in range(self.board, C.N_BOARDS):
            names = list(self.queue) if b == self.board else plan(b, P)
            for name in names:
                if pred(b, name):
                    return t, b, name
                t += self._exp_duration(P, b, name)
        return None

    # -------------------------------------------------------------- observation
    def _observation(self):
        P = self.obs_profile
        b = min(self.board, C.N_BOARDS - 1)
        act = self.activity
        cat = [float(act == c) for c in E.CATEGORIES]
        loc = [float(self.cur['loc'] == l and not self.done) for l in LOCS]
        nxt = self.boards[b + 1]['loc'] != 'stack' if b + 1 < C.N_BOARDS else False
        robot = [float(self.robot_state == s) for s in ROBOT_STATES]
        tool = self.next_tool()

        def until(pred):
            r = self.time_to(P, pred)
            return 1.0 if r is None else r[0] / 40.0

        obs = [
            self.board / C.N_BOARDS, *cat,
            float(E.needs_cut(b)), float(E.is_upper(b)), *loc,
            float(nxt), float(b + 1 < C.N_BOARDS),
            *robot, self.robot_ticks / 6.0,
            until(lambda bb, n: n == 'cut'), until(lambda bb, n: n == 'place'),
            until(lambda bb, n: n == 'measure' or n.startswith('reload')),
            float(tool == 'gun'), float(tool == 'cut_tools'),
            float(self.can_lift_hold()), float(self.can_steady()), float(self.bring_target() is not None),
            (P.height_m - 1.5) / 0.5, (P.reach_m - 0.5) / 0.3, (P.lift_capacity_kg - 15.0) / 40.0,
            (P.pace - 0.7) / 0.7, float(P.side), P.fatigue_rate / 0.002,
            E.load_ratio(P, b) / 2.0, float(E.hold_is_worth_it(P, b)),
        ]
        assert len(obs) == OBS_DIM, len(obs)
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _info(self):
        m = C.TICK_SECONDS / 60.0
        return {
            'ticks': self.time, 'minutes': self.time * m,
            'worker_wait_ticks': self.wait_ticks, 'interruptions': self.interruptions,
            'strain': round(self.strain, 3), 'heavy_ticks': self.heavy_ticks,
            'overhead_ticks': self.overhead_ticks,
            'unnecessary_commands': self.unnecessary, 'ignored_commands': self.ignored,
            'robot_exposure_ticks': self.exposure, 'holds': self.holds, 'boards_brought': self.brought,
            'steadied': self.steadied, 'handovers': self.handovers, 'wrong_side_handovers': self.wrong_side,
            'awkward_handovers': self.awkward, 'boards_installed': min(self.board, C.N_BOARDS),
            'final_fatigue': round(self.fatigue_multiplier, 3), 'activity_ticks': dict(self.activity_ticks),
            'profile': self.profile.name,
        }

    def render(self):
        return (f't={self.time:3d} board={min(self.board + 1, C.N_BOARDS)}/{C.N_BOARDS} '
                f'worker={self.activity:8s} rem={self.remaining:2d} robot={self.robot_state:18s} '
                f'act={action_label(self.last_action):34s} wait={self.wait_ticks:3d} strain={self.strain:5.1f}'
                f' | {self.last_event}')
