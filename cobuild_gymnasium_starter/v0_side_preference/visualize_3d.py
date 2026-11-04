"""3D view: watch a Cobuild strategy drive a Panda arm handing drywall panels to a worker.

The decisions (wait / deliver left / deliver right) come from exactly the same
abstract CobuildEnv and strategies used in evaluate.py. Each abstract tick is
then acted out in robosuite/MuJoCo: the arm picks a panel and carries it to the
chosen side, while the worker prepares each slot on the wall and takes the
panel when they need it.

The reported metrics (ticks, worker waiting, wrong-side deliveries) come from
CobuildEnv. How long each 3D motion takes is illustrative only.

Usage:
    mjpython visualize_3d.py --profile fast_left --strategy ppo          # interactive window (macOS needs mjpython)
    python visualize_3d.py --profile slow_right --video demo.mp4         # render to a video file instead
"""
import argparse
import time

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T

import scene_3d as S
from baselines import STRATEGIES, choose_action, load_ppo
from cobuild_env import CobuildEnv
from profiles import PROFILES

CONTROL_DT = 1.0 / 20
MIN_TICK_STEPS = 16
MAX_ARM_SPEED = 0.6  # fraction of the controller's 5 cm/step limit
HOME = np.array([*S.STACK_XY, 1.02])
VIA = np.array([0.12, 0.0, 1.15])
SIDE_NAME = {0: 'LEFT', 1: 'RIGHT'}
STRATEGY_LABELS = {
    'ppo': 'Cobuild: PPO agent using the worker profile',
    'reactive': 'A. Reactive: delivers only when the worker asks',
    'fixed_proactive': 'B. Proactive, not personalised (always RIGHT)',
    'profile_rule': 'Hand-written personalised rule (reference)',
}


def snapshot(env):
    return dict(transit=env.transit, ready=env.material_ready, completed=env.completed,
                idle=env.idle, remaining=env.remaining, time=env.time, mistakes=env.mistakes)


class Director:
    """Turns abstract CobuildEnv ticks into arm, worker, and panel motion in the 3D scene."""

    def __init__(self, scene, env, strategy, on_frame):
        self.scene = scene
        self.env = env
        self.strategy = strategy
        self.on_frame = on_frame
        self.pref = env.side
        self.gripper_down = scene.eef_mat()
        self.arm_target = HOME.copy()
        self.gripper = -1.0
        self.piece_holder = 'stack'
        self.piece_offset = np.zeros(3)
        self.hand_pos = {s: S.SHOULDER[s] + np.array([0.02, 0.0, -0.45]) for s in (0, 1)}
        self.hand_override = {}
        self.worker_mode = 'work'
        self.installed = 0
        self.progress = 0.0
        self.progress_target = 0.0
        self.unnecessary = 0
        self.robot_status = 'waiting at the material stack'
        self.shown = snapshot(env)  # counters on the HUD only update once a tick's animation ends
        self.steps = 0
        for side in (0, 1):
            self.scene.set_rgba(f'handover_{side}', [0.2, 0.8, 0.3, 0.6] if side == self.pref else [0.5, 0.5, 0.5, 0.25])

    # --- per-simulation-step update ---------------------------------------------------

    def step(self):
        eef = self.scene.eef_pos()
        action = np.zeros(self.scene.action_dim)
        action[:3] = np.clip((self.arm_target - eef) / 0.05, -MAX_ARM_SPEED, MAX_ARM_SPEED)
        # keep the gripper pointing down: rotate back toward the orientation it had at start
        rot_err = T.quat2axisangle(T.mat2quat(self.gripper_down @ self.scene.eef_mat().T))
        action[3:6] = np.clip(rot_err / 0.5, -1, 1)
        action[-1] = self.gripper
        self.scene.step(action)
        self.steps += 1
        self._update_worker()
        self._update_piece()
        self.on_frame(self)

    def _update_worker(self):
        slot = S.SLOTS[min(self.installed, len(S.SLOTS) - 1)]
        t = self.steps * CONTROL_DT
        for side in (0, 1):
            if side in self.hand_override:
                target = self.hand_override[side]
            elif self.worker_mode == 'wait' and side == self.pref:
                target = S.HANDOVER[side] + np.array([0.06, 0.0, 0.0])
            elif self.worker_mode == 'wait' or self.env.completed >= self.env.n_parts:
                target = S.SHOULDER[side] + np.array([0.02, 0.06 * (1 - 2 * side), -0.45])
            else:
                wobble = 0.035 * np.sin(2 * np.pi * 1.2 * t + side * np.pi)
                target = slot + np.array([-0.05, 0.07 * (1 - 2 * side), wobble])
            self.hand_pos[side] = self.hand_pos[side] + 0.25 * (target - self.hand_pos[side])
            self.scene.set_worker_arm(side, self.hand_pos[side])

        self.scene.set_rgba('worker_torso', S.WAIT_RGBA if self.worker_mode == 'wait' else S.VEST_RGBA)
        self.progress += 0.15 * (self.progress_target - self.progress)
        for i in range(len(S.SLOTS)):
            if i < self.installed:
                rgba = S.DRYWALL_RGBA
            elif i == self.installed:
                rgba = S.FRAME_RGBA + (S.PREP_RGBA - S.FRAME_RGBA) * self.progress
            else:
                rgba = S.FRAME_RGBA
            self.scene.set_rgba(f'slot_{i}', rgba)

    def _update_piece(self):
        if self.piece_holder == 'gripper':
            pos = self.scene.eef_pos() + self.piece_offset
        elif self.piece_holder == 'hand':
            pos = self.hand_pos[self.pref] + self.piece_offset
        else:
            pos = S.PIECE_ON_STACK
        self.scene.set_mocap('piece', pos)

    # --- small motion primitives (generators yield once per sim step) -----------------

    def move(self, target, tol=0.012, max_steps=120):
        self.arm_target = np.array(target, dtype=float)
        for _ in range(max_steps):
            if np.linalg.norm(self.scene.eef_pos() - self.arm_target) < tol:
                return
            yield

    def wait(self, n):
        for _ in range(n):
            yield

    def hand_to(self, side, pos, tol=0.02, max_steps=40):
        self.hand_override[side] = np.array(pos, dtype=float)
        for _ in range(max_steps):
            if np.linalg.norm(self.hand_pos[side] - self.hand_override[side]) < tol:
                return
            yield

    def pick(self):
        self.robot_status = 'picking a panel from the stack'
        self.gripper = -1.0
        yield from self.move(S.PIECE_ON_STACK + np.array([0, 0, 0.10]), tol=0.02)
        yield from self.move(S.PIECE_ON_STACK, tol=0.006)
        # the panel has no collision, so close the fingers only to roughly its width, then hold
        self.gripper = 1.0
        yield from self.wait(3)
        self.gripper = 0.0
        yield from self.wait(5)
        self.piece_offset = S.PIECE_ON_STACK - self.scene.eef_pos()
        self.piece_holder = 'gripper'
        yield from self.move(S.PIECE_ON_STACK + np.array([0, 0, 0.15]), tol=0.03)

    def hand_over(self, slot):
        self.robot_status = 'worker takes the panel'
        self.worker_mode = 'take'
        piece_pos = self.scene.eef_pos() + self.piece_offset
        yield from self.hand_to(self.pref, piece_pos)
        self.piece_offset = piece_pos - self.hand_pos[self.pref]
        self.piece_holder = 'hand'
        self.gripper = -1.0
        yield from self.wait(6)
        self.robot_status = 'returning to the material stack'
        self.arm_target = self.scene.eef_pos() + np.array([0, 0, 0.12])
        yield from self.hand_to(self.pref, S.SLOTS[slot] + np.array([-0.04, 0, 0]))
        self.arm_target = HOME.copy()
        self.installed = slot + 1
        self.piece_holder = 'stack'
        self.hand_override.pop(self.pref, None)
        self.progress = 0.0
        yield from self.wait(4)

    # --- one abstract tick --------------------------------------------------------------

    def run_tick(self, action):
        env = self.env
        pre = snapshot(env)
        launched = action in (1, 2) and pre['transit'] == 0 and not pre['ready']
        if action in (1, 2) and not launched:
            self.unnecessary += 1
        obs, _, terminated, truncated, _ = env.step(action)
        post = snapshot(env)
        self.shown = dict(pre, time=post['time'])
        consumed = post['completed'] > pre['completed']

        self.worker_mode = 'wait' if post['idle'] > pre['idle'] else 'work'
        self.progress_target = 1.0 if consumed or post['remaining'] == 0 else (env.pace - post['remaining']) / env.pace

        script = []
        if launched:
            sel = action - 1
            script.append(self.pick())
            script.append(self._deliver(sel))
        elif pre['transit'] == 2:
            self.robot_status = f"wrong side: worker reaches with the {SIDE_NAME[self.pref]} hand, robot must move"
        elif pre['transit'] == 1:
            script.append(self._reposition())
        elif action in (1, 2):
            self.robot_status = 'unnecessary delivery request (robot already busy)'
        elif not pre['ready'] and self.piece_holder == 'stack':
            self.robot_status = 'waiting at the material stack'

        if consumed:
            script.append(self.hand_over(pre['completed']))

        count = 0
        for gen in script:
            for _ in gen:
                self.step()
                count += 1
        while count < MIN_TICK_STEPS:
            self.step()
            count += 1

        if consumed:
            self.progress_target = (env.pace - post['remaining']) / env.pace
            self.worker_mode = 'work'
        self.shown = post
        return obs, terminated or truncated

    def _deliver(self, sel):
        if sel == self.pref:
            self.robot_status = f"delivering to the worker's {SIDE_NAME[sel]} (preferred side)"
        else:
            self.robot_status = f"delivering to {SIDE_NAME[sel]}, but the worker prefers {SIDE_NAME[self.pref]}!"
        yield from self.move(S.HANDOVER[sel])
        self.robot_status = ('panel ready at the preferred side' if sel == self.pref
                             else 'wrong side: worker cannot take it here')

    def _reposition(self):
        self.robot_status = 'repositioning to the preferred side'
        yield from self.move(VIA, tol=0.04)
        yield from self.move(S.HANDOVER[self.pref])
        self.robot_status = 'panel ready at the preferred side'

    # --- HUD ----------------------------------------------------------------------------

    def hud(self):
        env, shown = self.env, self.shown
        speed = 'fast' if env.pace <= 3 else 'slow'
        if shown['completed'] >= env.n_parts:
            worker = 'all panels installed'
        elif self.worker_mode == 'wait':
            worker = 'WAITING for a panel'
        elif self.worker_mode == 'take':
            worker = 'taking the panel and installing it'
        else:
            worker = f'preparing slot {self.installed + 1} ({int(round(self.progress * 100))}%)'
        return [
            ('Worker profile', f'{speed} ({env.pace} ticks per panel), prefers {SIDE_NAME[env.side]} side'),
            ('Strategy', STRATEGY_LABELS[self.strategy]),
            ('Tick', f"{shown['time']}    panels installed: {shown['completed']}/{env.n_parts}"),
            ('Worker waiting', f"{shown['idle']} ticks    wrong-side deliveries: {shown['mistakes']}    unnecessary: {self.unnecessary}"),
            ('Robot', self.robot_status),
            ('Worker', worker),
        ]


def set_camera(cam):
    cam.lookat[:] = (0.3, 0.0, 1.0)
    cam.distance = 2.6
    cam.azimuth = 65
    cam.elevation = -18


def run(args, scene, on_frame, keep_going=lambda: True):
    profile = None if args.profile == 'random' else PROFILES[args.profile]
    env = CobuildEnv(profile=profile)
    obs, _ = env.reset(seed=args.seed)
    model = load_ppo() if args.strategy == 'ppo' else None
    profile_dict = {'pace': env.pace, 'side': env.side}

    director = Director(scene, env, args.strategy, on_frame)
    for _ in range(30):  # settle the arm at home before the episode starts
        director.step()

    done = False
    while not done and keep_going():
        action = choose_action(args.strategy, env, profile_dict, obs, model)
        obs, done = director.run_tick(action)
    director.robot_status = 'episode finished'
    return director


def run_viewer(args, scene):
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(scene.mj_model, scene.mj_data, show_left_ui=False, show_right_ui=False)
    viewer.opt.geomgroup[0] = 0  # hide robosuite collision meshes
    set_camera(viewer.cam)
    step_wall = CONTROL_DT / args.speed
    last = [time.perf_counter()]

    def on_frame(director):
        if not viewer.is_running():
            raise SystemExit
        labels, values = zip(*director.hud())
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          '\n'.join(labels), '\n'.join(values)))
        viewer.sync()
        sleep = step_wall - (time.perf_counter() - last[0])
        if sleep > 0:
            time.sleep(sleep)
        last[0] = time.perf_counter()

    director = run(args, scene, on_frame, viewer.is_running)
    while viewer.is_running():  # keep the final scene up until the window is closed
        director.step()


def run_video(args, scene):
    import cv2

    width, height = 1280, 720
    scene.mj_model.vis.global_.offwidth = width
    scene.mj_model.vis.global_.offheight = height
    renderer = mujoco.Renderer(scene.mj_model, height, width)
    cam = mujoco.MjvCamera()
    set_camera(cam)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0
    fps = int(round(args.speed / CONTROL_DT))
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    def on_frame(director):
        renderer.update_scene(scene.mj_data, cam, opt)
        frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        lines = director.hud()
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (900, 22 + 26 * len(lines)), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        for i, (label, value) in enumerate(lines):
            y = 36 + 26 * i
            cv2.putText(frame, f'{label}:', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(frame, value, (190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(frame)

    director = run(args, scene, on_frame)
    for _ in range(int(2.0 / CONTROL_DT)):  # hold the final frame for a moment
        director.step()
    writer.release()
    print(f'Saved {args.video}  ({director.env.time} ticks, worker waited {director.env.idle} ticks, '
          f'{director.env.mistakes} wrong-side deliveries)')


def main():
    parser = argparse.ArgumentParser(description='Watch a Cobuild strategy in a 3D robosuite scene.')
    parser.add_argument('--profile', default='fast_left', choices=['random'] + list(PROFILES))
    parser.add_argument('--strategy', default='ppo', choices=['ppo'] + list(STRATEGIES))
    parser.add_argument('--speed', type=float, default=1.5, help='playback speed relative to real time')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video', help='write an .mp4 instead of opening a window')
    args = parser.parse_args()

    scene = S.CobuildHandover()
    scene.reset()
    if args.video:
        run_video(args, scene)
    else:
        run_viewer(args, scene)


if __name__ == '__main__':
    main()
