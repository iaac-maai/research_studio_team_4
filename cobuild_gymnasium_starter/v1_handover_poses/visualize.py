"""2D visualization of the drywall installation (schematic side view, pygame).

    python visualize.py --profile short_left --strategy C_cobuild_ppo
    python visualize.py --profile low_capacity_right --compare            # B (top) vs Cobuild rule (bottom)
    python visualize.py --profile random --compare --top A_reactive --bottom C_cobuild_ppo

In compare mode both robots assist the SAME worker with the SAME random
variability, side by side, so the difference you see is only the strategy.

What you see (side view, the worker faces the wall on the right):
    - material rack (boards + screws) on the left, the robot drives between rack and worker
    - the worker as a stick figure drawn to scale from their HEIGHT
    - green box = the worker's comfortable handover zone (from height and reach)
    - the ring where the robot presents material: L/R = side, green = comfortable, orange = awkward
    - worker turns RED while waiting; the robot turns red while hovering with an item nobody needs yet
    - the wall elevation (4 boards) and all metrics are on the right-hand panel

Keys: space pause · r restart · n new random worker · 1-9 personas · +/- speed · esc quit
"""
import argparse
import os

import config as C
import ergonomics as E
from cobuild_env import CobuildDrywallEnv, action_label
from profiles import GENERIC, PERSONAS, sample_population
from strategies import LABELS, STRATEGY_KEYS, make_strategy

PANEL_W, PANEL_H = 1240, 430
SCENE_W = 780
PX_PER_M = 135
RACK_X, WORKER_X, WALL_X = 90, 560, 690

BG = (247, 246, 242)
PANEL_BG = (255, 255, 255)
INK = (30, 30, 28)
MUTED = (110, 108, 102)
GRID = (222, 220, 214)
WORKER = (42, 120, 214)
WAITING = (214, 60, 60)
ROBOT = (235, 104, 52)
ROBOT_IDLE_HOLD = (214, 60, 60)
BOARD = (226, 205, 160)
BOARD_EDGE = (170, 148, 105)
SCREWS = (90, 90, 96)
GOOD = (27, 175, 122)
AWKWARD = (237, 161, 0)
ZONE = (200, 236, 216)


class Episode:
    def __init__(self, strategy_key, profile, seed):
        self.strategy = make_strategy(strategy_key)
        self.profile, self.seed = profile, seed
        self.reset()

    def reset(self):
        self.env = CobuildDrywallEnv(profile=self.profile, observe_profile=self.strategy.observe_profile)
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.fetch_total = 1
        self.action = 0

    @property
    def done(self):
        return self.env.done or self.env.time >= C.MAX_TICKS

    def step(self):
        if self.done:
            return
        env = self.env
        self.action = self.strategy.act(env, self.obs)
        was_free = env.robot_state == 'free'
        self.obs, _, _, _, self.info = env.step(self.action)
        if was_free and env.robot_state == 'fetching':
            self.fetch_total = env.robot_ticks + 1


def to_screen(x_m_from_worker, h_m, oy):
    """Scene coordinates: metres in front of the worker (towards the rack) and height above ground."""
    return int(WORKER_X - x_m_from_worker * PX_PER_M), int(oy + PANEL_H - 40 - h_m * PX_PER_M)


def draw_worker(screen, ep, oy, pg):
    env, p = ep.env, ep.profile
    ground = oy + PANEL_H - 40
    H = p.height_m * PX_PER_M
    color = WAITING if env.worker_state == 'waiting' else WORKER
    hip = (WORKER_X, int(ground - 0.53 * H))
    shoulder = (WORKER_X, int(ground - 0.82 * H))
    head = (WORKER_X, int(ground - 0.93 * H))
    pg.draw.line(screen, color, (WORKER_X - 12, ground), hip, 5)
    pg.draw.line(screen, color, (WORKER_X + 12, ground), hip, 5)
    pg.draw.line(screen, color, hip, shoulder, 7)
    pg.draw.circle(screen, color, head, int(0.07 * H))
    # arm target depends on what the worker is doing
    board = min(env.board, C.N_BOARDS - 1)
    upper = E.is_upper(board)
    step = env.step_name
    if env.worker_state == 'waiting' and C.NEEDS.get(step):
        hand = to_screen(0.45, p.comfort_handover_m, oy)  # reaching out towards the robot
    elif step == 'lift':
        top = (C.WALL_TOP_M if upper else C.WALL_TOP_M / 2) - 0.1
        hand = (WALL_X - 24, int(ground - min(top, p.overhead_reach_m) * PX_PER_M))
    elif step in ('fasten', 'measure'):
        h = 1.8 if upper else 0.8
        hand = (WALL_X - 14, int(ground - min(h, p.overhead_reach_m - 0.1) * PX_PER_M))
    else:  # cut: working on the board at waist height
        hand = (WORKER_X + 60, int(ground - 0.5 * H))
    pg.draw.line(screen, color, shoulder, hand, 5)
    pg.draw.circle(screen, color, hand, 5)
    # board being lifted / held against the wall
    if step == 'lift' and env.worker_state == 'working':
        base = 1.2 if upper else 0.0
        top_y = ground - (base + 1.2) * PX_PER_M
        pg.draw.rect(screen, BOARD, (WALL_X - 18, top_y, 12, 1.2 * PX_PER_M))
        pg.draw.rect(screen, BOARD_EDGE, (WALL_X - 18, top_y, 12, 1.2 * PX_PER_M), 1)


def robot_position(ep, oy):
    """(x, y) of the robot base and of its gripper."""
    env = ep.env
    ground = oy + PANEL_H - 40
    rack = (RACK_X + 50, ground)
    if env.robot_state in ('positioning', 'supporting'):
        return (WALL_X - 85, ground), (WALL_X - 20, int(ground - 1.6 * PX_PER_M))
    if env.robot_state in ('fetching', 'holding') and env.robot_pose is not None:
        _, h, d = E.POSES[env.robot_pose]
        target_grip = to_screen(d, h, oy)
        target_base = (target_grip[0] - 70, ground)
        if env.robot_state == 'holding':
            return target_base, target_grip
        progress = 1 - env.robot_ticks / max(ep.fetch_total, 1)
        bx = rack[0] + (target_base[0] - rack[0]) * progress
        return (int(bx), ground), (int(bx + 70 * progress), int(ground - 1.0 * PX_PER_M))
    return rack, (rack[0] + 30, int(ground - 1.0 * PX_PER_M))


def draw_scene(screen, font, small, ep, oy, pg):
    env, p = ep.env, ep.profile
    ground = oy + PANEL_H - 40
    pg.draw.rect(screen, PANEL_BG, (0, oy, SCENE_W, PANEL_H))
    pg.draw.line(screen, INK, (0, ground), (SCENE_W, ground), 2)
    # wall with the two board rows
    pg.draw.rect(screen, (235, 233, 226), (WALL_X, ground - C.WALL_TOP_M * PX_PER_M, 26, C.WALL_TOP_M * PX_PER_M))
    pg.draw.line(screen, GRID, (WALL_X, ground - 1.2 * PX_PER_M), (WALL_X + 26, ground - 1.2 * PX_PER_M), 1)
    screen.blit(small.render('wall 2.4 m', True, MUTED), (WALL_X - 8, ground - C.WALL_TOP_M * PX_PER_M - 18))
    # material rack
    for i in range(C.N_BOARDS - min(env.next_item_idx // 2 + (env.next_item_idx % 2), C.N_BOARDS)):
        pg.draw.rect(screen, BOARD, (RACK_X - 40 + i * 7, ground - 120, 6, 120))
    pg.draw.rect(screen, SCREWS, (RACK_X + 10, ground - 24, 26, 24))
    screen.blit(small.render('material rack', True, MUTED), (RACK_X - 45, ground + 6))
    # comfortable handover zone from the profile (height ± tolerance, out to the comfortable reach)
    zx0, zy0 = to_screen(p.reach_m, p.comfort_handover_m + C.HEIGHT_TOLERANCE_M, oy)
    zx1, zy1 = to_screen(0.15, p.comfort_handover_m - C.HEIGHT_TOLERANCE_M, oy)
    pg.draw.rect(screen, ZONE, (zx0, zy0, zx1 - zx0, zy1 - zy0))
    screen.blit(small.render('comfort zone', True, GOOD), (zx0, zy0 - 16))
    # robot
    base, grip = robot_position(ep, oy)
    hovering = env.robot_state == 'holding'
    rcolor = ROBOT_IDLE_HOLD if hovering and env.worker_state != 'waiting' else ROBOT
    pg.draw.rect(screen, rcolor, (base[0] - 30, base[1] - 34, 60, 30), border_radius=4)
    pg.draw.circle(screen, INK, (base[0] - 18, base[1] - 4), 5)
    pg.draw.circle(screen, INK, (base[0] + 18, base[1] - 4), 5)
    elbow = (base[0] + 10, base[1] - 90)
    pg.draw.line(screen, rcolor, (base[0], base[1] - 34), elbow, 6)
    pg.draw.line(screen, rcolor, elbow, grip, 6)
    if env.robot_item is not None:
        if env.robot_item[1] == 'board':
            pg.draw.rect(screen, BOARD, (grip[0] - 6, grip[1] - 60, 12, 120))
            pg.draw.rect(screen, BOARD_EDGE, (grip[0] - 6, grip[1] - 60, 12, 120), 1)
        else:
            pg.draw.rect(screen, SCREWS, (grip[0] - 10, grip[1] - 10, 20, 20))
    # handover ring with side label and comfort colour
    if env.robot_state in ('fetching', 'holding') and env.robot_pose is not None:
        pose = E.POSES[env.robot_pose]
        ticks, strain, _ = E.handover_cost(p, pose, env.robot_item[1])
        ring = to_screen(pose[2], pose[1], oy)
        col = AWKWARD if E.is_awkward(ticks, strain) else GOOD
        pg.draw.circle(screen, col, ring, 15, 3)
        lab = small.render(C.HANDOVER_SIDES[pose[0]][0].upper(), True, col)
        screen.blit(lab, (ring[0] - lab.get_width() // 2, ring[1] - lab.get_height() // 2))
    draw_worker(screen, ep, oy, pg)
    # status line inside the scene
    status = 'WAITING' if env.worker_state == 'waiting' else env.step_name.upper()
    screen.blit(font.render(f'worker: {status}   ·   robot: {action_label(ep.action)}', True, INK), (16, oy + 12))
    screen.blit(small.render(env.last_event, True, MUTED), (16, oy + 34))


def draw_hud(screen, font, small, big, ep, oy, pg):
    env, p = ep.env, ep.profile
    x = SCENE_W + 14
    pg.draw.rect(screen, BG, (SCENE_W, oy, PANEL_W - SCENE_W, PANEL_H))
    secs = env.time * C.TICK_SECONDS
    screen.blit(big.render(f'{secs // 60:02d}:{secs % 60:02d}', True, INK), (x, oy + 10))
    screen.blit(small.render(f'tick {env.time}  ·  board {min(env.board + 1, C.N_BOARDS)}/{C.N_BOARDS}', True, MUTED), (x + 130, oy + 24))
    screen.blit(font.render(LABELS[ep.strategy.key], True, INK), (x, oy + 52))
    lines = [
        f'worker: {p.name}',
        f'height {p.height_m:.2f} m · reach {p.reach_m:.2f} m · lifts {p.lift_capacity_kg:.0f} kg',
        f'pace x{p.pace:.2f} · {p.handedness}-handed · fatigue {p.fatigue_rate:.3f}',
        '',
        f'waiting:        {env.wait_ticks * C.TICK_SECONDS / 60:5.1f} min  ({env.interruptions} interruptions)',
        f'strain:         {env.strain:5.1f} pts   fatigue x{env.fatigue_multiplier:.2f}',
        f'robot hovering: {env.exposure * C.TICK_SECONDS / 60:5.1f} min',
        f'awkward handovers {env.awkward}/{env.handovers} · held boards {env.supports}',
    ]
    for i, line in enumerate(lines):
        screen.blit(small.render(line, True, INK if i >= 4 else MUTED), (x, oy + 80 + i * 18))
    # strain bar
    bar_y = oy + 236
    pg.draw.rect(screen, GRID, (x, bar_y, 260, 8), border_radius=4)
    pg.draw.rect(screen, AWKWARD, (x, bar_y, min(260, env.strain / 70 * 260), 8), border_radius=4)
    # wall elevation (front view): upper row boards 1-2, lower row 3-4
    wx, wy, bw, bh = x, oy + 262, 120, 60
    screen.blit(small.render('wall (front view)', True, MUTED), (wx, wy))
    for b in range(C.N_BOARDS):
        row, col = (0 if E.is_upper(b) else 1), b % 2
        rect = (wx + col * (bw + 4), wy + 20 + row * (bh + 4), bw, bh)
        if b < env.board:
            pg.draw.rect(screen, BOARD, rect)
        elif b == env.board and not env.done:
            frac = (env.step_idx + (0 if env.worker_state == 'waiting' else 0.5)) / len(C.STEPS)
            pg.draw.rect(screen, ZONE, (rect[0], rect[1], rect[2] * frac, rect[3]))
            screen.blit(small.render(env.step_name, True, INK), (rect[0] + 6, rect[1] + 6))
        pg.draw.rect(screen, BOARD_EDGE, rect, 1)
    if ep.done:
        msg = big.render(f'DONE in {secs // 60:02d}:{secs % 60:02d}', True, GOOD)
        screen.blit(msg, (SCENE_W - msg.get_width() - 20, oy + 12))


def resolve_profile(name, rng):
    if name == 'random':
        return sample_population(rng, name='random_worker')
    if name == 'generic_average':
        return GENERIC
    return PERSONAS[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', default='short_left', choices=['random', 'generic_average', *PERSONAS])
    parser.add_argument('--strategy', default='C_cobuild_ppo', choices=STRATEGY_KEYS)
    parser.add_argument('--compare', action='store_true', help='two strategies side by side (same worker, same variability)')
    parser.add_argument('--top', default='B_proactive_generic', choices=STRATEGY_KEYS)
    parser.add_argument('--bottom', default='C_cobuild_rule', choices=STRATEGY_KEYS)
    parser.add_argument('--fps', type=float, default=4.0, help='simulation ticks per second')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--screenshot', help='render headless, save a PNG at --at-tick and exit')
    parser.add_argument('--at-tick', type=int, default=40)
    args = parser.parse_args()
    if args.screenshot:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    import numpy as np
    import pygame as pg

    rng = np.random.default_rng(args.seed)
    profile = resolve_profile(args.profile, rng)
    keys = [args.top, args.bottom] if args.compare else [args.strategy]
    episodes = [Episode(k, profile, args.seed) for k in keys]

    pg.init()
    screen = pg.display.set_mode((PANEL_W, PANEL_H * len(episodes) + 4 * (len(episodes) - 1)))
    pg.display.set_caption('Cobuild — drywall installation (synthetic simulation)')
    font = pg.font.SysFont('helveticaneue,helvetica,arial', 16)
    small = pg.font.SysFont('menlo,monaco,courier', 13)
    big = pg.font.SysFont('helveticaneue,helvetica,arial', 30, bold=True)
    clock = pg.time.Clock()
    persona_keys = {getattr(pg, f'K_{i + 1}'): n for i, n in enumerate(PERSONAS)}
    fps, paused, running = args.fps, False, True

    def restart(new_profile=None):
        for ep in episodes:
            if new_profile is not None:
                ep.profile = new_profile
            ep.reset()

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    paused = not paused
                elif event.key == pg.K_r:
                    restart()
                elif event.key == pg.K_n:
                    restart(sample_population(rng, name='random_worker'))
                elif event.key in persona_keys:
                    restart(PERSONAS[persona_keys[event.key]])
                elif event.key in (pg.K_PLUS, pg.K_EQUALS):
                    fps = min(fps * 1.5, 60)
                elif event.key == pg.K_MINUS:
                    fps = max(fps / 1.5, 0.5)
        if not paused:
            for ep in episodes:
                ep.step()
        screen.fill(GRID)
        for i, ep in enumerate(episodes):
            oy = i * (PANEL_H + 4)
            draw_scene(screen, font, small, ep, oy, pg)
            draw_hud(screen, font, small, big, ep, oy, pg)
        pg.display.flip()
        if args.screenshot and all(ep.env.time >= args.at_tick or ep.done for ep in episodes):
            pg.image.save(screen, args.screenshot)
            print(f'saved {args.screenshot}')
            break
        clock.tick(fps if not args.screenshot else 0)
    pg.quit()


if __name__ == '__main__':
    main()
