"""2D visualization: watch one episode of CobuildEnv play out.

This is a schematic, geometric-shapes view of the scheduling simulation —
not a rendering of a real construction site or robot. It exists so the
behavior encoded in the reward/observation can be *seen*, for demos and to
sanity-check the trained policy.

Usage:
    python visualize.py --profile fast_left --strategy ppo
    python visualize.py --profile random --strategy profile_rule --fps 4

Controls while the window is open:
    1 / 2 / 3 / 4   switch to fast_left / fast_right / slow_left / slow_right and restart
    r               restart the current profile with a new episode
    space           pause / resume
    esc / close     quit
"""
import argparse

import pygame

from cobuild_env import CobuildEnv
from profiles import PROFILES
from baselines import STRATEGIES, choose_action, load_ppo

WIDTH, HEIGHT = 900, 450
GROUND_Y = 340
WORKER_X = 260
DOCK_X = WIDTH // 2
LEFT_MATERIAL_X = 90
RIGHT_MATERIAL_X = WIDTH - 90

BG = (245, 243, 238)
INK = (35, 33, 30)
MUTED = (120, 116, 108)
WORKER_COLOR = (60, 110, 180)
ROBOT_COLOR = (200, 90, 40)
MATERIAL_COLOR = (90, 160, 90)
BAD_COLOR = (190, 60, 60)
DOCK_COLOR = (210, 205, 195)

PROFILE_KEYS = {
    pygame.K_1: 'fast_left',
    pygame.K_2: 'fast_right',
    pygame.K_3: 'slow_left',
    pygame.K_4: 'slow_right',
}

ACTION_LABELS = {0: 'wait', 1: 'deliver left', 2: 'deliver right'}


class Episode:
    """Wraps one CobuildEnv rollout plus the bits of state the animation needs
    that aren't in the env (which side the robot is heading to, and how long
    the current delivery takes, so the robot can be drawn partway there)."""

    def __init__(self, profile_name, strategy, model, seed):
        self.profile_name = profile_name
        self.strategy = strategy
        self.model = model
        self.seed = seed
        self.reset()

    def reset(self):
        profile = PROFILES[self.profile_name] if self.profile_name != 'random' else None
        self.env = CobuildEnv(profile=profile)
        obs, _ = self.env.reset(seed=self.seed)
        self.obs = obs
        self.profile = {'pace': self.env.pace, 'side': self.env.side}
        self.last_action = 0
        self.delivery_total = 0
        self.delivery_side = None
        self.done = False

    def step(self):
        if self.done:
            return
        action = choose_action(self.strategy, self.env, self.profile, self.obs, self.model)
        self.last_action = action
        if action in (1, 2) and self.env.transit == 0 and not self.env.material_ready:
            # about to start a delivery: remember which side and how long it
            # will take so the robot's on-screen position can be interpolated
            self.delivery_side = action - 1
            self.delivery_total = 1 if self.delivery_side == self.env.side else 3
        obs, _, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info
        if self.env.transit == 0:
            self.delivery_total = 0
        self.done = terminated or truncated


def robot_position(ep):
    env = ep.env
    if env.transit <= 0 or ep.delivery_total == 0:
        return DOCK_X
    target_x = RIGHT_MATERIAL_X if ep.delivery_side == 1 else LEFT_MATERIAL_X
    progress = 1.0 - (env.transit / ep.delivery_total)
    return DOCK_X + (target_x - DOCK_X) * progress


def draw(screen, font, big_font, ep, paused):
    screen.fill(BG)
    env = ep.env
    pygame.draw.line(screen, INK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

    # material docks
    for x, label in [(LEFT_MATERIAL_X, 'LEFT'), (RIGHT_MATERIAL_X, 'RIGHT')]:
        pygame.draw.rect(screen, DOCK_COLOR, (x - 35, GROUND_Y - 30, 70, 30))
        text = font.render(label, True, MUTED)
        screen.blit(text, (x - text.get_width() // 2, GROUND_Y - 50))

    # worker
    pygame.draw.circle(screen, WORKER_COLOR, (WORKER_X, GROUND_Y - 40), 22)
    bar_w = 90
    progress = 0.0 if env.pace == 0 else (env.pace - env.remaining) / env.pace
    pygame.draw.rect(screen, MUTED, (WORKER_X - bar_w // 2, GROUND_Y - 90, bar_w, 10), 1)
    pygame.draw.rect(screen, WORKER_COLOR, (WORKER_X - bar_w // 2, GROUND_Y - 90, bar_w * progress, 10))
    side_label = 'prefers LEFT' if env.side == 0 else 'prefers RIGHT'
    screen.blit(font.render(side_label, True, MUTED), (WORKER_X - 45, GROUND_Y - 115))

    # material ready at the worker
    if env.material_ready:
        pygame.draw.rect(screen, MATERIAL_COLOR, (WORKER_X + 28, GROUND_Y - 55, 18, 18))

    # robot, in transit or docked
    rx = robot_position(ep)
    color = BAD_COLOR if (env.transit > 0 and ep.delivery_total == 3) else ROBOT_COLOR
    pygame.draw.rect(screen, color, (rx - 18, GROUND_Y - 36, 36, 36))
    if env.transit > 0:
        pygame.draw.rect(screen, MATERIAL_COLOR, (rx - 9, GROUND_Y - 50, 18, 12))

    # HUD
    lines = [
        f"profile: pace={env.pace}  side={'left' if env.side == 0 else 'right'}   strategy: {ep.strategy}",
        f"tick: {env.time}    completed: {env.completed}/{env.n_parts}    idle ticks: {env.idle}    wrong-side deliveries: {env.mistakes}",
        f"action: {ACTION_LABELS[ep.last_action]}",
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, INK), (20, 20 + i * 24))

    if ep.done:
        text = big_font.render('Episode complete — press R to replay, 1-4 to switch profile', True, INK)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 60))
    elif paused:
        text = big_font.render('Paused', True, INK)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 60))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description='Watch one CobuildEnv episode.')
    parser.add_argument('--profile', default='random',
                         choices=['random'] + list(PROFILES.keys()))
    parser.add_argument('--strategy', default='ppo',
                         choices=['ppo'] + list(STRATEGIES.keys()))
    parser.add_argument('--fps', type=float, default=3.0, help='simulation ticks per second')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model = load_ppo() if args.strategy == 'ppo' else None

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Cobuild — 2D visualization (synthetic MVP)')
    font = pygame.font.SysFont('menlo', 16)
    big_font = pygame.font.SysFont('menlo', 22, bold=True)
    clock = pygame.time.Clock()

    ep = Episode(args.profile, args.strategy, model, args.seed)
    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    ep.reset()
                    paused = False
                elif event.key in PROFILE_KEYS:
                    ep.profile_name = PROFILE_KEYS[event.key]
                    ep.reset()
                    paused = False

        if not paused and not ep.done:
            ep.step()

        draw(screen, font, big_font, ep, paused)
        clock.tick(args.fps if not ep.done else 30)

    pygame.quit()


if __name__ == '__main__':
    main()
