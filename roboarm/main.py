"""
main.py - application entry point: window, event loop and controls.

Run:            python main.py
Screenshot:     python main.py --plan --run --time 6 --screenshot shot.png
(see `python main.py --help`)
"""
import argparse
import os
import sys

import numpy as np

import config as cfg


def build_widgets(sim, renderer):
    from ui import Button, Slider
    x0 = cfg.WINDOW_WIDTH - cfg.PANEL_WIDTH + 14
    bw, bh, gap = (cfg.PANEL_WIDTH - 28 - 8) // 2, 30, 6
    show = renderer.show

    def toggle(key):
        return lambda: show.__setitem__(key, not show[key])

    def toggle_smoothing():
        sim.smoothing = not sim.smoothing
        sim.set_status(f"Path smoothing {'on' if sim.smoothing else 'off'} (applies to the next plan).")

    idle = lambda: not sim.busy
    specs = [
        ("Plan path", sim.request_plan, None, idle, "P"),
        (lambda: "Pause" if sim.state == "running" else "Start", sim.start_pause, None, None, "Space"),
        ("Reset", sim.reset, None, None, "R"),
        ("Regenerate", sim.request_regenerate, None, idle, "N"),
        ("Trajectory", toggle("trajectory"), lambda: show["trajectory"], None, "T"),
        ("RRT tree", toggle("tree"), lambda: show["tree"], None, "Y"),
        ("Ghost arms", toggle("ghosts"), lambda: show["ghosts"], None, "O"),
        ("Collisions", toggle("collision"), lambda: show["collision"], None, "C"),
        ("C-space", toggle("cspace"), lambda: show["cspace"], None, "K"),
        ("Smoothing", toggle_smoothing, lambda: sim.smoothing, None, "S"),
        ("Store goal", sim.store_custom_goal, None, None, "G"),
        ("Plan to goal", sim.request_joint_plan, None, idle, "J"),
        ("Default layout", sim.restore_default_layout, None, idle, "D"),
        ("Help", toggle("help"), lambda: show["help"], None, "H"),
    ]
    buttons = []
    y = 60
    for k, (label, fn, on, enabled, hint) in enumerate(specs):
        col, row = k % 2, k // 2
        buttons.append(Button((x0 + col * (bw + 8), y + row * (bh + gap), bw, bh), label, fn, on, enabled, hint))
    y += ((len(specs) + 1) // 2) * (bh + gap) + 10

    sw = cfg.PANEL_WIDTH - 28
    sliders = [Slider((x0, y, sw, 30), "Animation speed", *cfg.SPEED_SLIDER_RANGE,
                      lambda: sim.speed, lambda v: setattr(sim, "speed", v), "{:.2f}x")]
    y += 36
    lo, hi = cfg.LINK_LENGTH_SLIDER_RANGE
    for k in range(min(sim.robot.n_joints, 4)):
        sliders.append(Slider((x0, y, sw, 30), f"Link {k + 1} length", lo, hi,
                              lambda k=k: sim.robot.lengths[k], lambda v, k=k: sim.set_link_length(k, v),
                              "{:.2f} m", enabled=idle))
        y += 36
    sliders.append(Slider((x0, y, sw, 30), "Max planning iterations", *cfg.ITERATION_SLIDER_RANGE,
                          lambda: sim.max_iterations, lambda v: setattr(sim, "max_iterations", int(v)),
                          "{:,}", integer=True))
    y += 44
    return {"buttons": buttons, "sliders": sliders, "metrics_y": y}


def main(argv=None):
    parser = argparse.ArgumentParser(description="2D robotic arm motion-planning simulator (RRT-Connect).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help="random seed for repeatable plans")
    parser.add_argument("--planner", choices=["rrt_connect", "ompl"], default=cfg.PLANNER_BACKEND,
                        help="motion planner backend (ompl needs: pip install ompl)")
    parser.add_argument("--ompl-algorithm", default=cfg.OMPL_ALGORITHM,
                        help="OMPL geometric planner name, e.g. RRTConnect, RRT, PRM, KPIECE1, BITstar")
    parser.add_argument("--plan", action="store_true", help="plan immediately at start-up")
    parser.add_argument("--run", action="store_true", help="start the animation as soon as a plan is ready")
    parser.add_argument("--show", default="", help="comma list of views to enable: tree,cspace,collision,help")
    parser.add_argument("--time", type=float, default=0.0, help="(with --screenshot) simulated seconds to run")
    parser.add_argument("--screenshot", default="", help="render off-screen, save a PNG and exit")
    args = parser.parse_args(argv)
    cfg.OMPL_ALGORITHM = args.ompl_algorithm
    if args.planner == "ompl":
        from ompl_backend import _import_ompl
        _import_ompl()      # fail early with a clear message if OMPL is missing

    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from render import Renderer, in_view, to_world
    from simulation import Simulation

    pygame.init()
    pygame.display.set_caption("Robot arm motion planning - RRT-Connect pick and place")
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    sim = Simulation(seed=args.seed, backend=args.planner)
    renderer = Renderer(screen)
    for name in filter(None, args.show.split(",")):
        renderer.show[name.strip()] = True
    widgets = build_widgets(sim, renderer)
    if args.plan:
        sim.request_plan()

    if args.screenshot:
        return headless(sim, renderer, widgets, args, pygame)

    drag_last = None
    cursor = None
    running = True
    while running:
        dt = min(clock.tick(cfg.FPS) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if event.type == pygame.KEYDOWN:
                if not handle_key(event, sim, renderer, pygame):
                    running = False
                continue
            if any(w.handle(event) for w in widgets["sliders"]):
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if any(b.handle(event) for b in widgets["buttons"]):
                    continue
                if in_view(event.pos):
                    p = to_world(event.pos)
                    idx = sim.env.movable_obstacle_at(p)
                    if idx is not None:
                        if sim.busy:
                            sim.set_status("Wait for planning to finish (or press Esc) before moving obstacles.", "warn")
                        else:
                            renderer.drag_obstacle, drag_last = idx, p
            elif event.type == pygame.MOUSEMOTION:
                p = to_world(event.pos)
                if renderer.drag_obstacle is not None:
                    sim.env.move_obstacle(renderer.drag_obstacle, p - drag_last)
                    drag_last = p
                    sim.on_environment_changed()
                else:
                    renderer.hover_obstacle = sim.env.movable_obstacle_at(p) if in_view(event.pos) else None
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                renderer.drag_obstacle = None
        # mouse cursor feedback
        want = pygame.SYSTEM_CURSOR_HAND if renderer.hover_obstacle is not None or renderer.drag_obstacle is not None \
            else pygame.SYSTEM_CURSOR_ARROW
        if want != cursor:
            pygame.mouse.set_cursor(want)
            cursor = want
        # continuous jogging with the arrow keys
        keys = pygame.key.get_pressed()
        direction = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        if direction:
            fast = 3.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
            sim.jog(sim.selected_joint, -direction * np.radians(cfg.JOG_SPEED_DEG) * fast * dt)
        sim.update(dt)
        renderer.draw(sim, widgets)
        pygame.display.flip()
    sim.cancel_planning()
    pygame.quit()
    return 0


def handle_key(event, sim, renderer, pygame):
    """Keyboard shortcuts. Returns False to quit."""
    k, show = event.key, renderer.show
    toggles = {pygame.K_t: "trajectory", pygame.K_y: "tree", pygame.K_o: "ghosts",
               pygame.K_c: "collision", pygame.K_k: "cspace", pygame.K_h: "help"}
    if k == pygame.K_ESCAPE:
        if sim.busy:
            sim.cancel_planning()
        elif show["help"]:
            show["help"] = False
    elif k == pygame.K_q and (event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)):
        return False
    elif k in toggles:
        show[toggles[k]] = not show[toggles[k]]
    elif k == pygame.K_p and not sim.busy:
        sim.request_plan()
    elif k == pygame.K_SPACE:
        sim.start_pause()
    elif k == pygame.K_r:
        sim.reset()
    elif k == pygame.K_n and not sim.busy:
        sim.request_regenerate()
    elif k == pygame.K_d:
        sim.restore_default_layout()
    elif k == pygame.K_g:
        sim.store_custom_goal()
    elif k == pygame.K_j and not sim.busy:
        sim.request_joint_plan()
    elif k == pygame.K_s:
        sim.smoothing = not sim.smoothing
        sim.set_status(f"Path smoothing {'on' if sim.smoothing else 'off'} (applies to the next plan).")
    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        sim.speed = min(cfg.SPEED_SLIDER_RANGE[1], sim.speed * 1.25)
    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
        sim.speed = max(cfg.SPEED_SLIDER_RANGE[0], sim.speed / 1.25)
    elif k == pygame.K_UP:
        sim.selected_joint = (sim.selected_joint + 1) % sim.robot.n_joints
    elif k == pygame.K_DOWN:
        sim.selected_joint = (sim.selected_joint - 1) % sim.robot.n_joints
    elif pygame.K_1 <= k <= pygame.K_9 and k - pygame.K_1 < sim.robot.n_joints:
        sim.selected_joint = k - pygame.K_1
    return True


def headless(sim, renderer, widgets, args, pygame):
    """Off-screen run used for screenshots and automated checks."""
    if sim._worker is not None:
        sim._worker.join()
    sim.update(0.0)
    if args.run:
        sim.start_pause()
    t, dt = 0.0, 1.0 / 30
    while t < args.time:
        sim.update(dt)
        t += dt
    if sim._worker is not None:
        sim._worker.join()
        sim.update(0.0)
    renderer.draw(sim, widgets)
    pygame.image.save(renderer.screen, args.screenshot)
    print(f"saved {args.screenshot}  state={sim.state}  status={sim.status}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
