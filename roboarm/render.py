"""
render.py - everything that is drawn on screen.

World coordinates (metres, y up) are converted to screen pixels (y down)
with to_screen(); mouse clicks are converted back with to_world().
"""
import time

import numpy as np
import pygame
import pygame.gfxdraw

import config as cfg
from collision import CollisionChecker
from pick_place import Move

C = cfg.COLORS
VIEW_W = cfg.WINDOW_WIDTH - cfg.PANEL_WIDTH
SCALE = VIEW_W / (cfg.VIEW_X_RANGE[1] - cfg.VIEW_X_RANGE[0])           # pixels per metre
VIEW_TOP_Y = cfg.VIEW_Y_BOTTOM + (cfg.WINDOW_HEIGHT - cfg.TOP_BAR_HEIGHT) / SCALE


def to_screen(p):
    p = np.asarray(p, float)
    x = (p[..., 0] - cfg.VIEW_X_RANGE[0]) * SCALE
    y = cfg.TOP_BAR_HEIGHT + (VIEW_TOP_Y - p[..., 1]) * SCALE
    return np.stack([x, y], axis=-1)


def to_world(pos):
    return np.array([pos[0] / SCALE + cfg.VIEW_X_RANGE[0],
                     VIEW_TOP_Y - (pos[1] - cfg.TOP_BAR_HEIGHT) / SCALE])


def in_view(pos):
    return pos[0] < VIEW_W and pos[1] > cfg.TOP_BAR_HEIGHT


# --------------------------------------------------------------------------- #
# drawing primitives (anti-aliased)
# --------------------------------------------------------------------------- #
def aa_circle(surf, color, center, r):
    x, y, r = int(round(center[0])), int(round(center[1])), int(round(r))
    if r >= 1:
        pygame.gfxdraw.filled_circle(surf, x, y, r, color)
        pygame.gfxdraw.aacircle(surf, x, y, r, color)


def aa_polygon(surf, color, pts):
    pts = [(float(x), float(y)) for x, y in pts]
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    pygame.gfxdraw.aapolygon(surf, pts, color)


def capsule(surf, color, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = np.hypot(*d)
    if length > 1e-6:
        n = np.array([-d[1], d[0]]) / length * r
        aa_polygon(surf, color, [a + n, b + n, b - n, a - n])
    aa_circle(surf, color, a, r)
    aa_circle(surf, color, b, r)


def outlined_capsule(surf, fill, edge, a, b, r, w=2):
    capsule(surf, edge, a, b, r + w)
    capsule(surf, fill, a, b, r)


def dashed_line(surf, color, a, b, dash=8, width=2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.hypot(*(b - a))
    n = max(1, int(length / dash))
    for i in range(0, n, 2):
        p0 = a + (b - a) * i / n
        p1 = a + (b - a) * min(i + 1, n) / n
        pygame.draw.line(surf, color, p0, p1, width)


def lighten(c, f=0.35):
    return tuple(int(v + (255 - v) * f) for v in c[:3])


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.fonts = {
            "title": pygame.font.Font(None, 28),
            "big": pygame.font.Font(None, 40),
            "button": pygame.font.Font(None, 21),
            "text": pygame.font.Font(None, 22),
            "small": pygame.font.Font(None, 19),
            "tiny": pygame.font.Font(None, 16),
            "label": pygame.font.Font(None, 30),
        }
        self.show = {"trajectory": True, "tree": False, "ghosts": True, "collision": False,
                     "cspace": False, "help": False}
        self.hover_obstacle = None
        self.drag_obstacle = None
        self._layer_key = None
        self._layer = None
        self._cspace = None     # (key, surface, time)

    # ------------------------------------------------------------ main draw
    def draw(self, sim, widgets):
        s = self.screen
        s.fill(C["background"])
        world = s.subsurface(pygame.Rect(0, 0, VIEW_W, cfg.WINDOW_HEIGHT))
        self._draw_grid(world)
        self._draw_environment(world, sim)
        self._draw_plan_layer(world, sim)
        if sim.custom_goal is not None:
            self._draw_ghost_arm(world, sim, sim.custom_goal, C["custom_goal"], 200, width=5)
        report = sim.current_report()
        self._draw_object(world, sim)
        self._draw_robot(world, sim, report)
        if self.show["collision"]:
            self._draw_collision_view(world, sim, report)
        self._draw_readout(world, sim, report)
        if self.show["cspace"]:
            self._draw_cspace(world, sim)
        self._draw_top_bar(sim)
        self._draw_panel(sim, widgets)
        if sim.busy:
            self._draw_planning_overlay(world, sim)
        if self.show["help"]:
            self._draw_help()

    # ------------------------------------------------------------ world
    def _draw_grid(self, surf):
        x0, x1 = cfg.VIEW_X_RANGE
        for x in np.arange(np.ceil(x0 * 2) / 2, x1 + 0.01, 0.5):
            sx = to_screen([x, 0])[0]
            pygame.draw.line(surf, C["grid"], (sx, cfg.TOP_BAR_HEIGHT), (sx, cfg.WINDOW_HEIGHT), 1)
        for y in np.arange(0, VIEW_TOP_Y, 0.5):
            sy = to_screen([0, y])[1]
            pygame.draw.line(surf, C["grid"], (0, sy), (VIEW_W, sy), 1)

    def _draw_environment(self, surf, sim):
        env = sim.env
        f = self.fonts
        for i, o in enumerate(env.obstacles):
            pts = to_screen(o.points)
            if o.kind == "boundary":
                aa_polygon(surf, C["floor"] if o.name == "Floor" else C["boundary"], pts)
                if o.name == "Floor":
                    y = to_screen([0, 0])[1]
                    pygame.draw.line(surf, C["floor_line"], (0, y), (VIEW_W, y), 3)
                    for x in range(-20, VIEW_W, 22):
                        pygame.draw.line(surf, lighten(C["floor_line"], 0.3), (x, y + 4), (x + 14, y + 18), 1)
                continue
            if o.kind == "pedestal":
                aa_polygon(surf, C["pedestal"], pts)
                continue
            if o.kind == "table":
                aa_polygon(surf, C["table"], pts)
                pygame.draw.polygon(surf, C["table_edge"], pts, 2)
                top = pts[:, 1].min()
                pygame.draw.line(surf, C["table_edge"], (pts[:, 0].min(), top + 6), (pts[:, 0].max(), top + 6), 2)
                continue
            hovered = i in (self.hover_obstacle, self.drag_obstacle)
            aa_polygon(surf, C["obstacle_hover"] if hovered else C["obstacle"], pts)
            pygame.draw.polygon(surf, C["obstacle_edge"], pts, 2)
            # diagonal hatching = "solid matter"
            clip = surf.get_clip()
            lo, hi = pts.min(0), pts.max(0)
            surf.set_clip(pygame.Rect(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]))
            hatch = pygame.Surface((hi[0] - lo[0] + 1, hi[1] - lo[1] + 1), pygame.SRCALPHA)
            for k in range(-int(hi[1] - lo[1]), int(hi[0] - lo[0]), 14):
                pygame.draw.line(hatch, (255, 255, 255, 28), (k, hatch.get_height()), (k + hatch.get_height(), 0), 2)
            mask = pygame.Surface(hatch.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(mask, (255, 255, 255, 255), pts - lo)
            hatch.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            surf.blit(hatch, lo)
            surf.set_clip(clip)
            label = f["tiny"].render(o.name, True, (230, 234, 242))
            c = pts.mean(axis=0)
            if label.get_width() < hi[0] - lo[0] - 4:
                surf.blit(label, label.get_rect(center=(c[0], max(c[1], cfg.TOP_BAR_HEIGHT + 14))))
        # stations
        for name, st in env.stations.items():
            color = C["station_a"] if name == "A" else C["station_b"]
            p = to_screen([st.x, st.top / 2])
            aa_circle(surf, (255, 255, 255), p, 17)
            aa_circle(surf, color, p, 15)
            t = f["label"].render(name, True, (255, 255, 255))
            surf.blit(t, t.get_rect(center=(p[0], p[1] + 1)))
            if env.object_station != name:
                # dashed outline: where the object will be placed
                c = st.object_rest_center(env.object_size)
                corners = to_screen(sim.robot.box_corners(c, 0.0, env.object_half_size()))
                for a, b in zip(corners, np.roll(corners, -1, axis=0)):
                    dashed_line(surf, color, a, b, 6, 2)

    # ------------------------------------------------------------ plan layer
    def _plan_layer_key(self, sim):
        return (id(sim.visual), self.show["trajectory"], self.show["tree"], self.show["ghosts"],
                tuple(sim.robot.lengths))

    def _draw_plan_layer(self, surf, sim):
        if sim.visual is None:
            return
        key = self._plan_layer_key(sim)
        if key != self._layer_key:
            self._layer_key = key
            self._layer = self._build_plan_layer(sim)
        # fade the drawing if the plan is out of date
        self._layer.set_alpha(255 if sim.plan_valid or sim.state in ("running", "done") else 110)
        surf.blit(self._layer, (0, 0))

    def _build_plan_layer(self, sim):
        layer = pygame.Surface((VIEW_W, cfg.WINDOW_HEIGHT), pygame.SRCALPHA)
        vis = sim.visual
        if self.show["tree"]:
            for tree in vis["trees"]:
                color = C["tree_start"] if tree["is_start"] else C["tree_goal"]
                pts = to_screen(tree["tcp"])
                for i, p in enumerate(tree["parents"]):
                    if p >= 0:
                        pygame.draw.line(layer, color + (95,), pts[i], pts[p], 1)
                for p in pts:
                    pygame.draw.circle(layer, color + (150,), p, 2)
            for mv in vis["moves"]:
                if mv["raw_tcp"] is not None:
                    pts = to_screen(mv["raw_tcp"])
                    pygame.draw.lines(layer, C["raw_path"] + (200,), False, pts, 2)
        if self.show["ghosts"]:
            for mv in vis["moves"]:
                for q in mv["ghosts"]:
                    self._draw_ghost_arm(layer, sim, q, (205, 120, 40), 140, held=mv["held"])
        if self.show["trajectory"]:
            for mv in vis["moves"]:
                color = {"carry": C["object_edge"], "transit": C["path_tcp"]}.get(mv["kind"], (120, 30, 110))
                pts = to_screen(mv["tcp"])
                if len(pts) > 1:
                    pygame.draw.lines(layer, color + (230,), False, pts, 3)
                pygame.draw.circle(layer, color + (255,), pts[0], 4)
                pygame.draw.circle(layer, color + (255,), pts[-1], 4)
        return layer

    def _draw_ghost_arm(self, surf, sim, q, color, alpha, held=None, width=3):
        """A see-through "skeleton" of the arm at configuration q."""
        robot = sim.robot
        pts = to_screen(robot.forward_kinematics(q))
        rgba = tuple(color[:3]) + (alpha,)
        pygame.draw.lines(surf, rgba, False, pts, width)
        for p in pts[:-1]:
            pygame.draw.circle(surf, rgba, p, width + 2)
        starts, ends = robot.gripper_segments(q, robot.finger_closed if held is not None else robot.finger_open)
        for a, b in zip(to_screen(starts), to_screen(ends)):
            pygame.draw.line(surf, rgba, a, b, 2)
        if held is not None:
            c, ang = robot.object_pose(q, held)
            corners = to_screen(robot.box_corners(c, ang, held.half_size))
            pygame.draw.polygon(surf, C["object_edge"] + (alpha,), corners, 2)

    # ------------------------------------------------------------ robot & object
    def _draw_object(self, surf, sim):
        c, a = sim.object_pose()
        corners = to_screen(sim.robot.box_corners(c, a, sim.env.object_half_size()))
        aa_polygon(surf, C["object_edge"], corners)
        center = corners.mean(axis=0)
        inner = center + (corners - center) * 0.86
        aa_polygon(surf, C["object"], inner)
        # parcel tape
        mid_a = (corners[0] + corners[3]) / 2
        mid_b = (corners[1] + corners[2]) / 2
        pygame.draw.line(surf, lighten(C["object"], 0.45), mid_a + (center - mid_a) * 0.14,
                         mid_b + (center - mid_b) * 0.14, 4)

    def _draw_robot(self, surf, sim, report):
        robot = sim.robot
        q = sim.q
        pts = to_screen(robot.forward_kinematics(q))
        colliding = report["colliding"]
        # links
        for k in range(robot.n_joints):
            fill = C["link_colliding"] if k in colliding else C["link"]
            outlined_capsule(surf, fill, C["link_edge"], pts[k], pts[k + 1], robot.radii[k] * SCALE)
        # gripper
        starts, ends = robot.gripper_segments(q, sim.finger)
        radii = [robot.palm_radius, robot.finger_radius, robot.finger_radius]
        for j, (a, b, r) in enumerate(zip(to_screen(starts), to_screen(ends), radii)):
            fill = C["link_colliding"] if robot.n_joints + j in colliding else C["gripper"]
            outlined_capsule(surf, fill, (30, 30, 36), a, b, r * SCALE, 1)
        # joints
        for k in range(robot.n_joints):
            r = robot.radii[k] * SCALE * 0.78
            aa_circle(surf, C["joint"], pts[k], r)
            aa_circle(surf, C["joint_center"], pts[k], max(2, r * 0.35))
            if k == sim.selected_joint and sim.state != "running":
                pygame.gfxdraw.aacircle(surf, int(pts[k][0]), int(pts[k][1]), int(r + 7), C["custom_goal"])
                pygame.gfxdraw.aacircle(surf, int(pts[k][0]), int(pts[k][1]), int(r + 8), C["custom_goal"])
        aa_circle(surf, C["joint"], pts[-1], robot.radii[-1] * SCALE * 0.6)
        # tool centre point
        tcp = to_screen(robot.tcp(q))
        pygame.draw.line(surf, C["path_tcp"], tcp - (6, 0), tcp + (6, 0), 2)
        pygame.draw.line(surf, C["path_tcp"], tcp - (0, 6), tcp + (0, 6), 2)

    def _draw_collision_view(self, surf, sim, report):
        robot = sim.robot
        checker = sim.current_checker()
        sa, sb, radii = robot.collision_parts(sim.q, checker.finger_offset, sim.held)
        layer = pygame.Surface((VIEW_W, cfg.WINDOW_HEIGHT), pygame.SRCALPHA)
        for k, (a, b, r) in enumerate(zip(to_screen(sa[0]), to_screen(sb[0]), radii)):
            color = C["collision"] if k in report["colliding"] else (40, 40, 60)
            rr = max(1, int((r + checker.margin) * SCALE))
            d = b - a
            n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6) * rr
            pygame.draw.line(layer, color + (220,), a + n, b + n, 1)
            pygame.draw.line(layer, color + (220,), a - n, b - n, 1)
            pygame.draw.circle(layer, color + (220,), a, rr, 1)
            pygame.draw.circle(layer, color + (220,), b, rr, 1)
        surf.blit(layer, (0, 0))
        part, obstacle, pr, po, dist = report["closest"]
        color = C["collision"] if report["colliding"] else C["clearance"]
        a, b = to_screen(pr), to_screen(po)
        pygame.draw.line(surf, color, a, b, 3)
        aa_circle(surf, color, a, 4)
        aa_circle(surf, color, b, 4)
        t = self.fonts["small"].render(f"{dist:.3f} m  ({part} - {obstacle})", True, color)
        box = t.get_rect(topleft=(b[0] + 10, b[1] - 8)).inflate(8, 4)
        box.clamp_ip(surf.get_rect())
        pygame.draw.rect(surf, (255, 255, 255), box, border_radius=4)
        pygame.draw.rect(surf, color, box, 1, border_radius=4)
        surf.blit(t, t.get_rect(center=box.center))
        for k, msg in enumerate(report["pairs"][:4]):
            t = self.fonts["small"].render("COLLISION: " + msg, True, C["collision"])
            surf.blit(t, (12, cfg.TOP_BAR_HEIGHT + 10 + 18 * k))

    # ------------------------------------------------------------ C-space inset
    def _draw_cspace(self, surf, sim):
        robot = sim.robot
        if robot.n_joints < 2:
            return
        w, h = 300, 220
        box = pygame.Rect(VIEW_W - w - 12, cfg.TOP_BAR_HEIGHT + 10, w, h)
        plot = pygame.Rect(box.x + 34, box.y + 24, w - 44, h - 50)
        key = (sim.scene_version, sim.env.object_station, sim.held is None,
               tuple(np.round(sim.q[2:], 2)), tuple(robot.lengths))
        now = time.perf_counter()
        if self._cspace is None or (self._cspace[0] != key and now - self._cspace[2] > 0.25):
            nx, ny = 96, 72
            g1 = np.linspace(robot.lower[0], robot.upper[0], nx)
            g2 = np.linspace(robot.lower[1], robot.upper[1], ny)
            G = np.zeros((nx * ny, robot.n_joints))
            G[:, 0] = np.repeat(g1, ny)
            G[:, 1] = np.tile(g2, nx)
            G[:, 2:] = sim.q[2:]
            free = sim.current_checker().valid_batch(G).reshape(nx, ny)[:, ::-1]
            img = np.where(free[..., None], np.array(C["background"]), np.array(C["obstacle"]))
            self._cspace = (key, pygame.transform.scale(pygame.surfarray.make_surface(img.astype(np.uint8)),
                                                        plot.size), now)
        panel = pygame.Surface(box.size, pygame.SRCALPHA)
        panel.fill((255, 255, 255, 235))
        surf.blit(panel, box)
        pygame.draw.rect(surf, C["obstacle_edge"], box, 1, border_radius=4)
        surf.blit(self._cspace[1], plot)

        def cs(q):
            q = np.atleast_2d(q)
            x = plot.x + (q[:, 0] - robot.lower[0]) / (robot.upper[0] - robot.lower[0]) * plot.w
            y = plot.bottom - (q[:, 1] - robot.lower[1]) / (robot.upper[1] - robot.lower[1]) * plot.h
            return np.stack([x, y], axis=-1)

        clip = surf.get_clip()
        surf.set_clip(plot)
        if sim.visual is not None:
            if self.show["tree"]:
                for tree in sim.visual["trees"]:
                    color = C["tree_start"] if tree["is_start"] else C["tree_goal"]
                    pts = cs(tree["nodes"])
                    for i, p in enumerate(tree["parents"]):
                        if p >= 0:
                            pygame.draw.line(surf, color, pts[i], pts[p], 1)
            for mv in sim.visual["moves"]:
                color = C["object_edge"] if mv["held"] is not None else C["path_tcp"]
                pts = cs(mv["path"])
                if len(pts) > 1:
                    pygame.draw.lines(surf, color, False, pts, 2)
        if sim.custom_goal is not None:
            aa_circle(surf, C["custom_goal"], cs(sim.custom_goal)[0], 5)
        aa_circle(surf, (0, 0, 0), cs(sim.q)[0], 6)
        aa_circle(surf, C["accent"], cs(sim.q)[0], 4)
        surf.set_clip(clip)
        f = self.fonts["tiny"]
        rest = ", ".join(f"q{k + 1}={np.degrees(v):.0f}°" for k, v in enumerate(sim.q[2:], start=2))
        surf.blit(f.render(f"C-space slice (q1, q2) at {rest}", True, C["text"]), (box.x + 8, box.y + 7))
        surf.blit(f.render(f"q1 {np.degrees(robot.lower[0]):.0f}° to {np.degrees(robot.upper[0]):.0f}°",
                           True, C["text"]), (plot.x, plot.bottom + 5))
        surf.blit(f.render("dark = collision", True, C["obstacle"]), (plot.right - 80, plot.bottom + 5))
        lab = pygame.transform.rotate(f.render("q2", True, C["text"]), 90)
        surf.blit(lab, (box.x + 10, plot.centery - 8))

    # ------------------------------------------------------------ text
    def _draw_readout(self, surf, sim, report):
        q = np.degrees(sim.q)
        tcp = sim.robot.tcp(sim.q)
        joints = "  ".join(f"q{k + 1} {v:7.1f}°" for k, v in enumerate(q))
        clr = report["clearance"]
        color = C["error"] if report["colliding"] else C["text"]
        text = f"{joints}    TCP ({tcp[0]:.2f}, {tcp[1]:.2f}) m    clearance {clr:.3f} m"
        if sim.held is not None:
            text += "    holding object"
        t = self.fonts["small"].render(text, True, color)
        surf.blit(t, (10, cfg.WINDOW_HEIGHT - 22))
        hint = self.fonts["tiny"].render("H: help   1-9 or Up/Down: select joint   Left/Right: jog   drag blue blocks with the mouse",
                                         True, (90, 80, 64))
        surf.blit(hint, hint.get_rect(bottomright=(VIEW_W - 10, cfg.WINDOW_HEIGHT - 26)))

    def _draw_top_bar(self, sim):
        s = self.screen
        bar = pygame.Rect(0, 0, VIEW_W, cfg.TOP_BAR_HEIGHT)
        pygame.draw.rect(s, (255, 255, 255), bar)
        pygame.draw.line(s, C["grid"], (0, bar.bottom - 1), (VIEW_W, bar.bottom - 1), 1)
        state_colors = {"idle": (150, 150, 160), "planning": C["button_on"], "ready": C["ok"],
                        "running": C["accent"], "paused": C["warn"], "done": C["ok"], "failed": C["error"]}
        badge = pygame.Rect(10, 8, 92, 24)
        pygame.draw.rect(s, state_colors.get(sim.state, (150, 150, 160)), badge, border_radius=12)
        t = self.fonts["button"].render(sim.state.upper(), True, (255, 255, 255))
        s.blit(t, t.get_rect(center=badge.center))
        level = {"info": C["text"], "ok": (30, 130, 70), "warn": (170, 110, 0), "error": (200, 40, 40)}
        msg = sim.status
        font = self.fonts["text"]
        while font.size(msg)[0] > VIEW_W - 130 and len(msg) > 10:
            msg = msg[:-4] + "..."
        s.blit(font.render(msg, True, level[sim.status_level]), (114, 12))

    def _draw_panel(self, sim, widgets):
        s = self.screen
        x0 = VIEW_W
        pygame.draw.rect(s, C["panel"], (x0, 0, cfg.PANEL_WIDTH, cfg.WINDOW_HEIGHT))
        s.blit(self.fonts["title"].render("Robot arm motion planner", True, C["panel_text"]), (x0 + 16, 12))
        planner = f"OMPL {cfg.OMPL_ALGORITHM}" if sim.backend == "ompl" else "RRT-Connect (built-in)"
        s.blit(self.fonts["small"].render(f"{sim.robot.n_joints}-joint planar arm  ·  {planner}",
                                          True, C["panel_dim"]), (x0 + 16, 34))
        mouse = pygame.mouse.get_pos()
        for w in widgets["buttons"] + widgets["sliders"]:
            w.draw(s, self.fonts, mouse)
        # metrics
        y = widgets["metrics_y"]
        s.blit(self.fonts["button"].render("Last plan", True, C["panel_text"]), (x0 + 16, y))
        y += 22
        m = sim.metrics
        if not m:
            s.blit(self.fonts["small"].render("no plan yet", True, C["panel_dim"]), (x0 + 16, y))
            return
        ok = m.get("success")
        rows = [
            ("Result", "success" if ok else "FAILED", C["ok"] if ok else C["error"]),
            ("Planning time", f"{m['planning_time_s']:.2f} s  (RRT {m['rrt_time_s']:.2f} s)", None),
            ("Samples / tree nodes", f"{m['samples']:,} / {m['tree_nodes']:,}", None),
            ("Collision checks", f"{m['collision_checks']:,}", None),
        ]
        if "joint_path_length_rad" in m:
            rows += [
                ("Joint-space length", f"{m['joint_path_length_rad']:.2f} rad (raw {m['raw_joint_path_length_rad']:.1f})", None),
                ("Tool (TCP) path", f"{m['tcp_path_length_m']:.2f} m", None),
                ("Clearance free / min", f"{m['free_space_clearance_m']:.3f} / {m['min_clearance_m']:.3f} m", None),
            ]
        for label, value, color in rows:
            s.blit(self.fonts["small"].render(label, True, C["panel_dim"]), (x0 + 16, y))
            v = self.fonts["small"].render(value, True, color or C["panel_text"])
            s.blit(v, v.get_rect(topright=(x0 + cfg.PANEL_WIDTH - 14, y)))
            y += 18
        s.blit(self.fonts["tiny"].render("Every plan is appended to results/metrics_log.csv", True, C["panel_dim"]),
               (x0 + 16, y + 4))

    def _draw_planning_overlay(self, surf, sim):
        elapsed = time.perf_counter() - sim.planning_started
        dots = "." * (1 + int(elapsed * 3) % 3)
        text = f"Planning {sim.planning_what}{dots}  {elapsed:.1f} s   (Esc to cancel)"
        t = self.fonts["text"].render(text, True, (255, 255, 255))
        box = t.get_rect(center=(VIEW_W // 2, cfg.TOP_BAR_HEIGHT + 30)).inflate(24, 14)
        pygame.draw.rect(surf, C["button_on"], box, border_radius=10)
        surf.blit(t, t.get_rect(center=box.center))

    def _draw_help(self):
        lines = [
            "MOUSE   drag a blue obstacle to move it  ·  click the buttons and sliders on the right",
            "P       plan pick-and-place from the current pose       SPACE   start / pause",
            "R       reset arm and object                            N       regenerate obstacles (verified solvable)",
            "D       restore the default obstacle layout             Esc     cancel planning / close help",
            "1-9, Up/Down   select joint        Left/Right   jog the selected joint (hold Shift = faster)",
            "G       store current pose as joint goal               J       plan to the stored joint goal",
            "T  trajectory   Y  RRT tree   O  ghost arms   C  collision view   K  C-space view",
            "+ / -   animation speed                                 S       toggle path smoothing",
            "",
            "Colours: orange = robot, blue = movable obstacles, green box = object, magenta / green = tool path,",
            "blue / orange thin lines = the two RRT-Connect trees (drawn where the tool would be).",
        ]
        w, h = 880, 30 + 22 * len(lines)
        box = pygame.Rect((VIEW_W - w) // 2, 120, w, h)
        overlay = pygame.Surface(box.size, pygame.SRCALPHA)
        overlay.fill((30, 34, 46, 238))
        self.screen.blit(overlay, box)
        for k, line in enumerate(lines):
            self.screen.blit(self.fonts["text"].render(line, True, C["panel_text"]), (box.x + 20, box.y + 16 + 22 * k))
