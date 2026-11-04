"""Cobuild · Live Assist: camera worker calibration -> personalised robot simulation.

    camera -> pose tracking -> 10 s calibration -> worker profile (JSON)
           -> bridge (measured / stated / default provenance)
           -> Cobuild drywall simulation (no robot / reactive / generic / personalised)
           -> live view, status cards, analytics, 3D view

Every number on screen comes from tracking, calibration or the simulation; nothing is mocked.
RESEARCH PROTOTYPE: simulation only; never connect to a physical robot.
"""
from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageTk  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402  (also puts human_robot_coordination on sys.path)
import sim_client  # noqa: E402
from camera_worker import CameraWorker  # noqa: E402
from config import PROFILES_DIR  # noqa: E402
from human_tracking import TrackingStatus  # noqa: E402
from worker_profile import StatedPreferences, load_profile, save_profile  # noqa: E402

# ---- design tokens (dark industrial, yellow accent) --------------------------------------
BG, PANEL, PANEL2, LINE = "#0f1115", "#181b21", "#20242c", "#2b303a"
INK, MUTED = "#eceef2", "#98a0ad"
YELLOW, YELLOW_DK, YELLOW_BG = "#f5c518", "#d4a90f", "#2a2410"
GREEN, AMBER, RED = "#3ddc84", "#ffb020", "#ff5a5f"
BLUE, GREYBAR = "#6aa5e8", "#59616f"
FONT = "Segoe UI"
SOURCE_COL = {"measured": GREEN, "stated": BLUE, "manual": YELLOW, "default": AMBER}

SHORT = {"no_robot": "No robot", "A_reactive": "A · Reactive",
         "B_proactive_generic": "B · Generic robot", "C_cobuild_rule": "C · Personalised"}


def rgb(hex_):
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def make_demo_profile():
    """Profile from a SYNTHETIC stick-figure worker (no camera). Clearly not a real measurement."""
    sys.path.insert(0, str(bridge.HRC_DIR / "tests"))
    from config import CalibrationConfig, CameraConfig, TrackingConfig
    from human_tracking import MotionAnalyzer
    from synthetic import FPS, H, W, make_pose
    from worker_calibration import WorkerCalibrator

    def pose(t):
        if t < 3:
            return make_pose(0.0, 3.0)
        if t < 5:
            return make_pose(0.9 * (t - 3), 3.0)
        if t < 7:
            return make_pose(1.8 - 0.9 * (t - 5), 3.0)
        return make_pose(0.0, 3.0, arms="out")

    an = MotionAnalyzer(TrackingConfig(), CameraConfig())
    cal = WorkerCalibrator(CalibrationConfig(), CameraConfig().horizontal_fov_deg)
    cal.start(0.0)
    for i in range(int(10 * FPS) + 1):
        t = i / FPS
        cal.add(an.update(pose(t), t, (W, H)))
    prof = cal.build_profile()
    prof.profile_id = "worker-DEMO-synthetic"
    return prof


def _font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class App(tk.Tk):
    def __init__(self):
        enable_dpi_awareness()
        super().__init__()
        self.S = max(1.0, self.winfo_fpixels("1i") / 96.0)
        self.title("Cobuild · Live Assist (research prototype, simulation only)")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(self.px(1400), sw - 30)}x{min(self.px(860), sh - 80)}+10+10")
        self.minsize(self.px(1050), self.px(640))
        self.configure(bg=BG)
        self.camera = CameraWorker()
        self.profile = None
        self.sim_profile = None
        self.results = None
        self.reviewed = False
        self.events = queue.Queue()
        self._view_img = self._snap_img = None
        self._snapshot_pil = None          # latest 3D render (PIL), shown when the camera is off
        self._panel_tick = 0
        self.fonts = {}
        self._setup_style()
        self._build_ui()
        self.show_page("live")
        self.after(40, self._poll)

    def px(self, v):
        return int(v * getattr(self, "S", 1.0))

    # ----------------------------------------------------------------- styling --
    def _setup_style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", font=(FONT, 10), background=BG, foreground=INK, bordercolor=LINE,
                     lightcolor=PANEL, darkcolor=PANEL, focuscolor=PANEL)
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=PANEL, foreground=INK)
        st.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=(FONT, 9))
        st.configure("H.TLabel", background=PANEL, foreground=INK, font=(FONT, 12, "bold"))
        st.configure("TButton", background=PANEL2, foreground=INK, bordercolor=LINE, padding=(14, 8),
                     relief="flat", font=(FONT, 10))
        st.map("TButton", background=[("active", "#2b303a"), ("disabled", PANEL)],
               foreground=[("disabled", "#5d6572")])
        st.configure("Primary.TButton", background=YELLOW, foreground="#14120a", bordercolor=YELLOW,
                     font=(FONT, 10, "bold"), padding=(16, 8))
        st.map("Primary.TButton", background=[("active", YELLOW_DK), ("disabled", "#5b4f1c")],
               foreground=[("disabled", "#2a2410")])
        for w in ("TEntry", "TCombobox", "TSpinbox"):
            st.configure(w, fieldbackground=PANEL2, foreground=INK, bordercolor=LINE, insertcolor=INK, padding=5,
                         arrowcolor=INK)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL2)], foreground=[("readonly", INK)])
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=INK, borderwidth=0,
                     rowheight=self.px(30), font=(FONT, 10))
        st.configure("Treeview.Heading", background=PANEL2, foreground=MUTED, font=(FONT, 9, "bold"),
                     relief="flat", padding=(8, 6))
        st.map("Treeview", background=[("selected", "#2d3340")], foreground=[("selected", INK)])
        st.configure("Horizontal.TProgressbar", background=YELLOW, troughcolor=PANEL2, borderwidth=0,
                     thickness=self.px(8))

    def card(self, parent, title=None, icon=None, pack=None, **grid):
        """Card with optional header. `pack={...}` packs it; otherwise it is placed with grid(**grid)."""
        outer = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        if pack is not None:
            outer.pack(**pack)
        else:
            outer.grid(**grid)
        body = tk.Frame(outer, bg=PANEL)
        body.pack(fill="both", expand=True, padx=self.px(12), pady=self.px(6))
        if title:
            head = tk.Frame(body, bg=PANEL)
            head.pack(fill="x")
            if icon:
                tk.Label(head, text=icon, font=("Segoe UI Symbol", 13), bg=PANEL, fg=YELLOW).pack(side="left")
            tk.Label(head, text=("  " if icon else "") + title, font=(FONT, 11, "bold"), bg=PANEL, fg=INK).pack(side="left")
        return body

    @staticmethod
    def lab(parent, text="", fg=INK, size=10, bold=False, **kw):
        return tk.Label(parent, text=text, bg=kw.pop("bg", PANEL), fg=fg, font=(FONT, size, "bold" if bold else "normal"),
                        anchor="w", justify="left", **kw)

    # ---------------------------------------------------------------------- UI --
    def _build_ui(self):
        # --- header ---
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x")
        hrow = tk.Frame(header, bg=BG)
        hrow.pack(fill="x", padx=self.px(16), pady=self.px(10))
        logo = tk.Canvas(hrow, width=self.px(46), height=self.px(24), bg=BG, highlightthickness=0)
        logo.pack(side="left")
        for i in range(2):
            x = self.px(6 + i * 14)
            logo.create_polygon(x + self.px(8), 0, x + self.px(18), 0, x + self.px(10), self.px(24), x, self.px(24), fill=YELLOW)
        tk.Label(hrow, text=" COBUILD", font=(FONT, 17, "bold"), bg=BG, fg=INK).pack(side="left")
        tk.Frame(hrow, bg=LINE, width=1, height=self.px(28)).pack(side="left", padx=self.px(16))
        tk.Label(hrow, text="Live Assist", font=(FONT, 17, "bold"), bg=BG, fg=INK).pack(side="left")
        tk.Label(hrow, text="   Worker-aware human–robot collaboration, simulated", font=(FONT, 10), bg=BG,
                 fg=MUTED).pack(side="left", pady=(self.px(4), 0))
        tk.Label(hrow, text="  RESEARCH PROTOTYPE · SIMULATION ONLY  ", font=(FONT, 8, "bold"), bg=YELLOW,
                 fg="#14120a").pack(side="right")
        tk.Frame(self, bg=LINE, height=1).pack(fill="x")

        self.status = tk.StringVar(value="Ready. Start the camera and calibrate, load a profile, or use the demo worker.")
        tk.Label(self, textvariable=self.status, bg=BG, fg=MUTED, font=(FONT, 9), anchor="w",
                 padx=self.px(16), pady=self.px(6)).pack(fill="x", side="bottom")
        tk.Frame(self, bg=LINE, height=1).pack(fill="x", side="bottom")

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)
        self.nav = tk.Frame(main, bg=BG, width=self.px(190))
        self.nav.pack(side="left", fill="y")
        self.nav.pack_propagate(False)
        self.content = tk.Frame(main, bg=BG)
        self.content.pack(side="left", fill="both", expand=True, padx=(0, self.px(12)), pady=self.px(10))
        self.nav_items = {}
        for key, icon, text in (("overview", "⌂", "Overview"), ("configure", "☰", "Configure"),
                                ("live", "▷", "Live Operation"), ("analytics", "▥", "Analytics"),
                                ("settings", "⚙", "Settings")):
            self._nav_item(key, icon, text)

        self.pages = {k: tk.Frame(self.content, bg=BG) for k in self.nav_items}
        self._build_live(self.pages["live"])
        self._build_overview(self.pages["overview"])
        self._build_configure(self.pages["configure"])
        self._build_analytics(self.pages["analytics"])
        self._build_settings(self.pages["settings"])

    def _nav_item(self, key, icon, text):
        row = tk.Frame(self.nav, bg=BG, cursor="hand2")
        row.pack(fill="x", pady=self.px(2), padx=(self.px(8), self.px(8)))
        bar = tk.Frame(row, bg=BG, width=3)
        bar.pack(side="left", fill="y")
        ic = tk.Label(row, text=icon, font=("Segoe UI Symbol", 14), bg=BG, fg=MUTED, width=2)
        ic.pack(side="left", pady=self.px(9))
        tx = tk.Label(row, text=text, font=(FONT, 11), bg=BG, fg=MUTED, anchor="w")
        tx.pack(side="left", fill="x", expand=True)
        for w in (row, ic, tx, bar):
            w.bind("<Button-1>", lambda e, k=key: self.show_page(k))
        self.nav_items[key] = (row, bar, ic, tx)

    def show_page(self, key):
        for k, (row, bar, ic, tx) in self.nav_items.items():
            on = k == key
            bg = YELLOW_BG if on else BG
            for w in (row, ic, tx):
                w.config(bg=bg)
            bar.config(bg=YELLOW if on else bg)
            ic.config(fg=YELLOW if on else MUTED)
            tx.config(fg=YELLOW if on else MUTED, font=(FONT, 11, "bold" if on else "normal"))
            self.pages[k].pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        if key == "analytics" and self.results:
            self.reviewed = True
            self._refresh_panels()

    # ------------------------------------------------------------- live page --
    def _build_live(self, page):
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=0, minsize=self.px(340))
        page.rowconfigure(0, weight=1)
        page.rowconfigure(1, weight=0)

        # --- video card (left) ---
        vcard = tk.Frame(page, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        vcard.grid(row=0, column=0, sticky="nsew", padx=(0, self.px(10)), pady=(0, self.px(10)))
        self.view = tk.Canvas(vcard, bg="#0a0c10", highlightthickness=0)
        self.view.pack(fill="both", expand=True)
        self.view.bind("<Configure>", lambda e: self._render_view())
        bar = tk.Frame(vcard, bg=PANEL)
        bar.pack(fill="x", padx=self.px(10), pady=self.px(8))
        self.btn_cam = ttk.Button(bar, text="Start camera", command=self.toggle_camera)
        self.btn_cam.pack(side="left")
        self.btn_cal = ttk.Button(bar, text="Calibrate 10 s", style="Primary.TButton", command=self.calibrate,
                                  state="disabled")
        self.btn_cal.pack(side="left", padx=self.px(8))
        ttk.Button(bar, text="Load profile…", command=self.load_profile_dialog).pack(side="left")
        ttk.Button(bar, text="Demo worker (no camera)", command=self.use_demo).pack(side="left", padx=self.px(8))
        self.btn_run = ttk.Button(bar, text="▶  Run simulation", style="Primary.TButton", command=self.run_sim,
                                  state="disabled")
        self.btn_run.pack(side="right")

        # --- right cards ---
        rcol = tk.Frame(page, bg=BG)
        rcol.grid(row=0, column=1, sticky="nsew", pady=(0, self.px(10)))
        rcol.columnconfigure(0, weight=1)
        self.c_profile = self.card(rcol, "Worker Profile", "☺", row=0, column=0, sticky="ew")
        self.l_prof = [self.lab(self.c_profile, "", MUTED, 9) for _ in range(2)]
        for l in self.l_prof:
            l.pack(fill="x")
        c = self.card(rcol, "Task / Procedure", "▤", row=1, column=0, sticky="ew", pady=(self.px(4), 0))
        self.lab(c, "Drywall installation: carry → hold → fasten → repeat", INK, 9, wraplength=self.px(300)).pack(fill="x")
        c = self.card(rcol, "Current Activity", "◎", row=2, column=0, sticky="ew", pady=(self.px(4), 0))
        self.l_act = self.lab(c, "–", INK, 11, True)
        self.l_act.pack(fill="x")
        self.l_act2 = self.lab(c, "Camera off", MUTED, 9)
        self.l_act2.pack(fill="x")
        c = self.card(rcol, "Planner Decision", "✺", row=3, column=0, sticky="ew", pady=(self.px(4), 0))
        self.l_dec = tk.Label(c, text="Run the simulation", bg=YELLOW_BG, fg=YELLOW, font=(FONT, 10, "bold"),
                              anchor="w", padx=self.px(8), pady=self.px(4))
        self.l_dec.pack(fill="x", pady=(self.px(4), 0))
        self.l_dec2 = self.lab(c, "", MUTED, 9, wraplength=self.px(290))
        self.l_dec2.pack(fill="x", pady=(self.px(4), 0))
        c = self.card(rcol, "Next Robot Action", "»", row=4, column=0, sticky="ew", pady=(self.px(4), 0))
        self.l_next = self.lab(c, "Waiting for worker tracking.", MUTED, 9, wraplength=self.px(290))
        self.l_next.pack(fill="x")
        c = self.card(rcol, "Profile Data Quality / Readiness", "▮", row=5, column=0, sticky="ew", pady=(self.px(4), 0))
        qrow = tk.Frame(c, bg=PANEL)
        qrow.pack(fill="x", pady=(self.px(4), 0))
        self.q_canvas = tk.Canvas(qrow, height=self.px(10), bg=PANEL, highlightthickness=0)
        self.q_canvas.pack(side="left", fill="x", expand=True)
        self.l_q = self.lab(qrow, "–", INK, 10, True)
        self.l_q.pack(side="right", padx=(self.px(8), 0))
        self.l_ready = self.lab(c, "● Not ready · no profile", AMBER, 9)
        self.l_ready.pack(fill="x", pady=(self.px(4), 0))

        # --- bottom row ---
        bottom = tk.Frame(page, bg=BG)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        for i in range(3):
            bottom.columnconfigure(i, weight=1, uniform="b")
        c = self.card(bottom, "Task Flow", "▤", row=0, column=0, sticky="nsew", padx=(0, self.px(8)))
        self.flow = tk.Canvas(c, height=self.px(70), bg=PANEL, highlightthickness=0)
        self.flow.pack(fill="x", pady=(self.px(6), 0))
        self.flow.bind("<Configure>", lambda e: self._draw_flow())
        c = self.card(bottom, "Robot Status (simulated)", "⚙", row=0, column=1, sticky="nsew", padx=(0, self.px(8)))
        rs = tk.Frame(c, bg=PANEL)
        rs.pack(fill="x", pady=(self.px(6), 0))
        icon = tk.Canvas(rs, width=self.px(74), height=self.px(66), bg=PANEL, highlightthickness=0)
        icon.pack(side="left")
        self._draw_robot_icon(icon)
        rows = tk.Frame(rs, bg=PANEL)
        rows.pack(side="left", padx=self.px(10))
        self.rs_dots, self.rs_vals = {}, {}
        for i, key in enumerate(("Mode", "Tracking", "Profile", "Camera")):
            dot = tk.Label(rows, text="●", fg=MUTED, bg=PANEL, font=(FONT, 9))
            dot.grid(row=i, column=0, sticky="w")
            tk.Label(rows, text=key + ":", fg=MUTED, bg=PANEL, font=(FONT, 9, "bold")).grid(row=i, column=1, sticky="w", padx=4)
            val = tk.Label(rows, text="–", fg=INK, bg=PANEL, font=(FONT, 9), anchor="w")
            val.grid(row=i, column=2, sticky="w")
            self.rs_dots[key], self.rs_vals[key] = dot, val
        c = self.card(bottom, "Expected Value (simulated)", "▮", row=0, column=2, sticky="nsew")
        ev = tk.Frame(c, bg=PANEL)
        ev.pack(fill="x", pady=(self.px(6), 0))
        self.ev = {}
        for i, (key, icon_, label) in enumerate((("time", "◷", "Task time vs working alone"),
                                                 ("strain", "☹", "Worker strain vs working alone"),
                                                 ("cost", "✔", "Cost vs generic robot"))):
            ev.columnconfigure(i, weight=1, uniform="ev")
            f = tk.Frame(ev, bg=PANEL)
            f.grid(row=0, column=i, sticky="nsew")
            tk.Label(f, text=icon_, font=("Segoe UI Symbol", 12), fg=MUTED, bg=PANEL).pack()
            v = tk.Label(f, text="–", font=(FONT, 20, "bold"), fg=INK, bg=PANEL)
            v.pack()
            tk.Label(f, text=label, font=(FONT, 8), fg=MUTED, bg=PANEL, wraplength=self.px(120)).pack()
            self.ev[key] = v

    def _draw_robot_icon(self, c):
        p = self.px
        c.create_rectangle(p(10), p(62), p(44), p(70), fill="#3a404b", outline="")
        c.create_line(p(27), p(62), p(27), p(36), width=p(7), fill="#7b8494", capstyle="round")
        c.create_line(p(27), p(36), p(50), p(16), width=p(6), fill="#a2abb9", capstyle="round")
        c.create_line(p(50), p(16), p(66), p(24), width=p(5), fill=YELLOW, capstyle="round")
        for x, y in ((27, 36), (50, 16)):
            c.create_oval(p(x - 5), p(y - 5), p(x + 5), p(y + 5), fill=YELLOW, outline="")

    # ------------------------------------------------------- other pages --
    def _page_title(self, page, text, sub=""):
        tk.Label(page, text=text, font=(FONT, 16, "bold"), bg=BG, fg=INK, anchor="w").pack(fill="x", pady=(self.px(4), 0))
        if sub:
            tk.Label(page, text=sub, font=(FONT, 10), bg=BG, fg=MUTED, anchor="w", wraplength=self.px(900),
                     justify="left").pack(fill="x", pady=(0, self.px(10)))

    def _build_overview(self, page):
        self._page_title(page, "Overview", "Camera-measured worker profile → personalised robot, compared in simulation.")
        grid = tk.Frame(page, bg=BG)
        grid.pack(fill="x")
        for i in range(2):
            grid.columnconfigure(i, weight=1, uniform="o")
        c = self.card(grid, "How it works", "◎", row=0, column=0, sticky="nsew", padx=(0, self.px(8)))
        self.lab(c, ("1.  Track one worker with a webcam (MediaPipe pose, body landmarks only, no face "
                     "recognition).\n2.  Calibrate an anonymous profile in 10 s: speed, reach, body size, variability.\n"
                     "3.  Map it to the Cobuild drywall simulation and compare four robots on the SAME worker and the "
                     "same random variability: none, reactive, generic proactive, personalised.\n"
                     "4.  Review time, strain, charts and the 3D view in Analytics."), INK, 10, wraplength=self.px(430)
                 ).pack(fill="x", pady=(self.px(6), 0))
        c = self.card(grid, "Important limits", "⚠", row=0, column=1, sticky="nsew")
        self.lab(c, ("•  Academic simulation. Not certified; not for real construction sites; never connect to a "
                     "physical robot.\n•  Camera numbers are approximate (assumed 60° FOV, noisy single-camera depth).\n"
                     "•  Lifting capacity and fatigue cannot be measured by camera: default unless entered manually. "
                     "Pace comes from walking speed, only a proxy.\n•  Robot durations in the simulation are modelling "
                     "assumptions.\n•  The camera-side safety engine (clearance, prediction, motion-consistency, three "
                     "experimental conditions) is NOT built yet; today the only link is the tracking-lost SAFE-HOLD "
                     "indicator.\n•  PPO strategies excluded (need PyTorch); rule-based planners used."),
                 MUTED, 9, wraplength=self.px(430)).pack(fill="x", pady=(self.px(6), 0))

    def _build_configure(self, page):
        self._page_title(page, "Configure", "How camera measurements become the simulated worker. Source colours: "
                         "green measured, blue stated, yellow manual, amber default (not known).")
        c = self.card(page, "Profile → simulation mapping", "☰", pack=dict(fill="x"))
        cols = ("quantity", "camera", "simulation", "source")
        self.tree_profile = ttk.Treeview(c, columns=cols, show="headings", height=6)
        for col, wd in zip(cols, (180, 230, 140, 110)):
            self.tree_profile.heading(col, text=col.title(), anchor="w")
            self.tree_profile.column(col, width=self.px(wd), anchor="w")
        self.tree_profile.pack(fill="x", pady=(self.px(6), 0))
        for src, colr in SOURCE_COL.items():
            self.tree_profile.tag_configure(src, foreground=colr)
        self.tree_profile.tag_configure("warning", foreground=MUTED)

        row = tk.Frame(page, bg=BG)
        row.pack(fill="x", pady=(self.px(10), 0))
        row.columnconfigure(0, weight=3, uniform="c")
        row.columnconfigure(1, weight=2, uniform="c")
        c = self.card(row, "Manual overrides (researcher input)", "✎", row=0, column=0, sticky="nsew", padx=(0, self.px(8)))
        self.ov_vars = {k: tk.StringVar() for k in ("pace", "lift_capacity_kg", "fatigue_rate", "handedness")}
        g = tk.Frame(c, bg=PANEL)
        g.pack(fill="x", pady=(self.px(6), 0))
        for i, (k, hint) in enumerate((("pace", "0.8–1.3"), ("lift_capacity_kg", "kg, 22–50"),
                                       ("fatigue_rate", "0.0004–0.0016"), ("handedness", "left / right"))):
            self.lab(g, k, INK, 9).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(g, textvariable=self.ov_vars[k], width=10).grid(row=i, column=1, padx=8)
            self.lab(g, hint, MUTED, 9).grid(row=i, column=2, sticky="w")
        ttk.Button(c, text="Apply overrides", style="Primary.TButton", command=self.refresh_sim_profile).pack(
            anchor="w", pady=(self.px(8), 0))
        c = self.card(row, "Stated preferences", "☺", row=0, column=1, sticky="nsew")
        self.lab(c, "Told by the worker and applied at calibration. Never inferred.", MUTED, 9,
                 wraplength=self.px(300)).pack(fill="x", pady=(self.px(4), self.px(6)))
        self.var_side = tk.StringVar(value="none")
        self.var_clear = tk.StringVar()
        g = tk.Frame(c, bg=PANEL)
        g.pack(fill="x")
        self.lab(g, "Preferred tool side", INK, 9).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(g, textvariable=self.var_side, values=["none", "left", "right"], width=8,
                     state="readonly").grid(row=0, column=1, padx=8)
        self.lab(g, "Preferred clearance (m)", INK, 9).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(g, textvariable=self.var_clear, width=10).grid(row=1, column=1, padx=8)
        c = self.card(page, "Mapping assumptions", "ⓘ", pack=dict(fill="x", pady=(self.px(10), 0)))
        self.lab(c, "\n".join("•  " + n for n in bridge.MAPPING_NOTES), MUTED, 9, wraplength=self.px(950)).pack(
            fill="x", pady=(self.px(4), 0))

    def _build_analytics(self, page):
        self._page_title(page, "Analytics", "Paired comparison: the same worker and the same random variability for every robot.")
        c = self.card(page, "Performance by strategy", "▥", pack=dict(fill="x"))
        top = tk.Frame(c, bg=PANEL)
        top.pack(fill="x", pady=(self.px(6), 0))
        self.lab(top, "Repetitions", INK, 9).pack(side="left")
        self.var_runs = tk.IntVar(value=20)
        ttk.Spinbox(top, from_=1, to=200, textvariable=self.var_runs, width=5).pack(side="left", padx=8)
        self.btn_run2 = ttk.Button(top, text="▶  Run simulation", style="Primary.TButton", command=self.run_sim,
                                   state="disabled")
        self.btn_run2.pack(side="left")
        self.lbl_run = self.lab(top, "", MUTED, 9)
        self.lbl_run.pack(side="left", padx=10)
        cols = ("strategy", "time", "waiting", "interrupt.", "strain", "combined cost")
        self.tree_res = ttk.Treeview(c, columns=cols, show="headings", height=4)
        for col, wd in zip(cols, (200, 100, 100, 100, 100, 130)):
            self.tree_res.heading(col, text="time (min)" if col == "time" else col,
                                  anchor="w" if col == "strategy" else "center")
            self.tree_res.column(col, width=self.px(wd), anchor="w" if col == "strategy" else "center")
        self.tree_res.tag_configure("personalised", foreground=YELLOW)
        self.tree_res.pack(fill="x", pady=(self.px(8), 0))
        self.lbl_benefit = self.lab(c, "", MUTED, 9, wraplength=self.px(900))
        self.lbl_benefit.pack(fill="x", pady=(self.px(6), 0))
        self.fig = Figure(figsize=(8, 2.0), dpi=100 * self.S, facecolor=PANEL)
        self.canvas = FigureCanvasTkAgg(self.fig, master=c)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0, height=self.px(200))
        self.canvas.get_tk_widget().pack(fill="x")

        c3 = self.card(page, "3D comparison (generic robot B  vs  personalised robot C)", "◈",
                       pack=dict(fill="both", expand=True, pady=(self.px(10), 0)))
        bar = tk.Frame(c3, bg=PANEL)
        bar.pack(fill="x", pady=(self.px(6), 0))
        self.btn_snap = ttk.Button(bar, text="Render 3D snapshot", command=self.render_snapshot, state="disabled")
        self.btn_snap.pack(side="left")
        self.btn_vid = ttk.Button(bar, text="Render comparison video (1–2 min)", command=self.render_video,
                                  state="disabled")
        self.btn_vid.pack(side="left", padx=8)
        self.lbl_3d = self.lab(bar, "", MUTED, 9)
        self.lbl_3d.pack(side="left")
        self.snap_label = tk.Label(c3, bg="#0a0c10", fg=MUTED, text="No render yet", font=(FONT, 10))
        self.snap_label.pack(fill="both", expand=True, pady=(self.px(8), 0))

    def _build_settings(self, page):
        self._page_title(page, "Settings", "Applied the next time the camera is started.")
        c = self.card(page, "Camera and calibration", "⚙", pack=dict(fill="x"))
        self.set_vars = {"index": tk.StringVar(value="0"), "calib": tk.StringVar(value="10"),
                         "fov": tk.StringVar(value="60")}
        g = tk.Frame(c, bg=PANEL)
        g.pack(fill="x", pady=(self.px(6), 0))
        for i, (k, label, hint) in enumerate((("index", "Camera index", "0 = default webcam"),
                                              ("calib", "Calibration seconds", "default 10"),
                                              ("fov", "Assumed horizontal FOV (°)", "typical webcam 55-75"))):
            self.lab(g, label, INK, 9).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(g, textvariable=self.set_vars[k], width=8).grid(row=i, column=1, padx=10)
            self.lab(g, hint, MUTED, 9).grid(row=i, column=2, sticky="w")
        ttk.Button(c, text="Apply", style="Primary.TButton", command=self.apply_settings).pack(
            anchor="w", pady=(self.px(8), 0))

    def apply_settings(self):
        try:
            cfg = self.camera.cfg
            cfg.camera.index = int(self.set_vars["index"].get())
            cfg.calibration.duration_s = max(3.0, float(self.set_vars["calib"].get()))
            cfg.camera.horizontal_fov_deg = max(30.0, min(120.0, float(self.set_vars["fov"].get())))
        except ValueError:
            messagebox.showerror("Invalid setting", "Please enter numbers.")
            return
        self.btn_cal.config(text=f"Calibrate {cfg.calibration.duration_s:.0f} s")
        self.status.set("Settings applied. Restart the camera for them to take effect.")

    # --------------------------------------------------------------- live view --
    def _tracking_state(self):
        snap = self.camera.snapshot()
        return snap, snap["tracking"] if snap["running"] else None

    def _render_view(self):
        """Compose camera frame (or 3D snapshot) with callouts into the live-view canvas."""
        cw, ch = self.view.winfo_width(), self.view.winfo_height()
        if cw < 50 or ch < 50:
            return
        snap, tf = self._tracking_state()
        base = Image.new("RGBA", (cw, ch), rgb("#0a0c10") + (255,))
        frame, src_w, src_h, title = None, 640, 480, "Camera tracking"
        if snap["running"] and snap["frame_rgb"] is not None:
            frame = Image.fromarray(snap["frame_rgb"])
        elif self._snapshot_pil is not None:
            frame, title = self._snapshot_pil, "Simulation preview (3D)"
        if frame is not None:
            src_w, src_h = frame.size
            sc = min(cw / src_w, ch / src_h)
            nw, nh = int(src_w * sc), int(src_h * sc)
            ox, oy = (cw - nw) // 2, (ch - nh) // 2
            base.paste(frame.resize((nw, nh)).convert("RGBA"), (ox, oy))
        else:
            sc, ox, oy = 1.0, 0, 0
        ov = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        f_s, f_b, f_m = _font(self.px(12)), _font(self.px(13), True), _font(self.px(11))

        def callout(x, y, head, sub, anchor=None, col=YELLOW):
            wmax = max(d.textlength(head, f_b), d.textlength(sub, f_m)) + self.px(24)
            hh = self.px(46)
            x = max(8, min(cw - wmax - 8, x))
            y = max(8, min(ch - hh - (8 if anchor is None else self.px(56)), y))   # keep clear of the Cam chip
            if anchor is not None:
                ax, ay = anchor
                d.line([(x + wmax / 2, y + hh / 2), (ax, ay)], fill=rgb(col) + (230,), width=2)
                d.ellipse([ax - 5, ay - 5, ax + 5, ay + 5], fill=(255, 255, 255, 255), outline=rgb(col) + (255,), width=2)
            d.rounded_rectangle([x, y, x + wmax, y + hh], radius=self.px(7), fill=(14, 16, 20, 225),
                                outline=rgb(col) + (255,), width=2)
            d.text((x + self.px(12), y + self.px(6)), head, font=f_b, fill=rgb(col) + (255,))
            d.text((x + self.px(12), y + self.px(26)), sub, font=f_m, fill=(235, 238, 242, 255))

        live = snap["running"]
        chip = ("● LIVE" if live else "● PREVIEW")
        d.rounded_rectangle([self.px(12), self.px(12), self.px(12) + d.textlength(chip + "   " + title, f_b) + self.px(24),
                             self.px(12) + self.px(32)], radius=self.px(8), fill=(14, 16, 20, 225))
        d.text((self.px(24), self.px(18)), chip, font=f_b, fill=rgb(YELLOW if live else MUTED) + (255,))
        d.text((self.px(24) + d.textlength(chip + "   ", f_b), self.px(18)), title, font=f_b, fill=(235, 238, 242, 255))

        if frame is None:
            msg = "Start the camera to track a worker,\nor load a profile / demo worker and run the simulation."
            d.multiline_text((cw / 2, ch / 2), msg, font=f_b, fill=(152, 160, 173, 255), anchor="mm", align="center")
        elif live and tf is not None:
            pts = tf.landmarks_px
            if tf.status != TrackingStatus.LOST and pts is not None and pts[23, 2] > 0 and pts[24, 2] > 0:
                hip = ((pts[23, 0] + pts[24, 0]) / 2, (pts[23, 1] + pts[24, 1]) / 2)
                hx, hy = ox + hip[0] * sc, oy + hip[1] * sc
                pid = self.profile.profile_id if self.profile else "no profile"
                callout(hx + self.px(40), hy + self.px(20), "Worker", pid.replace("worker-", "ID: "), (hx, hy))
                wr = 16 if pts[16, 2] > 0 else 12
                wx, wy = ox + pts[wr, 0] * sc, oy + pts[wr, 1] * sc
                callout(wx - self.px(230), wy - self.px(80), "Current Activity",
                        f"{tf.activity.value.capitalize()} · {tf.posture.value}", (wx, wy))
            sim = self._robot_mode(tf)
            callout(self.px(16), ch - self.px(70), "Simulated robot", sim[0], None,
                    {"ok": YELLOW, "warn": AMBER, "bad": RED}[sim[1]])
            if tf.safe_hold_required:
                d.rectangle([0, 0, cw - 1, ch - 1], outline=rgb(RED) + (255,), width=5)
                d.text((cw / 2, self.px(60)), "WORKER LOST  →  SAFE-HOLD", font=_font(self.px(18), True),
                       fill=rgb(RED) + (255,), anchor="mm")
                if tf.reason:
                    d.text((cw / 2, self.px(86)), tf.reason, font=f_s, fill=(235, 238, 242, 255), anchor="mm")
            if snap["calibrating"]:
                p = snap["progress"]
                bx0, bx1, by = self.px(16), cw - self.px(16), ch - self.px(24)
                d.rounded_rectangle([bx0, by, bx1, by + self.px(10)], radius=5, fill=(40, 44, 52, 230))
                d.rounded_rectangle([bx0, by, bx0 + max(8, (bx1 - bx0) * p), by + self.px(10)], radius=5,
                                    fill=rgb(YELLOW) + (255,))
                left = self.camera.cfg.calibration.duration_s * (1 - p)
                d.text((bx0, by - self.px(22)), f"CALIBRATING  {left:3.1f}s   stand, walk a few steps, reach out both arms",
                       font=f_b, fill=rgb(YELLOW) + (255,))
        chip2 = "Cam 01 · Tracking view" if live else "3D · generic vs personalised"
        w2 = d.textlength(chip2, f_s) + self.px(24)
        d.rounded_rectangle([cw - w2 - self.px(12), ch - self.px(44), cw - self.px(12), ch - self.px(14)],
                            radius=self.px(8), fill=(14, 16, 20, 225))
        d.text((cw - w2, ch - self.px(37)), chip2, font=f_s, fill=(235, 238, 242, 255))
        self._view_img = ImageTk.PhotoImage(Image.alpha_composite(base, ov).convert("RGB"))
        self.view.delete("all")
        self.view.create_image(0, 0, image=self._view_img, anchor="nw")

    @staticmethod
    def _robot_mode(tf):
        if tf is None:
            return ("Idle", "warn")
        if tf.safe_hold_required:
            return ("SAFE-HOLD (worker lost)", "bad")
        if tf.status == TrackingStatus.DEGRADED:
            return ("Caution (partial tracking)", "warn")
        return ("Normal operation", "ok")

    # ---------------------------------------------------------- side/bottom panels --
    def _refresh_panels(self):
        snap, tf = self._tracking_state()
        # worker profile card
        if self.profile:
            tag = "synthetic demo" if "DEMO" in self.profile.profile_id else "calibrated profile"
            m = self.profile.measured
            self.l_prof[0].config(text=f"ID: {self.profile.profile_id}   |   {tag}")
            self.l_prof[1].config(text=(f"Height {m.height_m:.2f} m   |   Arm {m.arm_length_m:.2f} m" if m.height_m and m.arm_length_m
                                        else "Some body measurements missing"))
        else:
            self.l_prof[0].config(text="No profile loaded")
            self.l_prof[1].config(text="Calibrate, load, or use the demo worker")
        # activity
        if tf is not None and tf.status != TrackingStatus.LOST:
            self.l_act.config(text=tf.activity.value.capitalize())
            self.l_act2.config(text=f"Posture {tf.posture.value} · {tf.speed_mps:.2f} m/s · {snap['fps']:.0f} fps")
        elif tf is not None:
            self.l_act.config(text="Worker lost")
            self.l_act2.config(text=tf.reason or "No usable body in view")
        else:
            self.l_act.config(text="–")
            self.l_act2.config(text="Camera off")
        # next robot action (only behaviours that exist today)
        if tf is None:
            nxt = "Waiting for worker tracking. Start the camera to see the simulated robot react."
        elif tf.safe_hold_required:
            nxt = "Stay in safe-hold until the worker is back in view."
        elif tf.status == TrackingStatus.DEGRADED:
            nxt = "Move cautiously: tracking is partial."
        else:
            nxt = "Continue assisting: worker is tracked normally."
        self.l_next.config(text=nxt)
        # planner decision from simulation
        if self.results:
            st = self.results["strategies"]
            c_h, b_h = st["C_cobuild_rule"]["holds"], st["B_proactive_generic"]["holds"]
            head = "Hold every board" if c_h >= 3.5 else ("Free the arm, hand over tools" if c_h < 0.5
                                                         else "Hold boards selectively")
            self.l_dec.config(text="✓  " + head)
            self.l_dec2.config(text=f"Personalised planner held {c_h:.1f} of 4 boards on average for this profile; the "
                                    f"generic planner held {b_h:.1f}.")
        else:
            self.l_dec.config(text="Run the simulation")
            self.l_dec2.config(text="The planner's decision for this worker appears here.")
        # quality / readiness
        q = self.profile.calibration.tracked_fraction if self.profile else None
        w = max(self.q_canvas.winfo_width(), 10)
        self.q_canvas.delete("all")
        self.q_canvas.create_rectangle(0, 0, w, self.px(10), fill=PANEL2, outline="")
        if q is not None:
            self.q_canvas.create_rectangle(0, 0, w * q, self.px(10), fill=YELLOW, outline="")
        self.l_q.config(text="–" if q is None else f"{q * 100:.0f}%")
        if self.profile:
            ready = self.results is not None
            self.l_ready.config(text=("● Ready · simulation complete" if ready else "● Profile ready · run simulation"),
                                fg=GREEN if ready else AMBER)
        else:
            self.l_ready.config(text="● Not ready · no profile", fg=AMBER)
        # robot status
        mode = self._robot_mode(tf)
        colmap = {"ok": GREEN, "warn": AMBER, "bad": RED}
        self._status_row("Mode", mode[0], colmap[mode[1]])
        self._status_row("Tracking", tf.status.value if tf else "camera off", GREEN if tf and tf.status == TrackingStatus.TRACKING
                         else (RED if tf else MUTED))
        self._status_row("Profile", self.profile.profile_id if self.profile else "none", GREEN if self.profile else MUTED)
        self._status_row("Camera", f"{snap['fps']:.0f} fps" if snap["running"] else "off", GREEN if snap["running"] else MUTED)
        # expected value
        if self.results:
            st = self.results["strategies"]
            n, c_, b = st["no_robot"], st["C_cobuild_rule"], st["B_proactive_generic"]
            self._ev("time", (c_["minutes"] - n["minutes"]) / n["minutes"] * 100)
            self._ev("strain", (c_["strain"] - n["strain"]) / n["strain"] * 100)
            self._ev("cost", (c_["cost_minutes"] - b["cost_minutes"]) / b["cost_minutes"] * 100)
        self._draw_flow()

    def _status_row(self, key, text, colour):
        self.rs_dots[key].config(fg=colour)
        self.rs_vals[key].config(text=text)

    def _ev(self, key, pct):
        self.ev[key].config(text=f"{pct:+.0f}%".replace("-", "−"), fg=GREEN if pct < -0.5 else (RED if pct > 0.5 else INK))

    def _draw_flow(self):
        c = self.flow
        w = max(c.winfo_width(), 200)
        c.delete("all")
        running = self.camera.snapshot()["running"]
        done = [running or self.profile is not None, self.profile is not None, self.results is not None,
                self.results is not None and self.reviewed]
        cur = next((i for i, v in enumerate(done) if not v), None)
        labels = ["1. Track\nworker", "2. Calibrate\nprofile", "3. Run\nsimulation", "4. Review\nresults"]
        xs = [w * (i + 0.5) / 4 for i in range(4)]
        y, r = self.px(16), self.px(10)
        for i in range(3):
            c.create_line(xs[i] + r, y, xs[i + 1] - r, y, fill=YELLOW if done[i] else LINE, width=2)
        for i, x in enumerate(xs):
            if done[i]:
                c.create_oval(x - r, y - r, x + r, y + r, fill=YELLOW, outline="")
                c.create_text(x, y, text="✓", fill="#14120a", font=("Segoe UI Symbol", 10, "bold"))
            elif i == cur:
                c.create_oval(x - r, y - r, x + r, y + r, outline=YELLOW, width=3, fill=PANEL)
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=YELLOW, outline="")
            else:
                c.create_oval(x - r, y - r, x + r, y + r, outline=GREYBAR, width=2, fill=PANEL)
            c.create_text(x, y + r + self.px(18), text=labels[i], fill=INK if (done[i] or i == cur) else MUTED,
                          font=(FONT, 9, "bold" if i == cur else "normal"), justify="center")

    # ------------------------------------------------------------------ camera --
    def toggle_camera(self):
        if self.camera.running or (self.camera._thread and self.camera._thread.is_alive()):
            self.camera.stop()
            self.btn_cam.config(text="Start camera")
            self.btn_cal.config(state="disabled")
            self.status.set("Camera stopped.")
            self._render_view()
        else:
            self.status.set("Starting camera (first run downloads a ~9 MB pose model)…")
            self.camera.start()
            self.btn_cam.config(text="Stop camera")
            self.btn_cal.config(state="normal")

    def calibrate(self):
        side = self.var_side.get()
        try:
            clearance = float(self.var_clear.get()) if self.var_clear.get().strip() else None
        except ValueError:
            messagebox.showerror("Invalid value", "Preferred clearance must be a number in metres.")
            return
        self.camera.request_calibration(StatedPreferences(
            preferred_approach_side=None if side == "none" else side, preferred_clearance_m=clearance))
        self.status.set("Calibrating: stand, walk a few steps, reach out both arms. Keep the full body in view.")

    # ---------------------------------------------------------------- profiles --
    def set_profile(self, prof, source=""):
        self.profile = prof
        self.results = None
        self.refresh_sim_profile()
        for b in (self.btn_run, self.btn_run2, self.btn_snap, self.btn_vid):
            b.config(state="normal")
        self._refresh_panels()

    def load_profile_dialog(self):
        path = filedialog.askopenfilename(initialdir=str(PROFILES_DIR), filetypes=[("Profile JSON", "*.json")])
        if not path:
            return
        try:
            self.set_profile(load_profile(Path(path)), "loaded from file")
            self.status.set(f"Loaded {Path(path).name}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not load profile", str(exc))

    def use_demo(self):
        self.set_profile(make_demo_profile(), "synthetic")
        self.status.set("Synthetic demo worker loaded: a stick-figure test input, not a measurement.")

    def _overrides(self):
        ov = {}
        try:
            for k in ("pace", "lift_capacity_kg", "fatigue_rate"):
                if self.ov_vars[k].get().strip():
                    ov[k] = float(self.ov_vars[k].get())
        except ValueError:
            messagebox.showerror("Invalid override", "Overrides must be numbers.")
            return None
        h = self.ov_vars["handedness"].get().strip().lower()
        if h:
            if h not in ("left", "right"):
                messagebox.showerror("Invalid override", "handedness must be 'left' or 'right'.")
                return None
            ov["handedness"] = h
        if "pace" in ov:
            ov["pace"] = max(0.8, min(1.3, ov["pace"]))
        if "lift_capacity_kg" in ov:
            ov["lift_capacity_kg"] = max(15.0, min(60.0, ov["lift_capacity_kg"]))
        if "fatigue_rate" in ov:
            ov["fatigue_rate"] = max(0.0, min(0.005, ov["fatigue_rate"]))
        return ov

    def refresh_sim_profile(self):
        if self.profile is None:
            return
        ov = self._overrides()
        if ov is None:
            return
        self.sim_profile = bridge.to_sim_profile(self.profile, ov)
        m = self.profile.measured
        camera_vals = {
            "height_m": f"{m.height_m:.2f} m" if m.height_m else "n/a",
            "reach_m": f"arm {m.arm_length_m:.2f} m" if m.arm_length_m else "n/a",
            "pace": f"walking {m.walking_speed_mps:.2f} m/s" if m.walking_speed_mps else "no walking seen",
            "handedness": self.profile.stated_preferences.preferred_approach_side or "not stated",
            "lift_capacity_kg": "not measurable", "fatigue_rate": "not measurable"}
        self.tree_profile.delete(*self.tree_profile.get_children())
        for k in ("height_m", "reach_m", "pace", "handedness", "lift_capacity_kg", "fatigue_rate"):
            src = self.sim_profile.provenance[k]
            self.tree_profile.insert("", "end", values=(k, camera_vals[k], self.sim_profile.values[k], src), tags=(src,))
        for w in self.sim_profile.warnings + list(self.profile.calibration.warnings):
            self.tree_profile.insert("", "end", values=("⚠ " + w, "", "", ""), tags=("warning",))

    # -------------------------------------------------------------- simulation --
    def _spawn(self, fn, tag):
        def worker():
            try:
                self.events.put((tag, fn(), None))
            except Exception as exc:
                self.events.put((tag, None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def run_sim(self):
        if self.sim_profile is None:
            return
        runs = max(1, int(self.var_runs.get()))
        prof = self.sim_profile.sim_json()
        for b in (self.btn_run, self.btn_run2):
            b.config(state="disabled")
        self.lbl_run.config(text=f"Running {runs} repetitions × 4 strategies…")
        self.status.set("Running simulation…")
        self._spawn(lambda: sim_client.run_evaluation(prof, runs), "eval")

    def render_snapshot(self):
        prof = self.sim_profile.sim_json()
        self.lbl_3d.config(text="Rendering…")
        self._spawn(lambda: sim_client.render_snapshot(prof), "snap")

    def render_video(self):
        prof = self.sim_profile.sim_json()
        self.lbl_3d.config(text="Rendering video (1–2 min)…")
        self.btn_vid.config(state="disabled")
        self._spawn(lambda: sim_client.render_video(prof), "video")

    def show_results(self, res):
        self.results = res
        self.reviewed = False
        self.tree_res.delete(*self.tree_res.get_children())
        keys = list(SHORT)
        for k in keys:
            s = res["strategies"][k]
            self.tree_res.insert("", "end", tags=("personalised",) if k == "C_cobuild_rule" else (),
                                 values=(SHORT[k], f"{s['minutes']:.1f}", f"{s['worker_wait_minutes']:.2f}",
                                         f"{s['interruptions']:.1f}", f"{s['strain']:.0f}", f"{s['cost_minutes']:.1f}"))
        p = res["paired"]["B_proactive_generic__vs__C_cobuild_rule"]
        self.lbl_benefit.config(text=(
            f"Personalised vs generic robot over {res['runs']} paired runs: time {p['minutes']['mean_saving']:+.2f} ± "
            f"{p['minutes']['ci95']:.2f} min, strain {p['strain']['mean_saving']:+.1f} ± {p['strain']['ci95']:.1f} pts "
            f"(positive = personalised is better); better on combined cost in {p['cost_minutes']['pct_runs_better']:.0f}% "
            f"of runs, worse in {p['cost_minutes']['pct_runs_worse']:.0f}%. Simulated under the model's assumptions."))
        self._draw_chart(res, keys)
        self._refresh_panels()

    def _draw_chart(self, res, keys):
        self.fig.clear()
        labels = ["No\nrobot", "A\nreactive", "B\ngeneric", "C\npersonal."]
        colours = [GREYBAR, GREYBAR, BLUE, YELLOW]
        for i, (m, title, fmt) in enumerate((("minutes", "Task time (min)", "{:.1f}"),
                                             ("strain", "Physical strain (pts)", "{:.0f}"),
                                             ("cost_minutes", "Combined cost (min)", "{:.1f}"))):
            ax = self.fig.add_subplot(1, 3, i + 1, facecolor=PANEL)
            vals = [res["strategies"][k][m] for k in keys]
            bars = ax.bar(range(len(keys)), vals, color=colours, width=0.66)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(labels, fontsize=7.5, color=MUTED)
            ax.set_title(title, fontsize=10, fontweight="bold", color=INK, loc="left", pad=10)
            ax.set_ylim(0, max(vals) * 1.2)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(LINE)
            ax.set_yticks([])
            ax.tick_params(axis="x", length=0)
        self.fig.tight_layout(w_pad=2.0)
        self.canvas.draw()

    def set_snapshot(self, path):
        img = Image.open(path).convert("RGB")
        img.load()
        self._snapshot_pil = img
        self.update_idletasks()
        w = max(500, self.snap_label.winfo_width() - 10)
        self._snap_img = ImageTk.PhotoImage(img.resize((w, int(img.height * w / img.width))))
        self.snap_label.config(image=self._snap_img, text="")
        self._render_view()

    # -------------------------------------------------------------------- loop --
    def _poll(self):
        snap = self.camera.snapshot()
        if snap["error"]:
            self.status.set("Camera error: " + snap["error"])
            self.btn_cam.config(text="Start camera")
            self.btn_cal.config(state="disabled")
            self.camera.stop()
        if snap["running"]:
            self._render_view()
        self._panel_tick += 1
        if self._panel_tick % 6 == 0:
            self._refresh_panels()
        res, err = self.camera.take_calibration_result()
        if res is not None:
            path = save_profile(res)
            self.set_profile(res, "calibrated from camera")
            self.status.set(f"Calibration complete. Saved {path.name}. Press 'Run simulation'.")
        elif err:
            self.status.set("Calibration failed: " + err)
            messagebox.showwarning("Calibration failed", err)
        try:
            while True:
                tag, value, error = self.events.get_nowait()
                if tag == "eval":
                    for b in (self.btn_run, self.btn_run2):
                        b.config(state="normal")
                    self.lbl_run.config(text="")
                    if error:
                        self.status.set("Simulation failed: " + error[-300:])
                        messagebox.showerror("Simulation failed", error[-1200:])
                    else:
                        self.show_results(value)
                        self.status.set("Simulation complete. Rendering 3D preview…")
                        self.render_snapshot()
                elif tag == "snap":
                    self.lbl_3d.config(text="")
                    if error:
                        messagebox.showerror("3D render failed", error[-1200:])
                    else:
                        self.set_snapshot(value)
                        self.status.set("Simulation complete.")
                elif tag == "video":
                    self.btn_vid.config(state="normal")
                    self.lbl_3d.config(text="")
                    if error:
                        messagebox.showerror("Video render failed", error[-1200:])
                    else:
                        self.lbl_3d.config(text=f"Saved {value.name}")
                        try:
                            os.startfile(str(value))
                        except OSError:
                            pass
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def destroy(self):
        try:
            self.camera.stop()
        finally:
            super().destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
