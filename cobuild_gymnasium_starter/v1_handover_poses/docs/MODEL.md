# How the Cobuild drywall simulation works

This document explains **what we simulate, why, and what the results can and cannot tell us**.
It is written for the team and for our professors. For how to *run* things, see [../README.md](../README.md).

---

## 1. The question

> **Can worker-specific profiles improve task efficiency compared to non-personalized proactive robotic assistance in construction?**

To answer it fairly we need three things:

1. A task where a robot assistant really can help: **drywall (tablaroca) installation**.
2. Workers that are **physically different** (height, reach, strength, handedness, pace) so the profile can matter.
3. A comparison where the **only** difference between robots is whether they use the profile, repeated many times with realistic variability, so we can measure *how much work* the profile saves and *for whom*.

The simulation is a **scheduling model with simple ergonomics**. It is not a physics or robotics simulation, and all numbers are synthetic assumptions (see §9). Its value is to compare strategies under identical, explicit assumptions and to show which assumptions the conclusion depends on.

---

## 2. The task: covering one wall with drywall

The wall is 2.4 m high and is covered by **4 standard boards** (1.2 × 2.4 m, about 26 kg each), placed horizontally. As is common practice, the **upper row goes first**.

```
 wall (front view)            every board goes through 4 steps:
 ┌──────────┬──────────┐
 │ board 1  │ board 2  │  ← upper row: the board must be pushed up to 2.4 m
 ├──────────┼──────────┤      measure ─► cut ─────► lift & position ─► fasten
 │ board 3  │ board 4  │  ← lower row                needs           (robot may     needs
 └──────────┴──────────┘                            BOARD            hold it)      SCREWS
```

| Step | Base duration (average worker) | Needs from the robot | Intrinsic strain |
|---|---|---|---|
| measure & mark | 3 ticks | – | 0.2 |
| cut (score, snap, trim) | 4 ticks | **board** | 0.5 |
| lift & position | 3 ticks × load factor × height factor | – (robot can **hold** the board) | see §3 |
| fasten (screw) | 8 ticks (+ overhead slowdown on the upper row) | **screws** | 1.0 (+ overhead) |

1 tick ≈ 15 s, so an average worker needs about 20–25 minutes per wall in this model.

Every step's real duration = expected duration × **random variability** (lognormal, σ = 0.15) × **fatigue**. Variability is why we repeat each installation many times.

---

## 3. The worker profile: preferences + physical characteristics

| Trait | Range in the population | How it enters the model |
|---|---|---|
| **Height** (stature) | 1.55 – 1.95 m | Comfortable handover height = 0.60 × height. Overhead reach = 1.25 × height: if this is below the 2.4 m wall top, lifting and fastening upper boards is slower and more straining. |
| **Comfortable reach** | 0.52 – 0.78 m (correlated with height) | If the robot presents material farther away, the worker must step or lean: +1 tick, +1.5 strain. |
| **Lifting capacity** | 20 – 50 kg | Load ratio = 26 kg / capacity. Above 0.6 the lift gets slower. Strain of lifting alone = 3 × load ratio × height factor. Board handovers in awkward postures strain more for weaker workers. |
| **Handedness** | left / right (50/50 in training) | Receiving material on the wrong side: +1 tick (turn, switch hands), +1 strain. |
| **Pace** | 0.8 (fast) – 1.3 (slow) | Multiplies the duration of every step. A robot that expects the wrong pace arrives too late (the worker waits) or too early (it hovers in the worker's space). |
| **Fatigue rate** | 0.002 – 0.008 | Every strain point makes the following steps that much slower: `duration × (1 + rate × accumulated strain)`. This is how physical strain turns into lost time. |

The anthropometric ratios are approximations based on the classic Drillis & Contini segment proportions (elbow ≈ 0.63 H, shoulder ≈ 0.82 H, arm ≈ 0.44 H). In a real Cobuild deployment these traits would be *measured* (or entered once) for each worker.

**Personas** (synthetic, in [profiles.py](../profiles.py)): `expert_tall_right`, `expert_average_left`, `novice_average_right`, `short_left`, `short_strong_right`, `tall_novice_left`, `low_capacity_right`, `low_capacity_short_left`, plus `generic_average`, which is the population median the non-personalized robot is designed for.

### Handover cost
When the worker takes an item from the robot at pose (side, height, distance):

```
ticks  = 1 if wrong side       + 1 if |height − comfort| > 15 cm   + 1 if distance > reach
strain = 1 if wrong side       + 2 per 10 cm beyond a 5 cm tolerance + 1.5 if distance > reach
strain × (1 + load ratio)  when the item is a 26 kg board
```

### Holding the board (support)
If the robot holds the board on the wall, the lift takes a fixed **4 ticks** (coordination included) with only **0.5** strain. Lifting alone can take anywhere from **2.4 ticks** (tall, strong, experienced) to **10+ ticks** (short, low capacity, upper row). The catch is that while the robot is holding the board it **cannot fetch the screws**, so the worker waits about 3 ticks for them. Holding pays off for some workers and costs time for others. That trade-off is exactly what a profile can resolve.

---

## 4. The robot (what the agent controls)

Every tick the robot picks **one of 14 actions**:

| Action | Meaning |
|---|---|
| `0` wait | do nothing this tick |
| `1–12` deliver at pose *k* | fetch the next item in the procedure (board or screws) and present it at one of 12 poses = side {left, right} × height {0.90, 1.05, 1.20 m} × distance {near 0.45 m, far 0.65 m} |
| `13` hold board | move to the wall and hold the board during the next lift (only possible while the worker is cutting) |

Rules of the world:
- Fetching takes 3 ticks for a board and 2 for screws, **+1 tick if delivering near** (the robot slows down close to people).
- The robot carries **one item at a time** and **holds it until the worker takes it** (direct handover). The worker takes an item only when they start the step that needs it.
- Commands sent while the robot is busy are ignored (small penalty). Asking for a delivery when nothing is left, or to hold a board when it is not possible, is an **unnecessary delivery**.

### What happens in one tick
1. The robot command is applied.
2. The robot advances (travelling, arriving, getting into position).
3. The worker advances: works on the current step, or, if they need an item or the robot's support, takes it if available or **waits**.
4. Metrics and reward are updated.

---

## 5. What the agent observes (27 numbers, all scaled to 0–1)

| Group | Features |
|---|---|
| Task progress | board index; current step (one-hot ×4); worker waiting?; upper-row board?; ticks spent in this step |
| Timing estimates | expected ticks left in the current step; **expected ticks until the next item is needed** (both computed from the profile) |
| Robot state | free / fetching / holding / positioning / supporting (one-hot ×5); ticks left in the current robot motion |
| Next item | is it a board? screws? for the current board? |
| **Worker profile** | height, reach, lifting capacity, pace, handedness, fatigue rate |
| Profile implications | expected duration and strain of the next lift if done alone |

The robot does **not** observe the random variability or the worker's accumulated strain. It has to plan with expectations, like a real robot would.

**Ablation**: the "PPO without profile" agent gets exactly the same observation, but the profile features (and everything computed from them) always describe the generic average worker.

---

## 6. The reward (what PPO is trying to maximise)

| Component | Value | Behaviour it encourages |
|---|---|---|
| every tick | −1 | finish the wall quickly |
| worker waiting | −1 extra per tick | deliver *before* the worker needs it (no interruptions) |
| worker strain | −0.5 per strain point | right side, right height, within reach; hold heavy boards when that helps this worker |
| robot hovering with an item | −0.1 per tick | don't arrive too early and occupy the worker's space |
| unnecessary delivery / invalid hold | −1 | don't send useless requests |
| command ignored (robot busy) | −0.2 | learn to wait while busy |
| wall finished | +10 | complete the task |

The weights express a value judgement ("one strain point is worth half a tick"). The sensitivity analysis (§8) shows the conclusions hold when it is halved or doubled.

---

## 7. The strategies we compare

| Strategy | Anticipates? | Uses the individual profile? | How |
|---|---|---|---|
| **A · Reactive** | no | no | waits until the worker stops and asks, then fetches; presents at the generic pose; never holds boards |
| **B · Proactive (average worker)** | yes | no | just-in-time planner tuned for the population-average worker (timing, pose, holding decisions) |
| **C · Cobuild rule** | yes | **yes** | **the same planner code as B**, given the individual's profile |
| **C · Cobuild PPO** | yes (learned) | **yes** | PPO policy trained on 2M ticks of random workers |
| **PPO without profile** (ablation) | yes (learned) | no | identical training, but the profile in the observation is always the generic worker |

### Why the comparison is fair
- **Same information.** Rule-based strategies read the task state (current step, elapsed time, robot state), which is the same information PPO observes. None of them sees the hidden random variability. (In v0, baseline B read the worker's exact remaining time and so was secretly personalised on pace.)
- **B vs. C-rule differ only in the profile.** They run the same code. With the generic worker they give identical results (this is checked by an automated test).
- **PPO vs. PPO-without-profile differ only in the profile.** They use the same algorithm, hyperparameters, seed and training budget.
- **Common random numbers.** For a given (worker, seed), every strategy faces exactly the same variability, so every difference is *paired* and only due to the strategy.
- **Unseen workers.** PPO is trained on random workers and evaluated on 100 *different* random workers plus the named personas.

---

## 8. Metrics and analyses

| Metric | Meaning |
|---|---|
| **Total time** (min) | time to finish the wall: the main efficiency measure |
| **Worker waiting** (min) | time the worker stood idle waiting for material or for the robot |
| **Interruptions** | how many times the worker had to stop and wait |
| **Strain** (points) | accumulated physical load: the ergonomic "work" the worker did |
| **Unnecessary deliveries** | useless delivery / hold requests |
| **Robot exposure** (min) | time the robot hovered in the worker's space holding something not yet needed |
| **Awkward handovers** | handovers that cost a tick or more than 0.5 strain |

- `evaluate.py`: every strategy × (9 personas × 50 runs + 100 unseen workers × 5 runs). Reports means, **paired savings with 95 % confidence intervals**, and the share of runs where each strategy was better or worse.
- `analysis.py` **trait contribution**: personalises one trait group at a time to show *which characteristics* produce the savings.
- `analysis.py` **sensitivity**: repeats B vs. C-rule under 10 alternative assumptions (no fatigue, double fatigue, low/high variability, milder ergonomic penalties, wrong side costing no time, different strain weights, a faster robot, a more conservative B).

---

## 9. Assumptions and limitations (please read before presenting results)

- **All durations, strain values and anthropometric ratios are illustrative assumptions** (all in [config.py](../config.py)). The *absolute* minutes are not predictions for a real site. The *comparison* between strategies under the same assumptions is the result.
- **The ergonomics are deliberately simple.** "Strain points" are a relative unit, not a validated score. A natural next step is to map handover postures to REBA/RULA scores.
- **The handover model is discrete** (12 poses). A real robot would choose continuous positions.
- **The rule-based Cobuild planner knows the ergonomic model exactly.** It is an upper reference for what a personalised planner could achieve *if the model were right*. PPO has to learn the model from experience.
- **PPO does not find every good behaviour.** In our runs it never learned to hold boards (the action is rarely useful, so exploration
  seldom finds it), and it prefers a "safe" near handover that is slightly slower for long-reach workers. It does use the profile
  (it beats the no-profile PPO in 99 % of runs), but it stays below the rule-based planner. See README → *Honest caveats about PPO*.
- **Profiles are assumed accurate.** We do not yet test what happens when a profile is wrong or outdated (a good next experiment).
- **One worker, one robot, one wall, no safety model**, no travel obstacles, no other trades on site.

### What to measure to make this real
1. Real step durations for measure / cut / lift / fasten (video of 2–3 installers, a stopwatch is enough) → replaces `BASE_TICKS` and `DURATION_NOISE_SIGMA`.
2. Height, reach and a comfortable lifting estimate for each participant → real profiles.
3. Time and posture for handovers at different heights, sides and distances → calibrates `handover_cost`.
4. Lift time alone vs. with a helper holding the board, on upper and lower rows → calibrates `lift_unsupported` and `SUPPORTED_LIFT_TICKS`.

---

## 10. What changed from v0 and why

The previous prototype is kept in [../v0_side_preference/](../v0_side_preference/). A review of it found:

| v0 issue | Consequence | v1 fix |
|---|---|---|
| Fully deterministic: with a fixed profile every run was identical (the seed did nothing) | could not repeat the installation or compute statistics | per-step random variability, drawn per seed with common random numbers |
| Baseline B read `env.remaining`, the individual worker's exact remaining time | B was secretly personalised on pace, so only left/right was really tested | rules plan with expected durations from the profile they are given |
| B always delivered on the right | half the profiles lost exactly 3 ticks per delivery by construction, so the result followed directly from the rules | B = the same planner, tuned for the average worker |
| "Unnecessary deliveries" and "interruptions" were not measured | two metrics from the brief were missing | both are tracked, plus strain, exposure and awkward handovers |
| Trained and evaluated on the same 4 profiles | no evidence of generalisation | trained on random workers, evaluated on 100 unseen ones |
| Only pace and side | the question about physical characteristics could not be studied | height, reach, lifting capacity and fatigue, plus a real 4-step drywall procedure and the option to hold boards |
