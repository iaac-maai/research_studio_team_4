# How the Cobuild drywall simulation works (v2)

For the team and for our professors. How to *run* things: [../README.md](../README.md). Where the numbers come from: [RESEARCH.md](RESEARCH.md).

---

## 1. The question

> **Can worker-specific profiles improve task efficiency compared to non-personalised proactive robotic assistance in construction?**

To answer it fairly we need a task where a robot arm really helps (drywall installation), workers who are **physically different**, and a comparison in which the **only** difference between robots is whether they use the profile, repeated many times so we can say *how much* work it saves and *for whom*.

The model is a **scheduling model with simple ergonomics**, not a physics simulation. What is calibrated to published data and what is assumed is listed in §9.

---

## 2. The task: covering one wall with drywall

A 4.5 × 2.44 m wall, studs every 16", **4 boards of 1.22 × 2.44 m (34 kg)** laid horizontally. The upper row goes first. The right-hand board of each row is trimmed to fit, so the wall mixes **2 whole boards and 2 boards that need cutting**, like the 50/50 mix observed in the field (Yuan et al. 2016).

```
 whole board:    prep → [stool] ─────────────────────────────── place → screw (a) → reload (a) → screw (b) → reload (b)
 trimmed board:  prep → [stool] → measure → cut → (carry to wall) → place → screw (a) → reload (a) → screw (b) → reload (b)
                                   at the wall   at the cart
```

If the worker needs something the robot is not providing (the board at the cart, the board at the wall, the tools), they **do it alone**: they lift and carry the board, place it and hold it up for the first screws, fetch the tape and cutter, reload the screw gun themselves. Heavy phases cost strain that depends on their body.

**Calibration.** Durations are set so the average worker alone needs ~5 min per board (Yuan: ~4.8 min per piece; Dasgupta: 4.5 min per panel for a pair) and the time shares are close to Yuan's activity table (an automated test checks this):

| Activity | Simulation (average worker alone) | Yuan et al. 2016 |
|---|---|---|
| measure + cut | ~20 % | 19.8 % |
| lift + carry the board | ~20 % | 19.1 % |
| place / hold | ~6 % | 4.8 % |
| screw | ~19 % | 15.9 % |
| reload the screw gun | ~16 % | 19.1 % |
| other (prep, marking, stool) | ~20 % | 21.4 % |

---

## 3. The worker profile

| Trait | Range in the population | What it changes |
|---|---|---|
| **Height** | 1.55–1.95 m | Comfortable tool-handover height (0.60 × H). Overhead reach (1.25 × H): the upper row is within reach or not. A short worker needs a stool and works slower and with more strain overhead. |
| **Comfortable reach** | 0.52–0.78 m | A tool presented farther away costs the worker a step or a lean. |
| **Lifting capacity** | 20–50 kg | Load ratio = 34 kg / capacity. Above 1.0 lifts get slower, and strain of lifting, carrying, placing and holding scales with it. |
| **Handedness** | left / right | Side on which the worker takes tools (screw gun). The wrong side costs a little time and strain. |
| **Pace** | 0.8 (fast) – 1.3 (slow) | Multiplies every duration. A robot that assumes the wrong pace arrives too late or too early. |
| **Fatigue rate** | small | Each strain point makes later steps slightly slower, so strain also costs time. |

Eight synthetic **personas** plus a **generic average worker** (the one a non-personalised robot is designed for). For training and for the "unseen workers" evaluation, workers are drawn from a synthetic population. The published drywall studies include almost only right-handed men (RESEARCH.md §6), so **short, left-handed and low-capacity personas are hypotheses, not data**.

---

## 4. The robot (what the agent controls)

**One arm, one job at a time.** Every tick (6 s) it chooses one of **16 actions**:

| Action | Task | What it does |
|---|---|---|
| 0 wait | | nothing |
| 1 **bring board** | 3 | stack → cut station, stack → wall staging, or cut station → wall staging |
| 2 **steady board** | 5 | goes to the cut station and steadies the board while the worker measures and cuts (cut 15 % faster, 30 % less strain: **assumption**). If it is still holding the board when the cut ends, it carries it to the wall. |
| 3 **lift & hold** | 1 + 2 | lifts the staged board onto the studs, aligns it and **holds it until the board is fastened**. The worker only directs the alignment. |
| 4–15 **hand tool** | 4 | hands over the next tool (cut tools, or a **pre-loaded screw gun** that swaps the empty one) at one of 12 poses: side × height × distance. |

Holding the board **blocks the single arm**: it cannot hand over the screw gun, so the worker reloads alone. That is the trade-off a profile can resolve: a weak or short worker gains a lot from holding; a tall, strong worker loses time and may prefer the arm to stay free.

---

## 5. What the agent observes (43 numbers in [0, 1])

Task progress (board, activity, trimmed/upper board); where the current board is (stack / cut station / staged / held / wall) and whether the next one is already staged; robot state and ticks left; **expected ticks until the worker needs the next board, the wall, or a tool** (computed from the profile); which tool comes next; which actions are possible; the **worker profile** (height, reach, capacity, pace, handedness, fatigue); and what the profile implies (load ratio of the board, whether holding pays off).

The robot does **not** see the worker's random variability or accumulated strain: it plans with expectations. Impossible actions are **masked** (MaskablePPO), so the agent can discover rarely useful ones such as holding.

**Ablation:** the *PPO without profile* agent has the same observation, but the profile entries always describe the generic worker.

---

## 6. Reward

| Component | Value | Encourages |
|---|---|---|
| every tick | −1 | finish quickly |
| worker waiting | −1 extra per tick | provide things before they are needed |
| worker strain | −0.5 per point | choose poses, holds and carries that spare this body |
| robot hovering with a tool nobody has taken | −0.1 per tick | arrive just in time, not too early |
| nonsensical command / robot busy | −1 / −0.2 | learn what is possible |
| wall finished | +10 | complete the task |

"One strain point is worth half a tick" is a **value judgement**. The sensitivity analysis halves and doubles it.

---

## 7. The strategies and why the comparison is fair

| Strategy | Anticipates | Uses the profile | How |
|---|---|---|---|
| **0 · No robot** | | | the worker does everything alone (reference for "how much work is saved") |
| **A · Reactive** | no | no | the worker asks (1 tick), the robot reacts; always holds when asked; generic handover pose |
| **B · Proactive (average worker)** | yes | no | just-in-time planner tuned for the average worker |
| **C · Cobuild rule** | yes | **yes** | **the same planner as B**, given the individual's profile |
| **C · Cobuild PPO** | yes | **yes** | learned policy trained on random workers |
| **PPO without profile** | yes | no | identical training; profile replaced by the generic worker |

- B and the Cobuild rule run **the same code**; with the generic worker they coincide (tested).
- PPO and PPO-without-profile differ **only** in the profile.
- For a given (worker, seed) every strategy faces **the same variability** (common random numbers), so differences are paired.
- PPO is evaluated on **100 unseen workers**.
- The rule-based planners see the same task information as PPO, but not the hidden variability.

**"Combined cost"** = minutes + 0.05 × strain points. It is the planners' own trade-off in minutes, used because the profile often trades one for the other.

---

## 8. Analyses

- `evaluate.py`: personas × 50 runs + 100 unseen workers × 5 runs; means, **paired savings with 95 % CI**, runs better/worse, time **by activity**.
- `analysis.py`: **trait contribution** (personalise one trait group at a time) and **sensitivity** to 19 modelling assumptions.

---

## 9. Assumptions and limitations (read before presenting)

- **Calibrated to data:** the time per board and its split by activity (Yuan et al.), the order and structure of the procedure, board weight, stud spacing. **Everything the robot does** (carry 4–5 ticks, align 2 ticks, tool fetch 2 ticks, steadying effect, hold until fastened) **is an assumption**: nobody has published field data for it. All are in [config.py](../config.py), tagged `[ASSUME]`.
- **Strain points** are a relative unit, not a clinical score (a later version could use REBA/RULA). Our simulated strain reduction from the robot (−59 %) is of the same order as Dasgupta's −52 % low-back compressive force for a *partial* aid; it was **not tuned** to match.
- **Time savings from holding are model-driven.** Dasgupta found that supporting the panel did **not** shorten the cycle (5:00 vs 4:30 on a ladder). In our model the robot saves time mostly by removing carrying and placing from the worker's critical path. If real alignment takes longer than assumed, the time saving shrinks (see the "robot alignment takes 4 ticks" scenario).
- Movement between stations takes no time, one worker, one robot, one wall, no safety model, profiles assumed accurate.
- The 3D scene is schematic. A real robot that lifts 34 kg to 2.4 m is a mobile manipulator with a lift column; it is not a Franka Panda.

### What to measure to make this real
1. Stopwatch times per step (measure, cut, carry, place, screw, reload) for 2–3 installers: replaces `BASE` and the noise.
2. Height, reach and a comfortable lifting capacity for each participant.
3. Time and posture for the same task **with a helper holding the board** (a person can stand in for the robot): replaces the robot assumptions.
4. Handover time and discomfort for the 12 poses.

---

## 10. What changed

| Version | What it modelled | Why it was replaced |
|---|---|---|
| **v0** (`v0_side_preference/`) | pace + left/right preference; deterministic; 6 identical components | Deterministic (the seed did nothing), baseline B secretly used the worker's exact remaining time, B always delivered on the right by construction, "unnecessary deliveries" and "interruptions" were not measured |
| **v1** (`v1_handover_poses/`) | adds height, reach, capacity, fatigue; but the robot only *delivered* boards and screws at 12 poses; abstract 4-step board task | The robot's role was not a realistic one (it did not do the tasks a robot arm would do in drywall), and the task was abstract |
| **v2** (this folder) | procedure from field studies; robot tasks 1–5; time by activity; calibration test; 3D scene | |
