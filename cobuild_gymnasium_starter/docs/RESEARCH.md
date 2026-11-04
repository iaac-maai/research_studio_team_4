# Research notes: how drywall is installed, and what a robot arm can do

Purpose: ground the simulation's **procedure, robot tasks and timings** in published sources instead of guesses.
Scope for now: **installation only** (measure, cut, lift, place, fasten). Taping, mudding and sanding are out of scope.

Reliability legend: **[A]** peer-reviewed / field data · **[B]** patent, project page or trade press · **[C]** weak (content-farm pages, vendor claims) · **[Inference]** our own reasoning.

---

## 1. The procedure

Order of work: ceilings first, then walls; on walls the **upper row is hung first**, then the lower row [C: how-to pages, consistent with each other].
Per board, field observation of installers (Yuan et al. 2016 [A]) identifies two work cycles, each used about half the time:

| Cycle | Steps |
|---|---|
| **Whole sheet** | lift from the stack → carry to the wall → hold/place (adjust position) → screw → reload the screw gun |
| **Partial sheet** (needs trimming) | measure/cut (sheet stored vertically on a cart) → carry → hold in place while screwing → screw → reload the screw gun |

Other facts used in the model:

- Boards: 1.22 × 2.44 m (4 × 8 ft). A 5/8" sheet weighs roughly 31–34 kg (68–75 lb) by most tables [C]; Yuan et al. use a density of 881 kg/m³, which implies ~42 kg [A]. We use **34 kg** and test 26 and 42 kg in the sensitivity analysis. 4 × 12 and 4 × 16 sheets exist and raise the loads a lot (+12 % and +28 % muscle/joint forces vs 4 × 8) [A: Yuan & Buchholz 2014].
- Studs every 16" (0.406 m) on walls. Screws about every 16" in the field and 8" on edges for walls; ~12" on ceilings [C]. This gives about 28 screws per full board.
- Installers screw with a powered drywall screwdriver (~12 N). **Cutting** uses a cutter and tape measure (a "drywall hammer cutter" in Yuan).

## 2. How installers spend their time (the calibration data)

Yuan et al. 2016 [A] observed 8 installers on a condominium site (Boston, 2003–04) at 60-second intervals: **126 observations**.

| Activity | % of observations | Load in hands | Trunk posture |
|---|---|---|---|
| Cut / measure | 19.8 % | very light | mostly neutral |
| **Lift** | **6.4 %** | **heavy (100 %)** | 5 of 8 with bend + twist + flexion |
| Carry | 12.7 % | light to heavy | almost always neutral |
| **Hold / place** | **4.8 %** | **heavy (100 %)** | mostly non-neutral |
| Screw | 15.9 % | light | mixed |
| **"In between"** | **19.1 %** | light | neutral. *Exclusively loading and adjusting the screw gun; always follows screwing.* |
| Other (climb, communicate, mark) | 21.4 % | very light | mixed |

How we use it:

- An installer finishes **~50 whole + ~50 partial sheets per 8-hour day** (field interviews in the same paper): **≈ 4.8 min per piece**. Dasgupta et al. 2016 [A] measured 4 min 30 s per 4 × 8 panel for a pair working on a ladder. We calibrate the average worker to **~4.5 min per board**.
- **Lift + hold/place are only ~11 % of the time but nearly all of the heavy load.** The robot's main value is expected to be physical strain, not minutes.
- **Reloading the screw gun is 19 % of the time.** It is a concrete target for a robot that hands over a pre-loaded tool (task 4).

Limits: 126 observations, 8 workers, one site, a 2-hour observation window, and percentages of *snapshots*, not stopwatch times. We only use them to set the **proportions between steps**.

## 3. Strain and what supporting the board does

- The most hazardous activity is **lifting the panel and holding it overhead with both arms** (Dasgupta et al. 2016 [A]); a survey of 60 carpenters says the same [A: Pan et al. 1999].
- Dasgupta's field study with 5 installers and 46 hours of observation tested a support aid (an electric lift plus a "deadman" holding the board against the ceiling) against ladders:
  - low-back compressive force **2690 → 1280 N (−52 %)**, left-shoulder moment **−35 %**, both arms overhead **40 % → 27 %** of the time, heavy-load handling **3.1 % → 0 %**;
  - but the cycle time per panel did **not** improve: 4:30 (ladder), 5:30 (manual lift), 5:00 (electric lift + deadman).
- Yuan et al. [A]: lifting a whole sheet gives average L4/L5 disc compression close to 7000 N, above NIOSH's 6400 N maximum limit. Lifting a 4 × 12 or 4 × 16 sheet alone is "physically too difficult or even impossible" for one person. Storing sheets **vertically instead of flat** cuts the low-back/shoulder loads ~8 %.
- Injuries: about half are sprains and strains, mostly of the back [CDC 203662, B/A].

**Consequence for the model:** a robot that holds the board should cut strain by roughly half, and should **not be assumed to save time** unless it also removes work from the critical path (carrying, reloading). We therefore report time and strain separately, and compare our simulated strain reduction with the −52 % field figure as a plausibility check (not as something we tuned to match).

## 4. What a robot arm can do on this task

| # | Robot task | Evidence | Role in the simulation |
|---|---|---|---|
| 1 | **Hold the board against the studs** while the worker fastens | Patents (US 10526799 and US 9995047: suction frame, optional screw-gun arm [B]); Dasgupta's "deadman" [A] | Action *lift and hold*. Removes the worker's holding strain |
| 2 | **Lift and position** the board at wall height | Same patents [B]; KUKA 6-DOF drywall studies in simulation [B: arXiv 2307.04195, 2306.09639]; AIST prototype [B] | Same action: positioning replaces the worker's heavy "place" step with a short alignment |
| 3 | **Carry / stage boards** from the stack to the cut station or the wall | Etex/BRE: material handling ≈ 18 % of inefficient time [B]; Yuan: carry = 12.7 % [A] | Action *bring board* |
| 4 | **Hand over tools**: cutter and tape, and a **pre-loaded screw gun** | Handover literature: pose depends on both people's heights and arm lengths, and personalising it improves comfort [A/B: PLOS ONE 2019, ICRA 2023, Cakmak et al.]; reloading = 19 % [A] | Action *hand tool at pose* (12 poses: side × height × distance) |
| 5 | **Steady the board at the cutting station** | [Inference]. Yuan describes cutting on a vertical cart; no study of robot help here | Action *steady board*. A small effect that is an **assumption**, flagged in `config.py` |
| – | Fasten (robot drives the screws) | Patents, prototypes [B] | **Not modelled yet** (changes the division of labour; later phase) |
| – | Tape, mud, sand | Canvas (commercial) [B/C] | **Out of scope** (finishing phase) |

Single-arm constraint: **the robot can do only one thing at a time**. Holding a board blocks handing tools and staging the next board. This is what makes the *when and for whom* question non-trivial, and where a worker profile can matter.

## 5. Handovers and profiles

- Preferred handover height and distance depend on **both partners' heights and arm lengths** (PLOS ONE 2019 [A]); user-specific adjustment improved perceived comfort (ICRA 2023 workshop [B]). Almost nothing is published about handover **timing**.
- Capacity matters: a biomechanical simulation of 60–100 lb sheets estimated that only a limited share of the male population has enough strength for the lift [B/A: Pan & Chiou 1999, as cited].
- Yuan et al. generate subjects from normal height/weight distributions (about 176 ± 10 cm, 83 ± 11 kg, range 160–200 cm). Our synthetic population (1.55–1.95 m) is a bit wider on purpose, to include short workers.

## 6. What we still do not know

- **No credible stopwatch study** of minutes per step. Cost-estimating pages contradict each other (e.g. 1 to 12 h per 100 ft²). The proportions in §2 are the best we have.
- **No data on robot-assisted drywall hanging in the field** (only patents, prototypes and simulations), so every robot-side duration is an assumption (`config.py`).
- No data on how much a steadying robot helps at the cutting station.
- Little data on **women, left-handed or short workers**: Dasgupta's 5 participants were all men, 21–60, right-handed; Yuan's modelled subjects were healthy right-handed men, and anthropometry was not recorded for the observed crew. Our left-handed and short personas are therefore **hypothetical**, not drawn from these studies.
- A real robot able to lift a 34 kg board to 2.4 m is a mobile manipulator with a lift column and a payload far above a Franka Panda. The 3D scene is a **schematic stand-in**, not a specific product.

## Sources

- [A] Yuan, Buchholz, Punnett, Kriebel (2016). *An integrated biomechanical modeling approach to the ergonomic evaluation of drywall installation.* Applied Ergonomics 53:52–63. [CDC 202815](https://stacks.cdc.gov/view/cdc/202815/cdc_202815_DS1.pdf)
- [A] Yuan & Buchholz (2014). *The effects of position and size of drywall on the physical demands for installers.* HFES 58th Annual Meeting. [CDC 200964](https://stacks.cdc.gov/view/cdc/200964/cdc_200964_DS1.pdf)
- [A] Dasgupta, Punnett, Moir, Kuhn, Buchholz (2016). *Does drywall installers' innovative idea reduce the ergonomic exposures of ceiling installation: a field case study.* Applied Ergonomics 55:183–193. [CDC 203662](https://stacks.cdc.gov/view/cdc/203662/cdc_203662_DS1.pdf), [PubMed](https://pubmed.ncbi.nlm.nih.gov/26995048/)
- [A] Washington State ergonomics demonstration project: wallboard. [ELCOSH](https://elcosh.org/document/1939/d000862/Ergonomics%2BDemonstration%2BProject%253A%2BWallboard%2B2002.html)
- [B] US patents: [10526799](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10526799) (automated drywall cutting and hanging), [9995047](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9995047) (construction board installation robot).
- [B] [arXiv 2307.04195](https://arxiv.org/pdf/2307.04195) and [arXiv 2306.09639](https://arxiv.org/pdf/2306.09639.pdf): robot-assisted drywall installation, human–robot collaboration in simulation.
- [B] WallMoBot (Aalborg University): [project page](https://vbn.aau.dk/en/projects/wallmobot/).
- [B] Etex/BRE time study summarised by [Specfinish](https://www.specfinish.co.uk/?p=10430).
- [A/B] Handovers: [PLOS ONE 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6588208/), [ICRA 2023 workshop](https://nms.kcl.ac.uk/icra2023.explainablerobotics/papers/xro-lagomarsino.pdf), [Cakmak et al. 2011](https://personalrobotics.cs.washington.edu/publications/cakmak2011human.pdf).
- [C] Procedure and weight/spacing tables from how-to and spec pages: [Canadian Woodworking](https://canadianwoodworking.com/techniques_and_tips/19-tips-for-better-drywall-installation/), [EngineerFix](https://engineerfix.com/what-is-the-proper-ceiling-drywall-screw-spacing/). Verify against the Gypsum Association (GA-216) or the manufacturer's guide before relying on exact numbers.
