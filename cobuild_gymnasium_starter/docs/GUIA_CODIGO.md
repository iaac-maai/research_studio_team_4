# Guía del código: cómo leerlo, modificarlo y entenderlo

Para quien necesite explicar o tocar el código del proyecto Cobuild (v2, instalación de tablaroca). Está en español; el código y los demás documentos están en inglés.

Documentos relacionados:
- Cómo correr las cosas: [../README.md](../README.md)
- Cómo funciona el modelo y por qué la comparación es justa: [MODEL.md](MODEL.md)
- De dónde salen los números: [RESEARCH.md](RESEARCH.md)

Todos los comandos se corren desde la carpeta `cobuild_gymnasium_starter/`.

---

## Paso 0 · La idea en 30 segundos

Hay un **simulador** de un trabajador instalando 4 placas de tablaroca en un muro, y un **robot** que puede ayudarle. Se repite la instalación muchas veces con distintos robots y distintos trabajadores, y se mide el tiempo y el esfuerzo físico.

```
profiles.py ──► cobuild_env.py ◄── config.py        (el mundo: trabajador, tarea, números)
                    │   ▲
                    │   └── ergonomics.py            (cuerpo → tiempo y esfuerzo)
                    ▼
              strategies.py                          (los "cerebros" del robot)
              ┌─────┴──────┐
        train.py          evaluate.py ──► plots.py   (entrenar / medir / graficar)
      (crea models/)      analysis.py                (qué rasgos importan, sensibilidad)
                          visualize_3d.py + scene3d.py   (ver la simulación en 3D)
```

---

## Paso 1 · Mapa de archivos (en el orden en que conviene leerlos)

| # | Archivo | Líneas | Qué contiene | Lo tocas cuando… |
|---|---|---|---|---|
| 1 | [config.py](../config.py) | ~105 | **Todos los números** del modelo (tiempos, pesos, recompensa), etiquetados `[DATA]`, `[LIT]` o `[ASSUME]` | quieres cambiar un supuesto |
| 2 | [profiles.py](../profiles.py) | ~86 | La clase `WorkerProfile`, las 8 personas, el trabajador genérico y el generador de trabajadores aleatorios | quieres otro perfil o otras características |
| 3 | [ergonomics.py](../ergonomics.py) | ~179 | Convierte un cuerpo en tiempo y esfuerzo para cada tarea; decide si conviene sostener la placa | cambias cómo el cuerpo afecta una tarea |
| 4 | [cobuild_env.py](../cobuild_env.py) | ~520 | **El entorno de Gymnasium**: estado, acciones, observación, recompensa, métricas | cambias las reglas de la simulación |
| 5 | [strategies.py](../strategies.py) | ~125 | Las 6 estrategias (sin robot, A, B, C regla, C PPO, PPO sin perfil) | cambias cómo decide el robot |
| 6 | [train.py](../train.py) | ~64 | Entrena el PPO | quieres reentrenar |
| 7 | [evaluate.py](../evaluate.py) | ~169 | Repite la instalación muchas veces y guarda resultados | quieres más corridas o más métricas |
| 8 | [plots.py](../plots.py) | ~195 | Figuras 1–5 | quieres otra gráfica |
| 9 | [analysis.py](../analysis.py) | ~202 | Contribución de cada rasgo + sensibilidad (figuras 6–7) | quieres probar otro supuesto |
| 10 | [scene3d.py](../scene3d.py), [visualize_3d.py](../visualize_3d.py) | ~299 / ~393 | Escena 3D y director de los videos | quieres cambiar cómo se ve |
| 11 | [tests/test_env.py](../tests/test_env.py) | ~144 | 48 pruebas automáticas | siempre, después de cambiar algo |

**Orden recomendado de lectura:** `config.py` → `profiles.py` → los docstrings de `ergonomics.py` → `cobuild_env.py` (`reset`, `step`, `_observation`) → `strategies.py` → `evaluate.py`. Con eso se entiende el 90 %. El 3D se puede ignorar al principio: no afecta ningún resultado.

---

## Paso 2 · Las herramientas existentes que usamos

Instaladas con [requirements.txt](../requirements.txt). **No modificamos ninguna librería**; escribimos código nuestro encima de ellas.

| Herramienta | Para qué existe | Cómo la usamos | Dónde |
|---|---|---|---|
| **Gymnasium** | Estándar de la industria para entornos de RL: define `reset()`, `step()`, `action_space`, `observation_space` | Escribimos nuestra propia clase `CobuildDrywallEnv(gym.Env)` | `cobuild_env.py` |
| **Stable-Baselines3** | Implementaciones de algoritmos de RL ya probadas | Solo sus utilidades: `make_vec_env` (8 entornos en paralelo) y `BaseCallback` (imprimir progreso) | `train.py` |
| **sb3-contrib → `MaskablePPO`** | PPO con **enmascaramiento de acciones**: el agente no puede elegir acciones imposibles | Es el algoritmo que entrenamos. El entorno expone `action_masks()` y el modelo lo usa | `train.py`, `strategies.py` |
| **NumPy** | Arreglos y números aleatorios | El ruido de duraciones (`lognormal`), la observación, el muestreo de trabajadores | en todos |
| **pandas** | Tablas | `evaluate.py` guarda una fila por episodio en `results/episodes.csv`; las gráficas la leen | `evaluate.py`, `plots.py` |
| **tabulate** | Tablas en markdown | Escribe `results/summary.md` y `analysis.md` | `evaluate.py`, `analysis.py` |
| **matplotlib** | Gráficas | Las 7 figuras de `results/` | `plots.py`, `analysis.py` |
| **MuJoCo** | Motor de física/render 3D | Solo para **dibujar**. Los cuerpos son "mocap" (los movemos nosotros), sin física | `scene3d.py` |
| **OpenCV** | Video | Escribe el mp4 (códec H.264 `avc1`) y la ventana `--live` | `visualize_3d.py` |
| **Pillow** | Texto sobre imágenes | El panel con reloj y métricas en cada cuadro | `visualize_3d.py` |
| **pytest** | Pruebas automáticas | `python -m pytest -q tests` | `tests/` |

### Lo que es nuestro y no viene de ninguna librería
- **El modelo de la tarea** (qué pasa en cada tick): `cobuild_env.py` y `ergonomics.py`.
- **La calibración** con Yuan et al. 2016 y Dasgupta et al. 2016: `config.py` y la prueba en `tests/`.
- **Las estrategias de reglas** (B y C regla): `strategies.py`.
- **La comparación pareada** con números aleatorios comunes: `evaluate.py`.

---

## Paso 3 · Qué hace cada pieza, a detalle

### 3.1 `config.py`: los números
Un solo lugar con todos los supuestos. Cada línea trae una etiqueta:
- `[DATA]`: calibrado con una medición publicada.
- `[LIT]`: la dirección o el tamaño tiene respaldo en la literatura.
- `[ASSUME]`: lo supusimos nosotros. **Estos son los que hay que defender o medir.**

Bloques principales:
- **Tarea:** `TICK_SECONDS = 6`, `N_BOARDS = 4`, `BOARD_KG = 34`.
- **`BASE`**: duración en ticks de cada fase para el trabajador promedio solo (medir 7, cortar 12, recargar 4.5…). `DURATION_NOISE_SIGMA = 0.15` es la variabilidad aleatoria.
- **Cuerpo:** `LIFT_LOAD_KNEE`, `OVERHEAD_FACTOR_PER_M`, `FATIGUE_SCALE`.
- **Robot:** `BRING_TICKS`, `ALIGN_TICKS`, `FETCH_TOOL_TICKS`, `STEADY_CUT_TIME_FACTOR`, `HOLD_RELEASE_AFTER`.
- **Poses de entrega de herramienta:** 2 lados × 3 alturas × 2 distancias = 12.
- **Recompensa:** `R_TICK`, `R_WORKER_WAIT`, `R_STRAIN`, `R_EXPOSURE`, `R_UNNECESSARY`, `R_IGNORED`, `R_COMPLETE`.

> Regla de oro: **si quieres cambiar un número, cámbialo aquí, no en otro archivo.**

### 3.2 `profiles.py`: los trabajadores
- `WorkerProfile` es un dataclass con: `height_m`, `reach_m`, `lift_capacity_kg`, `pace`, `handedness`, `fatigue_rate`.
- `GENERIC` es el trabajador promedio para quien se diseña el robot no personalizado (B).
- `PERSONAS` son 8 casos nombrados (por ejemplo `short_left`, `low_capacity_right`).
- `sample_population(rng)` crea un trabajador aleatorio (altura normal, capacidad normal, ritmo uniforme 0.8–1.3, mano 50/50). Se usa para **entrenar** y para los **trabajadores no vistos** en la evaluación.

### 3.3 `ergonomics.py`: del cuerpo al tiempo y al esfuerzo
Funciones puras (sin estado). Las centrales:
- `load_ratio(P, b)`: peso de la placa ÷ capacidad del trabajador.
- `overhead_deficit(P, b)`: cuánto le falta de alcance para la fila de arriba.
- `phase_cost(P, b, name, mode)`: **la función más importante**. Devuelve tiempo extra y esfuerzo de una fase, dado el cuerpo del trabajador y quién la hace (solo o con el robot).
- `hold_is_worth_it(P, b)`: ¿conviene que el robot sostenga la placa para este trabajador? Es lo que hace distinto a un robot personalizado.
- `best_tool_pose(P)`, `handover_cost(P, pose)`: la mejor pose para entregar una herramienta y cuánto cuesta una pose mala.

### 3.4 `cobuild_env.py`: el corazón
La clase `CobuildDrywallEnv(gym.Env)`.

**Flujo de un episodio**
1. `reset(seed)`: elige el perfil (fijo o aleatorio), sortea **una vez** el ruido de todas las fases y empieza la placa 0. El ruido se sortea antes de cualquier acción del robot: por eso todas las estrategias con la misma semilla enfrentan al mismo trabajador (**números aleatorios comunes**).
2. `step(action)` en cada tick de 6 s:
   1. suma `R_TICK`,
   2. `_robot_command(action)`: ejecuta lo que pidió el robot (o castiga si no tiene sentido),
   3. `_advance_robot()`: avanza el estado del robot,
   4. `_advance_worker()`: avanza la fase del trabajador,
   5. suma `R_EXPOSURE`, `R_STRAIN × (esfuerzo nuevo)` y `R_COMPLETE` si terminó.
3. Termina cuando las 4 placas están atornilladas (`terminated`) o se llega a `MAX_TICKS` (`truncated`).

**Las 16 acciones:** 0 esperar · 1 traer placa · 2 estabilizar en el corte · 3 levantar y sostener · 4–15 entregar herramienta en una de las 12 poses.

**Máscaras:** `action_masks()` marca como imposibles las acciones que no aplican (por ejemplo, sostener si no hay una placa lista). Una cosa más: el brazo es **uno solo**, así que mientras sostiene no puede entregar herramientas.

**Observación:** `_observation()` arma 43 números entre 0 y 1: avance de la tarea, dónde está la placa, estado del robot, ticks esperados hasta que el trabajador necesite algo (`time_to`), el perfil y lo que implica (si conviene sostener).

**Interruptor `observe_profile`:** si es `False`, el perfil que ve el agente es siempre el genérico. Así se construye el modelo de control "PPO sin perfil".

**Métricas:** `_info()` devuelve minutos, espera, esfuerzo, tiempo cargando, minutos por actividad, etc. `evaluate.py` las guarda.

### 3.5 `strategies.py`: los cerebros
Cada estrategia es un objeto `Strategy` con una función `act(env, obs) → acción`.

| Clave | Cómo decide |
|---|---|
| `no_robot` | siempre espera; el trabajador hace todo solo |
| `A_reactive` | `reactive_action`: espera a que el trabajador pida ayuda |
| `B_proactive_generic` | `planner_action(env, GENERIC)` |
| `C_cobuild_rule` | `planner_action(env, env.profile)`: **la misma función que B**, con el perfil real |
| `C_cobuild_ppo` | carga `models/cobuild_ppo_profile.zip` |
| `ablation_ppo_no_profile` | carga `models/cobuild_ppo_no_profile.zip` y no ve el perfil |

`planner_action(env, P)` es el planificador de reglas. Prioridad: sostener si conviene → traer la siguiente placa justo a tiempo → entregar la herramienta justo a tiempo en la mejor pose → estabilizar. "Justo a tiempo" usa `env.time_to(...)` más un margen de seguridad (`SAFETY_MARGIN_TICKS = 3`).

> Por eso la comparación B vs C regla es limpia: **solo cambia el perfil `P`**.

### 3.6 `train.py`: entrenar
- Crea 8 entornos en paralelo con `make_vec_env`.
- `MaskablePPO('MlpPolicy', …)`, red de 2 capas de 128, `gamma=0.995`, `ent_coef=0.01`.
- Cada episodio usa un trabajador aleatorio nuevo, así que el agente aprende una regla general y no memoriza personas.
- Guarda en `models/cobuild_ppo_profile.zip` y `models/cobuild_ppo_no_profile.zip`.

```bash
python train.py --timesteps 3000000          # ambos modelos, 3 M de pasos
python train.py --variant profile --seed 7   # solo el que ve el perfil, otra semilla
```
> Ojo: el valor por defecto del archivo es **2 M** de pasos (`--timesteps`). Los resultados reportados en el README son de 3 M; pásalo explícitamente si quieres reproducirlos.
> Ojo 2: `--seed` cambia la semilla, pero cada modelo se guarda con el mismo nombre y **sobrescribe** al anterior. Para probar varias semillas hay que copiar o renombrar los modelos entre corridas.

### 3.7 `evaluate.py`: medir
- `run_episode(strategy, profile, seed)` corre un muro completo y devuelve una fila de métricas.
- Grupos: las 8 personas × 50 corridas y 100 trabajadores no vistos × 5 corridas (900 corridas por estrategia).
- Para cada (trabajador, semilla) corre **todas** las estrategias con el mismo ruido.
- Guarda `results/episodes.csv` (una fila por episodio), `results/summary.md` y llama a `plots.py`.
- `paired(...)` calcula ahorros pareados con intervalo de confianza del 95 %.
- **Costo combinado** = minutos + 0.05 × puntos de esfuerzo.

### 3.8 `plots.py` y `analysis.py`
- `plots.py` lee `episodes.csv` y genera las figuras 1–5. Cada figura es una función (`fig_overview`, `fig_activity`, `fig_work_saved`, `fig_profile_value`, `fig_paired`). Puedes regenerarlas sin reentrenar: `python plots.py`.
- `analysis.py` hace dos cosas: **(a)** `trait_contribution`: el planificador conoce solo un grupo de rasgos y se mide cuánto ayuda; **(b)** `sensitivity`: 19 escenarios con supuestos distintos. Escribe `results/analysis.md` y las figuras 6–7.

### 3.9 `scene3d.py` y `visualize_3d.py`: ver la simulación
- `scene3d.py` construye la escena en MuJoCo: muro con postes, tarima, mesa de corte, trabajador (con la altura de su perfil), robot, placas y herramientas. Solo dibuja.
- `visualize_3d.py` corre el **mismo entorno y las mismas estrategias** de la evaluación, interpola posiciones entre ticks y escribe el video. `pose_of(env, P)` traduce el estado del entorno a posiciones 3D.
- **Importante:** el 3D no cambia ningún resultado. Si el video se ve raro, el error está en el dibujo y no en los números.

### 3.10 `tests/test_env.py`: la red de seguridad
48 pruebas. Las más útiles para entender el diseño:
- el entorno cumple la API de Gymnasium (`check_env`),
- las máscaras de acciones son correctas,
- misma semilla ⇒ mismo trabajador en todas las estrategias,
- B y C regla dan **idéntico** resultado con el trabajador genérico,
- el tiempo del trabajador solo se parece a Yuan et al.,
- la decisión de sostener la placa cambia según la persona.

---

## Paso 4 · Cómo correr todo, en orden

```bash
python -m pytest -q tests          # 1. ¿todo bien?
python train.py --timesteps 3000000   # 2. entrenar (~10 min por modelo)
python evaluate.py                 # 3. medir + figuras + tablas
python analysis.py                 # 4. rasgos + sensibilidad (~3 min)
python visualize_3d.py --profile short_left --compare --top no_robot --bottom C_cobuild_ppo --video videos/demo.mp4
```

Sin entrenar: `python evaluate.py --no-ppo` usa solo las estrategias de reglas.

---

## Paso 5 · Recetas: quiero cambiar X

| Quiero… | Qué tocar | Después |
|---|---|---|
| Cambiar la duración de una fase del trabajador | `BASE` en `config.py` | tests (hay uno de calibración) → `evaluate.py` |
| Cambiar qué tan rápido es el robot | `BRING_TICKS`, `ALIGN_TICKS`, `FETCH_TOOL_TICKS`, `SWAP_TICKS` en `config.py` | **reentrenar** el PPO → `evaluate.py` |
| Cambiar el peso de la placa | `BOARD_KG` | reentrenar → evaluar |
| Cambiar cuánto valoramos el esfuerzo frente al tiempo | `R_STRAIN` en `config.py`. En `evaluate.py` el costo combinado se deriva solo de `R_STRAIN`; en `analysis.py` `COST_PER_STRAIN_MIN` está fijo en 0.05 a propósito (para que los escenarios sean comparables) | reentrenar → evaluar |
| Agregar un perfil nombrado | `PERSONAS` en `profiles.py` | añadirlo a la lista de pruebas si aplica; aparece solo en `evaluate.py` y en `--profile` |
| Agregar una característica al perfil (por ejemplo, edad) | `WorkerProfile` y `sample_population` en `profiles.py`; su efecto en `ergonomics.py`; añadirla a `_observation()` en `cobuild_env.py` (cambia `OBS_DIM`, definido en `cobuild_env.py`) | **reentrenar obligatoriamente** (la red cambia de tamaño) |
| Agregar una tarea del robot (por ejemplo, atornillar) | nuevo número de acciones y estado en `cobuild_env.py`, su máscara en `action_masks()`, su lógica en `planner_action` | reentrenar, pruebas, evaluar |
| Agregar una estrategia nueva | `STRATEGY_KEYS`, `LABELS` y `make_strategy` en `strategies.py`; color en `plots.py` | evaluar |
| Evaluar con más corridas | `python evaluate.py --runs 100 --unseen 200` | — |
| Probar otro supuesto en la sensibilidad | la lista de escenarios en `sensitivity()` de `analysis.py` | `python analysis.py` |
| Cambiar solo cómo se ven las gráficas | `plots.py` | `python plots.py` (sin reentrenar) |
| Cambiar solo cómo se ve el 3D | `scene3d.py` / `visualize_3d.py` | regenerar el video |

### ¿Cuándo hay que reentrenar?
Reentrena si cambias **cualquier cosa que el PPO vea o sienta**: números de `config.py` que afectan la simulación, la recompensa, la observación, las acciones o el perfil. Las estrategias de reglas no necesitan reentrenar, pero sí volver a evaluar. Si solo cambias gráficas o 3D, no hace falta.

> Peligro: un PPO viejo con un entorno nuevo **no falla con error claro** (o falla por tamaño, o peor, corre y da resultados engañosos). Si cambias el modelo, reentrena y vuelve a correr `evaluate.py`.

---

## Paso 6 · Trampas y detalles que conviene saber

1. **Un tick son 6 segundos**; los tiempos del código están en ticks. Multiplica por `TICK_SECONDS / 60` para minutos.
2. **El robot no ve el ruido ni el esfuerzo acumulado**; planea con valores esperados.
3. **El esfuerzo son "puntos relativos"**, no una unidad clínica.
4. **Una sola semilla de entrenamiento** (42) hoy. Para resultados del PPO más fiables hacen falta varias; ver la discusión en el README.
5. **Los modelos no se versionan**: `models/*.zip` se sobrescribe al reentrenar.
6. **Los videos** se escriben con códec H.264 (`avc1`); con `mp4v` no se reproducían en QuickTime ni en VS Code.
7. Lo que **no** es RL: B y C regla son reglas escritas a mano (a propósito: son la referencia limpia).

---

## Paso 7 · Mini-glosario

| Término | Significa aquí |
|---|---|
| **Tick** | 6 segundos de trabajo simulado |
| **Episodio** | un muro completo (4 placas) |
| **Persona** | un perfil de trabajador con nombre |
| **Trabajador no visto** | un perfil aleatorio no usado en las personas |
| **Máscara de acciones** | lista de acciones permitidas en este momento |
| **Ablación** | quitar una pieza (el perfil) para ver cuánto importaba |
| **Números aleatorios comunes** | misma variabilidad para todas las estrategias, para comparar de forma pareada |
| **Costo combinado** | minutos + 0.05 × puntos de esfuerzo |
| **B / C regla** | mismo planificador; B asume trabajador promedio, C usa el perfil real |
