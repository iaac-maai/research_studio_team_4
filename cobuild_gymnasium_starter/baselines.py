"""Rule-based delivery strategies compared against the trained PPO agent.

Each strategy is a function (env, profile) -> action. They read the
environment's ground-truth internal state (env.remaining, env.transit,
env.material_ready) directly rather than from the observation vector: these
are meant to represent simple, transparent planners, not agents that must
infer hidden state the way PPO does.

Actions: 0 = wait, 1 = deliver left, 2 = deliver right.
"""


def reactive(env, profile):
    """A. Reactive: only requests material once the worker has stopped and needs it."""
    worker_waiting = env.remaining == 0 and not env.transit and not env.material_ready
    return profile['side'] + 1 if worker_waiting else 0


def fixed_proactive(env, profile):
    """B. Proactive, non-personalized: anticipates the need one tick ahead,
    but always delivers from the right regardless of the worker's preference."""
    about_to_need_material = env.remaining == 1 and not env.transit and not env.material_ready
    return 2 if about_to_need_material else 0


def profile_rule(env, profile):
    """Transparent personalized planner: proactive like B, but delivers to
    the worker's preferred side. Serves as a hand-written upper reference
    for what a personalized policy could achieve."""
    about_to_need_material = env.remaining == 1 and not env.transit and not env.material_ready
    return profile['side'] + 1 if about_to_need_material else 0


STRATEGIES = {
    'reactive': reactive,
    'fixed_proactive': fixed_proactive,
    'profile_rule': profile_rule,
}


def load_ppo():
    """Load the trained PPO model, or None if train.py hasn't been run."""
    from pathlib import Path
    model_path = Path(__file__).with_name('cobuild_ppo.zip')
    if not model_path.exists():
        return None
    from stable_baselines3 import PPO
    return PPO.load(str(model_path))


def choose_action(strategy, env, profile, obs, model):
    """Action from a rule-based strategy, or from the PPO model when strategy == 'ppo'."""
    rule = STRATEGIES.get(strategy)
    if rule is not None:
        return rule(env, profile)
    if model is None:
        raise SystemExit('No trained model found at cobuild_ppo.zip — run train.py first, '
                         'or pass --strategy reactive/fixed_proactive/profile_rule.')
    action, _ = model.predict(obs, deterministic=True)
    return int(action)
