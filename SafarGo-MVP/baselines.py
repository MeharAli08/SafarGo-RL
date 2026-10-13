def baseline_action(env, policy='demand'):
    """Choose a simple rule action: demand heuristic or no repositioning."""
    if policy == 'stay':
        return env.zones
    # Estimate demand from static zone weights and local supply; deterministic proxy.
    score = env.base_demand / (1 + env.supply)
    target = int(score.argmax())
    return target if env.supply.max() > 1 else env.zones

