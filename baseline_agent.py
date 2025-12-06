import numpy as np

def get_baseline_action(state):
    """
    A simple heuristic agent.
    State indices (16 elements):
    0-2: Danger (Straight/Right/Left)
    3-6: Direction
    7-10: Food Dir
    11-14: Poison Dir (NEW - Baseline ignores this, causing it to fail!)
    15: Tail Prox
    """
    danger = state[0:3] 
    
    # Priority 1: Avoid immediate collision (Walls/Self)
    # Straight
    if danger[0] == 0: 
        return 1
    # Right (if straight is blocked)
    if danger[1] == 0:
        return 2
    # Left (if straight and right are blocked)
    if danger[2] == 0:
        return 0
    
    return 1 # Dead end
    
def get_baseline_agent(state):
    return get_baseline_action(state)