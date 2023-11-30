DEFAULT_DISCRETE_INTENTIONS = ["forward", "right", "left"]

def intention_to_idx(intention): 
    return DEFAULT_DISCRETE_INTENTIONS.index(intention)

def idx_to_intention(idx): 
    return DEFAULT_DISCRETE_INTENTIONS[idx]