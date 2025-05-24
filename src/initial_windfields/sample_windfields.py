import numpy as np
import json

def random_unit_vector():
    """Sample a random 2D unit vector (direction)."""
    angle = np.random.uniform(0, 2 * np.pi)
    return (float(np.cos(angle)), float(np.sin(angle)))

def sample_windfield():
    """Sample a random windfield configuration, including evolution parameters."""
    wind_init_params = {
        'base_speed': float(np.random.uniform(2.0, 4.0)),
        'base_direction': random_unit_vector(),
        'pattern_scale': int(np.random.choice([16, 32, 64, 96, 128])),
        'pattern_strength': float(np.random.uniform(0.1, 0.8)),
        'strength_variation': float(np.random.uniform(0.1, 0.6)),
        'noise': float(np.random.uniform(0.05, 0.3))
    }
    wind_evol_params = {
        'wind_change_prob': float(np.random.uniform(0.0, 1.0)),
        'pattern_scale': int(np.random.choice([16, 32, 64, 96, 128])),
        'perturbation_angle_amplitude': float(np.random.uniform(0.0, 0.2)),
        'perturbation_strength_amplitude': float(np.random.uniform(0.0, 0.2)),
        'rotation_bias': float(np.random.uniform(-0.2, 0.2)),
        'bias_strength': float(np.random.uniform(0.0, 1.0))
    }
    return {
        'wind_init_params': wind_init_params,
        'wind_evol_params': wind_evol_params
    }
    
    
def sample_windfield2():
    """Sample a random windfield configuration, including evolution parameters."""
    wind_init_params = {
        'base_speed': float(np.random.uniform(1.5, 5.0)),  # wider range
        'base_direction': random_unit_vector(),
        'pattern_scale': int(np.random.choice([8, 16, 32, 64, 96, 128, 192])),  # more options
        'pattern_strength': float(np.random.uniform(0.05, 1.0)),  # wider range
        'strength_variation': float(np.random.uniform(0.05, 0.8)),  # wider range
        'noise': float(np.random.uniform(0.01, 0.5))  # wider range
    }
    wind_evol_params = {
        'wind_change_prob': float(np.random.uniform(0.0, 1.0)),
        'pattern_scale': int(np.random.choice([8, 16, 32, 64, 96, 128, 192])),
        'perturbation_angle_amplitude': float(np.random.uniform(0.0, 0.4)),  # wider range
        'perturbation_strength_amplitude': float(np.random.uniform(0.0, 0.4)),  # wider range
        'rotation_bias': float(np.random.uniform(-0.4, 0.4)),  # wider range
        'bias_strength': float(np.random.uniform(0.0, 1.5))  # wider range
    }
    return {
        'wind_init_params': wind_init_params,
        'wind_evol_params': wind_evol_params
    }

def windfield_hash(windfield):
    """Hash a windfield for uniqueness (rounded for floating point stability)."""
    p = windfield['wind_init_params']
    e = windfield['wind_evol_params']
    key = (
        round(p['base_speed'], 2),
        round(p['base_direction'][0], 2),
        round(p['base_direction'][1], 2),
        p['pattern_scale'],
        round(p['pattern_strength'], 2),
        round(p['strength_variation'], 2),
        round(p['noise'], 2),
        e['pattern_scale'],
        round(e['wind_change_prob'], 2),
        round(e['perturbation_angle_amplitude'], 2),
        round(e['perturbation_strength_amplitude'], 2),
        round(e['rotation_bias'], 2),
        round(e['bias_strength'], 2)
    )
    return key

def sample_unique_windfields(n, exclude_hashes=None):
    """Sample n unique windfields, avoiding hashes in exclude_hashes."""
    windfields = []
    hashes = set() if exclude_hashes is None else set(exclude_hashes)
    while len(windfields) < n:
        wf = sample_windfield()
        h = windfield_hash(wf)
        if h not in hashes:
            windfields.append(wf)
            hashes.add(h)
    return windfields, hashes

if __name__ == "__main__":
    # Sample 500 unique training windfields
    train_windfields, train_hashes = sample_unique_windfields(500)
    # Sample 50 unique test windfields, different from training
    test_windfields, _ = sample_unique_windfields(50, exclude_hashes=train_hashes)

    # Save to JSON files
    with open("sampled_train_windfields.json", "w") as f:
        json.dump(train_windfields, f, indent=2)
    with open("sampled_test_windfields.json", "w") as f:
        json.dump(test_windfields, f, indent=2)

    print("Sampled 500 training and 50 testing windfields.")