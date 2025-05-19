"""
Predefined initial windfields for the sailing challenge.
Each initial windfield defines a starting wind configuration.
Wind evolution is handled by the environment.
"""

# Define common evolution parameters (same as in SailingEnv but duplicated to avoid circular imports)
COMMON_WIND_EVOL_PARAMS = {
    'wind_change_prob': 1.0,      # Wind field updates on every step
    'pattern_scale': 128,          # Scale of spatial perturbation patterns
    'perturbation_angle_amplitude': 0.1,  # Angle perturbation per step
    'perturbation_strength_amplitude': 0.1,  # Strength variation per step
    'rotation_bias': 0.02,        # Clockwise rotational bias (positive = clockwise)
    'bias_strength': 1.0          # Full strength of rotational bias
}

# Training Initial Windfield 1: North-Northwest Wind
# Characteristics: Starting with NNW wind
TRAINING_INITIAL_WINDFIELD_1 = {
    'wind_init_params': {
        'base_speed': 3.0,
        'base_direction': (-0.8, -0.2),  # NNW wind
        'pattern_scale': 32,
        'pattern_strength': 0.3,
        'strength_variation': 0.4,
        'noise': 0.1
    },
    # Include evolution params for backward compatibility
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Training Initial Windfield 2: North-Northeast Wind
# Characteristics: Starting with NNE wind
TRAINING_INITIAL_WINDFIELD_2 = {
    'wind_init_params': {
        'base_speed': 3.0,
        'base_direction': (-0.2, 0.8), 
        'pattern_scale': 128,
        'pattern_strength': 0.6,
        'strength_variation': 0.3,
        'noise': 0.1
    },
    # Include evolution params for backward compatibility
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Training Initial Windfield 3: Pure North Wind
# Characteristics: Starting with N wind, smaller pattern scale
TRAINING_INITIAL_WINDFIELD_3 = {
    'wind_init_params': {
        'base_speed': 3.0,
        'base_direction': (0.2, -0.8),  
        'pattern_scale': 32,           
        'pattern_strength': 0.4,
        'strength_variation': 0.2,
        'noise': 0.1
    },
    # Include evolution params for backward compatibility
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Simple Static Initial Windfield: Stable NE wind with minimal variations
# This is a reference configuration - use static_wind=True when creating the environment
SIMPLE_STATIC_INITIAL_WINDFIELD = {
    'wind_init_params': {
        'base_speed': 3.0,
        'base_direction': (-0.7, -0.7),  # NE wind
        'pattern_scale': 32,
        'pattern_strength': 0.1,     # Very small variations
        'strength_variation': 0.1,   # Very small variations
        'noise': 0.05               # Minimal noise
    },
    # Static params for backward compatibility
    # New code should use SailingEnv(..., static_wind=True) instead
    'wind_evol_params': {
        'wind_change_prob': 0.0,
        'pattern_scale': 128,
        'perturbation_angle_amplitude': 0.0,
        'perturbation_strength_amplitude': 0.0,
        'rotation_bias': 0.0,
        'bias_strength': 0.0
    }
}



# Training Initial Windfield 4: Strong East Wind (with Upwind Component)
# Characteristics: East-southeast wind pushing the agent off-course (against north)
TRAINING_INITIAL_WINDFIELD_4 = {
    'wind_init_params': {
        'base_speed': 3.5,
        'base_direction': (1.0, -0.3),  # East-Southeast wind (upwind)
        'pattern_scale': 64,
        'pattern_strength': 0.5,
        'strength_variation': 0.35,
        'noise': 0.15
    },
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

# Training Initial Windfield 5: Chaotic Southwest Wind (with Upwind Component)
# Characteristics: Turbulent SW wind, largely pushing against northward sailing
TRAINING_INITIAL_WINDFIELD_5 = {
    'wind_init_params': {
        'base_speed': 2.8,
        'base_direction': (-0.6, -0.6),  # Southwest wind (upwind)
        'pattern_scale': 16,
        'pattern_strength': 0.7,
        'strength_variation': 0.5,
        'noise': 0.2
    },
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}

TRAINING_INITIAL_WINDFIELD_6 = {
    'wind_init_params': {
        'base_speed': 3.5,
        'base_direction': (0.2, -0.9),  # Upwind with slight east bias
        'pattern_scale': 32,
        'pattern_strength': 0.5,
        'strength_variation': 0.3,
        'noise': 0.2
    },
    'wind_evol_params': COMMON_WIND_EVOL_PARAMS.copy()
}
TRAINING_INITIAL_WINDFIELD_7 = {
    'wind_init_params': {
        'base_speed': 2.0,
        'base_direction': (0.6, 0.6),  # NE wind
        'pattern_scale': 96,
        'pattern_strength': 0.4,
        'strength_variation': 0.2,
        'noise': 0.2
    },
    'wind_evol_params': {
        **COMMON_WIND_EVOL_PARAMS,
        'rotation_bias': 0.1,  # Significant clockwise rotation
        'bias_strength': 1.0
    }
}

TRAINING_INITIAL_WINDFIELD_8 = {
    'wind_init_params': {
        'base_speed': 3.2,
        'base_direction': (0.0, -1.0),  # Pure upwind
        'pattern_scale': 32,
        'pattern_strength': 0.7,
        'strength_variation': 0.4,
        'noise': 0.25
    },
    'wind_evol_params': {
        **COMMON_WIND_EVOL_PARAMS,
        'wind_change_prob': 0.0,
        'perturbation_angle_amplitude': 0.0,
        'perturbation_strength_amplitude': 0.0,
        'rotation_bias': 0.0,
        'bias_strength': 0.0
    }
}



# Dictionary mapping initial windfield names to their parameters
INITIAL_WINDFIELDS = {
    'training_1': TRAINING_INITIAL_WINDFIELD_1,
    'training_2': TRAINING_INITIAL_WINDFIELD_2,
    'training_3': TRAINING_INITIAL_WINDFIELD_3,
    'simple_static': SIMPLE_STATIC_INITIAL_WINDFIELD
}
# Register them in the windfield dictionary
INITIAL_WINDFIELDS['training_4'] = TRAINING_INITIAL_WINDFIELD_4
INITIAL_WINDFIELDS['training_5'] = TRAINING_INITIAL_WINDFIELD_5
INITIAL_WINDFIELDS['training_6'] = TRAINING_INITIAL_WINDFIELD_6
INITIAL_WINDFIELDS['training_7'] = TRAINING_INITIAL_WINDFIELD_7
INITIAL_WINDFIELDS['training_8'] = TRAINING_INITIAL_WINDFIELD_8

def get_initial_windfield(name):
    """
    Get the parameters for a specific initial windfield.
    
    Args:
        name: String, one of ['training_1', 'training_2', 'training_3', 'simple_static']
        
    Returns:
        Dictionary containing wind_init_params and wind_evol_params
        
    Note:
        To create a static environment (no wind evolution), use:
        env = SailingEnv(**get_initial_windfield(name), static_wind=True)
        
        To create a dynamic environment (default):
        env = SailingEnv(**get_initial_windfield(name))
    """
    if name not in INITIAL_WINDFIELDS:
        raise ValueError(f"Unknown initial windfield '{name}'. Available initial windfields: {list(INITIAL_WINDFIELDS.keys())}")
    return INITIAL_WINDFIELDS[name] 