"""
Numerical and configuration constants for GCS analysis.

Centralizes all magic numbers and threshold values for reproducibility.
"""

import torch

# =============================================================================
# Numerical Stability Constants
# =============================================================================
NUMERICAL_EPSILON = 1e-8
EIGENVALUE_THRESHOLD = 1e-12
SYMMETRY_TOLERANCE = 1e-6
DIV_ZERO_PROTECTION = 1e-12

# =============================================================================
# Geometric Analysis Thresholds
# =============================================================================
LOW_DIVERSITY_THRESHOLD = 0.5
NEIGHBOR_DIVERSITY_RATIO = 0.9
PATH_IDENTITY_THRESHOLD = 0.001
DEFAULT_VARIANCE_THRESHOLD = 0.95

# =============================================================================
# Default Experiment Parameters
# =============================================================================
DEFAULT_PRIME = 113
DEFAULT_TRAIN_FRACTION = 0.3
DEFAULT_RANDOM_SEED = 777
DEFAULT_DEVICE = 'cuda:0'

# =============================================================================
# Training Defaults
# =============================================================================
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1.0
DEFAULT_TOTAL_EPOCHS = 15000

# =============================================================================
# GMN-TSA Analysis Defaults
# =============================================================================
DEFAULT_N_SAMPLES = 200
DEFAULT_K_NEIGHBORS = 15
DEFAULT_MANIFOLD_DIM = 8

# =============================================================================
# Model Architecture Defaults
# =============================================================================
DEFAULT_D_MODEL = 128
DEFAULT_D_MLP = 512
DEFAULT_D_HEAD = 32
DEFAULT_NUM_HEADS = 4
DEFAULT_N_CTX = 3
DEFAULT_NUM_LAYERS = 1

# =============================================================================
# Checkpoint and Logging Defaults
# =============================================================================
DEFAULT_RECORD_FREQUENCY = 200
DEFAULT_GCS_FREQUENCY = 200


def get_device(preferred_device: str = DEFAULT_DEVICE) -> str:
    """Get available device, falling back to CPU if needed."""
    if preferred_device.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    return preferred_device


def get_default_config(num_layers: int = 1, act_type: str = 'ReLU', task: str = 'add'):
    """Get default configuration for experiments."""
    return {
        # Data
        'p': DEFAULT_PRIME,
        'fraction': DEFAULT_TRAIN_FRACTION,
        'seed': DEFAULT_RANDOM_SEED,
        'd_vocab': DEFAULT_PRIME + 1,
        'task': task,

        # Model
        'num_layers': num_layers,
        'd_model': DEFAULT_D_MODEL,
        'd_mlp': DEFAULT_D_MLP,
        'd_head': DEFAULT_D_HEAD,
        'num_heads': DEFAULT_NUM_HEADS,
        'n_ctx': DEFAULT_N_CTX,
        'act_type': act_type,
        'use_cache': False,
        'use_ln': False,

        # Training
        'total_epochs': DEFAULT_TOTAL_EPOCHS,
        'lr': DEFAULT_LEARNING_RATE,
        'weight_decay': DEFAULT_WEIGHT_DECAY,
        'optimizer': 'AdamW',
        'betas': (0.9, 0.98),

        # Analysis
        'analysis_N': DEFAULT_N_SAMPLES,
        'analysis_k': DEFAULT_K_NEIGHBORS,
        'analysis_d': DEFAULT_MANIFOLD_DIM,
        'record_frequency': DEFAULT_RECORD_FREQUENCY,
        'gcs_frequency': DEFAULT_GCS_FREQUENCY,

        # Device
        'device': get_device()
    }
