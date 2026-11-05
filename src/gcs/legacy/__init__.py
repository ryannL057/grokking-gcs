"""Preserved experiment protocols; invoke through scripts/experiment.py."""
import torch

# These protocols were specified in double precision, including fresh processes
# that only analyze checkpoints and never call setup_training_environment.
torch.set_default_dtype(torch.float64)
