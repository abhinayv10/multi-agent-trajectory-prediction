"""
models package — trajectory prediction sub-modules.

Re-exports the public model classes so callers can do
`from models import TrajectoryModel`.
"""

from .temporal_encoder import TemporalEncoder
from .social_gat import SocialGAT
from .map_encoder import MapEncoder
from .goal_predictor import GoalPredictor
from .cvae import CVAE
from .decoder import Decoder
from .trajectory import TrajectoryModel
from .baselines import ConstantVelocity, LSTMEncDec

MODELS = {
    "trajectory": TrajectoryModel,
    "lstm": LSTMEncDec,
    "cv": ConstantVelocity,
}


def build_model(config: dict, device):
    """Instantiate the model selected by config["model"]["type"] (default
    "trajectory")."""
    kind = config["model"].get("type", "trajectory")
    if kind not in MODELS:
        raise ValueError(f"Unknown model type '{kind}'. Expected one of {list(MODELS)}.")
    return MODELS[kind](config, device)

__all__ = [
    "TemporalEncoder",
    "SocialGAT",
    "MapEncoder",
    "GoalPredictor",
    "CVAE",
    "Decoder",
    "TrajectoryModel",
    "ConstantVelocity",
    "LSTMEncDec",
    "build_model",
]
