from pymab.logging_config import setup_logging


setup_logging()

from pymab.game import Game
from pymab.policies.thompson_sampling import ThompsonSamplingPolicy
