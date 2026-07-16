# Hyperparameters and CV search space (solution "Hyperparameters")
from __future__ import annotations

from dataclasses import dataclass

M_REPEAT = 1e6
H_P_LONG_DEFAULT = 3000.0
H_P_SHORT = 730.0
POP_WINDOW_DEFAULT = 1500.0
LAMBDA_DEFAULT = 1.0
RECENT_SEASONS = 5
DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)
WORST_YEAR_ALPHA = 0.5
SEED = 42

CV_YEARS = [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2006, 2007, 2008]
MODEL_A_SEED_TIME = "2005-01-01"
MODEL_B_SEED_TIME = "2010-01-01"

H_P_LONG_GRID = [1500.0, 3000.0]
POP_WINDOW_GRID = [1095.0, 1500.0, 1825.0]
LAMBDA_GRID = [0.0, 1.0, 2.0]
WEIGHT_SIMPLEX_STEP = 0.1
SURVIVAL_WEIGHT_CAP = 0.3


@dataclass(frozen=True)
class HeuristicConfig:
    num_dst: int
    eval_k: int
    h_p_long: float = H_P_LONG_DEFAULT
    h_p_short: float = H_P_SHORT
    pop_window_days: float = POP_WINDOW_DEFAULT
    lam: float = LAMBDA_DEFAULT
    weights: tuple = DEFAULT_WEIGHTS
    m_repeat: float = M_REPEAT
    recent_seasons: int = RECENT_SEASONS
