import math
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------
# Utilities: normal CDF (no SciPy required)
# ---------------------------
def std_norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (accurate enough for our use)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
