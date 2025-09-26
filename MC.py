import math
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def std_norm_cdf(x: float) -> float:
    #math.erf is the error function and is generally used for normal distribtuions
    return 0.5 * (1.0 + math.erf(x /2.0**.5))
