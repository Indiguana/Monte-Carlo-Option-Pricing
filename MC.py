import math
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def std_norm_cdf(x: float) -> float:
    #math.erf is the error function and is generally used for normal distribtuions
    return 0.5 * (1.0 + math.erf(x /2.0**.5))

#This function only takes in floats and outputs a float, which is the price of the option
#S0 = stock price today; K = Strike Price; r = risk free interest rate; T = Time; sigma is the volatility of the stock
def black_scholes_call(S0, K, r, sigma, T):

    #Black-Scholes formula for a European CALL, where the individual has the right to excersise the call option but is not required
    if T <= 0: #This is if the option has expired
        return max(S0 - K, 0.0)
    if sigma <= 0:
        ST = S0 * math.exp(r * T) #Stock price at expiration
        return math.exp(-r * T) * max(ST - K, 0.0) #gives the present value of the call option
    
    #these two variables measure how many standard deviations is
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T)) #d1 differs from d2 because d1 considers both the drift and volatility
    d2 = d1 - sigma * math.sqrt(T)
    #Calculates the normal cdf
    try:
        from scipy.stats import norm
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
    except Exception:
        N_d1 = std_norm_cdf(d1)
        N_d2 = std_norm_cdf(d2)
    #This is the equation to calculate the blackscholes function
    return S0 * N_d1 - K * math.exp(-r * T) * N_d2