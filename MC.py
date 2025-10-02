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
#This function gets the price series of a stock (sequence of prices) from Yahoo Finance. It takes in stings and outputs a pandas series
def fetch_price_series(ticker, start, end):
   # Download price data using yfinance and return a price series start/end in 'YYYY-MM-DD' format.
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check ticker and date range.")
    if "Adj Close" in df.columns:
        series = df["Adj Close"].dropna()
    elif "Close" in df.columns:
        series = df["Close"].dropna()
    else:
        raise ValueError("Downloaded data does not contain Close or Adj Close columns.")
    return series


#This function estimates the drift of a stock (mu) and the volatility of the stokc (sigma). It takes in the price series as a Pandas Series and the number of trading days in a year
def estimate_annual_mu_sigma(prices, trading_days):
    if len(prices) < 2:
        raise ValueError("Need at least 2 price points to compute returns.")
    #This calculates the logarithmic returns of the prices compared to the previous entry in the price series. Logarithms are used because theyc compound over time.
    logrets = np.log(prices / prices.shift(1)).dropna() 
    mu = logrets.mean() * trading_days #Gets the annual log returns
    sigma = logrets.std(ddof=1) * math.sqrt(trading_days)
    return float(mu), float(sigma)