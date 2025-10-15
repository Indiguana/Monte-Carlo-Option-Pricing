import math
from typing import Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def std_norm_cdf(x):
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
def estimate_annual_mu_sigma(prices, trading_days=252):
    if len(prices) < 2:
        raise ValueError("Need at least 2 price points to compute returns.")
    #This calculates the logarithmic returns of the prices compared to the previous entry in the price series. Logarithms are used because theyc compound over time.
    logrets = np.log(prices / prices.shift(1)).dropna() 
    mu = logrets.mean() * trading_days #Gets the annual log returns
    sigma = logrets.std(ddof=1) * math.sqrt(trading_days)
    return float(mu), float(sigma)


#This function is calculates the monte carlo approximation by just averaging the expected price of random events. Outputs a tuple of (est_price, avg_error)
def mc_call_price_gbm(S0, K, r, sigma, T, N= 100_000, seed = None):
    #Here, n represents the number of monte carlo simulations, which defaults to 100k if no input is provided
    # Additionally, seed is an optional RNG
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)#These will be used to estimate the S_T = Stock price at the specified time
    drift = (r - 0.5 * sigma * sigma) * T #expect average rate of change of stock
    diffusion = sigma * math.sqrt(T) * Z #how much stock price will change (volatility)
    ST = S0 * np.exp(drift + diffusion)
    payoffs = np.maximum(ST - K, 0.0) #Ammount the option holder will get if they use the option on expiration date
    discounted = np.exp(-r * T) * payoffs#Adjusts payoffs to get their value today (adjusted by the interest rate)
    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / math.sqrt(N)) #formula to calculate standard error
    return price, se


#This function uses a better version of the montecarlo approach from the previous function because it reduces random noise (variance) in the estimate by pairing each random number with its negative counterpart (-Z).
#Takes in floats and outputs a tuple of (estimated option price, standard error), similar to before
def mc_call_price_antithetic(S0: float, K: float, r: float, sigma: float, T: float,
                             N: int = 100_000, seed: Optional[int] = None) -> Tuple[float, float]:

    rng = np.random.default_rng(seed)

   
    half = (N + 1) // 2 #THis time, we only need half the number of random numbers because for each one, we are going to consider the negative counterpart as well
    Z = rng.standard_normal(half)
    Z_full = np.concatenate([Z, -Z])[:N] #combines both Zs to get a list of both
    
    #rest of the code is the same
    drift = (r - 0.5 * sigma * sigma) * T       
    diffusion = sigma * math.sqrt(T) * Z_full       
    ST = S0 * np.exp(drift + diffusion)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())
    se = float(discounted.std(ddof=1)/math.sqrt(N))
    return (price, se)

def main():
    ticker = "NVDA"  # example stock
    start = "2020-01-01"
    end = "2024-01-01"
    print(f"Downloading {ticker} prices {start} to {end} ...")
    prices = fetch_price_series(ticker, start, end)
    prices = prices.dropna()
    prices = prices[prices > 0]

    S0 = float(prices.iloc[-1])
    print(f"Last price (S0) = {S0:.2f}")

    mu_hist, sigma_hist = estimate_annual_mu_sigma(prices)
    print(f"Estimated annual historical mu = {mu_hist:.4f}, sigma = {sigma_hist:.4f}")

    # Model inputs| can be modified
    K = S0 * 1.05
    T = 1.0
    r = 0.03
    sigma = sigma_hist  # use historical volatility
    if sigma_hist > 3:
        sigma_hist /= 100.0
    print(f"Inputs: K={K:.2f}, T={T}, r={r}, sigma={sigma:.4f}")

    bs_price = black_scholes_call(S0, K, r, sigma, T)
    print(f"Black-Scholes price = {bs_price:.6f}")

    N = 200_000
    mc_price, mc_se = mc_call_price_gbm(S0, K, r, sigma, T, N=N, seed=123)
    mc_ant_price, mc_ant_se = mc_call_price_antithetic(S0, K, r, sigma, T, N=N, seed=123)

    print(f"MC plain     : price={mc_price:.6f}, SE={mc_se:.6f}")
    print(f"MC antithetic: price={mc_ant_price:.6f}, SE={mc_ant_se:.6f}")

    # Simulate terminal prices
    rng = np.random.default_rng(123)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)

    #This one outputs the expected stock price after T time has passed
    plt.figure(figsize=(8, 4))
    plt.hist(ST, bins=60, alpha=0.8, color="skyblue")
    plt.axvline(K, color="red", linestyle="--", label=f"Strike K={K:.2f}")
    plt.title(f"Simulated Terminal Stock Prices ST (T={T} year)")
    plt.xlabel("Stock Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #This graph displays the call option payoffs with simulation
    plt.figure(figsize=(8, 4))
    plt.hist(payoffs, bins=60, alpha=0.8, color="lightgreen")
    plt.axvline(payoffs.mean(), color="red", linestyle="--", label=f"Expected payoff ≈ {payoffs.mean():.2f}")
    plt.title(f"Simulated Call Option Payoffs (max(ST-K,0))")
    plt.xlabel("Option Payout")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()



main()




