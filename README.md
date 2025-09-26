This is my first attempt at a quantative finance project. 

Monte Carlo simulation is a method to price European call options using real stock market data.
This project also compares the results against the Black–Scholes formula, which calculates the same thing, but using a formula instead of a large number of simulations.

Functionality:
1. Fetch Data: Stock prices are pulled using Yahoo Finance (yfinance) and the programme calulates the expected annual return (drift) and volatility (standard deviation of results)
2. 
Monte Carlo Simulation:
- Simulate thousands of possible future stock prices at expiration.

Black–Scholes Comparison:
- Use the closed-form Black–Scholes formula to compare against the MC calculations

Plotting:
- Plot the distribution of simulated stock prices.
