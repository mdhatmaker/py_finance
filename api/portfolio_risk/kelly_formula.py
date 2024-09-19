import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://blogs.cfainstitute.org/investor/2018/06/14/the-kelly-criterion-you-dont-know-the-half-of-it/


# Kelly % = edge/odds  (from Fortune's Formula)

# Kelly % = W – [(1 – W)/R]   (original formula)
# W is win probability
# R is ratio between profit and loss
# ex: Win  P/L=20%   Probability=60%
#     Loss P/L=20%   Probability=40%
#     Kelly % = 60% – [(1 – 60%) / (20%/20%)] = 20%

# Kelly % = W/A – (1 – W)/B   (modified formula, for downside-case loss less than 100%)
# W is win probability
# B is profit in case of win
# A is potential loss
# ex: Win  P/L=20%   Probability=60%
#     Loss P/L=20%   Probability=40%
#     Kelly % = 60%/20% – (1 – 60%)/20% = 100%


# Kelly Simulation, Binary Security

def generate_simulations(winprob=0.6, profit_loss=np.array([0.2, -0.2]), competing_allocations=np.array([0.2, 1, 1.5]), trials=1000, periods=100):
    # trials = 1000  # Repeat the simulation this many times
    # periods = 100  # Periods per simulation
    # winprob = 0.6  # Win probability per period
    # profit_loss = np.array([0.2, -0.2])  # Profit if win, Loss if lose
    # competing_allocations = np.array([0.2, 1, 1.5])  # Competing allocations to test

    np.random.seed(136)
    wealth = np.zeros((trials, len(competing_allocations), periods))
    wealth[:, :, 0] = 1  # Initial wealth is 1 in period 1

    # Simulation loop
    print('trials: ', end='')
    for trial in range(trials):
        if trial % 100 == 0: print(f"{trial} ", end='')
        outcome = np.random.binomial(n=1, p=winprob, size=periods)
        ret = np.where(outcome, profit_loss[0], profit_loss[1])
        for i in range(1, periods):
            # print('.', end='')
            for j in range(len(competing_allocations)):
                bet = competing_allocations[j]
                wealth[trial, j, i] = wealth[trial, j, i-1] * (1 + bet * ret[i])
    print()

    # Convert wealth array to a pandas DataFrame for easier plotting
    wealth_df = pd.DataFrame(
        wealth.reshape(trials * len(competing_allocations), periods),
        columns=[f'Period {i+1}' for i in range(periods)]
    )
    wealth_df['Fraction'] = np.repeat(competing_allocations, trials)
    wealth_df['Trial'] = np.tile(np.arange(trials), len(competing_allocations))

    # Melt the DataFrame for seaborn
    wealth_melted = pd.melt(wealth_df, id_vars=['Fraction', 'Trial'], var_name='Period', value_name='Wealth')

    return wealth_melted


def plot_results(wealth_melted, winprob, profit_loss):
    # Plotting the results
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=wealth_melted, x='Period', y='Wealth', hue='Fraction')
    plt.title(f'Kelly Simulation, Binary Security (W={winprob}, B={profit_loss[0]}, A={profit_loss[1]})')
    plt.xlabel('Period')
    plt.ylabel('Wealth')
    plt.legend(title='Fraction')
    plt.show()


def run_kelly_formula(winprob=0.6, profit_loss=np.array([0.2, -0.2]), competing_allocations=np.array([0.2, 1, 1.5]), trials=1000, periods=100):
    wealth_melted = generate_simulations(winprob, profit_loss, competing_allocations, trials, periods)
    print(wealth_melted)
    plot_results(wealth_melted, winprob, profit_loss)


if __name__ == "__main__":
    run_kelly_formula(winprob=0.6, profit_loss=np.array([0.2, -0.2]), competing_allocations=np.array([0.2, 1, 1.5]))
    run_kelly_formula(winprob=0.6, profit_loss=np.array([0.2, -0.1]), competing_allocations=np.array([0.2, 1, 1.5]))
    run_kelly_formula(winprob=0.65, profit_loss=np.array([0.2, -0.1]), competing_allocations=np.array([0.2, 1, 1.5]))
    run_kelly_formula(winprob=0.85, profit_loss=np.array([0.2, -0.1]), competing_allocations=np.array([0.2, 1, 1.5]))


"""
The theoretical downside for all capital market investments is -100%. Bad things happen. Companies go
bankrupt. Bonds default and are sometimes wiped out. Fair enough.

But for an analysis of the securities in the binary framework implied by the edge/odds formula, the
downside-scenario probability must be set to the probability of a total capital loss, not the much
larger probability of some loss.

There are many criticisms of the Kelly criterion. And while most are beyond the scope of this article,
one is worth addressing. A switch to the “correct” Kelly formula — Kelly % = W/A – (1 – W)/B — often
leads to significantly higher allocations than the more popular version.

Most investors won’t tolerate the volatility and resulting drawdowns and will opt to reduce the
allocation. That’s well and good — both variations of the formula can be scaled down — but the
“correct” version is still superior. Why? Because it explicitly accounts for and encourages investors
to think through the downside scenario.
"""

