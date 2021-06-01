import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from recochef.models.bandits import EpsilonBandit


class EpsilonGreedy:
    """
    Class that is used to run a Epsilon-greedy multi-armed bandits test.

    Attributes:
        epsilon: Percentage of exploration.
        batch_size: Number of examples per batch.
    
    Methods:
        add_bandit: Adds a new BetaBandit to the test.
        update_bandit: Updates the priors of the BetaBandit at index idx.
        add_best_bandit: Adds the best bandit of the current batch.
        bandit_batch: Determines how many times each bandit gets used in the running batch.
    """

    def __init__(self, epsilon: float=0.2, batch_size: int=1000):
        """
        Initializes a new instance of EpsilonGreedy with the passed parmeters

        Args:
            epsilon: Percentage of exploration.
            batch_size: Number of examples per batch.
        """
    
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.bandits = list()
        self.best_bandits = list()
    
    def add_bandit(self, positive_examples: int=0):
        """
        Adds a new EpsilonBandit to the test.

        Args:
            positive_examples: Number of positive examples.
        """
        self.bandits.append(EpsilonBandit(positive_examples))

    def update_bandit(self, idx: int, positive_examples: int=0):
        """
        Updates the priors of the BetaBandit at index idx.

        Args:
            idx: Index of the bandit to be updated.
            positive_examples: Number of positive examples.
        """

        self.bandits[idx].update(positive_examples)

    def add_best_bandit(self):
        """
        Adds the best bandit of the current batch.
        """

        idx = 0
        maxVal = 0
        for i in range(len(self.bandits)):
            val = self.bandits[i].get_value()
            if val > maxVal:
                maxVal = val
                idx = i
        self.best_bandits.append(idx)

    def bandit_batch(self) -> Tuple[int, int]:
        """
        Determines which bandit is the best and how many times examples
        get used for exploration.

        Returns:
            The number of exploratory examples and the best bandit's index.
        """

        self.add_best_bandit()
        n_bandits = len(self.bandits)
        exploration_total = self.batch_size * self.epsilon
        exploration = int(exploration_total / n_bandits)

        return exploration, self.best_bandits[-1]


class EpsilonGreedyRunner:
    """
    Class that is used to run simulations of Thompson sampling tests.

    Attributes:
        bandit_returns: List of average returns per bandit.
        epsilon: Percentage of exploration.
        batch_size: Number of examples per batch.
        batches: Number of batches.
        simulations: Number of simulations.
    
    Methods:
        init_bandits: Prepares everything for new simulation.
        run: Runs the simulations and tracks performance.
    """

    def __init__(self, bandit_returns: List[float], epsilon: float=0.2, batch_size: int=10000, batches: int=10, simulations: int=100):
        """
        Initializes a new instance of RunEpsilonGreedy with the passed parameters.

        Attributes:
            bandit_returns: List of average returns per bandit.
            epsilon: Percentage of exploration.
            batch_size: Number of examples per batch.
            batches: Number of batches.
            simulations: Number of simulations.
        """

        self.bandit_returns = bandit_returns
        self.n_bandits = len(bandit_returns)
        self.bandits = list(range(self.n_bandits))

        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batches = batches
        self.simulations = simulations

        self.df_bids = pd.DataFrame(columns=self.bandit_returns)
        self.df_clicks = pd.DataFrame(columns=self.bandit_returns)

    def init_bandits(self):
        """
        Prepares everything for new simulation.
        """

        self.first_batch = True
        self.bandit_positive_examples = [0] * self.n_bandits
        self.bandit_total_examples = [0] * self.n_bandits
        self.eps = EpsilonGreedy(self.epsilon, self.batch_size)
        for i in self.bandits:
            self.eps.add_bandit()
    
    def run(self):
        """
        Runs the simulations and tracks performance.
        """

        for j in range(self.simulations):
            self.init_bandits()
            for i in range(self.batches):
                exploration_examples, best_bandit = self.eps.bandit_batch()
                if self.first_batch:
                    self.first_batch = False
                    exploration_examples = self.batch_size // self.n_bandits
                for idx in self.bandits:
                    self.bandit_total_examples[idx] += exploration_examples
                    positive_examples = np.random.binomial(exploration_examples, self.bandit_returns[idx])
                    self.bandit_positive_examples[idx] += positive_examples
                    self.eps.update_bandit(idx, positive_examples)
                
                exploitation_examples = self.batch_size - exploration_examples * self.n_bandits
                self.bandit_total_examples[best_bandit] += exploitation_examples
                self.bandit_positive_examples[best_bandit] += np.random.binomial(exploitation_examples, self.bandit_returns[best_bandit])

                if self.df_bids.shape[0] < self.batches:
                    self.df_bids.loc[i] = self.bandit_total_examples
                    self.df_clicks.loc[i] = self.bandit_positive_examples
                else:
                    self.df_bids.loc[i] += self.bandit_total_examples
                    self.df_clicks.loc[i] += self.bandit_positive_examples
        self.df_bids /= self.simulations
        self.df_clicks /= self.simulations