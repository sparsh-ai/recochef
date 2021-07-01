from random import random
from typing import List, Dict
from collections import Counter
import scipy
import scipy.stats as stats
import numpy as np
import pandas as pd

from recochef.models.bandits import BetaBandit


class WeightedChoiceFailed(Exception):
    """
    Custom exception class that is used to format excpetion messages.

    Attributes:
        relative_frequencies: List of frequencies that lead to the exception.
        message: Message to be returned.
    """

    def __init__(self, relative_frequencies: List[float], message: str='Weighted choice failed:'):
        """
        Initializes a new instance of WeightedChoiceFailed.
        """
        self.relative_frequencies = relative_frequencies
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """
        Returns a formated string of the exception.
        """
        return f'{self.relative_frequencies} -> {self.message}'


class ThompsonSampling:
    """
    Class that is used to run a Thompson sampling test.

    Attributes:
        sample_size: Number of examples sampled in each batch.
        batch_size: Number of examples per batch.
    
    Methods:
        add_bandit: Adds a new BetaBandit to the test.
        update_bandit: Updates the priors of the BetaBandit at index idx.
        generate_relative_frequencies: Generates relative frequencies for each bandit in the test that will later be used in the weighted lottery.
        weighted_choice: Performes a run of the weighted lottery using relative frequencies as weights, and returns the index of the bandit that was drawn.
        bandit_batch: Determines how many times each bandit gets used in the running batch.
    """

    def __init__(self, sample_size: int=1000, batch_size: int=1000):
        """
        Initializes a new instance of ThompsonSampling with the passed parmeters

        Args:
            sample_size: Number of examples sampled in each batch.
            batch_size: Number of examples per batch.
        """

        self.sample_size = sample_size
        self.batch_size = batch_size
        self.bandits = list()
        self.relative_frequencies = list()
    
    def add_bandit(self, positive_examples: int=0, negative_examples: int=0, alpha_prior: float=1., beta_prior: float=1.):
        """
        Adds a new BetaBandit to the test.

        Args:
            positive_examples: Number of positive examples.
            negative_examples: Number of negative examples.
            alpha_prior: Prior for the alpha parameter of the underlying beta distribution.
            beta_prior: Prior for the beta parameter of the underlying beta distribution.
        """

        self.bandits.append(BetaBandit(positive_examples, negative_examples, alpha_prior, beta_prior))

    def update_bandit(self, idx: int, positive_examples: int=0, negative_examples: int=0):
        """
        Updates the priors of the BetaBandit at index idx.

        Args:
            idx: Index of the bandit to be updated.
            positive_examples: Number of positive examples.
            negative_examples: Number of negative examples.
        """

        self.bandits[idx].update(positive_examples, negative_examples)

    def generate_relative_frequencies(self):
        """
        Generates relative frequencies for each bandit in the test
        that will later be used in the weighted lottery.
        """

        bandit_samples = dict()
        self.relative_frequencies = [0] * len(self.bandits)
        for i in range(len(self.bandits)):
            bandit = self.bandits[i]
            bandit_samples[i] = bandit.sample(self.sample_size)
        bandit_df = pd.DataFrame(bandit_samples)
        for i in range(bandit_df.shape[0]):
            self.relative_frequencies[bandit_df.iloc[i].idxmax()] += 1. / self.sample_size
    
    def weighted_choice(self) -> int:
        """
        Performes a run of the weighted lottery using relative frequencies as
        weights, and returns the index of the bandit that was drawn.

        Returns:
            The index of the bandit that won the weighted lottery.
        """

        r = random()

        for i in range(len(self.relative_frequencies)):
            r -= self.relative_frequencies[i]
            if r < 0:
                return i

        raise WeightedChoiceFailed(self.relative_frequencies)

    def bandit_batch(self) -> Dict[int, int]:
        """
        Determines how many times each bandit gets used in the running batch.

        Returns:
            A dictionary that tells how many times each bandit is
            applied in the running batch.
        """

        self.generate_relative_frequencies()
        strategy = [self.weighted_choice() for _ in range(self.batch_size)]
        counter = Counter(strategy)
        return dict(counter)


class ThompsonSamplingRunner:
    """
    Class that is used to run simulations of Thompson sampling tests.

    Attributes:
        bandit_returns: List of average returns per bandit.
        alpha_priors: List of alpha priors for each bandit.
        beta_priors: List of beta priors for each bandit.
        sample_size: Sample size of BetaBandit pulls per batch.
        batch_size: Number of examples per batch.
        batches: Number of batches.
        simulations: Number of simulations.
    
    Methods:
        init_bandits: Prepares everything for new simulation.
        run: Runs the simulations and tracks performance.
    """

    def __init__(self, bandit_returns: List[float], alpha_priors: List[float]=None, beta_priors: List[float]=None, sample_size: int=1000, batch_size: int=1000, batches: int=10, simulations: int=2):
        """
        Initializes a new instance of RunThompsonSampling with the passed parameters.

        Args:
            bandit_returns: List of average returns per bandit.
            alpha_priors: List of alpha priors for each bandit.
            beta_priors: List of beta priors for each bandit.
            sample_size: Sample size of BetaBandit pulls per batch.
            batch_size: Number of examples per batch.
            batches: Number of batches.
            simulations: Number of simulations.
        """

        self.bandit_returns = bandit_returns
        self.n_bandits = len(bandit_returns)
        self.bandits = list(range(self.n_bandits))

        self.sample_size = sample_size
        self.batch_size = batch_size
        self.batches = batches
        self.simulations = simulations

        self.df_bids = pd.DataFrame(columns=self.bandit_returns)
        self.df_clicks = pd.DataFrame(columns=self.bandit_returns)

        if alpha_priors is None:
            alpha_priors = [1.] * self.n_bandits
        if beta_priors is None:
            beta_priors = [1.] * self.n_bandits
    
    def init_bandits(self):
        """
        Prepares everything for new simulation.
        """

        self.bandit_positive_examples = [0] * self.n_bandits
        self.bandit_total_examples = [0] * self.n_bandits
        self.thomsam = ThompsonSampling(self.sample_size, self.batch_size)
        for i in self.bandits:
            self.thomsam.add_bandit()

    def run(self):
        """
        Runs the simulations and tracks performance.
        """

        for j in range(self.simulations):
            self.init_bandits()
            for i in range(self.batches):
                for key, val in self.thomsam.bandit_batch().items():
                    self.bandit_total_examples[key] += val
                    self.bandit_positive_examples[key] += np.random.binomial(val, self.bandit_returns[key])
                    self.thomsam.update_bandit(key, self.bandit_positive_examples[key], self.bandit_total_examples[key] - self.bandit_positive_examples[key])
                if self.df_bids.shape[0] < self.batches:
                        self.df_bids.loc[i] = self.bandit_total_examples
                        self.df_clicks.loc[i] = self.bandit_positive_examples
                else:
                    self.df_bids.loc[i] += self.bandit_total_examples
                    self.df_clicks.loc[i] += self.bandit_positive_examples
        self.df_bids /= self.simulations
        self.df_clicks /= self.simulations


class ThompsonSimulation():
    def __init__(self, n_bandits=3):
        self.n_bandits = n_bandits
        self.trials = [0] * self.n_bandits
        self.wins = [0] * self.n_bandits
    def pull(self, i, p_bandits):
        if np.random.rand() < p_bandits[i]:
            return 1
        else:
            return 0
    def step(self, p_bandits):
        # Define the prior based on current observations
        bandit_priors = [stats.beta(a=1+w, b=1+t-w) for t, w in zip(self.trials, self.wins)]
        # Sample a probability theta for each bandit
        theta_samples = [d.rvs(1) for d in bandit_priors]
        # choose a bandit
        chosen_bandit = np.argmax(theta_samples)
        # Pull the bandit
        x = self.pull(chosen_bandit, p_bandits)
        # Update trials and wins (defines the posterior)
        self.trials[chosen_bandit] += 1
        self.wins[chosen_bandit] += x
        return self.trials, self.wins