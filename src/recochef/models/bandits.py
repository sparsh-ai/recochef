import math

import numpy as np


class BetaBandit:
    """
    Bandit class that is used in Thompson sampling.

    Attributes:
        alpha: Alpha parameter of the beta distribution (number of positive examples).
        beta: Beta parameter of the beta distribution (number of negative examples).
    
    Methods:
        update: Updates alpha and beta priors of the BetaBandit.
        sample: Samples the BetaBandit's distribution n times.
    """
    def __init__(self, alpha: int=0, beta: int=0, alpha_prior: float=1., beta_prior: float=1.):
        """
        Initializes new BetaBandit with passed parameters.

        Args:
            alpha: Alpha parameter of the beta distribution (number of positive examples).
            beta: Beta parameter of the beta distribution (number of negative examples).
            alpha_prior: The prior for alpha parameter.
            beta_prior: The prior for beta parameter.
        """
        self.alpha = alpha + alpha_prior
        self.beta = beta + beta_prior
    
    def update(self, positive_examples: int=0, negative_examples: int=0):
        """
        Updates alpha and beta priors of the BetaBandit.

        Args:
            positive_examples: Number of positive examples.
            negative_examples: Number of negative examples.
        """
        self.alpha += positive_examples
        self.beta += negative_examples
    
    def sample(self, n: int) -> np.ndarray:
        """
        Samples the BetaBandit's distribution n times.

        Args:
            n: Sample size.
        Returns:
            An array filled with n examples sampled
            from the BetaBandit's distribution.
        """
        return np.random.beta(self.alpha, self.beta, n)


class EpsilonBandit:
    """
    Bandit class that is used in Epsilon-greedy multi-armed bandits.

    Attributes:
        positive_examples: Number of positive examples.
    
    Methods:
        update: Updates the number of positive examples of the EpsilonBandit.
        get_value: Gets the number of positive examples of the EpsilonBandit.
    """
    def __init__(self, positive_examples: int=0):
        """
        Initializes a new EpsilonBandit and sets its positive examples.

        Args:
            positive_examples: Number of positive examples.
        """
        self.positive_examples = positive_examples
    
    def update(self, positive_examples: int=0):
        """
        Updates the number of positive examples of the EpsilonBandit.

        Args:
            positive_examples: Number of positive examples.
        """
        self.positive_examples += positive_examples
    
    def get_value(self) -> int:
        """
        Gets the number of positive examples of the EpsilonBandit.

        Returns:
            The number of positive examples of the EpsilonBandit.
        """
        return self.positive_examples