import numpy as np
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

"""
rewards in -1 to 1
"""


class ActionReward(ABC):

    def __init__(self, mean, r_min=-1, r_max=1):
        self.mean = mean
        self.r_min = r_min
        self.r_max = r_max

    def get_reward(self):
        return np.clip(self._get_reward(), self.r_min, self.r_max)

    @abstractmethod
    def _get_reward(self):
        pass

    def get_mean(self):
        return self.mean

    @staticmethod
    @abstractmethod
    def is_valid_reward_params(params):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @staticmethod
    @abstractmethod
    def calc_mean(params):
        pass

    @staticmethod
    @abstractmethod
    def generate_params():
        pass

    def plot(self, ax=None):
        if ax is None:
            plt.hist([self.get_reward() for _ in range(10000)], bins=100)
        else:
            ax.hist([self.get_reward() for _ in range(10000)], bins=100)

    @classmethod
    def plot_family(cls):
        fig, ax = plt.subplots(6, 6, sharex=True, figsize=(20, 20))
        for i in range(6):
            for j in range(6):
                cls(*cls.generate_params()).plot(ax[i][j])
        fig.show()


class GaussianReward(ActionReward):

    def __init__(self, mean, var):
        super().__init__(mean)
        self.var = var

    def _get_reward(self):
        return np.random.normal(self.mean, self.var)

    @staticmethod
    def generate_params():
        return np.random.uniform(-0.5, 0.1), np.random.uniform(0, 0.3)

    @staticmethod
    def is_valid_reward_params(params):
        if len(params) > 1:
            negative_mean_exists = False
            positive_mean_exists = False
            for (mean, _) in params:
                if mean < 0:
                    negative_mean_exists = True
                elif mean > 0:
                    positive_mean_exists = True
            if negative_mean_exists and positive_mean_exists:
                return True
        return False

    def get_params(self):
        return self.mean, self.var

    @staticmethod
    def calc_mean(params):
        return params[0]


class MixOfBetas(ActionReward):

    def __init__(self, alphas, betas):
        super().__init__(self._calc_mean(alphas, betas))
        self.alphas = alphas
        self.betas = betas
        self.beta_dist_n = len(alphas)

    def _get_reward(self):
        return sum([np.random.beta(a, b) - 0.5 for a, b in zip(self.alphas, self.betas)]) * (2 / self.beta_dist_n)

    @staticmethod
    def calc_mean(params):
        alphas, betas = params
        return MixOfBetas._calc_mean(alphas, betas)

    @staticmethod
    def _calc_mean(alphas, betas):
        return sum([1 / (1 + b / a) - 0.5 for a, b in zip(alphas, betas)]) * (2 / len(alphas))

    @staticmethod
    def generate_params():
        mixed_dist_n = 3
        return np.random.uniform(0.001, 1.5, (2, mixed_dist_n)).tolist()

    @staticmethod
    def is_valid_reward_params(params):
        if len(params) > 1:
            negative_mean_exists = False
            positive_mean_exists = False
            for (alphas, betas) in params:
                mean = MixOfBetas._calc_mean(alphas, betas)
                if mean < 0:
                    negative_mean_exists = True
                elif mean > 0:
                    positive_mean_exists = True
            if negative_mean_exists and positive_mean_exists:
                return True
        return False

    def get_params(self):
        return self.alphas, self.betas
