from typing import List, Tuple

import numpy as np

from MAB.Rewards import MixOfBetas, ActionReward


class Game:

    def __init__(self, actions, reward_type: ActionReward, reward_params=None):

        self.actions = actions
        self.action_rewards = []
        self.reward_type = reward_type
        if reward_params is None:
            reward_params = self.generate_params_for_actions()
        assert (len(actions) == len(reward_params))
        for params in reward_params:
            self.action_rewards.append(reward_type(*params))

        self.reward_means = np.zeros_like(self.actions, dtype=float)
        for i, a in enumerate(self.actions):
            self.reward_means[i] = self.action_rewards[a].get_mean()
        self.best_mean = self.reward_means.max()

    def get_reward_for_action(self, action):
        return self.action_rewards[action].get_reward()

    def get_best_action(self):
        return self.actions[np.argmax(self.reward_means)]

    def get_best_mean(self):
        return self.best_mean

    def get_mean_for_action(self, a):
        return self.reward_means[a]

    def get_action_count(self):
        return len(self.actions)

    def get_actions(self):
        return self.actions

    def get_reward_type(self):
        return self.reward_type

    def get_reward_params(self):
        return [ar.get_params() for ar in self.action_rewards]

    def get_params(self):
        return {"actions": self.get_actions(),
                "reward_type": self.get_reward_type(),
                "reward_params": self.get_reward_params()}

    def get_action_means(self):
        action_means = []
        for a, params in zip(self.actions, self.get_reward_params()):
            action_means.append(self.reward_type.calc_mean(params))
        return action_means

    def generate_params_for_actions(self):
        params = []
        while not self.is_valid_reward_params(params):
            params = [self.reward_type.generate_params() for _ in self.actions]
        return params

    def is_valid_reward_params(self, params):
        if len(params) > 1:
            negative_mean_exists = False
            positive_mean_exists = False
            for (alphas, betas) in params:
                mean = self.reward_type.calc_mean((alphas, betas))
                if mean < 0:
                    negative_mean_exists = True
                elif mean > 0:
                    positive_mean_exists = True
            if negative_mean_exists and positive_mean_exists:
                return True
        return False


class CostRewardGame(Game):

    def __init__(self, actions, reward_type, reward_params: List[Tuple], costs_type, costs_params: List[Tuple]):
        super().__init__(actions, reward_type, reward_params)
        self.action_costs = {}
        for a, params in zip(actions, costs_params):
            self.action_costs[a] = costs_type(*params)

    def get_cost_for_action(self, action):
        return self.action_costs[action].get_reward()

    def get_best_action(self):
        reward_means = np.zeros_like(self.actions, dtype=float)
        cost_means = np.zeros_like(self.actions, dtype=float)
        for i, a in enumerate(self.actions):
            reward_means[i] = self.action_rewards[a].get_mean()
            cost_means[i] = self.action_costs[a].get_mean()

        return self.actions[np.argmax(reward_means / cost_means)]
