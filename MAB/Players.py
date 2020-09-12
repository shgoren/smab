from abc import ABC, abstractmethod
from MAB.Util import *

import numpy as np

from MAB.Game import *

from Util import argmin


class Player(ABC):

    def __init__(self, game):
        self.game = game
        self.actions = game.get_actions()
        self.past_rewards = []
        self.past_actions = []

    def choose_action(self):
        a = self._choose_action()
        self.past_actions.append(a)
        return a

    @abstractmethod
    def _choose_action(self):
        pass

    @abstractmethod
    def take_action(self, action):
        pass

    def update(self, action, reward):
        pass

    def get_game(self):
        return self.game

    def take_step(self):
        return self.take_action(self.choose_action())

    # TODO: compare means instead of empirical regret (check if that is correct to do, what is pseudo regret?)
    def get_regret(self):
        return self.game.get_best_mean() * self.get_step() - sum(
            [self.game.get_mean_for_action(a) for a in self.past_actions])

    def get_total_rewards(self):
        return sum(self.past_rewards)

    def get_step(self):
        return len(self.past_rewards)

    def get_past_actions(self):
        return self.past_actions

    def avg_reward(self):
        return mean(self.past_rewards)

    def get_step_type(self):
        return []

    def game_desc(self):
        return {}


class SurvivalPlayer(Player, ABC):

    def __init__(self, game: Game, budget: float):
        super().__init__(game)
        self.budget = budget
        self.init_budget = budget
        self.active = True

    def take_action(self, action):
        if not self.active:
            raise Exception("Player no longer active")
        r = self.game.get_reward_for_action(action)
        self.budget += r
        self.update(action, r)
        self.past_rewards.append(r)
        if self.budget < 0:
            self.active = False
        return r

    def get_init_budget(self):
        return self.init_budget

    def get_budget(self):
        return self.budget

    def game_desc(self):
        return {"is_ruined": self.active}

    def is_active(self):
        return self.active

    def get_params(self):
        return {"budget": self.budget}


class BudgetPlayer(Player, ABC):

    def __init__(self, game: CostRewardGame, budget: float):
        super().__init__(game)
        self.past_costs = []
        self.budget = budget
        self.active = True

    def take_action(self, action):
        """
        if budget is positive, performs and action and logs it's results (budget can be negative) 
        :param action:
        :return:
        """
        if not self.active:
            raise Exception("Player no longer active")
        r = self.game.get_reward_for_action(action)
        c = self.game.get_cost_for_action(action)
        self.past_rewards.append(r)
        self.past_costs.append(c)
        self.budget -= c
        if self.budget < 0:
            self.active = False
        return r, c


class RandomSurvivalPlayer(SurvivalPlayer):

    def _choose_action(self):
        return np.random.choice(self.actions)


class RandomBudgetPlayer(SurvivalPlayer):

    def _choose_action(self):
        return np.random.choice(self.actions)


class SimpleUCBSurvivalPlayer(SurvivalPlayer):

    def __init__(self, game, budget):
        super().__init__(game, budget)
        self.action_data = {a: {"avg": 0, "n": 0} for a in self.actions}
        self.is_safe_step = []

    def _choose_action(self):
        # possible_actions = [a for a in self.actions if self.is_safe(a)]
        # if len(possible_actions) > 0:
        #     best_action = self.best_action(possible_actions)
        #     self.is_safe_step.append(True)
        # else:
        #     best_action = self.safest_action(self.actions)
        #     self.is_safe_step.append(False)
        # return best_action
        return self.best_action(self.actions)

    def update(self, a, r):
        action_record = self.action_data[a]
        avg, n = action_record["avg"], action_record["n"]
        action_record["avg"] = (avg * n + r) / (n + 1)
        action_record["n"] = n + 1

    def is_safe(self, a):
        return self.budget - self.lower_bound(a) > 0

    def best_action(self, action_list):
        return argmax([self.upper_bound(a) for a in action_list])

    def worst_action(self, action_list):
        return argmin([self.lower_bound(a) for a in action_list])

    def get_confidence_bound(self, a):
        return np.sqrt((np.log(self.budget)) / (self.action_data[a]["n"] + 1))

    def lower_bound(self, a):
        return self.action_data[a]["avg"] - self.get_confidence_bound(a)

    def upper_bound(self, a):
        return self.action_data[a]["avg"] + self.get_confidence_bound(a)

    def safest_action(self, action_list):
        # TODO: change according to some risk averse measure (std or quantile based)
        return self.best_action(self.actions)

    def game_desc(self):
        return {"safe_steps": self.is_safe_step}.update(super().game_desc())


class MVsurvivalPlayer(SurvivalPlayer):

    def __init__(self, game, budget, risk_aversion=1):
        super().__init__(game, budget)
        self.risk_aversion = risk_aversion
        self.action_rewards = {a: [] for a in self.actions}
        self.action_mv = {a: [0] for a in self.actions}
        self.risk_factor = []

    def get_mv(self, a):
        return self.action_mv[a][-1]

    def calc_mv(self, a):
        rewards = self.action_rewards[a]
        m = mean(rewards)
        v = var(rewards, m)
        return v - self.risk_factor * m

    def calc_mv_with_ci(self, a):
        delta = 0.95
        mv = self.calc_mv(a)
        return mv - (self.risk_factor) * np.sqrt((np.log(1 / delta)) / (2 * self.action_count(a) + 1))

    def action_count(self, a):
        return len(self.action_rewards[a])

    def _choose_action(self):
        return argmin([l[-1] for l in self.action_mv.values()])

    def update(self, action, reward):
        self.action_rewards[action].append(reward)
        self.update_risk_factor()
        self.update_mv()

    def update_risk_factor(self):
        self.risk_factor = self.budget * np.exp(self.avg_reward()) / self.risk_aversion

    def update_mv(self):
        for a in self.actions:
            self.action_mv[a].append(self.calc_mv_with_ci(a))


class BufferedUCBPlayer(SurvivalPlayer):

    def __init__(self, game, budget):
        super().__init__(game, budget)
        self.action_data = {a: {"avg": 0, "n": 0} for a in self.actions}
        self.positive_avg_actions = set()
        self.negative_avg_actions = set(self.actions)
        self.pos_steps = 0
        self.neg_steps = 0
        self.step_types = []

    def _choose_action(self):
        if len(self.positive_avg_actions) > 0 and self.budget < self.get_min_budget():
            upper_bounds = self.pos_group_upper_bounds()
            step_type = True
        else:
            upper_bounds = self.neg_group_upper_bounds()
            step_type = False
        a = max(upper_bounds.keys(), key=lambda k: upper_bounds[k])
        self.step_types.append(step_type)
        return a

    def update(self, action, reward):
        if self.step_types[-1] == True:
            self.pos_steps += 1
        else:
            self.neg_steps += 1
        self.update_avg(action, reward)
        self.update_groups()

    def update_avg(self, a, r):
        action_record = self.action_data[a]
        avg, n = action_record["avg"], action_record["n"]
        action_record["avg"] = (avg * n + r) / (n + 1)
        action_record["n"] = n + 1

    def update_groups(self):
        pos_set = set()
        for a in self.actions:
            if self.action_data[a]["avg"] > 0:
                pos_set.add(a)
        self.positive_avg_actions = pos_set

    def pos_group_upper_bounds(self):
        d = {}
        for a in self.positive_avg_actions:
            n = self.action_data[a]["n"]
            avg = self.action_data[a]["avg"]
            d[a] = avg + np.sqrt(np.log(self.pos_steps) / n)
        return d

    def neg_group_upper_bounds(self):
        d = {}
        for a in self.negative_avg_actions:
            n = self.action_data[a]["n"]
            avg = self.action_data[a]["avg"]
            d[a] = avg + np.sqrt(np.log(self.neg_steps) / n)
        return d

    def get_min_budget(self):
        delta = 0.7
        upper_bounds = self.pos_group_upper_bounds()
        a = max(upper_bounds.keys(), key=lambda k: upper_bounds[k])
        self.positive_avg_actions
        return -np.log(delta)/np.square(self.action_data[a]["avg"])

    def get_step_type(self):
        return self.step_types

