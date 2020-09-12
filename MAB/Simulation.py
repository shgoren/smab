import operator
from collections import Counter
from pprint import pprint

import numpy as np

from MAB.Players import SimpleUCBSurvivalPlayer
from MAB.Rewards import MixOfBetas
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import math


def true_false_sections(tfsequence):
    safe_section = None
    begin_idx = 0
    safe_sections = []
    unsafe_sections = []
    for now_idx, safe_now in enumerate(tfsequence):
        if safe_section is None:
            safe_section = safe_now
        if safe_section != safe_now or now_idx == len(tfsequence) - 1:
            if safe_section:
                safe_sections.append((begin_idx, now_idx))
            else:
                unsafe_sections.append((begin_idx, now_idx))
            safe_section = safe_now
            begin_idx = now_idx
    return safe_sections, unsafe_sections


class Simulation:

    def __init__(self, game_const, reward_const, player_const, budget_tightness=0.1, timeout=10000):
        self.timeout = timeout
        self.budget_tightness = budget_tightness
        self.player_const = player_const
        self.reward_const = reward_const
        self.game_const = game_const
        self.run_log = []

    def run_count(self):
        return len(self.run_log)

    def new_agent(self, game_params=None, player_params=None):
        if game_params is None:
            action_n = np.random.randint(2, 100)
            actions = list(range(action_n))
            reward_type = self.reward_const
            game_params = {"actions": actions,
                           "reward_type": reward_type}
        game = self.game_const(**game_params)
        if player_params is None:
            action_n = len(game.get_actions())
            budget = np.random.uniform(action_n * self.budget_tightness,
                                       len(game.get_actions()) * 5 * self.budget_tightness)
            player_params = {"budget": budget}
        player_params.update({"game": game})
        player = self.player_const(**player_params)
        return player

    def simulate(self, n_runs, game_params=None, player_params=None):
        for i in range(self.run_count(), self.run_count() + n_runs):
            regret = []
            budget = []
            steps = 0
            agent = self.new_agent(game_params, player_params)
            run_params = agent.get_game().get_params()
            while agent.is_active() and steps < self.timeout:
                agent.take_step()
                steps += 1
                regret.append(agent.get_regret())
                budget.append(agent.get_budget())
            self.run_log.append({"run_params": run_params,
                                 "init_budget": agent.get_init_budget(),
                                 "budget": budget,
                                 "regret": regret,
                                 "agent_specific_log": agent.game_desc(),
                                 "past_actions": agent.get_past_actions(),
                                 "avg_reward": agent.avg_reward(),
                                 "is_ruined": agent.is_active(),
                                 "agent_type": self.player_const,
                                 "step_type": agent.get_step_type()
                                 })

    def set_player(self, new_player):
        self.player_const = new_player

    def simulate_comparison(self, runs_to_simulate, times):
        for i in runs_to_simulate:
            for _ in range(times):
                game_params = self.fetch_log(i, "run_params")
                player_params = {"budget": self.fetch_log(i, "init_budget")}
                self.simulate(1, game_params, player_params)

    def summary(self):
        t = PrettyTable(["",
                         "steps",
                         "final_regret",
                         "initial budget",
                         f"final budget",
                         "min budget",
                         "most freq actions",
                         "actions"], align='l')

        for i, rlog in enumerate(self.run_log):
            t.add_row([i,
                       len(rlog['regret']),
                       round((rlog['regret'][-1])),
                       round(self.fetch_log(i, 'init_budget'), 2),
                       round(rlog['budget'][-1], 2),
                       round(min(rlog['budget']), 2),
                       max(Counter(self.fetch_log(i, 'past_actions')).items(), key=operator.itemgetter(1))[0],
                       len(rlog['run_params']['actions'])])
        print(t)

    def describe_round(self, round_n, plot_action_rewards=True, plot_regret=True, plot_budget=True):
        reward_params, actions, init_budget = self.fetch_log(round_n, 'reward_params'), self.fetch_log(round_n,
                                                                                                       "actions"), \
                                              self.fetch_log(round_n, "init_budget")
        action_n = len(actions)
        means = [round(self.reward_const.calc_mean(params), 5) for params in reward_params]
        ending_reason = "timeout" if self.fetch_log(round_n, "is_ruined") else "destruction"
        best_action_idx = np.array(means).argmax()
        avg_reward, past_actions = self.fetch_log(round_n, "avg_reward"), self.fetch_log(round_n, "past_actions")
        top_freq_action = max(Counter(past_actions).items(), key=operator.itemgetter(1))[0]

        t = PrettyTable(["", f"round-{round_n}"])
        t.align = "l"
        t.add_row(["ending reason:", ending_reason])
        t.add_row(["initial budget:", init_budget])
        t.add_row(["min budget:", min(self.fetch_log(round_n, "budget"))])
        t.add_row(["number of actions:", action_n])
        t.add_row(["best action:", actions[best_action_idx]])
        t.add_row(["best action mean:", means[best_action_idx]])
        t.add_row(["action means:", means])
        t.add_row(["average reward:", avg_reward])
        t.add_row(["most frequent action:", top_freq_action])
        print(t)
        if plot_regret:
            self.plot_regret(round_n)
        if plot_regret:
            self.plot_budget(round_n)
        if plot_action_rewards:
            self.plot_action_rewards(reward_params)

    def plot_action_rewards(self, reward_params):
        action_n = len(reward_params)
        plot_size = math.ceil(math.sqrt(action_n))
        fig, axes = plt.subplots(plot_size, plot_size, sharex=True,
                                 figsize=(max([6, int(plot_size * 1.5)]), max([6, int(plot_size * 1.5)])))
        for i, params in enumerate(reward_params):
            ax = axes[i // plot_size][i % plot_size]
            ax.set_title(i)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            self.reward_const(*params).plot(ax)
        fig.show()

    def plot_regret(self, round_n):
        plt.plot(self.run_log[round_n]['regret'])
        plt.title("regret over time (best mean - real rewards)")
        plt.show()

    def plot_regret_all_rounds(self):
        flag = None
        for i, rlog in enumerate(self.run_log):
            agent_type = self.fetch_log(i, "agent_type")
            c = 'red' if agent_type == SimpleUCBSurvivalPlayer else "blue"
            lbl = str(agent_type).replace("'>", "").split(".")[-1] if agent_type != flag else None
            plt.plot(rlog['regret'], color=c, label=lbl)
            flag = agent_type
        plt.title("regret over time (best mean - real rewards)")
        plt.legend()
        plt.show()

    def plot_round_actions(self, round_n, show=True):
        c = Counter(self.run_log[round_n]["past_actions"])
        actions = list(c.keys())
        freq = list(c.values())

        fig, ax = plt.subplots(figsize=(max([6, 6 * len(actions) / 25]), max([3, 3 * len(actions) / 30])))

        plt.bar(actions, freq)
        plt.title("Actions played frequency")
        plt.xlabel("Actions")
        plt.ylabel("Frequency")
        ax.set_xticks(actions)
        if show:
            fig.show()

    def plot_budget(self, round_n, show=True):
        fig, ax = plt.subplots(1)
        ax.plot(self.run_log[round_n]['budget'])
        ax.set_title("budget over time (best mean - real rewards)")

        step_type = self.run_log[round_n]["step_type"]
        if len(step_type) > 0:
            safe_sections, unsafe_sections = true_false_sections(self.run_log[round_n]["step_type"])
            [ax.axvspan(xmin=s[0], xmax=s[1], facecolor='green', alpha=0.4) for s in safe_sections]
            [ax.axvspan(xmin=s[0], xmax=s[1], facecolor='red', alpha=0.4) for s in unsafe_sections]
        if show:
            fig.show()

    def fetch_log(self, run, field):
        if field in self.run_log[run].keys():
            return self.run_log[run][field]
        if field in self.run_log[run]["run_params"].keys():
            return self.run_log[run]["run_params"][field]
        if field in self.run_log[run]["agent_specific_log"].keys():
            return self.run_log[run]["agent_specific_log"][field]
        else:
            raise KeyError("log field not found")
