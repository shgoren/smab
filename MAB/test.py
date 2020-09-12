from MAB.Game import Game
from MAB.Players import RandomSurvivalPlayer, SimpleUCBSurvivalPlayer, MVsurvivalPlayer, BufferedUCBPlayer
from MAB.Simulation import *
from MAB.Rewards import GaussianReward, MixOfBetas

from matplotlib import pyplot as plt
import numpy as np

sim = Simulation(Game, GaussianReward, BufferedUCBPlayer)
sim.simulate(1,
             game_params={"actions": list(range(10)),
                          "reward_type": GaussianReward,
                          "reward_params": [(-0.1, 0.25) for _ in range(9)] + [(0.1, 0.25)]},
             player_params={"budget": 7})
sim.simulate_comparison([0], 9)
sim.summary()
sim.describe_round(0, plot_action_rewards=False, plot_budget=True, plot_regret=True)
sim.plot_round_actions(0)
sim.set_player(SimpleUCBSurvivalPlayer)
sim.simulate_comparison([0], 10)
sim.summary()
sim.plot_regret_all_rounds()
