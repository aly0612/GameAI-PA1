"""
@Author: 	Austin Ly
@Date:		2024-02-02
@Description: CS 4900 GameAI PA1, Running RL using stable-baselines3 and WandB on Connect4 Game.
train.py is a script to train a PPO model using the Connect4Env environment. It uses stable-baselines3 and WandB for logging.
Connect4Env comes from https://www.askpython.com/python/examples/connect-four-game
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import wandb

# Assuming connect4_env.py contains your custom Connect4Env environment
from connect4_env import Connect4Env

# Initialize wandb
wandb.init(project='PA1', entity='al044516', config={
    "architecture": "CNN",
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gae_lambda": 0.95,
})

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, log_interval=10):
        super(WandbCallback, self).__init__(verbose)
        self.log_interval = log_interval  # Interval for logging
        self.total_episodes = 0  # Total episodes encountered
        self.total_wins = 0  # Total wins encountered

    def _on_step(self):
        for info in self.locals["infos"]:
            episode_info = info.get("episode")
            if episode_info:
                self.total_episodes += 1
                if episode_info["r"] == 1:  # Assuming a reward of 1 indicates a win
                    self.total_wins += 1

                # Calculate cumulative win rate
                cumulative_win_rate = self.total_wins / self.total_episodes if self.total_episodes > 0 else 0

                # Log cumulative win rate at specified intervals
                if self.total_episodes % self.log_interval == 0:
                    wandb.log({"cumulative_win_rate": cumulative_win_rate}, step=self.total_episodes)
        return True


def make_env():
    def _init():
        env = Connect4Env()
        env = Monitor(env)
        return env
    return _init

vec_env = DummyVecEnv([make_env()]) #had to do this, not sure why

#Using PPO algorithm
model = PPO("MlpPolicy", vec_env, learning_rate=wandb.config.learning_rate, verbose=1,
            n_steps=wandb.config.n_steps, batch_size=wandb.config.batch_size,
            n_epochs=wandb.config.n_epochs, gae_lambda=wandb.config.gae_lambda)

model.learn(total_timesteps=50000, callback=WandbCallback())

model.save("connect4_model")

vec_env.close()

wandb.finish()
