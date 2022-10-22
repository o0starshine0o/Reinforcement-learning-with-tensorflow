import random

import gym
import numpy as np

from utils import process_frame


class GameWrapper:
    """Wrapper for the environment provided by Gym"""

    def __init__(self, env_name, no_op_steps=10, history_length=4):
        self.env = gym.make(env_name, render_mode='rgb_array').unwrapped
        self.no_op_steps = no_op_steps
        self.history_length = history_length

        # (84, 84, 4)
        self.state = None
        self.frame = None
        self.last_lives = 0

    def reset(self, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame, _ = self.env.reset()
        self.last_lives = 0

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(0)

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

    def step(self, action):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal, truncated, info = self.env.step(action)

        # In the commonly ignored 'info' or 'meta' data returned by env.step
        # we can get information such as the number of lives the agent has.

        # We use this here to find out when the agent loses a life, and
        # if so, we set life_lost to True.

        # We use life_lost to force the agent to start the game
        # and not sit around doing nothing.
        # 生命丢失的时候, env会空白几秒, 通过这种方式判断是否生命丢失, 加快学习速度
        if info['lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['lives']

        processed_frame = process_frame(new_frame)
        # 抽取最后n-1帧, 再添加1帧, 保证self.state的shape永远是(84, 84, 4)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        # (84, 84, 1)
        return processed_frame, reward, terminal, life_lost
