import os
import random

import numpy as np

from config import (INPUT_SHAPE)


class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""

    def __init__(self, size=1000000, input_shape=INPUT_SHAPE, history_length=4, use_per=True):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
            use_per: Use PER instead of classic experience replay
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent performed
            frame: A (84, 84) frame of the game in grayscale
            reward: A float determining the reward the agent received for performing an action
            terminal: A bool stating whether the episode terminated
            clip_reward: 是否对奖励裁剪
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        if clip_reward:
            reward = np.sign(reward)

        # Write memory
        self.actions[self.current] = action
        # 省略所有的冒号, 等同于self.frames[self.current], self.frames[self.current, :, :]
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_mini_batch(self, batch_size=32, priority_scale=0.0):
        """Returns a mini batch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a mini batch')

        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = self.priorities[self.history_length:self.count - 1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities
                # based on priority weights
                if self.use_per:
                    index = np.random.choice(np.arange(self.history_length, self.count - 1), p=sample_probabilities)
                else:
                    index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.
                # If either are True, the index is invalid.
                # current是插入的分界线, 如果选到了分界线后面4个,无法构成连续的状态,需要跳过
                if index >= self.current >= index - self.history_length:
                    continue
                # 终止标识也是一样, 无法构成连续的状态,跳过
                if self.terminal_flags[index - self.history_length:index].any():
                    continue

                # 跳出寻找index的过程, 表明选择的index是可以被接受的
                break
            indices.append(index)

        # Retrieve states from memory
        # 取出用于训练的state和对应的下一帧
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx - self.history_length:idx, ...])
            new_states.append(self.frames[idx - self.history_length + 1:idx + 1, ...])

        # (4, 84, 84) -> (batch_size, 4, 84, 84) -> (batch_size, 84, 84, 4)
        # 转成需要的形状
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1 / self.count * 1 / sample_probabilities[[index - self.history_length for index in indices]]
            importance = importance / importance.max()

            return (states, self.actions[indices], self.rewards[indices], new_states,
                    self.terminal_flags[indices]), importance, indices
        else:
            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')
