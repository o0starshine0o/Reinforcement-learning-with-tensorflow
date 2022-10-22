import json
import os

import numpy as np
import tensorflow as tf

from config import (BATCH_SIZE, DISCOUNT_FACTOR, INPUT_SHAPE,
                    PRIORITY_SCALE)


class Agent(object):
    """Implements a standard Double-DQN agent"""

    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape=INPUT_SHAPE,
                 batch_size=32,
                 history_length=4,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000,
                 use_per=True):
        """
        Arguments:
            dqn: A DQN (returned by the DQN function) to predict moves
            target_dqn: A DQN (returned by the DQN function) to predict target-q values.
                This can be initialized in the same way as the dqn argument
            replay_buffer: A ReplayBuffer object for holding all previous experiences
            n_actions: Number of possible actions for the given environment
            input_shape: Tuple/list describing the shape of the pre-processed environment
            batch_size: Number of samples to draw from the replay memory every updating session
            history_length: Number of historical frames available to the agent
            eps_initial: Initial epsilon value. e-greedy, 探索概率
            eps_final: The "half-way" epsilon value.  The epsilon value decreases more slowly after this
            eps_final_frame: The final epsilon value
            eps_evaluation: The epsilon value used during evaluation
            eps_annealing_frames: Number of frames during which epsilon will be annealed(退火)
                to eps_final, then eps_final_frame
            replay_buffer_start_size: Size of replay buffer before beginning to learn
                (after this many frames, epsilon is decreased more slowly)
            max_frames: Number of total frames the agent will be trained for
            use_per: Use PER instead of classic experience replay
        """

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope * self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (
                self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            frame_number: Global frame number (used for epsilon)
            evaluation: True if the model is evaluating,
                False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        else:
            limit = self.replay_buffer_start_size + self.eps_annealing_frames
            if frame_number < limit:
                return self.slope * frame_number + self.intercept
            else:
                return self.slope_2 * frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for, (84, 84, 4) , (宽, 高, 连续n帧)
            evaluation: True if the model is evaluating,
                False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """
        # Calculate epsilon based on the frame number
        epsilon = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < epsilon:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        # 需要把state转换成DQN能够接受的shape
        state = state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length))
        # DQN返回的是(1, 4)形式,第二维数组代表了每个动作的执行概率
        # 因为只需要预测一次,这里使用call方法比predict方法快10倍
        action_probabilities = self.DQN(state).numpy()[0]
        # 返回概率最大的那个动作索引
        return action_probabilities.argmax()

    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):
        """
        Get the output of a hidden layer inside the model.  This will be/is used for visualizing model
        Arguments:
            state: The input to the model to get outputs for hidden layers from
            layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
            stack_state: Stack `state` four times so the model can take input on a single (84, 84, 1) frame
        Returns:
            Outputs to the hidden layers specified, in the order they were specified.
        """
        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.history_length, axis=2)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))

    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)

    def learn(self, frame_number, batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR, priority_scale=PRIORITY_SCALE):
        """Sample a batch and use it to improve the DQN
        Arguments:
            frame_number: Global frame number (used for calculating importances)
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            priority_scale: How much to weight priorities when sampling the replay buffer.
                0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """

        if self.use_per:
            (states, actions, rewards, new_states,
             terminal_flags), importance, indices = self.replay_buffer.get_mini_batch(batch_size=self.batch_size,
                                                                                      priority_scale=priority_scale)
            importance = importance ** (1 - self.calc_epsilon(frame_number))
        else:
            # states: (32, 84, 84, 4), actions: 32, rewards: 32, new_states: (32, 84, 84, 4), terminal_flags: 32
            states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_mini_batch(
                batch_size=self.batch_size, priority_scale=priority_scale)

        # Main DQN estimates(估计) best action in new states
        # new_state(32, 84, 84, 4)放到dqn中进行预测, 得到32个动作的价值(32, 4),
        # 再选取每个state对应action的最大价值的动作的**索引**, shape(32)
        # ** Double DQN最大的不同:用DQN网络来选取new_state下最优action的索引(用当前的Q网络来选择动作) **
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        # (32, 4)
        future_q_vals = self.target_dqn.predict(new_states)
        # DQN网络选取的动作,到target网络中的动作的**价值**, shape(32)
        # ** Double DQN最大的不同:用target网络根据DQN网络选择的action,计算动作价值 **
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        # Double-DQN算法:
        # 1, 从dqn网络中选取action(argmax)
        # 2, 把action放入target网络中,计算Q值, shape: (32)
        target_q = rewards + (gamma * double_q * (1 - terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            # states: (32, 84, 84, 4), q_values: (32, 4)
            # 这里是调用call方法, 结果约等同于predict(), 返回是tensor, predict()返回类型是ndarray
            # 这是DQN的预测值
            q_values = self.DQN(states)

            # 因为当时选择动作时, 是有随机选择的情况在里面的, 所以这里不能简单的用argmax来设置Q
            # using tf.one_hot causes strange errors
            # actions: (32), one_hot_actions: (32, 4)
            # 只有对应的action才为1, 其余位置都为0
            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)
            # multiply: (32, 4), 只保留选取动作的action概率, action的概率皆为0
            # Q:(32), 只保留选取动作的action概率
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            # shape: (32)
            # 公式中应该是target_q-Q, 后面要用到平方或者绝对值, 所以这里没啥区别
            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        # 计算梯度
        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        # 反向传播, 更新dqn的参数
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current},
                                **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
            load_replay_buffer: 是否同时也加载经验回放
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta
