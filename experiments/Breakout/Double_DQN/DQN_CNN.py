import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers.initializers_v2 import VarianceScaling
from tensorflow.python.keras.layers import Add, Conv2D, Dense, Flatten, Input, Lambda, Subtract
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from agent import Agent
from config import *
from game_wrapper import GameWrapper
from replay_buffer import ReplayBuffer


def init_env():
    game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
    action_space = game_wrapper.env.action_space
    action_means = game_wrapper.env.get_action_meanings()
    print(action_space.n, ' actions: ', action_means)
    return game_wrapper, action_space.n


def build_q_network(action_count: int, learning_rate=LEARNING_RATE, input_shape=INPUT_SHAPE, history_length=4):
    """Builds a dueling DQN as a Keras model
    Arguments:
        action_count: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    # (None, 84, 84, 4)
    model_input = Input((input_shape[0], input_shape[1], history_length))
    # normalize by 255
    # (None, 84, 84, 4)
    x = Lambda(lambda layer: layer / 255)(model_input)

    # (None, 12, 12, 32), 12 = (84 + 2 * 0 - 12) / 6 + 1
    x = Conv2D(32, 8, 4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    # (None, 5, 5, 64), 5 = (12 + 2 * 0 - 4) / 2 + 1
    x = Conv2D(64, 4, 2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    # (None, 5, 5, 64), 5 = (12 + 2 * 0 - 4) / 2 + 1
    x = Conv2D(64, 3, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    # (None, 1, 1, 128), 1 = (5 + 2 * 0 - 5) / 1 + 1
    x = Conv2D(1024, 7, 1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    # Split into value(None, 1, 1, 64) and advantage(None, 1, 1, 64) streams
    # custom splitting layer
    # 这是duelling-DQN的架构
    value_stream, advantage_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)

    # (None, 64)
    value_stream = Flatten()(value_stream)
    # (None, 1)
    value = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(value_stream)

    # (None, 128)
    advantage_stream = Flatten()(advantage_stream)
    # (None, n_actions=4)
    advantage = Dense(action_count, kernel_initializer=VarianceScaling(scale=2.))(advantage_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    # reduce_mean(adv):(None, 1)
    # Subtract()([adv, reduce_mean(adv)]): (None, 4)
    # (None, 4)
    # duelling-DQN的Q值计算: a_value = value + (advantage - mean(advantage))
    q_values = Add()([value, Subtract()([advantage, reduce_mean(advantage)])])

    # Build model
    model = Model(model_input, q_values)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model


@tf.function
def get_model(dqn: Model):
    _input = np.full(shape=(1, 84, 84, 4), fill_value=2, dtype='float32')
    return dqn(_input)


def init_agent(action_count: int):
    # Build main and target networks
    main_dqn = build_q_network(action_count, LEARNING_RATE)
    _input = get_model(main_dqn)
    target_dqn = build_q_network(action_count)

    # replay buffer
    replay_buffer = ReplayBuffer(size=MAX_REPLAY_BUFFER_SIZE, use_per=USE_PER)

    # agent
    return Agent(main_dqn, target_dqn, replay_buffer, action_count, batch_size=BATCH_SIZE, use_per=USE_PER)


def init_meta(load_from=LOAD_FROM, is_load_replay_buffer=LOAD_REPLAY_BUFFER):
    # 初始化元数据, 断点学习
    if load_from is None:
        return 0, [], []
    else:
        print('Loading from: ', load_from)
        meta = agent.load(load_from, is_load_replay_buffer)

        # Apply information loaded from meta
        return meta['frame_number'], meta['rewards'], meta['loss_list']


def action_step(frame_number: int, is_train=True, life_lost=False):
    loss = None

    # Get action
    # Breakout requires a "fire" action (action #1) to start the game each time a life is lost.
    # Otherwise, the agent would sit around doing nothing.
    # 训练的时候让模型自己去学会这个,评估的时候人工干预
    action = ACTION_FIRE if not is_train and life_lost else agent.get_action(frame_number, game.state, not is_train)

    # Take step
    processed_frame, reward, terminal, life_lost = game.step(action)

    # Add experience to replay memory
    if is_train:
        # 只提取最新的一帧(84, 84)
        agent.add_experience(action, processed_frame[:, :, 0], reward, life_lost)

    # Update agent
    # 每操作4步,DQN执行一次梯度下降
    if is_train and frame_number % UPDATE_DQN_FRAMES == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
        loss, error = agent.learn(frame_number)

    # Update target network
    # 把DQN的所有参数赋给target
    if is_train and frame_number % UPDATE_TARGET_FRAMES == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
        agent.update_target_network()

    return reward, terminal, life_lost, loss


def train(frame_number: int, rewards: list, loss_list: list):
    """Evaluation every `FRAMES_BETWEEN_EVAL` frames

    Arguments:
        frame_number: Number of current number
        rewards: 收益列表
        loss_list: 损失列表
    """
    epoch_frame = 0
    start_time = time.time()
    while epoch_frame < TRAIN_FRAMES_BETWEEN_EVALUATION:
        if len(rewards) % 10 == 0:
            start_time = time.time()
        game.reset()
        reward_sum = 0
        for _ in range(MAX_TRAIN_EPISODE_FRAMES):
            reward, terminal, life_lost, loss = action_step(frame_number)
            frame_number += 1
            epoch_frame += 1
            reward_sum += reward
            if loss:
                loss_list.append(loss)

            # Break the loop when the game is over
            if terminal:
                break

        rewards.append(reward_sum)

        # Output the progress every 10 games
        if len(rewards) % 10 == 0:
            # Write to TensorBoard
            if WRITE_TENSORBOARD:
                tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                writer.flush()

            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                  f'Game number: {str(len(rewards)).zfill(6)}  '
                  f'Frame number: {str(frame_number).zfill(8)}  '
                  f'Average reward: {np.mean(rewards[-10:]):0.1f}  '
                  f'Time taken: {(time.time() - start_time):0.1f}s')

    return frame_number


def evaluation(frame_number: int):
    """Evaluation every `FRAMES_BETWEEN_EVAL` frames

    Arguments:
        frame_number: Number of current number
    """
    terminal = True
    life_lost = False
    eval_rewards = []
    episode_reward_sum = 0
    start_time = time.time()

    for _ in range(EVAL_FRAMES):
        if terminal:
            game.reset(evaluation=True)
            episode_reward_sum = 0

        reward, terminal, life_lost, loss = action_step(frame_number, is_train=False, life_lost=life_lost)

        episode_reward_sum += reward

        # On game-over
        if terminal:
            eval_rewards.append(episode_reward_sum)

    # 得分为每场的均值, 如果只有一场(可能还没结束), 就是用本场的得分
    evaluation_score = np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum

    # Print score and write to tensorboard
    print('Evaluation score:', evaluation_score, ' Time taken: ', time.time() - start_time)
    if WRITE_TENSORBOARD:
        tf.summary.scalar('Evaluation score', evaluation_score, frame_number)
        writer.flush()


def get_save_path():
    try:
        return input('Would you like to save the trained model? If so, type in a save path, '
                     'otherwise, interrupt with ctrl+c. ')
    except KeyboardInterrupt:
        print('\nExiting...')


def save_model(frame_number: int, save_path=SAVE_PATH, support_keyboard_input=False, **kwargs):
    if save_path is None and support_keyboard_input:
        save_path = get_save_path()

    if save_path is None:
        return

    path = f'{save_path}/save-{str(frame_number).zfill(8)}'
    kwargs['frame_number'] = frame_number
    agent.save(path, **kwargs)


def run():
    # 绘制计算图
    is_graph = False

    # meta data
    frame_number, rewards, loss_list = init_meta()

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                # Training
                frame_number = train(frame_number, rewards, loss_list)
                # Evaluation, 经验池都没有填到位的话,没必要评价
                if frame_number > MIN_REPLAY_BUFFER_SIZE:
                    evaluation(frame_number)
                # 展示计算图
                if not is_graph:
                    is_graph = True
                    print('draw graph done')
                    tf.summary.trace_export('train_func', step=0)
                # Save model
                if len(rewards) > 300:
                    save_model(frame_number, rewards=rewards, loss_list=loss_list)
            # 最后了还是需要保存下模型的
            save_model(frame_number, rewards=rewards, loss_list=loss_list)

    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        save_model(frame_number, support_keyboard_input=True, rewards=rewards, loss_list=loss_list)


if __name__ == "__main__":
    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
    tf.summary.trace_on()

    # 游戏环境
    game, n_actions = init_env()

    # agent
    agent = init_agent(n_actions)

    run()
