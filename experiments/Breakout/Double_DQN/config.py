# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'ALE/Breakout-v5'

# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_FROM = None
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard/'

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
# Since Breakout is a simple game, I wouldn't recommend using it here.
USE_PER = False

# How much the replay buffer should sample based on priorities.
# 0 = complete random samples, 1 = completely aligned with priorities
PRIORITY_SCALE = 0.7
# Any positive reward is +1, and negative reward is -1, 0 is unchanged
CLIP_REWARD = True

# Total number of frames to train for, 总计用于训练的帧数, 平均2000帧用时1分钟, 60000 / 2000  = 30 minute
TOTAL_FRAMES = 30000000
# Randomly perform this number of actions before every evaluation to give it an element of randomness 环境重置时随机运行几帧
MAX_NOOP_STEPS = 20

# 模型架构
INPUT_SHAPE = (84, 84)  # Size of the preprocessed input frame. Anything below ~80 won't work.
BATCH_SIZE = 32  # Number of samples the agent learns from at once

# 经验回放
MIN_REPLAY_BUFFER_SIZE = 50000  # The minimum size the replay buffer must be before we start to update the agent
MAX_REPLAY_BUFFER_SIZE = 1000000  # The maximum size of the replay buffer

# 训练
# Maximum length of an episode (in frames). 10800 frames / 60 fps = 180 second = 3 minute
MAX_TRAIN_EPISODE_FRAMES = 18000
# Number of frames between evaluations, 每次评估之间,训练的帧数
# 前期没什么经验随便选, 每盘就200帧左右, 10盘就2000帧的样子
TRAIN_FRAMES_BETWEEN_EVALUATION = 100000

# 学习
LEARNING_RATE = 0.00001
DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
UPDATE_DQN_FRAMES = 4  # Number of actions between gradient descent steps
UPDATE_TARGET_FRAMES = 4  # Number of actions chosen between updating the target network

# 评估
EVAL_FRAMES = 10000  # Number of frames to evaluate for, 每次评估需要产生的帧数
