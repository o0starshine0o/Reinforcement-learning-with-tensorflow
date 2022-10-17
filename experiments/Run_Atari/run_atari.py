import gym

env = gym.make('ALE/Breakout-v5', render_mode='human')
env.action_space.seed(1)
height, width, channels = env.observation_space.shape
action_means = env.get_action_meanings()
observation, info = env.reset(seed=1)
print('info: ', info)
# 打到砖块时得分
score = 0
terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    score += reward
    print('action: ', action_means[action], ', reward:', reward, ', score:', score)
env.close()
exit()
