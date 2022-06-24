import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

#train and evaluate
env = make_atari_env('ALE/Breakout-v5', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
env.render()

log_path = os.path.join('Training', 'Logs', 'Breakout')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=1000000)

model.save(os.path.join('Training', 'saved_models', 'Breakout'))


#load and evaluate
env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
model = A2C.load(os.path.join('Training', 'saved_models', 'Breakout'), env)
result = evaluate_policy(model, env, n_eval_episodes=50, render=True)
print(result)