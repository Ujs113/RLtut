import os
import gym
from stable_baselines3 import PPO, ppo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

log_path = os.path.join('Training', 'Logs')
save_path = os.path.join('Training', 'saved_models')

env = gym.make('CartPole-v0')

def test(env):
    episodes = 5
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
        print("Episode:{}, Score:{}".format(episode, score))
    env.close()

#TODO: Try changing this to **kwargs
def train(env, save=False, path=save_path):
    DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=20000)
    if(save):
        model.save(os.path.join(save_path, path))
    return model

def load(env, path):
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    # model.load(os.path.join(save_path, path), env=env) # Not working for some reason
    model.set_parameters(os.path.join(save_path, path))
    return model

# model = train(env, True, 'PPO_cartpole')
model = load(env, 'PPO_cartpole')
# result = evaluate_policy(model, env, n_eval_episodes=10, render=True)
# print(result)
test(env)