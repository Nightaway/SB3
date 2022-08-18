from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

# Install sb3
# pip install stable-baselines3

checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./checkpoint/')
event_callback = EveryNTimesteps(n_steps=10000, callback=checkpoint_on_event)

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env('PongNoFrameskip-v4', n_envs=16, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# MlpPolicy
model = A2C('CnnPolicy', env=env, verbose=1)
model.learn(total_timesteps=100000000, callback=event_callback)
