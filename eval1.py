from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C.load("checkpoint/rl_model_4600000_steps")
model.set_env(env)

obs = env.reset()
for i in range(100000):
    action, _state = model.predict(obs, deterministic=False )
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()