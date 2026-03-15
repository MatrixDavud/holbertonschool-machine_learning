#!/usr/bin/env python3
"""
Train a DQN agent to play Breakout using Gymnasium, Keras, and Keras-RL2
"""
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class GymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to make Gymnasium compatible with Keras-RL2
    (converts (obs, info) to obs and (obs, reward, term, trunc, info)
    to (obs, reward, done, info))
    """
    def reset(self, **kwargs):
        """Reset the environment"""
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """Step the environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


def create_model(window_length, n_actions):
    """
    Creates a CNN model for Atari Breakout
    """
    model = Sequential()
    # Keras-RL expects (window, H, W). We permute to (H, W, window) for Keras
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    return model


if __name__ == '__main__':
    # Setup Environment
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = GymnasiumWrapper(env)

    n_actions = env.action_space.n
    window_length = 4

    model = create_model(window_length, n_actions)

    # Configure Keras-RL2
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory,
                   nb_steps_warmup=50000, target_model_update=10000,
                   policy=policy, train_interval=4, delta_clip=1.)

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Training
    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)

    # Save policy
    dqn.save_weights('policy.h5', overwrite=True)
