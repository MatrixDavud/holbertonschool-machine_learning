#!/usr/bin/env python3
"""
Display a game played by a trained DQN agent
"""
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


class GymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to make Gymnasium compatible with Keras-RL2
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


if __name__ == '__main__':
    # Setup Environment (Human render mode for display)
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = GymnasiumWrapper(env)

    n_actions = env.action_space.n
    window_length = 4

    # Recreate the architecture
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory,
                   policy=GreedyQPolicy())

    dqn.compile(optimizer='adam')

    # Load the weights
    dqn.load_weights('policy.h5')

    # Play
    dqn.test(env, nb_episodes=5, visualize=True)
