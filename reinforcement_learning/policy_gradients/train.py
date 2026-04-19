#!/usr/bin/env python3
"""Training module for policy gradient reinforcement learning."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implement a full training using Monte-Carlo policy gradient (REINFORCE).

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True, render the environment every 1000 episodes

    Returns:
        scores: list of scores (sum of rewards) for each episode
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_gradients = []
        episode_rewards = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if show_result and (episode % 1000 == 0):
                env.render()

            action, gradient = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_gradients.append(gradient)
            episode_rewards.append(reward)

            state = next_state

        score = sum(episode_rewards)
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

        for t in range(len(episode_rewards)):
            G = sum(gamma ** (k - t) * episode_rewards[k]
                    for k in range(t, len(episode_rewards)))
            weight += alpha * G * episode_gradients[t]

    return scores
