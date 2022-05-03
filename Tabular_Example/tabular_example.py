# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:43:11 2022

@author: almsk
"""

import numpy as np


def get_action(state, Q, epsilon=0.7):

    if np.random.rand() > epsilon:
        if Q[state, 1] > Q[state, 0]:
            return 1

        elif Q[state, 1] < Q[state, 0]:
            return 0

        else:
            return np.random.choice([0, 1])
    else:
        return np.random.choice([0, 1])


def take_action(state, action):

    if state == 0:
        if action == 0:
            return 1, -4
        return 5, 1

    elif state == 1:
        if action == 0:
            return 4, 3
        return 2, 5

    elif state == 2:
        if action == 0:
            return 1, 5
        return 3, 4

    elif state == 3:
        if action == 0:
            return 2, 4
        return 4, 3

    elif state == 4:
        if action == 0:
            return 3, 3
        return 1, 3

    elif state == 5:
        if action == 0:
            return 6, -3
        return 8, -1

    elif state == 6:
        if action == 0:
            return 7, -2
        return 5, -3

    elif state == 7:
        if action == 0:
            return 8, -1
        return 6, -2

    elif state == 8:
        if action == 0:
            return 5, -1
        return 7, -1

    return 0, 0


def update_Q(state, action, Q, reward):
    Q[state, action] = Q[state, action] + 0.7*(reward - Q[state, action])
    return Q


def update_Q_SARSA(state, action, Q, reward, new_state, epsilon, end):
    new_action = get_action(new_state, Q, epsilon)

    if end:
        Q[state, action] = Q[state, action] + 0.7*(reward - Q[state, action])
    else:
        Q[state, action] = Q[state, action] + 0.7*(reward + 0.7*Q[new_state, new_action] - Q[state, action])

    return Q


def update_Q_Q(state, action, Q, reward, new_state, end):
    if end:
        Q[state, action] = Q[state, action] + 0.7*(reward - Q[state, action])
    else:
        Q[state, action] = Q[state, action] + 0.7*(reward + 0.7*np.max(Q[new_state, :]) - Q[state, action])

    return Q


if __name__ == "__main__":

    N_episodes = 100000

    N_steps = 10

    # State space
    SP = np.arange(9)

    # Action space
    AP = [0, 1]  # 0 = left, 1 = right

    # Q-table
    Q = np.zeros([len(SP), len(AP)])

    for e in range(N_episodes):
        # Start state
        state_old = 0
        epsilon = 1 - e/N_episodes
        for n in range(N_steps):

            if n == N_steps - 1:
                end = True
            else:
                end = False

            # Find best action according to policy
            action = get_action(state_old, Q, epsilon)

            # Take the action in created env.
            state_new, reward = take_action(state_old, action)

            # Update the Q-table
            # Q = update_Q(state_old, action, Q, reward)
            Q = update_Q_SARSA(state_old, action, Q, reward, state_new, epsilon, end)
            # Q = update_Q_Q(state_old, action, Q, reward, state_new, end)

            # Update the state
            state_old = state_new
