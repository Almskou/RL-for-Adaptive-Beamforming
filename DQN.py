# -*- coding: utf-8 -*-
"""
Taken from https://medium.com/@aniket.tcdav/deep-q-learning-with-tensorflow-2-686b700c868b
and modified by:
@Nicolai
@Victor
"""


# %% ---------- Imports ----------
import gym  # Environemnt is taken from this package
import time
import itertools
import numpy as np
import tensorflow as tf
from statistics import mean
from collections import namedtuple


from classes_DQN import Model, ReplayMemory, EpsilonGreedyStrategy, DQN_Agent

# Initialize tensorboard object
name = f'DQN_logs_{time.time()}'
summary_writer = tf.summary.create_file_writer(logdir=f'logs/{name}/')


# %% ---------- Functions ----------
def copy_weights(Copy_from, Copy_to):
    """
    Function to copy weights of a model to other
    """
    variables2 = Copy_from.trainable_variables
    variables1 = Copy_to.trainable_variables
    for v1, v2 in zip(variables1, variables2):
        v1.assign(v2.numpy())


# %% ---------- Main ----------
if __name__ == "__main__":

    # Initialize the parameters
    batch_size = 64
    gamma = 0.99
    eps_start = 1
    eps_end = 0.000
    eps_decay = 0.001
    target_update = 25
    memory_size = 100000
    lr = 0.01
    epochs = 1000
    hidden_units = [200, 200]

    # Initialize the environment
    env = gym.make('CartPole-v0')

    """
    Notice that we are not using any function to make the states discrete here as DQN
    can handle discrete state spaces.
    """

    # Initialize Class variables
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = DQN_Agent(strategy, env.action_space.n)
    memory = ReplayMemory(memory_size)

    # Experience tuple variable to store the experience in a defined format
    Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones'])

    # Initialize the policy and target network
    policy_net = Model(len(env.observation_space.sample()), hidden_units, env.action_space.n)
    target_net = Model(len(env.observation_space.sample()), hidden_units, env.action_space.n)

    # Copy weights of policy network to target network
    copy_weights(policy_net, target_net)

    optimizer = tf.optimizers.Adam(lr)

    total_rewards = np.empty(epochs)

    for epoch in range(epochs):
        state = env.reset()
        ep_rewards = 0
        losses = []

        for timestep in itertools.count():
            # Take action and observe next_stae, reward and done signal
            action, rate, flag = agent.select_action(state, policy_net)
            next_state, reward, done, _ = env.step(action)
            ep_rewards += reward

            # Store the experience in Replay memory
            memory.push(Experience(state, action, next_state, reward, done))
            state = next_state

            if memory.can_provide_sample(batch_size):
                # Sample a random batch of experience from memory
                experiences = memory.sample(batch_size)
                batch = Experience(*zip(*experiences))

                # batch is a list of tuples, converting to numpy array here
                states = np.asarray(batch[0])
                actions = np.asarray(batch[1])
                rewards = np.asarray(batch[3])
                next_states = np.asarray(batch[2])
                dones = np.asarray(batch[4])

                # Calculate TD-target
                q_s_a_prime = np.max(target_net(np.atleast_2d(next_states).astype('float32')), axis=1)
                q_s_a_target = np.where(dones, rewards, rewards+gamma*q_s_a_prime)
                q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype='float32')

                # Calculate Loss function and gradient values for gradient descent
                with tf.GradientTape() as tape:
                    q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(states).astype('float32'))
                                               * tf.one_hot(actions, env.action_space.n), axis=1)
                    loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

                # Update the policy network weights using ADAM
                variables = policy_net.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))

                losses.append(loss.numpy())

            else:
                losses.append(0)

            # If it is time to update target network
            if timestep % target_update == 0:
                copy_weights(policy_net, target_net)

            if done:
                break

        total_rewards[epoch] = ep_rewards
        avg_rewards = total_rewards[max(0, epoch - 100):(epoch + 1)].mean()  # Running average reward of 100 iterations

        # Good old book-keeping
        with summary_writer.as_default():
            tf.summary.scalar('Episode_reward', total_rewards[epoch], step=epoch)
            tf.summary.scalar('Running_avg_reward', avg_rewards, step=epoch)
            tf.summary.scalar('Losses', mean(losses), step=epoch)

        if epoch % 1 == 0:
            print(f"Episode:{epoch} Episode_Reward:{total_rewards[epoch]} Avg_Reward:{avg_rewards: 0.1f} \
                  Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")

    env.close()
