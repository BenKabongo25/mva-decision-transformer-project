# Deep Learning
# January 2024
#
# Ben Kabongo
# M2 MVA, ENS Paris-Saclay


import numpy as np
from tqdm import tqdm
from utils.replay_buffer import FixedReplayBuffer


def format_data(
        num_buffers, 
        num_steps, 
        game="Breakout", 
        data_dir_prefix="../data/atari/",
        trajectories_per_buffer=10, 
        replay_idx=1,
        update_horizon=1,
        gamma=0.99,
        batch_size=32,
        replay_capacity=100_000
    ):

    states, actions, returns, done_idxs, stepwise_returns = [], [], [0], [], []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0

    with tqdm(total=num_steps) as pbar:

        while len(states) < num_steps:
            buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
            i = transitions_per_buffer[buffer_num]

            replay_buffer = FixedReplayBuffer(
                data_dir=data_dir_prefix + game + f'/{replay_idx}/replay_logs',
                replay_suffix=buffer_num,
                observation_shape=(84, 84),
                stack_size=4,
                update_horizon=update_horizon,
                gamma=gamma,
                observation_dtype=np.uint8,
                batch_size=batch_size,
                replay_capacity=replay_capacity
            )

            if replay_buffer._loaded_buffers:
                done = False
                curr_num_transitions = len(states)
                trajectories_to_load = trajectories_per_buffer

                while not done:
                    sample = replay_buffer.sample_transition_batch(batch_size=1, indices=[i])
                    state, action, reward, _, _, _, terminal, _ = sample
                    state = state.transpose((0, 3, 1, 2))
                    state, action, reward, terminal = state[0], action[0], reward[0], terminal[0]
                    states.append(state)
                    actions.append(action)
                    stepwise_returns.append(reward)
                    pbar.update(1)

                    if terminal:
                        done_idxs.append(len(states))
                        returns.append(0)
                        if trajectories_to_load == 0:
                            done = True
                        else:
                            trajectories_to_load -= 1

                    returns[-1] += reward
                    i += 1
                    if i >= replay_capacity:
                        states = states[:curr_num_transitions]
                        actions = actions[:curr_num_transitions]
                        stepwise_returns = stepwise_returns[:curr_num_transitions]
                        returns[-1] = 0
                        i = transitions_per_buffer[buffer_num]
                        done = True

                num_trajectories += (trajectories_per_buffer - trajectories_to_load)
                transitions_per_buffer[buffer_num] = i

    states = np.array(states)
    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs, dtype=int)

    start_index = 0
    rtgs = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1):
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtgs[j] = sum(rtg_j)
        start_index = i

    start_index = 0
    timesteps = np.zeros(len(actions) + 1, dtype=int)
    for i in done_idxs:
        timesteps[start_index : i + 1] = np.arange(i + 1 - start_index)
        start_index = i + 1

    return states, actions, returns, done_idxs, rtgs, timesteps
