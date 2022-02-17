import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
import time

from collections import namedtuple

from trajectory_utils import Primitive, BreakPoint

NUM_PRIMITIVE = 2
NUM_BP = 20  # -------------------------------------- HERE
NUM_SCALE = 10
NUM_ACTION = NUM_PRIMITIVE * NUM_SCALE
NUM_STATE = NUM_BP * NUM_PRIMITIVE

NUM_EPISODE = 5000  # -------------------------------------- HERE
NUM_CURVES = 1000  # -------------------------------------- HERE

ERROR_THRESHOLD = 1.0
REWARD_FACTOR = 5  # -------------------------------------- HERE
REWARD_STEP = -10
REWARD_FINISH = 2000  # -------------------------------------- HERE
REWARD_DECAY_FACTOR = 0.9

GAMMA = 0.99
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.8
MAX_GREEDY_STEP = 10
Q_SAVE = 50


Curve_Error = namedtuple("Curve_Error", ('primitive', 'max_error'))


def read_data(dir_path):
    all_data = []
    col_names = ['X', 'Y', 'Z', 'direction_x', 'direction_y', 'direction_z']
    count = 0

    for file in os.listdir(dir_path):
        count += 1
        if count > NUM_CURVES:
            break
        file_path = dir_path + os.sep + file
        data = pd.read_csv(file_path, names=col_names)
        curve_x = data['X'].tolist()
        curve_y = data['Y'].tolist()
        curve_z = data['Z'].tolist()
        curve = np.vstack((curve_x, curve_y, curve_z)).T
        all_data.append(curve)

    return all_data


def greedy_data_to_dict(greedy_data: pd.DataFrame):
    ret_dict = dict()
    for index, row in greedy_data.iterrows():
        ret_dict[row['id']] = row['n_primitives']
    return ret_dict


def longest_fitting_primitive(last_bp, curve, p=[]):
    longest_fit = {'moveC': None, 'moveL': None}

    search_left_c = search_left_l = last_bp
    search_right_c = search_right_l = len(curve)
    search_point_c = search_right_c
    search_point_l = search_right_l

    step_count = 0
    while step_count < MAX_GREEDY_STEP:
        step_count += 1

        # moveC
        left_bp = BreakPoint(idx=last_bp)
        right_bp = BreakPoint(idx=search_point_c)
        target_curve = curve[last_bp:int(search_point_c) + 1]
        # print(len(target_curve), last_bp, search_point_c)

        if len(target_curve) >= 2:
            primitive_c = Primitive(start=left_bp, end=right_bp, move_type='C')
            left_bp.right = primitive_c
            right_bp.left = primitive_c
            # print("---", len(target_curve), last_bp, search_point_c, step_count)
            curve_fit, max_error = primitive_c.curve_fit(target_curve, p=p)
            if max_error <= ERROR_THRESHOLD:
                if longest_fit['moveC'] is None or primitive_c > longest_fit['moveC'].curve:
                    longest_fit['moveC'] = Curve_Error(primitive_c, max_error)

                search_left_c = search_point_c
                search_point_c = np.floor(np.mean([search_point_c, search_right_c]))
            else:
                search_right_c = search_point_c
                search_point_c = np.floor(np.mean([search_left_c, search_point_c]))

        # moveL
        left_bp = BreakPoint(idx=last_bp)
        right_bp = BreakPoint(idx=search_point_l)
        target_curve = curve[last_bp:int(search_point_l) + 1]

        primitive_l = Primitive(start=left_bp, end=right_bp, move_type='L')
        left_bp.right = primitive_l
        right_bp.left = primitive_l
        curve_fit, max_error = primitive_l.curve_fit(target_curve)
        if max_error <= ERROR_THRESHOLD:
            if longest_fit['moveL'] is None or primitive_l > longest_fit['moveL'].curve:
                longest_fit['moveL'] = Curve_Error(primitive_l, max_error)

            search_left_l = search_point_l
            search_point_l = np.floor(np.mean([search_point_l, search_right_l]))
        else:
            search_right_l = search_point_l
            search_point_l = np.floor(np.mean([search_left_l, search_point_l]))

    return longest_fit


def feature_to_state(last_bp, curve_c, curve_l):
    if curve_c is None and curve_l is None:
        curve_choice = 0
    elif curve_c is None:
        curve_choice = 0
    elif curve_l is None:
        curve_choice = 1
    else:
        curve_choice = 0 if len(curve_l) > len(curve_c) else 1
    state = curve_choice * NUM_BP + last_bp
    return state


def eps_greedy(q, state, eps_thresh, curve_c=None, curve_l=None):
    sample = random.random()
    if sample > eps_thresh:
        if curve_c is None:
            action = np.argmax(q[state, 0:NUM_SCALE])
        elif curve_l is None:
            action = np.argmax(q[state, NUM_SCALE:NUM_ACTION])
        else:
            action = np.argmax(q[state, :])
    else:
        if curve_c is None:
            action = np.random.randint(0, NUM_SCALE)
        elif curve_l is None:
            action = np.random.randint(NUM_SCALE, NUM_ACTION)
        else:
            action = np.random.randint(0, NUM_ACTION)
    return action


def step(action, longest_fits, current_curve, bp_idx):
    primitive_type = 'moveL' if action < NUM_SCALE else 'moveC'
    action = action - NUM_SCALE if action >= NUM_SCALE else action
    primitive_length = (action + 1) * 0.1
    primitive_end_idx = int(np.ceil(len(longest_fits[primitive_type].curve.curve) * primitive_length))

    # current_last_bp = len(current_curve)
    # new_last_bp = current_last_bp + primitive_end_idx
    # closest_bp = 0
    # for i, bp in enumerate(bp_idx):
    #     if bp - new_last_bp >= 0:
    #         closest_bp = i
    #         break
    # primitive_end_idx = new_last_bp - current_last_bp

    new_curve = longest_fits[primitive_type].curve.curve[0:primitive_end_idx]
    # new_ref_curve = ref_curve[current_last_bp:current_last_bp+len(new_curve)]
    # new_curve_error = np.max(np.linalg.norm(new_ref_curve - new_curve, axis=1))
    # error_diff = new_curve_error - longest_fits[primitive_type].max_error

    last_bp = len(bp_idx)
    current_curve = np.concatenate([current_curve, new_curve], axis=0)
    for i in range(len(bp_idx) - 1):
        if bp_idx[i] <= len(current_curve) <= bp_idx[i + 1]:
            last_bp = i
            break

    return current_curve, last_bp, (primitive_type, new_curve)


def reward_function(i_step, done):
    reward = REWARD_STEP
    if done:
        reward += REWARD_FINISH * (REWARD_DECAY_FACTOR ** i_step)

    return reward


def train_rl(q, curves, greedy_stat=None, n_episode=NUM_EPISODE):

    timer = time.time_ns()
    episode_rewards = []
    episode_steps = []
    greedy_rl_ratios = []
    q_visited = np.zeros_like(q)
    learning_data = {"state": [], "action": [], "reward": [], "next_state": [], "next_action": []}

    for i_episode in range(n_episode):

        epsilon = 0.01 - i_episode * (0.01 - 0.001) / n_episode
        # epsilon = 0.5 - i_episode * (0.5 - 0.001) / n_episode  # fix after 2000 episodes?
        # epsilon = 0.5 * i_episode ** (1/3)
        episode_reward = 0

        lr = LEARNING_RATE if i_episode < LEARNING_RATE_DECAY * n_episode else LEARNING_RATE * 0.25

        random_curve_idx = np.random.randint(0, NUM_CURVES)
        curve = curves[random_curve_idx]
        fit_curve = [curve[0]]
        bp_index = [int(np.ceil(x * len(curve)/NUM_BP)) for x in range(NUM_BP+1)]
        last_bp = 0

        done = False
        i_step = 0

        longest_fit = longest_fitting_primitive(len(fit_curve)-1, curve, p=fit_curve[-1])
        curve_c = None
        curve_l = None
        if longest_fit['moveC'] is not None:
            curve_c = longest_fit['moveC'].curve.curve
            curve_c_error = longest_fit['moveC'].max_error
        if longest_fit['moveL'] is not None:
            curve_l = longest_fit['moveL'].curve.curve
            curve_l_error = longest_fit['moveL'].max_error
        if longest_fit['moveC'] is None and longest_fit['moveL'] is None:
            print("No valid fit")
            done = True

        state = feature_to_state(last_bp, curve_c=curve_c, curve_l=curve_l)
        action = eps_greedy(q, state, epsilon, curve_c=curve_c, curve_l=curve_l)

        while not done:
            i_step += 1

            next_fit_curve, last_bp, _ = step(action, longest_fit, fit_curve, bp_index)
            if last_bp >= len(bp_index):
                done = True
                reward = reward_function(i_step=i_step, done=done)
                q[state, action] += lr * (reward - q[state, action])
                q_visited[state, action] += 1
                fit_curve = next_fit_curve
                episode_reward += reward
                break

            next_longest_fit = longest_fitting_primitive(len(next_fit_curve)-1, curve, p=next_fit_curve[-1])
            curve_c = None
            curve_l = None
            if next_longest_fit['moveC'] is not None:
                curve_c = next_longest_fit['moveC'].curve.curve
                curve_c_error = next_longest_fit['moveC'].max_error
            if next_longest_fit['moveL'] is not None:
                curve_l = next_longest_fit['moveL'].curve.curve
                curve_l_error = next_longest_fit['moveL'].max_error
            if next_longest_fit['moveC'] is None and next_longest_fit['moveL'] is None:
                print("No valid fit")
                done = True

            if len(next_fit_curve) >= len(curve):
                done = True

            next_state = feature_to_state(last_bp, curve_c=curve_c, curve_l=curve_l)
            reward = reward_function(i_step=i_step, done=done)
            next_action = eps_greedy(q, next_state, epsilon, curve_c=curve_c, curve_l=curve_l)
            q[state, action] += lr * (reward + GAMMA * q[next_state, next_action] * (1 - done) - q[state, action])

            learning_data["state"].append(state)
            learning_data["action"].append(action)
            learning_data["reward"].append(reward)
            learning_data["next_state"].append(next_state)
            learning_data["next_action"].append(next_action)

            q_visited[state, action] += 1
            state, action = next_state, next_action

            fit_curve = next_fit_curve
            longest_fit = next_longest_fit
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_steps.append(i_step)
        greedy_rl_ratio = greedy_stat[random_curve_idx] / i_step
        greedy_rl_ratios.append(greedy_rl_ratio)

        print("Episode {} / {} --- {} Steps --- Reward: {:.3f} --- {:.3f}s Curve {} --- {:.3f}"
              .format(i_episode + 1, NUM_EPISODE, i_step, episode_reward,
                      (time.time_ns() - timer) / (10 ** 9 * (i_episode + 1)),
                      random_curve_idx, greedy_rl_ratio))

        if (i_episode+1) % Q_SAVE == 0:
            print("Saving model")
            np.savetxt('model/q_value_{}.txt'.format(i_episode+1), q, delimiter=',')
            np.savetxt('model/q_visited_{}.txt'.format(i_episode + 1), q_visited, delimiter=',')

            fit_curve_eval, primitives_eval, target_curve_eval = evaluate_rl(q, curves[0])
            plt.figure()
            ax3 = plt.axes(projection='3d')
            ax3.plot3D(target_curve_eval[:, 0], target_curve_eval[:, 1], target_curve_eval[:, 2], 'gray')
            ax3.plot3D(fit_curve_eval[:, 0], fit_curve_eval[:, 1], fit_curve_eval[:, 2], 'green')
            plt.savefig('plots/fitted_curve_{}.jpg'.format(i_episode+1))

            plt.figure()
            ax4 = plt.axes(projection='3d')
            for move_type, path in primitives_eval:
                color = 'b' if move_type == 'moveC' else 'r'
                ax4.plot3D(path[:, 0], path[:, 1], path[:, 2], color)
                ax4.plot3D(path[0, 0], path[0, 1], path[0, 2], 'gx')
                ax4.plot3D(path[-1, 0], path[-1, 1], path[-1, 2], 'gx')
            plt.savefig('plots/fitted_primitives_{}.jpg'.format(i_episode+1))

            running_rewards_train = running_rewards(episode_rewards)
            plt.figure()
            ax2 = plt.axes()
            ax2.plot([x for x in range(len(running_rewards_train))], running_rewards_train, 'b')
            plt.savefig('plots/training_curve_{}.jpg'.format(i_episode+1))

            plt.figure()
            ax5 = plt.axes()
            ax5.plot([x for x in range(len(greedy_rl_ratios))], greedy_rl_ratios, 'b')
            plt.savefig('plots/greedy_ratio_{}.jpg'.format(i_episode+1))

            learning_data_df = pd.DataFrame(learning_data)
            learning_data_df.to_csv("learning_data.csv", index=False)

    return q, episode_rewards, greedy_rl_ratios


def evaluate_rl(q, curve):
    episode_reward = 0
    primitives = []

    fit_curve = [curve[0]]
    bp_index = [int(np.ceil(x * len(curve) / NUM_BP)) for x in range(NUM_BP + 1)]
    last_bp = 0

    done = False
    i_step = 0

    longest_fit = longest_fitting_primitive(len(fit_curve) - 1, curve, p=fit_curve[-1])
    curve_c = None
    curve_l = None
    if longest_fit['moveC'] is not None:
        curve_c = longest_fit['moveC'].curve.curve
        curve_c_error = longest_fit['moveC'].max_error
    if longest_fit['moveL'] is not None:
        curve_l = longest_fit['moveL'].curve.curve
        curve_l_error = longest_fit['moveL'].max_error
    if longest_fit['moveC'] is None and longest_fit['moveL'] is None:
        print("No valid fit")
        done = True

    state = feature_to_state(last_bp, curve_c=curve_c, curve_l=curve_l)
    action = eps_greedy(q, state, 0, curve_c=curve_c, curve_l=curve_l)

    while not done:
        i_step += 1

        next_fit_curve, last_bp, primitive_type = step(action, longest_fit, fit_curve, bp_index)
        primitives.append(primitive_type)

        if last_bp >= len(bp_index):
            done = True
            reward = reward_function(i_step=i_step, done=done)
            fit_curve = next_fit_curve
            episode_reward += reward
            break

        next_longest_fit = longest_fitting_primitive(len(next_fit_curve) - 1, curve, p=next_fit_curve[-1])
        curve_c = None
        curve_l = None
        if next_longest_fit['moveC'] is not None:
            curve_c = next_longest_fit['moveC'].curve.curve
            curve_c_error = next_longest_fit['moveC'].max_error
        if next_longest_fit['moveL'] is not None:
            curve_l = next_longest_fit['moveL'].curve.curve
            curve_l_error = next_longest_fit['moveL'].max_error
        if next_longest_fit['moveC'] is None and next_longest_fit['moveL'] is None:
            # print("No valid fit")
            done = True

        if len(next_fit_curve) >= len(curve):
            done = True

        next_state = feature_to_state(last_bp, curve_c=curve_c, curve_l=curve_l)
        reward = reward_function(i_step=i_step, done=done)
        next_action = eps_greedy(q, next_state, 0, curve_c=curve_c, curve_l=curve_l)
        state, action = next_state, next_action

        fit_curve = next_fit_curve
        longest_fit = next_longest_fit
        episode_reward += reward

    return fit_curve, primitives, curve


def running_rewards(rewards, n=20):
    running_reward = []
    for i, r in enumerate(rewards):
        if i < n:
            running_reward.append(np.mean(rewards[0:i+1]))
        else:
            running_reward.append(np.mean(rewards[i-n:i+1]))
    return running_reward


def tabular_rl_fit(curves, greedy_stat, q_file=None):

    if q_file is None:
        q = np.random.random((NUM_STATE, NUM_ACTION))
    else:
        q_df = pd.read_csv(q_file, header=None)
        q = q_df.values
    q, reward_train, greedy_rl_ratio = train_rl(q, curves, greedy_stat=greedy_stat)
    np.savetxt('q_value.txt', q, delimiter=',')

    running_rewards_train = running_rewards(reward_train)
    fig = plt.figure()
    ax2 = plt.axes()
    ax2.plot([x for x in range(len(running_rewards_train))], running_rewards_train, 'b')
    plt.savefig('plots/training_curve.jpg')

    fit_curve, primitives, target_curve = evaluate_rl(q, curves[0])
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot3D(target_curve[:, 0], target_curve[:, 1], target_curve[:, 2], 'gray')
    ax3.plot3D(fit_curve[:, 0], fit_curve[:, 1], fit_curve[:, 2], 'green')
    plt.savefig('plots/fitted_curve.jpg')

    plt.figure()
    ax4 = plt.axes(projection='3d')
    for move_type, path in primitives:
        color = 'b' if move_type == 'moveC' else 'r'
        ax4.plot3D(path[:, 0], path[:, 1], path[:, 2], color)
        ax4.plot3D(path[0, 0], path[0, 1], path[0, 2], 'gx')
        ax4.plot3D(path[-1, 0], path[-1, 1], path[-1, 2], 'gx')
    plt.savefig('plots/fitted_primitives.jpg')

    plt.figure()
    ax5 = plt.axes()
    ax5.plot([x for x in range(len(greedy_rl_ratio))], greedy_rl_ratio, 'b')
    plt.savefig('plots/greedy_ratio.jpg')


def main():
    data = read_data("data/pert")
    greedy_data = pd.read_csv("greedy_data.csv")
    greedy_dict = greedy_data_to_dict(greedy_data)

    tabular_rl_fit(data, greedy_dict, q_file="q_value.txt")


if __name__ == '__main__':
    main()
