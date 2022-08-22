import copy
import os
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import time, sys
sys.path.append('../toolbox/')

from rl_ilc_env import ILCEnv
from ilc_replayer import Replayer
from td3_agent import TD3Agent
from robots_def import abb6640
import matplotlib.pyplot as plt


def get_args(message=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_frequency", type=int, default=50)
    parser.add_argument("--train_episode_start", type=int, default=0)
    parser.add_argument("--max_train_episode", type=int, default=int(1e3))
    parser.add_argument("--curve_feature_dim", type=int, default=32)
    parser.add_argument("--state_robot_dim", type=int, default=4)
    parser.add_argument("--replayer_capacity", type=int, default=int(1e6))
    parser.add_argument("--curve_normalize_dim", type=int, default=50)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--max_action", type=float, default=2.)
    parser.add_argument("--warm_up", type=int, default=10)

    args = parser.parse_args() if message is None else parser.parse_args(message)
    return args


def read_data(curve_idx, data_dir):
    base_path = data_dir + "base/curve_{}.csv".format(curve_idx)
    js_path = data_dir + "js/curve_{}.csv".format(curve_idx)

    base_data = pd.read_csv(base_path, header=None).values
    js_data = pd.read_csv(js_path, header=None).values

    curve = base_data[:, :3]
    curve_normal = base_data[:, 3:]
    curve_js = js_data

    return curve, curve_normal, curve_js


def train(agent: TD3Agent, data_dir, args):
    train_timer = time.time()

    eval_result = {"Max Error": [], "Iteration": [], 'Episode': []}

    forward_data_dir = data_dir + os.sep + 'forward' + os.sep
    reverse_data_dir = data_dir + os.sep + 'reverse' + os.sep

    robot = abb6640(d=50)
    replayer = Replayer(args, capacity=args.replayer_capacity)

    q_values = {'step': [], 'optimal': [], 'extreme': [], 'policy': [], 'best': [], 'x': [], 'y': [], 'z': []}

    eps_start = 0.5
    train_step = 0

    actions_values = np.linspace(-args.max_action, args.max_action, 5)

    for episode in range(args.train_episode_start, args.max_train_episode):
        epsilon = eps_start - (episode/args.max_train_episode) * eps_start

        curve_idx = np.random.randint(100)
        data_dir = forward_data_dir if episode % 2 == 0 else reverse_data_dir
        print(data_dir + "curve{}".format(curve_idx))
        curve, curve_normal, curve_js = read_data(curve_idx, data_dir)

        env = ILCEnv(curve, curve_normal, curve_js, robot, 100)
        state, status, message = env.reset()
        done = False

        opt_action = np.zeros((env.n, env.action_dim))
        opt_action[:, 0] = 1.
        extreme_action = np.ones((env.n, env.action_dim)) * -2.
        extreme_action[:, 0] = 2.

        if status:
            episode_reward = 0
            is_warm_up = (episode - args.train_episode_start) < args.warm_up

            print("[Episode {:>6}] {:>6.2f} {}".format(episode+1, (time.time()-train_timer)/(episode+1), message))

            while not done:
                train_step += 1
                step_timer = time.time()

                curve_error, curve_target, robot_pose, is_start, is_end = state
                state_error = np.array([x.T.flatten() for x in curve_error])
                state_target = np.array([x.T.flatten() for x in curve_target])
                state_curve = np.hstack([state_error, state_target])
                state_robot = np.hstack([np.array(robot_pose), is_start.reshape(-1, 1), is_end.reshape(-1, 1)])

                if is_warm_up:
                    actions = (np.random.rand(env.n, env.action_dim) * 2 - 1) * args.max_action
                    # actions = opt_action if episode % 2 == 0 else extreme_action
                    # actions = opt_action
                # elif np.random.rand() < epsilon:
                #     actions = np.random.rand(env.n, env.action_dim) * 2 - 1
                else:
                    actions, policy_q = agent.select_action(state_curve, state_robot, use_noise=True, return_q=True)
                next_state, reward, done, message, success = env.step(actions)

                episode_reward += reward
                replayer.store(state, actions, reward, next_state, done, success)

                copy_env = copy.deepcopy(env)
                opt_next_state, opt_reward, opt_done, opt_message, opt_success = copy_env.step(opt_action)
                replayer.store(state, opt_action, opt_reward, opt_next_state, opt_done, opt_success)

                # ========== For debugging Q value estimation
                q_optimal = agent.q_value(state_curve, state_robot, opt_action)
                q_extreme = agent.q_value(state_curve, state_robot, extreme_action)
                _, q_policy = agent.select_action(state_curve, state_robot, use_noise=False, return_q=True)

                highest_q_action = np.array([0, 0, 0])
                highest_q_value = None
                for x in range(5):
                    for y in range(5):
                        for z in range(5):
                            sample_action = np.array([actions_values[x], actions_values[y], actions_values[z]]) * np.ones((env.n, env.action_dim))
                            sample_q_value = np.mean(agent.q_value(state_curve, state_robot, sample_action))
                            if highest_q_value is None or sample_q_value > highest_q_value:
                                highest_q_action = np.array([actions_values[x], actions_values[y], actions_values[z]])
                                highest_q_value = sample_q_value

                q_values['optimal'].append(np.mean(q_optimal))
                q_values['extreme'].append(np.mean(q_extreme))
                q_values['policy'].append(np.mean(q_policy))
                q_values['best'].append(highest_q_value)
                q_values['x'].append(highest_q_action[0])
                q_values['y'].append(highest_q_action[1])
                q_values['z'].append(highest_q_action[2])
                q_values['step'].append(train_step)
                # ========== For debugging Q value estimation

                agent.train(replayer, gradient_steps=5)

                state = next_state
                done = np.sum(done) > 0

                print("\r\tStep {:>3}: {}\t[Reward: {:>8.2f} {:>8.2f} {:>8.2f}]\tEps={:.3f}\t({:>6.2f} s)".format(env.itr, message, np.mean(reward), np.max(reward), np.min(reward), epsilon, time.time()-step_timer))
        else:
            print("[Episode {:>6}] Invalid curve. Skipped.".format(episode + 1))

        if episode == 0 or (episode + 1) % args.eval_frequency == 0:
            eval_error, eval_itr = evaluate(agent, 'eval_data/curve1', render=True, render_dir='render/eval')
            eval_result["Max Error"].append(eval_error)
            eval_result["Iteration"].append(eval_itr)
            eval_result["Episode"].append(episode + 1)
            eval_df = pd.DataFrame(eval_result)
            eval_df.to_csv('Eval Result.csv', index=False)

            save_dir = 'model/{}'.format(episode+1)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            agent.save(save_dir)
            replayer.save_to_file('model/replayer_memory')
            print("[Model Saved]")

        if episode % 10 == 0:
            q_df = pd.DataFrame(q_values)
            q_df.to_csv('train_q_values.csv', index=False)


def evaluate(agent, data_dir, render=False, render_dir="", env_mode='robot_studio'):
    data_dir = data_dir + os.sep

    robot = abb6640(d=50)
    exec_error = 0
    num_itr = 0
    num_curve = 0

    all_q_optimal = []
    all_q_extreme = []
    all_q_policy = []
    all_itr = []

    for i in range(10, 11):

        curve, curve_normal, curve_js = read_data(i, data_dir)
        env = ILCEnv(curve, curve_normal, curve_js, robot, 100, mode=env_mode)
        state, status, message = env.reset()
        done = False
        print("[EVAL] Curve {:>5} {}".format(i, message))

        opt_action = np.zeros((env.n, env.action_dim))
        opt_action[:, 0] = 1.
        extreme_action = np.ones((env.n, env.action_dim)) * -2.
        extreme_action[:, 0] = 2.

        if status:
            num_curve += 1

            if render:
                env.render(i, save=True, save_dir=render_dir)

            i_step = 0
            while not done:
                i_step += 1
                curve_error, curve_target, robot_pose, is_start, is_end = state

                state_error = np.array([x.T.flatten() for x in curve_error])
                state_target = np.array([x.T.flatten() for x in curve_target])
                state_curve = np.hstack([state_error, state_target])
                state_robot = np.hstack([np.array(robot_pose), is_start.reshape(-1, 1), is_end.reshape(-1, 1)])

                actions = agent.select_action(state_curve, state_robot, use_noise=False)

                plot_action(actions, i_step)
                # actions[:, 0] = 1
                # actions[:, 1] = 0
                # actions[:, 2] = 0

                q_optimal = agent.q_value(state_curve, state_robot, opt_action)
                q_extreme = agent.q_value(state_curve, state_robot, extreme_action)
                q_policy = agent.q_value(state_curve, state_robot, actions)
                all_q_optimal.append(np.mean(q_optimal))
                all_q_extreme.append(np.mean(q_extreme))
                all_q_policy.append(np.mean(q_policy))
                all_itr.append(i_step)

                next_state, reward, done, message, success = env.step(actions)

                # plot_error_reward(curve_error, next_state, reward, i_step)

                state = next_state
                done = np.sum(done) > 0

                if render:
                    env.render(i, save=True, save_dir=render_dir)

                print("\r\t[EVAL] Step {:>3}: {}".format(env.itr, message))

            exec_error += env.max_exec_error
            num_itr += env.itr
        else:
            time.sleep(2)
            print("[EVAL] Invalid curve. Skipped.")
    exec_error /= num_curve
    num_itr /= num_curve

    return exec_error, num_itr
    # return all_q_optimal, all_q_extreme, all_q_policy, all_itr


def plot_action(actions, i_step=0):
    plt.figure()
    plt.scatter(np.arange(1, actions.shape[0]+1), actions[:, 0], label='axis 1')
    plt.scatter(np.arange(1, actions.shape[0]+1), actions[:, 1], label='axis 2')
    plt.scatter(np.arange(1, actions.shape[0]+1), actions[:, 2], label='axis 3')
    plt.xlabel('Breakpoint Index')
    plt.ylabel('Action Value')
    plt.title("Iteration {}".format(i_step))
    plt.legend()
    plt.ylim(-2.2, 2.2)
    plt.savefig('action/actions_itr{}.png'.format(i_step))
    plt.close()


def plot_error_reward(curve_error, next_state, reward, i_step):
    next_error, _, _, _, _ = next_state
    state_max_error = np.max(np.linalg.norm(np.array(curve_error), axis=-1), axis=-1)
    next_state_max_error = np.max(np.linalg.norm(np.array(next_error), axis=-1), axis=-1)
    error_improve = (next_state_max_error - state_max_error)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    p1 = ax.plot(np.arange(1, error_improve.shape[0] + 1), -error_improve, 'r', label='$\Delta E_{max}$')
    ax.set_xlabel('Breakpoint Index')
    ax.set_ylabel('$\Delta E$')
    ax.set_ylim(-2, 2)
    p2 = ax2.plot(np.arange(1, reward.shape[0] + 1), reward, 'b', label='reward')
    ax2.set_ylabel("Reward")
    ax2.set_ylim(-150, 150)

    lines = p1 + p2
    labels = [p.get_label() for p in lines]
    ax.legend(lines, labels)

    fig.savefig('action/reward_{}.png'.format(i_step))


def evaluate_all(agent, data_dir):
    all_eval_result = {'Max Error': [], 'Iteration': [], 'Episode': []}

    for i in range(100, 1700, 100):
        print('Evaluating Model at Episode {}'.format(i))
        agent.load('model/{}'.format(i))
        eval_error, eval_itr = evaluate(agent, data_dir, render=False, env_mode='robot_studio')
        all_eval_result['Max Error'].append(eval_error)
        all_eval_result['Iteration'].append(eval_itr)
        all_eval_result['Episode'].append(i)
        df = pd.DataFrame(all_eval_result)
        df.to_csv('Eval Result New.csv', index=False)


def main():
    args = get_args()
    data_dir = 'train_data/curve1'
    eval_dir = 'eval_data/curve1'

    agent = TD3Agent(args)

    # agent.load('model/gaussian_200/1500')
    # args.train_episode_start = 1400
    train(agent, data_dir, args)

    # agent.load('model/50')
    # eval_error, eval_itr = evaluate(agent, eval_dir, render=True, render_dir='render/eval', env_mode='robot_studio')

    # q_optimal = []
    # q_extreme = []
    # q_policy = []
    # itr = []
    # model = []
    # all_idx = [1] + [i for i in range(100, 1600, 100)]
    # for model_idx in all_idx:
    #     agent.load("model/gaussian_200/{}".format(model_idx))
    #     model_optimal, model_extreme, model_policy, model_itr = evaluate(agent, eval_dir, render=False, render_dir='render/eval', env_mode='robot_studio')
    #     q_optimal += model_optimal
    #     q_extreme += model_extreme
    #     q_policy += model_policy
    #     itr += model_itr
    #     model += [model_idx] * len(model_itr)
    #
    # df = pd.DataFrame({'optimal': q_optimal, 'extreme': q_extreme, 'policy': q_policy, 'itr': itr, 'episode': model})
    # df.to_csv('q_values.csv', index=False)

    # df = pd.read_csv('q_values.csv')
    # #
    # for i in range(1, 10):
    #     itr_df = df.loc[df['itr'] == i]
    #
    #     steps = itr_df['episode'].values
    #     q_optimal = itr_df['optimal'].values
    #     q_extreme = itr_df['extreme'].values
    #     q_policy = itr_df['policy'].values
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(steps, q_optimal, label='optimal')
    #     ax.plot(steps, q_extreme, label='extreme')
    #     ax.plot(steps, q_policy, label='policy')
    #     ax.legend()
    #     ax.set_ylim(-10000, 100)
    #     ax.set_title('Q values - Iteration {}'.format(i))
    #     ax.set_xlabel('Train Episode')
    #     ax.set_ylabel('Q Value')
    #     fig.savefig('q_value/eval_q_val_itr_{}.png'.format(i), dpi=300)
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(steps, (q_optimal - q_policy), label='$\Delta Q_{optimal}$')
    #     ax.plot(steps, (q_extreme - q_policy), label='$\Delta Q_{extreme}$')
    #     ax.legend()
    #     ax.set_ylim(-20, 5)
    #     ax.set_title('Q value diff - Iteration {}'.format(i))
    #     ax.set_xlabel('Train Episode')
    #     ax.set_ylabel('$\Delta Q$')
    #     fig.savefig('q_value/eval_q_diff_itr_{}.png'.format(i), dpi=300)

    #
    #     mean_curve = itr_df['mean'].values
    #     min_curve = itr_df['min'].values
    #     max_curve = itr_df['max'].values
    #     models = itr_df['model'].values
    #
    #     fig, ax = plt.subplots()
    #
    #     ax.plot(models, mean_curve, label='average')
    #     ax.plot(models, min_curve, label='max')
    #     ax.plot(models, max_curve, label='min')
    #     ax.plot(models, np.zeros(len(models)), label='0')
    #     ax.fill_between(models, min_curve, max_curve, alpha=0.2)
    #     ax.set_xlim(-100, 1700)
    #     ax.set_ylim(-30, 100)
    #     ax.set_xlabel('Train Episode')
    #     ax.set_ylabel('Q value diff')
    #     ax.set_title('Q value difference - Iteration {}'.format(i))
    #     ax.legend()
    #
    #     fig.savefig('q_value/itr_{}.png'.format(i))
    #

if __name__ == '__main__':
    main()
