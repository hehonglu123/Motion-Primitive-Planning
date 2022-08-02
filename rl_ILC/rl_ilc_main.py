import os
import pandas as pd
import numpy as np
import argparse
import time, sys
sys.path.append('../toolbox/')

from rl_ilc_env import ILCEnv
from ilc_replayer import Replayer
from td3_agent import TD3Agent
from robots_def import abb6640,m710ic


def get_args(message=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--max_train_episode", type=int, default=int(2e3))
    parser.add_argument("--curve_feature_dim", type=int, default=32)
    parser.add_argument("--state_robot_dim", type=int, default=4)
    parser.add_argument("--replayer_capacity", type=int, default=int(1e6))
    parser.add_argument("--curve_normalize_dim", type=int, default=50)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--max_action", type=float, default=1)
    parser.add_argument("--warm_up", type=int, default=5)

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

    eps_start = 0.5

    for episode in range(args.max_train_episode):
        epsilon = eps_start - (episode/args.max_train_episode) * eps_start

        curve_idx = np.random.randint(100)
        # curve_idx = 0
        data_dir = forward_data_dir if episode % 2 == 0 else reverse_data_dir
        print(data_dir + "curve{}".format(curve_idx))
        curve, curve_normal, curve_js = read_data(curve_idx, data_dir)

        env = ILCEnv(curve, curve_normal, curve_js, robot, 100)
        state, status = env.reset()
        done = False

        if status:
            episode_reward = 0
            is_warm_up = episode < args.warm_up

            print("[Episode {:>6}] {:>6.2f}".format(episode+1, (time.time()-train_timer)/(episode+1)))

            while not done:
                step_timer = time.time()

                # env.render(curve_idx, save=True)

                curve_error, curve_target, robot_pose, is_start, is_end = state
                state_error = np.array([x.T.flatten() for x in curve_error])
                state_target = np.array([x.T.flatten() for x in curve_target])
                state_curve = np.hstack([state_error, state_target])
                state_robot = np.hstack([np.array(robot_pose), is_start.reshape(-1, 1), is_end.reshape(-1, 1)])

                if is_warm_up:
                    actions = np.random.rand(env.n, env.action_dim) * 2 - 1
                elif np.random.rand() < epsilon:
                    actions = np.random.rand(env.n, env.action_dim) * 2 - 1
                else:
                    actions = agent.select_action(state_curve, state_robot, use_noise=False)
                next_state, reward, done, message = env.step(actions)

                episode_reward += reward
                replayer.store(state, actions, reward, next_state, done)

                if not is_warm_up:
                    agent.train(replayer)

                state = next_state
                done = np.sum(done) > 0

                print("\r\tStep {:>3}: {}\t[Reward: {:>8.2f}]\tEps={:.3f}\t({:>6.2f} s)".format(env.itr, message, np.mean(reward), epsilon, time.time()-step_timer))
        else:
            print("[Episode {:>6}] Invalid curve. Skipped.".format(episode + 1))

        if (episode + 1) % args.eval_frequency == 0:
            eval_error, eval_itr = evaluate(agent, 'eval_data/curve1', render=True)
            eval_result["Max Error"].append(eval_error)
            eval_result["Iteration"].append(eval_itr)
            eval_result["Episode"].append(episode + 1)
            eval_df = pd.DataFrame(eval_result)
            eval_df.to_csv('Eval Result.csv', index=False)

            save_dir = 'model/{}'.format(episode+1)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            agent.save(save_dir)
            print("[Model Saved]")


def evaluate(agent, data_dir, render=False, render_dir="", env_mode='robot_studio'):
    data_dir = data_dir + os.sep

    robot = abb6640(d=50)

    if env_mode=='roboguide' or env_mode=='fanuc':
        robot = m710ic(d=50)

    exec_error = 0
    num_itr = 0
    num_curve = 0

    for i in range(10,11):

        curve, curve_normal, curve_js = read_data(i, data_dir)
        env = ILCEnv(curve, curve_normal, curve_js, robot, 100, mode=env_mode)
        state, status = env.reset()
        done = False
        print("[EVAL] Curve {:>5}".format(i))

        if render:
            env.render(i, save=True, save_dir=render_dir)

        if status:
            num_curve += 1

            while not done:
                curve_error, curve_target, robot_pose, is_start, is_end = state
                state_error = np.array([x.T.flatten() for x in curve_error])
                state_target = np.array([x.T.flatten() for x in curve_target])
                state_curve = np.hstack([state_error, state_target])
                state_robot = np.hstack([np.array(robot_pose), is_start.reshape(-1, 1), is_end.reshape(-1, 1)])

                actions = agent.select_action(state_curve, state_robot, use_noise=False)
                next_state, reward, done, message = env.step(actions)

                state = next_state
                done = np.sum(done) > 0

                if render:
                    env.render(i, save=True, save_dir=render_dir)

                print("\r\t[EVAL] Step {:>3}: {}".format(env.itr, message))

            exec_error += env.max_exec_error
            num_itr += env.itr
        else:
            print("[EVAL] Invalid curve. Skipped.")
    exec_error /= num_curve
    num_itr /= num_curve

    return exec_error, num_itr


def main():
    args = get_args()
    # data_dir = 'train_data/curve1'
    # eval_dir = 'eval_data/curve1'

    eval_dir = 'eval_data/curve2_fanuc'

    agent = TD3Agent(args)
    agent.load('model/1600')
    # train(agent, data_dir, args)
    # eval_error, eval_itr = evaluate(agent, eval_dir, render=True, render_dir='render/curve1', env_mode='abb')
    eval_error, eval_itr = evaluate(agent, eval_dir, render=True, render_dir='render/curve2_fanuc', env_mode='roboguide')


if __name__ == '__main__':
    main()




