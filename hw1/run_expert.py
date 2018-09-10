#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import helpers

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--output_dir', type=str, default='expert_data')
    parser.add_argument('--save_output', action='store_true')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = helpers.load_policy('experts/'+args.envname+'.pkl')
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()
        data = helpers.run_policy(args, policy_fn)

    print('mean return', np.mean(data['returns']))
    print('std of return', np.std(data['returns']))

    if args.save_output:
        with open(os.path.join(args.output_dir, args.envname + '.pkl'), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
