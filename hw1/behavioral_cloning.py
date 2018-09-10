import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tf_util
import pickle
import math
import helpers

def build_model(args, indim, outdim):
    model = keras.Sequential()
    #Input layer
    model.add(keras.layers.Dense(units=args.num_neurons, activation=args.activation, input_shape=indim))
    #Middle layers
    for l in range(args.num_layers - 2):
        model.add(keras.layers.Dense(units=args.num_neurons, activation=args.activation))
    #Output layer
    model.add(keras.layers.Dense(units=outdim[0]))
    #Compile
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

def split_validation(data, validation_prcnt=.75):
    train_size = math.ceil(len(data) * .75)
    train = np.squeeze(data[:train_size])
    val = np.squeeze(data[train_size:])
    return train, val

def concat_dicts(d1, d2, axis=0):
    for k in d1.keys():
        d1[k] = np.concatenate((d1[k], d2[k]), axis=axis)
    return d1

def append_dicts(d1, d2):
    for k in d1.keys():
        d1[k].append(d2[k])
    return d1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_neurons', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--compare_expert', action='store_true')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--dagger_iters', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='model_results')
    args = parser.parse_args()

    with open('expert_data/' + args.envname + '.pkl', 'rb') as f:
        rollouts = pickle.loads(f.read())

    obs_train, obs_val = split_validation(rollouts['observations'])
    action_train, action_val = split_validation(rollouts['actions'])

    indim = (rollouts['observations'].shape[-1], )
    outdim = (rollouts['actions'].shape[-1], )

    with tf.Session():
        tf_util.initialize()
        model = build_model(args, indim, outdim)
        model.fit(obs_train, action_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(obs_val, action_val))

        def policy_fn(obs):
            return model.predict(obs)

        if args.dagger:
            results = {'observations': [], 'actions': [], 'returns': [], 'dagger_iter': []}
            for i in range(args.dagger_iters):
                print('dagger iter ' + str(i))
                model_results = helpers.run_policy(args, policy_fn)
                model_results['dagger_iter'] = i
                results = append_dicts(results, model_results)
                rollouts = concat_dicts(rollouts, model_results)
                obs_train, obs_val = split_validation(rollouts['observations'])
                action_train, action_val = split_validation(rollouts['actions'])
                model.fit(obs_train, action_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(obs_val, action_val))

        else:
            results = helpers.run_policy(args, policy_fn)

        if args.save_output:
            with open(os.path.join(args.output_dir, args.envname + '-model.pkl'), 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        print('model mean return', np.mean(results['returns']))
        print('model std of return', np.std(results['returns']))

    if args.compare_expert:
        with tf.Session():
            tf_util.initialize()

            expert_policy_fn = helpers.load_policy('experts/'+args.envname+'.pkl')
            expert_results = helpers.run_policy(args, expert_policy_fn)

            if args.save_output:
                with open(os.path.join(args.output_dir, args.envname + '-expert.pkl'), 'wb') as f:
                    pickle.dump(expert_results, f, pickle.HIGHEST_PROTOCOL)

            print('expert mean returns', np.mean(expert_results['returns']))
            print('expert std of returns', np.std(expert_results['returns']))

if __name__ == '__main__':
    main()
