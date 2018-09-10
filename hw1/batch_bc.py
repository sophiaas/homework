import os
import argparse

envnames = ['Reacher-v2']
num_neurons = [64]
learning_rates = [.1, .075, .05, .01, .005, .001]
num_layers = [3]
batch_size = [16]
num_epochs = [10]
num_rollouts = 20

for e in envnames:
    for n in num_neurons:
        for lr in learning_rates:
            for l in num_layers:
                for b in batch_size:
                    for ne in num_epochs:
                        run_type = 'n'+str(n)+'_lr'+str(lr)+'_l'+str(l)+'_b'+str(b)+'_ne'+str(ne)
                        print(run_type)
                        output_dir = 'model_results/'+run_type+'/'
                        if not os.path.isdir(output_dir):
                            os.mkdir(output_dir)
                        os.system('python behavioral_cloning.py ' + e + ' --num_neurons ' + str(n) + ' --learning_rate ' + str(lr) + ' --num_layers ' + str(l) + ' --batch_size ' + str(b) + ' --save_output' + ' --output_dir ' + output_dir + ' --num_rollouts ' + str(num_rollouts) + ' --epochs ' + str(ne))
