import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generate_rollouts', action='store_true')
parser.add_argument('--problem_2_1', action='store_true')
parser.add_argument('--problem_2_2', action='store_true')
parser.add_argument('--problem_3', action='store_true')
args = parser.parse_args()

"""
To replicate results, run as, e.g.:

python homework.py --generate_rollouts --problem_2_1 --problem_2_2 --problem_3

To view plots, run jupyter notebook
"""

if args.generate_rollouts:
    envnames = ['Reacher-v2', 'Humanoid-v2']
    num_rollouts = 1
    for e in envnames:
        os.system('python run_expert.py ' + e + ' --num_rollouts ' + str(num_rollouts) + ' --save_output --output_dir expert_data')

if args.problem_2_1:
    #Train models
    os.system('python behavioral_cloning.py Humanoid-v2 --save_output --output_dir model_results --epochs 50')
    os.system('python behavioral_cloning.py Reacher-v2 --save_output --output_dir model_results')
    #Collect expert results
    os.system('python run_expert.py Humanoid-v2 --save_output --output_dir expert_results --num_rollouts 20')
    os.system('python run_expert.py Reacher-v2 --save_output --output_dir expert_results --num_rollouts 20')

if args.problem_2_2:
    os.system('python batch_bc.py')

if args.problem_3:
    os.system('python behavioral_cloning.py Humanoid-v2 --save_output --output_dir dagger --dagger --dagger_iters 5')
