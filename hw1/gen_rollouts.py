import os

envnames = ['Walker2d-v2', 'Reacher-v2', 'Humanoid-v2', 'Hopper-v2', 'HalfCheetah-v2', 'Ant-v2']
num_rollouts = 1000

for e in envnames:
    os.system('python run_expert.py' + e + ' --num_rollouts ' + str(num_rollouts))
