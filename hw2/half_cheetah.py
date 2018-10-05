import os

# batch_sizes = [10000, 30000, 50000]
# lrs = [0.005, 0.01, 0.02]
#
# for b in batch_sizes:
#   for lr in lrs:
#       os.system('python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} -rtg --nn_baseline --exp_name hc_b{}_r{}'.format(b, lr, b, lr))

b = 10000
lr = 0.01

commands = [
# 'python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.005 --exp_name hc_b10000_r0.005',
# 'python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} -rtg --exp_name hc_b{}_r{}_rtg'.format(b, lr, b, lr),
# 'python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} --nn_baseline --exp_name hc_b{}_r{}_baseline'.format(b, lr, b, lr),
'python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} -rtg --nn_baseline --exp_name hc_b{}_r{}_rtg_baseline'.format(b, lr, b, lr)]

for c in commands:
    os.system(c)
