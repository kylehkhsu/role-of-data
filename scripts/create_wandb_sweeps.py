import os
import ipdb
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dry', type=int, default=1)
args = parser.parse_args()

yamls = os.popen('ls ./sweeps/rebuttal').read()
yamls = yamls.split('\n')
yamls = list(filter(None, yamls))
yamls = [os.path.join('./sweeps/rebuttal', yaml) for yaml in yamls]

with open('./scripts/tests/wandb.txt', 'r') as f:
    wandb = f.read()

dump = '#!/usr/bin/env bash\n'
count = 0
for yaml in yamls:
    if 'mlp' not in yaml:
        continue
    count += 1
    print(yaml)
    if not args.dry:
        sweep_msg = os.popen(f'wandb sweep {yaml}').read()

        p = re.compile('.*Create sweep with ID: (\w+)\n.*', re.DOTALL)
        id = p.match(sweep_msg)
        if id:
            id = id.group(1)
        else:
            print(f'regex failed to find id for yaml {yaml}')
            break
        dump += (f'wandb agent {id}\n')

    # os.popen(f'srun --mem=12G -c 8 --gres=gpu:1 -p p100 wandb agent {id}')
print(f'count: {count}')
if not args.dry:
    with open('./scripts/rebuttal_sgd_mlp.sh', 'w+') as f:
        f.write(dump)
