import os
import ipdb
import re

yamls = os.popen('ls ./sweeps/final').read()
yamls = yamls.split('\n')
yamls = list(filter(None, yamls))
yamls = [os.path.join('./sweeps/final', yaml) for yaml in yamls]

with open('./scripts/tests/wandb.txt', 'r') as f:
    wandb = f.read()

dump = ''
for yaml in yamls:
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

with open('./scripts/final.sh', 'w+') as f:
    f.write(dump)
