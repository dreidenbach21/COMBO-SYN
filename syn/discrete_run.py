import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#### os.environ["CUDA_VISIBLE_DEVICES"]="8,1"
os.environ["CUDA_VISIBLE_DEVICES"]="6" #"9" 7
import sys
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code')
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code/hgraph2graph')
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code/MolecularTransformer')

import fire
import combo_mole_discrete
from offlinerl.utils.config import parse_config
import discrete_mole_config

def run_algo(**kwargs):

    print("start", kwargs)
    algo_config = parse_config(discrete_mole_config)
    algo_config.update(kwargs)
    print("config update", algo_config)

    algo_init_fn, algo_trainer_obj = combo_mole_discrete.algo_init, combo_mole_discrete.AlgoTrainer

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)


    algo_trainer.train()

if __name__ == "__main__":
    fire.Fire(run_algo)