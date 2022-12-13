import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="9"
import sys
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code')
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code/hgraph2graph')
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code/MolecularTransformer')

import fire
# from offlinerl.algo import algo_select
# from offlinerl.data import load_data_from_neorl
# from offlinerl.evaluation import get_defalut_callback, OnlineCallBackFunction
import combo_mole
from offlinerl.utils.config import parse_config
import mole_config

def run_algo(**kwargs):

    print("start", kwargs)
    algo_config = parse_config(mole_config)
    algo_config.update(kwargs)
    print("config update", algo_config)

    algo_init_fn, algo_trainer_obj = combo_mole.algo_init, combo_mole.AlgoTrainer

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)


    algo_trainer.train()

if __name__ == "__main__":
    fire.Fire(run_algo)