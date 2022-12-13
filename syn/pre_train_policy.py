import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import sys
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code')
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code/hgraph2graph')
sys.path.insert(0, '/home/yangk/dreidenbach/rl/molecule/code/MolecularTransformer')
import torch
import pickle
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net, build_mlp
from offlinerl.utils.net.tanhpolicy import CategoricalPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition

from utils.molecular_transformer import *
from utils.latent_model import *
from tdc import Oracle
from collections import OrderedDict, defaultdict
from rdkit import Chem
from mlp import MLPPolicy
with open("/home/yangk/dreidenbach/rl/molecule/data/combo_buffer_JNK3_V3_discrete_map_inv.pkl", "rb") as f:
    action_map_inv = pickle.load(f)
with open("/home/yangk/dreidenbach/rl/molecule/data/combo_buffer_JNK3_V3_discrete_map.pkl", "rb") as f:
    action_map = pickle.load(f)


device = 'cuda:0'

def build_actor(obs_shape = 32, action_shape = 4344, hidden_layer_size=6000, num_layers = 3):
    # net_a = Net(layer_num=num_layers, 
    #         state_shape=obs_shape,
    #         output_shape=action_shape,
    #         hidden_layer_size=hidden_layer_size).to(device)

    # actor = CategoricalPolicy(preprocess_net=net_a,
    #                         action_shape=action_shape,
    #                         hidden_layer_size=hidden_layer_size,
    #                         conditioned_sigma=False).to(device)
    # actor_optim = torch.optim.Adam(actor.parameters(), lr = 1e-4)
    actor = MLPPolicy(ac_dim = action_shape, ob_dim = obs_shape, n_layers = num_layers, size = hidden_layer_size, learning_rate = 1e-4, device = device)
    actor_optim = actor.optimizer 
    return actor, actor_optim

def load_train_data():
    with open("/home/yangk/dreidenbach/rl/molecule/data/combo_buffer_JNK3_discrete_exact_V3.pkl", "rb") as f:
            checked_buffer = pickle.load(f)
    data = []
    states = set()
    sa_map = {}
    for sa, p, r in checked_buffer:
        s, a = sa
        if s not in sa_map:
            sa_map[s] = set()
        sa_map[s].add(a)
        states.add(s)
        val = (s, a, p, scale_reward(r))
        data.append(val)
    lens = [len(sa_map[x]) for x in sa_map.keys()]
    print(f"Mean Unique SA Pairs: {np.mean(lens)}, Std Unique SA Pairs: {np.std(lens)}, Max Unique SA Pairs: {np.max(lens)}")
    return data, states


def scale_reward(r, c = 100, scale_rewards = 3):
    if scale_rewards == 0:
        return r
    elif scale_rewards == 1:
        return max(c*r*r+1, r)
    elif scale_rewards == 2:
        return max(c*r+1, r)
    elif scale_rewards == 3:
        return max(c*np.sqrt(r)+1, r)
    else:
        return r

def discretize(actions):
        return torch.Tensor([action_map[smi] for smi in actions])

def undiscretize(action):
    # choices = torch.argmax(actions, dim = -1).detach().cpu().numpy()
    return [action_map_inv[x] for x in actions.detach().cpu().numpy()]

def encode_states(states, load = True):
    if load:
        with open("/home/yangk/dreidenbach/rl/molecule/data/policy_pretrain_encode_states.pkl", "rb") as f:
            result = pickle.load(f)
        print(len(result), "Encode Map", len(states))
        return result
    print("Encoding States...")
    latent_space = LatentSpaceModel()
    all_states = list(states)
    result = {}
    batch_size = 100
    for idx in range(len(all_states)//batch_size+1):
        smi_actions = all_states[idx*batch_size:(idx+1)*batch_size]
        latent_actions = latent_space.encode(smi_actions)
        for smi, z in zip(smi_actions, latent_actions):
            result[smi] = z.detach().cpu().numpy()

    with open("/home/yangk/dreidenbach/rl/molecule/data/policy_pretrain_encode_states.pkl", "wb") as f:
        pickle.dump(result, f)
    return result


def train_actor(data, encode_map, epochs = 10, batch_size = 10, imitation = False):
    # import ipdb; ipdb.set_trace()
    actor, optimizer = build_actor()
    LOG = []
    for epoch in range(epochs):
        log = {}
        print("\n", epoch)
        losses = []
        for idx in range(len(data)//batch_size+1):
            chunk = data[idx*batch_size : (idx+1)*batch_size]
            obs = [x[0] for x in chunk]
            acs = [x[1] for x in chunk]
            obs_next = [x[2] for x in chunk]
            r = [x[3] for x in chunk]
            observations = torch.Tensor([encode_map[x] for x in obs]).to(device)
            actions = discretize(acs).to(device)
            action_distribution = actor(observations)
            if not imitation:
                adv_n = torch.Tensor(r).to(device)
                loss = - action_distribution.log_prob(actions)*adv_n
            else:
                loss = - action_distribution.log_prob(actions)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log[idx] = loss.item()
            losses.append(loss.item())
            print(f".", end = "", flush=True)
        stats(losses)
        LOG.append(log)
        with open("/home/yangk/dreidenbach/rl/molecule/logs/policy_pretrain_encode_losses.pkl", "wb") as f:
            pickle.dump(LOG, f)
        if imitation:
            torch.save(actor.state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/pre_train_policy_imitation_{epoch}.pt")
        else:
            torch.save(actor.state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/pre_train_policy_pg_{epoch}.pt")

def stats(data):
    mean = np.mean(data)
    std = np.std(data)
    mmax = np.max(data)
    mmin = np.min(data)
    print(f"Loss --> mean:{mean} std:{std} max:{mmax} min:{mmin}")

if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    imitation = True
    load_encode = True
    epochs = 100
    data, states = load_train_data()
    encode_map = encode_states(states, load_encode)
    train_actor(data, encode_map, imitation=imitation, epochs = epochs)
    print("END")
  

