# COMBO: Conservative Offline Model-Based Policy Optimization
# http://arxiv.org/abs/2102.08363
# No available code

import torch
import pickle
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition

from utils.molecular_transformer import *
from utils.latent_model import *
from tdc import Oracle
from collections import OrderedDict, defaultdict
from rdkit import Chem
# import ipdb; ipdb.set_trace()

def algo_init(args):
    logger.info('Run algo_init function')
    print(torch.__version__)
    setup_seed(args['seed'])
    
#     if args["obs_shape"] and args["action_shape"]:
#         obs_shape, action_shape = args["obs_shape"], args["action_shape"]
#     elif "task" in args.keys():
#         from offlinerl.utils.env import get_env_shape
#         obs_shape, action_shape = get_env_shape(args['task'])
#         args["obs_shape"], args["action_shape"] = obs_shape, action_shape
#     else:
#         raise NotImplementedError
    
#     obs_shape, action_shape = (1,32), (1,32)
    if args['discrete']:
        obs_shape, action_shape = 32, #TODO _run preorocessing #3248 actions
        actor = None # TODO: create categorical distribution
    else:
        obs_shape, action_shape = 32, 32
        net_a = Net(layer_num=args['hidden_layers'], 
                state_shape=obs_shape, 
                hidden_layer_size=args['hidden_layer_size']).to(args['device'])

        actor = TanhGaussianPolicy(preprocess_net=net_a,
                                action_shape=action_shape,
                                hidden_layer_size=args['hidden_layer_size'],
                                conditioned_sigma=True).to(args['device'])

    args['target_entropy'] = - float(np.prod(action_shape))
    
#     transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'], args['transition_init_num']).to(args['device'])
#     transition_optim = torch.optim.Adam(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)
    latent_space = LatentSpaceModel()
    transition = MolecularTransformer()
    transition_optim = None
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])


    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["critic_lr"])

    q1 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    q2 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])

    log_beta = torch.zeros(1, requires_grad=True, device=args['device'])
    beta_optimizer = torch.optim.Adam([log_beta], lr=args["critic_lr"])

    return {
        "transition" : {"net" : transition, "opt" : transition_optim},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer},
        "critic" : {"net" : [q1, q2], "opt" : critic_optim},
        "log_beta" : {"net" : log_beta, "opt" : beta_optimizer},
        "latent_space": latent_space
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        self.discrete = args["discrete"]
        self.transition = algo_init['transition']['net']
        # self.transition_optim = algo_init['transition']['opt']
        # self.transition_optim_secheduler = torch.optim.lr_scheduler.ExponentialLR(self.transition_optim, gamma=0.99)
        self.selected_transitions = None

        self.latent_space = algo_init["latent_space"]

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.log_beta = algo_init['log_beta']['net']
        self.log_beta_optim = algo_init['log_beta']['opt']

        self.device = args['device']
        
        self.oracle = Oracle(args['oracle'])
        self.data_root = args['data_root']

        self.react_cache = OrderedDict()
        self.latent_cache = OrderedDict()
        self.reward_cache = OrderedDict()
        self.cache_size = args['cache_size']
        self.scale_rewards = args['scale_rewards']
        self.save_name = args['save_name']
        # import ipdb; ipdb.set_trace()
        
    def train(self, train_buffer = None, val_buffer = None, callback_fn = None):
        # import ipdb; ipdb.set_trace()
        if self.args['dynamics_path'] is not None:
            self.transition = MolecularTransformer(self.args['dynamics_path']) #torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.transition = MolecularTransformer()
#             self.train_transition(train_buffer)
#         self.transition.requires_grad_(False)   
        if train_buffer == None:
            train_buffer = self.build_train_buffer()
        policy = self.train_policy(train_buffer, val_buffer, self.transition, callback_fn)
    
    def get_policy(self):
        return self.actor

    def make_numpy(self, x):
        return np.asarray(x) #, dtype = object)

    def save_cache(self, cache, name):
        with open(self.data_root + f"/data/{name}.pkl", "wb") as f:
            pickle.dump(cache, f)
    
    def load_cache(self, name):
        try:
            with open(self.data_root + f"/data/{name}.pkl", "rb") as f:
                cache = pickle.load(f)
        except:
            print(f"No {name} cache found. Starting Fresh")
            cache = OrderedDict()
        return cache

    def build_train_buffer(self):
        # import ipdb; ipdb.set_trace()
        with open(self.data_root + "/data/combo_buffer_JNK3_V2.pkl", "rb") as f:
            checked_buffer = pickle.load(f)
        self.train_buffer = ModelBuffer(2*len(checked_buffer))
        
        reacts, partners, prods = [], [], []
        rewards = []
        for ab, prod, reward in checked_buffer:
        # for ab, prod in checked_buffer:
            reacts.append(ab[0])
            partners.append(ab[1])
            prods.append(prod)
            # reward = np.random.random_sample()
            rewards.append(reward)
            self.reward_cache[prod] = reward

        # import ipdb; ipdb.set_trace() #TODO make a depth counter list
        batch_data = Batch({"obs" : self.make_numpy(reacts),
                            "act" : self.make_numpy(partners),
                            "rew" : np.asarray(rewards), #self.make_numpy(self.oracle(prods)), # Pre calculated
                            "done" : np.asarray([0]*len(prods)),
                            "obs_next" : self.make_numpy(prods),
                        })
        self.train_buffer.put(batch_data)
        return self.train_buffer

    # DONE: need encode action cache and reaction cache
    # TODO: figure out what the rdkit condition is for encoding and add a filter step for the data
    # TODO: reward scaling

    def reward(self, smiles, k = 1, scale = False):
        rewards = []
        if len(smiles) <=0:
            return [0]*k
        for mole in smiles:
            if mole not in self.reward_cache:
                val = self.oracle(mole)
                if scale:
                    val = self.scale_reward(val)
            else:
                val = self.reward_cache[mole]

            self.reward_cache[mole] = val
            rewards.append(val)
        self.rebalance(self.reward_cache)
        return rewards


    def encode(self, smiles):
        # TODO: encode error check
        dirty_smiles = {}
        count = 0
        invalids = set()
        for smi in smiles:
            if smi in invalids:
                continue
            if smi not in self.latent_cache:
                if None == Chem.MolFromSmiles(smi) or not self.latent_space.check_encodable(smi):
                    invalids.add(smi)
                    self.latent_cache[smi] = "INVALID"
                else:
                    dirty_smiles[smi] = 0
            else:
                move_up = self.latent_cache[smi]
                self.latent_cache[smi] = move_up
                count += 1
        # if len(smiles) - count > 0:
        #     print("Encode Cache Hits = ", count, "Encode Cache Misses = ", len(smiles) - count)
        smi_actions = list(dirty_smiles.keys())
        if len(smi_actions) > 0:
            latent_actions = self.latent_space.encode(smi_actions)
            for smi, z in zip(smi_actions, latent_actions):
                self.latent_cache[smi] = z.detach().cpu().numpy()
                del z
            del latent_actions
        results = []
        bad_idx = []
        for idx, smi in enumerate(smiles):
            z = self.latent_cache[smi]
            if z == "INVALID":
                bad_idx.append(idx)
            else:
                results.append(z)
        
        self.rebalance(self.latent_cache)
        torch.cuda.empty_cache()
        return torch.from_numpy(np.stack(results, axis = 0)), bad_idx
        # return torch.stack(results, dim = 0)
    def scale_reward(self, r, c = 100):
        return max(c*r*r+1, r)

    def react(self, A, B):
        dirty_A = []
        dirty_B = []
        set_a, set_b = set(), set()
        count = 0
        for a,b in zip(A,B):
            key = f"{a}<>{b}"
            key2 = f"{b}<>{a}"
            if key not in self.react_cache and key2 not in self.react_cache:
                # cond = False
                # if a not in set_a:
                #     set_a.add(a)
                #     cond = True
                # if b not in set_b:
                #     set_b.add(b)
                #     cond = True
                # if cond:
                dirty_A.append(a)
                dirty_B.append(b)
            else:
                if key not in self.react_cache:
                    key = key2
                move_up = self.react_cache[key]
                self.react_cache[key] = move_up
                count += 1
        if len(A) - count > 0:
            print("React Cache Hits = ", count, "React Cache Misses = ", len(A) - count)
        # else:
        #     print(f"|R:H{count}M{len(A) - count}", end="")
        if len(dirty_A) > 0:
            _, next_obs = self.transition.react(dirty_A, dirty_B)
            for a, b, prod in zip(dirty_A, dirty_B, next_obs):
                key = f"{a}<>{b}"
                self.react_cache[key] = prod[0]
        results = []
        for a,b in zip(A,B):
            key = f"{a}<>{b}"
            if key not in self.react_cache:
                key = f"{b}<>{a}"
                # print("react cache flip flop", key in self.react_cache)
            results.append(self.react_cache[key])
        
        self.rebalance(self.react_cache)
        return results
        
    def rebalance(self, cache):
        size = len(cache)
        if size < self.cache_size:
            return
        print("Rebalancing cache")
        for _ in range(size - self.cache_size):
            cache.popitem()

    def check_gpu(self):
        import torch
        import gc
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass
    
    def prune(self, data, indicies):
       return np.asarray([j for i, j in enumerate(data) if i not in indicies])

    def train_policy(self, train_buffer, val_buffer, transition, callback_fn):
        real_batch_size = int(self.args['policy_batch_size'] * self.args['real_data_ratio'])
        model_batch_size = self.args['policy_batch_size']  - real_batch_size
        
        # model_buffer = ModelBuffer(self.args['buffer_size'])

        self.model_buffer_map = {}
        for i in range(self.args["horizon"]):
            self.model_buffer_map[i] = ModelBuffer(self.args['buffer_size'])

        self.log_list = []
        self.latent_cache = self.load_cache("latent_cache_test")
        self.react_cache = self.load_cache("react_cache_test")
        print(type(self.latent_cache))

        for epoch in range(self.args['max_epoch']):
            # collect data
            LOG = {}
            for t in range(self.args['horizon']): # number of reactions
                print(f"Epoch {epoch} step {t}")
                log = defaultdict(list)
                LOG[t] = log
                with torch.no_grad():
                    if t == 0:
                        train_samples = train_buffer.sample(int(self.args['data_collection_per_epoch']))
                        smile_obs = train_samples['obs']
                        smile_acs = train_samples['act']
                        all_next_obs = train_samples['obs_next']
                        rewards = train_samples['rew']
                        if self.scale_rewards:
                            rewards = [self.scale_reward(x) for x in rewards]
                    else:
                        # self.pdb() #TODO: maybe do multiple model buffers so we actually sample those that are not OG and from prev step
                        # model_samples = model_buffer.sample(int(self.args['data_collection_per_epoch']))
                        model_samples = self.model_buffer_map[t-1].sample(int(self.args['data_collection_per_epoch']))
                        smile_obs = model_samples['obs_next']
                        obs, _ = self.encode(smile_obs)
                        noise_dist = self.actor(obs.to(self.device))
                        noise = noise_dist.sample()
                        action = self.latent_space.combo_sample(obs.to(self.device), noise, scale = 1) #.detach().cpu().numpy() # torch 1.9.0 has linear size miss match error on gpu
                        smile_acs = self.latent_space.decode(action)

                        bad_acs_idx = []
                        idx = 0
                        for smi, z in zip(smile_acs, action):
                            if None == Chem.MolFromSmiles(smi): # or not self.latent_space.check_encodable(smi): #may not need this as we decoding
                                bad_acs_idx.append(idx)
                                self.latent_cache[smi] = "INVALID"
                            else:
                                self.latent_cache[smi] = z.detach().cpu().numpy()
                            idx += 1
                        del action

                        if len(bad_acs_idx) > 0:
                            print(f"Latent Action Decode: {len(bad_prod_idx)} invalid molecules out of {len(all_next_obs)}")
                            smile_obs = self.prune(smile_obs, bad_acs_idx)

                        print("Reacting...")
                        all_next_obs = self.react(smile_obs, smile_acs)
                        _, bad_prod_idx = self.encode(all_next_obs)
                        # _, all_next_obs = transition.react(smile_obs, smile_acs)
                        if len(bad_prod_idx) > 0:
                            print(f"Chemical Reaction: {len(bad_prod_idx)} invalid molecules out of {len(all_next_obs)}")
                            smile_obs = self.prune(smile_obs, bad_prod_idx)
                            smile_acs = self.prune(smile_acs, bad_prod_idx)
                            all_next_obs = self.prune(all_next_obs, bad_prod_idx)

                        # REWARDS CAN BE CALCUALTED LATER FOR BETTER EFFICIENCY
                        rewards = self.reward(all_next_obs, k = len(smile_obs), scale = self.scale_rewards)

                    rewards = torch.Tensor(rewards)
                    mean_reward = rewards.mean()
                    print('Gather Step: average reward:', mean_reward.item(), "max reward", rewards.max().item(), "std reward", rewards.std().item())
                    log["total_mean_offline_reward"].append(mean_reward.item())

                    if t == self.args['horizon'] -1:
                        dones = torch.ones_like(rewards)
                    else:
                        dones = torch.zeros_like(rewards)
                    # ONLY CLEAN MOLECULES GET ADDED TO THE BUFFER
                    batch_data = Batch({
                            "obs" : smile_obs,
                            "act" : smile_acs,
                            "rew" : rewards,
                            "done" : dones.cpu(),
                            "obs_next" : all_next_obs,
                        })
                    # self.pdb()
                    # model_buffer.put(batch_data)
                    self.model_buffer_map[t].put(batch_data)

                # update
                update_steps = self.args['steps_per_epoch']

                print("CQL Update ...")
                print(f"Real Batch Size: {real_batch_size} Model Batch Size: {model_batch_size}")
                for cql_idx in range(update_steps):
                    batch = train_buffer.sample(real_batch_size) # TB is large so not guarenteed a cache hit
                    # model_batch = model_buffer.sample(model_batch_size)
                    model_batch = self.model_buffer_map[t].sample(model_batch_size)
                    batch = Batch.cat([batch, model_batch], axis=0)
                    
                    smi_obs = batch['obs']
                    smi_action = batch['act']
                    smi_next_obs = batch['obs_next']

                    # self.pdb()
                    # TODO: can merge these encode calls so that there is only one
                    batch['obs'], bad_obs_idx = self.encode(smi_obs) #.detach().cpu() cant cause now a tuple
                    batch['act'], bad_acs_idx  = self.encode(smi_action)
                    batch['obs_next'], bad_prod_idx  = self.encode(smi_next_obs)
                    assert(len(bad_prod_idx) == 0)
                    assert(len(bad_acs_idx) == 0)
                    assert(len(bad_obs_idx) == 0)

                    
                    # batch.to_torch(device=self.device)
                    batch.to_torch(device = "cpu")
                    print(".", end = "", flush = True)
                    # self.pdb()
                    losses = self._cql_update(batch, t)
                    self.update_log(losses, log)
                print("")
                self.save_cache(self.latent_cache, "latent_cache_test")
                self.save_cache(self.react_cache, "react_cache_test")

            # res = callback_fn(self.get_policy())
            
#             res['disagreement_uncertainty'] = disagreement_uncertainty.mean().item()
#             res['aleatoric_uncertainty'] = aleatoric_uncertainty.mean().item()
            # res['beta'] = torch.exp(self.log_beta.detach()).item()
            # res['reward'] = reward.mean().item()
            # self.log_res(epoch, res)
            self.log_list.append(LOG)
            with open(f"/home/yangk/dreidenbach/rl/molecule/logs/log_{self.save_name}.pkl", "wb") as f:
                pickle.dump(self.log_list, f)

            torch.save(self.get_policy().state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/policy_{self.save_name}_{epoch}.pt")
        return self.get_policy()
        
    # def scale_reward(self, rewards):
    #     mean = np.mean(rewards)
    #     std = np.std(rewards)
    #     scale = lambda x, bound: 1 + min(1, 1/bound*x)
    #     new_rewards = [scale(x, mean + std) for x in rewards]
    #     return torch.Tensor(new_rewards)
    
    def update_log(self, losses, log):
        for k, v in losses.items():
            log[k].append(v)
        
    def pdb(self):
        import ipdb; ipdb.set_trace()
        return

    def _cql_update(self, batch_data, t):
        obs = batch_data['obs'].to(device=self.device)
        action = batch_data['act'].to(device=self.device)
        next_obs = batch_data['obs_next'].to(device=self.device)
        reward = batch_data['rew'].to(device=self.device)
        done = batch_data['done'].to(device=self.device)
#         batch_size = done.shape[0]
        batch_size = obs.shape[0]
        # self.pdb()
        '''update critic''' # max q back up in cql update propagagte info change scritic target currently we just sample one

        # normal bellman backup loss
        obs_action = torch.cat([obs, action], dim=-1)
        _q1 = self.q1(obs_action)
        _q2 = self.q2(obs_action)
        # self.pdb()
    #     if self.discrete:
    #         assert(1==0)
    # #         with torch.no_grad():
    # #             next_noise_dist = self.actor(next_obs)
    # #             next_noise = next_noise_dist.sample()
    # #             next_action = None #TODO discrete action selection
    # # #             log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
    # #             log_prob = noise_dist.log_prob(next_noise).sum(dim=-1, keepdim=True)
    # #             next_obs_action = torch.cat([next_obs, next_action], dim=-1)
    # #             _target_q1 = self.target_q1(next_obs_action)
    # #             _target_q2 = self.target_q2(next_obs_action)
    # #             alpha = torch.exp(self.log_alpha)
    # #             y = reward + self.args['discount'] * (1 - done) * (torch.min(_target_q1, _target_q2) - alpha * log_prob)
    #     else:
        with torch.no_grad():
            next_noise_dist = self.actor(next_obs)
            next_noise = next_noise_dist.sample()
            next_action = next_noise + next_obs # NOISE POLICY
#             log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            log_prob = next_noise_dist.log_prob(next_noise).sum(dim=-1, keepdim=True) #used to train alpha
            next_obs_action = torch.cat([next_obs, next_action], dim=-1)
            _target_q1 = self.target_q1(next_obs_action)
            _target_q2 = self.target_q2(next_obs_action)
            alpha = torch.exp(self.log_alpha)
            y = reward + self.args['discount'] * (1 - done) * (torch.min(_target_q1, _target_q2) - alpha * log_prob)
        # max cql q value anikait suggested. alpha*mean + beta*std for K action samples
        # http://proceedings.mlr.press/v139/ghasemipour21a/ghasemipour21a.pdf 
        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()
        critic_loss_pre = critic_loss.item()
        # print("critic loss", critic_loss_pre)
        # attach the value penalty term
        random_actions = torch.rand(self.args['num_samples'], batch_size, action.shape[-1]).to(action) * 2 - 1
        noise_dist = self.actor(obs)
        sampled_actions = torch.stack([noise_dist.rsample() + obs for _ in range(self.args['num_samples'])], dim=0)

        random_next_actions = torch.rand(self.args['num_samples'], batch_size, action.shape[-1]).to(action) * 2 - 1
        next_noise_dist = self.actor(next_obs)
        sampled_next_actions = torch.stack([next_noise_dist.rsample() + next_obs for _ in range(self.args['num_samples'])], dim=0)
        # self.pdb()
        sampled_actions = torch.cat([random_actions, sampled_actions], dim=0)
        repeated_obs = torch.repeat_interleave(obs.unsqueeze(0), sampled_actions.shape[0], 0)
        sampled_q1 = self.q1(torch.cat([repeated_obs, sampled_actions], dim=-1))
        sampled_q2 = self.q2(torch.cat([repeated_obs, sampled_actions], dim=-1))

        sampled_next_actions = torch.cat([random_next_actions, sampled_next_actions], dim=0)
        repeated_next_obs = torch.repeat_interleave(next_obs.unsqueeze(0), sampled_next_actions.shape[0], 0)
        sampled_next_q1 = self.q1(torch.cat([repeated_next_obs, sampled_next_actions], dim=-1))
        sampled_next_q2 = self.q2(torch.cat([repeated_next_obs, sampled_next_actions], dim=-1))

        sampled_q1 = torch.cat([sampled_q1, sampled_next_q1], dim=0)
        sampled_q2 = torch.cat([sampled_q2, sampled_next_q2], dim=0)        

        if self.args['with_important_sampling']:
            # perform important sampling
            _random_log_prob = torch.ones(self.args['num_samples'], batch_size, 1).to(sampled_q1) * action.shape[-1] * np.log(0.5)
            _log_prob = action_dist.log_prob(sampled_actions[self.args['num_samples']:]).sum(dim=-1, keepdim=True)
            _next_log_prob = next_action_dist.log_prob(sampled_next_actions[self.args['num_samples']:]).sum(dim=-1, keepdim=True)
            is_weight = torch.cat([_random_log_prob, _log_prob, _random_log_prob, _next_log_prob], dim=0)
            sampled_q1 = sampled_q1 - is_weight
            sampled_q2 = sampled_q2 - is_weight

        q1_penalty = (torch.logsumexp(sampled_q1, dim=0) - _q1) * self.args['base_beta']
        q2_penalty = (torch.logsumexp(sampled_q2, dim=0) - _q2) * self.args['base_beta']

        if self.args['learnable_beta']:
            # update beta
            beta_loss = - torch.mean(torch.exp(self.log_beta) * (q1_penalty - self.args['lagrange_thresh']).detach()) - \
                torch.mean(torch.exp(self.log_beta) * (q2_penalty - self.args['lagrange_thresh']).detach())

            self.log_beta_optim.zero_grad()
            beta_loss.backward()
            self.log_beta_optim.step()

        q1_penalty = q1_penalty * torch.exp(self.log_beta)
        q2_penalty = q2_penalty * torch.exp(self.log_beta)

        critic_loss = critic_loss + torch.mean(q1_penalty) + torch.mean(q2_penalty)
        # print("critic loss with penalty", critic_loss.item())
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])


        '''update actor'''
        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.args['target_entropy']).detach())
            # print("alpha loss", alpha_loss.item())
            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

        # norm actor loss
        # obs = obs.detach()
        
        actor_noise_dist = self.actor(obs)
        new_noise = actor_noise_dist.rsample()
        new_action = new_noise + obs
        # expert_noise = action - obs #TODO
        # if t < 0: #for now ignore behavior cloning
        #     action_log_prob = actor_noise_dist.log_prob(expert_noise)
        #     new_action = action
        action_log_prob = actor_noise_dist.log_prob(new_noise)
        new_obs_action = torch.cat([obs, new_action], dim=-1)
        qv = torch.min(self.q1(new_obs_action), self.q2(new_obs_action))
        # actor_loss2 = - qv.mean() + torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1).mean()
        actor_loss = (torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1) - qv).mean()
        # print("actor loss", actor_loss.item()) #,actor_loss2.item() )
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "actor_loss": actor_loss.item(), 
            "alpha_loss": alpha_loss.item(), 
            "critic_loss_pre":critic_loss_pre, 
            "critic_loss":critic_loss.item(),
            "cql_update_mean_reward": reward.mean().item(),
        }



#     def _select_best_indexes(self, metrics, n):
#         pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
#         pairs = sorted(pairs, key=lambda x: x[0])
#         selected_indexes = [pairs[i][1] for i in range(n)]
#         return selected_indexes

#     def _train_transition(self, transition, data, optim):
#         data.to_torch(device=self.device)
#         dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
#         loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
#         loss = loss.mean()

#         loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

#         optim.zero_grad()
#         loss.backward()
#         optim.step()
        
#     def _eval_transition(self, transition, valdata):
#         with torch.no_grad():
#             valdata.to_torch(device=self.device)
#             dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
#             loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1,2))
#             return list(loss.cpu().numpy())


# Batch(
#     obs: array(['C1CNCCN1', 'ClCc1ccc2c(c1)OCO2', 'Cc1ccc(S(=O)(=O)Cl)cc1',
#                 'C[Si](C)(C)CCOCn1ccc2c(-c3cn[nH]c3)ncnc21', 'O=C(Cl)CCCBr'],
#                dtype='<U59'),
#     act: array(['c1ccc2c(OCC3CO3)cccc2c1', 'O=C(NC1CCNCC1)c1ccccc1',
#                 'OCCC(O)c1ccccc1', 'CSCCC=CC#N', 'c1ccc2c(c1)CCCN2'], dtype='<U59'),
#     rew: tensor([0.0300, 0.0000, 0.0000, 0.0700, 0.0000]),
#     done: tensor([0., 0., 0., 0., 0.]),
#     obs_next: array(['OC(COc1cccc2ccccc12)CN1CCNCC1',
#                      'O=C(NC1CCN(Cc2ccc3c(c2)OCO3)CC1)c1ccccc1',
#                      'Cc1ccc(S(=O)(=O)OCCC(O)c2ccccc2)cc1',
#                      'CSCCC(CC#N)n1cc(-c2ncnc3c2ccn3COCC[Si](C)(C)C)cn1',
#                      'O=C(CCCBr)N1CCCc2ccccc21'], dtype='<U79'),
# )

# Batch(
#     obs: array(['Cc1ccc(S(=O)(=O)Cl)cc1', 'ClCc1ccc2c(c1)OCO2', 'O=C(Cl)CCCBr',
#                 'C1CNCCN1', 'Cc1ccc(S(=O)(=O)Cl)cc1'], dtype='<U59'),
#     act: array(['OCCC(O)c1ccccc1', 'O=C(NC1CCNCC1)c1ccccc1', 'c1ccc2c(c1)CCCN2',
#                 'c1ccc2c(OCC3CO3)cccc2c1', 'OCCC(O)c1ccccc1'], dtype='<U59'),
#     rew: tensor([0.0000, 0.0000, 0.0000, 0.0300, 0.0000]),
#     done: tensor([0., 0., 0., 0., 0.]),
#     obs_next: array(['Cc1ccc(S(=O)(=O)OCCC(O)c2ccccc2)cc1',
#                      'O=C(NC1CCN(Cc2ccc3c(c2)OCO3)CC1)c1ccccc1',
#                      'O=C(CCCBr)N1CCCc2ccccc21', 'OC(COc1cccc2ccccc12)CN1CCNCC1',
#                      'Cc1ccc(S(=O)(=O)OCCC(O)c2ccccc2)cc1'], dtype='<U79'),
# )