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
import torch
# import ipdb; ipdb.set_trace()

def algo_init(args):
    logger.info('Run algo_init function')
    print(torch.__version__)
    # torch.autograd.set_detect_anomaly(True)
    setup_seed(args['seed'])
    
    obs_shape, action_shape = 32, 4344 #3109 # 4344 actions in V3 maps
    # net_a = Net(layer_num=args['hidden_layers'], 
    #         state_shape=obs_shape,
    #         output_shape=action_shape,
    #         hidden_layer_size=args['hidden_layer_size']).to(args['device2'])

    # actor = CategoricalPolicy(preprocess_net=net_a,
    #                         action_shape=action_shape,
    #                         hidden_layer_size=args['hidden_layer_size'],
    #                         conditioned_sigma=False).to(args['device2'])
    if args['use_learned_actor']:
        actor = MLPPolicy(ac_dim = action_shape, ob_dim = obs_shape, n_layers = args['hidden_layers'], size = args['hidden_layer_size'], learning_rate = args['actor_lr'], device = args['device2'])
        # actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])
        if args['use_pretrain_policy']:
            print("Loading Pretrained Policy Weights Trained via Imitation Learning")
            actor.load_state_dict(torch.load(args['pretrain_policy_path'], map_location=torch.device(args['device2'])))
        actor_optim = actor.optimizer #torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])
    else:
        actor = lambda q, x: torch.distributions.Categorical(torch.softmax(q(x)))
        actor_optim = None

    
    args['target_entropy'] = - float(np.prod(action_shape))

    latent_space = LatentSpaceModel()
    transition = MolecularTransformer()
    transition_optim = None
    
    


    log_alpha = torch.zeros(1, requires_grad=True, device=args['device2'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["critic_lr"])

    q1 = MLP(obs_shape, action_shape, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device2'])
    q2 = MLP(obs_shape, action_shape, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device2'])
   
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])

    log_beta = torch.zeros(1, requires_grad=True, device=args['device2'])
    beta_optimizer = torch.optim.Adam([log_beta], lr=args["critic_lr"])

    return {
        "transition" : {"net" : transition, "opt" : transition_optim},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer},
        "critic" : {"net" : [q1, q2], "opt" : critic_optim},
        "log_beta" : {"net" : log_beta, "opt" : beta_optimizer},
        "latent_space": latent_space,
        "action_shape": action_shape
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        self.discrete = args["discrete"]
        self.transition = algo_init['transition']['net']

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
        self.device2 = args['device2']
        self.property_name = args['oracle']
        self.oracle = Oracle(args['oracle'])
        self.data_root = args['data_root']

        self.react_cache = OrderedDict()
        self.latent_cache = OrderedDict()
        self.reward_cache = OrderedDict()

        self.cache_size = args['cache_size']
        self.save_name = args['save_name']
        self.num_actions = algo_init['action_shape']
        self.scale_rewards = args['scale_rewards']
        self.data_collection_samples = args['data_collection_samples']
        self.max_q_back_up = args['max_q_back_up']
        self.inject = args['inject']
        self.use_eps_greedy = args['use_eps_greedy']
        self.eps = args['eps']
        self.learning_start_epoch = args['learning_start_epoch']
        self.use_pretrain_policy  = args['use_pretrain_policy']
        self.clean_up = args['clean_up']
        self.use_learned_actor = args['use_learned_actor']
        self.entropy_beta_decay = args['entropy_beta_decay']
        self.no_done = args['no_done']
        # import ipdb; ipdb.set_trace()
        
    def train(self, train_buffer = None, val_buffer = None, callback_fn = None):
        # import ipdb; ipdb.set_trace()
        if self.args['dynamics_path'] is not None:
            self.transition = MolecularTransformer(self.args['dynamics_path']) #torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.transition = MolecularTransformer()
        if train_buffer == None:
            train_buffer = self.build_train_buffer()
        policy = self.train_policy(train_buffer, val_buffer, self.transition, callback_fn)
    
    def clean_up_buffer(self, buffer, t):
        rewards = buffer.data['rew']
        if type(rewards) != np.ndarray:
            rewards = rewards.detach().cpu().numpy()
        mean_score = np.mean(rewards)
        bad_idx = []
        good_idx = []
        for idx, batch in enumerate(buffer.data):
            if batch['rew'] >= mean_score: #/(max(1, 2/t)): # TODO: play with balance in good and bad 
                good_idx.append(idx)
            else:
                bad_idx.append(idx)
        good_idx.extend(np.random.choice(bad_idx,len(bad_idx)//2,replace=False))
        good_idx.sort()
        print(f"Clean up Reduction {len(rewards)} --> {len(good_idx)} Mean Score = {mean_score}, bad_idx_len = {len(bad_idx)}")
        buffer.data = buffer.data[good_idx]


    def get_policy(self):
        return self.actor

    def make_numpy(self, x):
        return np.asarray(x) #, dtype = object)

    def save_cache(self, cache, name):
        with open(self.data_root + f"/data/{name}_{self.save_name}.pkl", "wb") as f:
            pickle.dump(cache, f)
    
    def load_cache(self, name):
        try:
            with open(self.data_root + f"/data/{name}_{self.save_name}.pkl", "rb") as f: 
            # with open(self.data_root + f"/data/{name}_discrete_buffer_test_V3_mean_cut", "rb") as f:
                cache = pickle.load(f)
            print(f"cache size {len(cache)}")
        except:
            print(f"No {name} cache found. Starting Fresh")
            cache = OrderedDict()
        return cache

    def build_train_buffer(self):
        # import ipdb; ipdb.set_trace() # 91761 in V3 exact
        with open(self.data_root + f"/data/combo_buffer_{self.property_name}_discrete_exact_V3.pkl", "rb") as f:
            checked_buffer = pickle.load(f)
        self.train_buffer = ModelBuffer(int(1.1*len(checked_buffer)))
        
        reacts, partners, prods = [], [], []
        rewards = []
        for ab, prod, reward in checked_buffer:
            reacts.append(ab[0])
            partners.append(ab[1])
            prods.append(prod)
            rewards.append(reward)
            self.reward_cache[prod] = reward

        batch_data = Batch({"obs" : self.make_numpy(reacts),
                            "act" : self.make_numpy(partners),
                            "rew" : np.asarray(rewards), #self.make_numpy(self.oracle(prods)), # Pre calculated
                            "done" : np.asarray([0]*len(prods)),
                            "obs_next" : self.make_numpy(prods),
                            # "epoch": self.make_numpy([-1]*len(prods))
                        })
        self.train_buffer.put(batch_data)

        with open(self.data_root + "/data/combo_buffer_JNK3_V3_discrete_map_inv.pkl", "rb") as f:
            self.action_map_inv = pickle.load(f)
        with open(self.data_root + "/data/combo_buffer_JNK3_V3_discrete_map.pkl", "rb") as f:
            self.action_map = pickle.load(f)
        return self.train_buffer

    def reward(self, smiles, k = 1, scale = False):
        rewards = []
        if len(smiles) <=0:
            return [0]*k
        for mole in smiles:
            if mole not in self.reward_cache:
                try:
                    val = self.oracle(mole)
                except:
                    print("Oracle error", mole)
                    val = 0
                if scale:
                    val = self.scale_reward(val)
            else:
                val = self.reward_cache[mole]

            self.reward_cache[mole] = val
            rewards.append(val)
        self.rebalance(self.reward_cache)
        return rewards



    # def encode(self, smiles, verbose = False):
    #     dirty_smiles = {}
    #     count = 0
    #     invalids = set()
    #     if verbose:
    #         print("Checking Validity", flush = True)
    #     for idx, smi in enumerate(smiles):
    #         if smi in invalids:
    #             continue
    #         if verbose:
    #             print(idx, smi)
    #         if smi not in self.latent_cache:
    #             if verbose:
    #                 print("miss", flush = True)
    #             if not self.latent_space.check_encodable(smi): #None == Chem.MolFromSmiles(smi) or 
    #                 invalids.add(smi)
    #                 self.latent_cache[smi] = "INVALID"
    #             else:
    #                 dirty_smiles[smi] = 0
    #                 self.latent_cache[smi] = "VALID"
    #         else:
    #             if verbose:
    #                 print("hit", flush = True)
    #             move_up = self.latent_cache[smi]
    #             if move_up == "INVALID" or move_up == "VALID":
    #                 continue
    #             self.latent_cache[smi] = move_up
    #             count += 1

    #     smi_actions = list(dirty_smiles.keys())
    #     if len(smi_actions) > 0:
    #         # latent_actions = self.latent_space.encode(smi_actions)
    #         batch_size = 100
    #         if verbose:
    #             print("Encode start", len(smi_actions), flush = True)
    #         for b in range(len(smi_actions)//batch_size + 1):
    #             to_encode = smi_actions[b*batch_size : (b+1)*batch_size]
    #             if verbose:
    #                 print("Encode batch", b, flush = True)
    #             if len(to_encode) == 0:
    #                 continue
    #             latent_actions = self.latent_space.encode(to_encode)
    #             for smi, z in zip(to_encode, latent_actions):
    #                 self.latent_cache[smi] = z.detach().cpu().numpy()
    #                 # del z # dels were here before
    #             # del latent_actions
    #     results = []
    #     bad_idx = []
    #     if verbose:
    #         print("Encode End", flush = True)
    #     for idx, smi in enumerate(smiles):
    #         z = self.latent_cache[smi]
    #         if z == "INVALID":
    #             bad_idx.append(idx)
    #         else:
    #             results.append(z)
        
    #     self.rebalance(self.latent_cache)
    #     if verbose:
    #         print("torch cache dump", flush = True)
    #     torch.cuda.empty_cache()
    #     return torch.from_numpy(np.stack(results, axis = 0)), bad_idx

    def encode(self, smiles, verbose = False):
        dirty_smiles = {}
        count = 0
        for idx, smi in enumerate(smiles):
            if smi not in self.latent_cache:
                dirty_smiles[smi] = 0
                self.latent_cache[smi] = "MAYBE"
            else:
                move_up = self.latent_cache[smi]
                if move_up == "MAYBE":
                    continue
                self.latent_cache[smi] = move_up
                count += 1

        smi_actions = list(dirty_smiles.keys())
        if len(smi_actions) > 0:
            batch_size = 100
            for b in range(len(smi_actions)//batch_size + 1):
                to_encode = smi_actions[b*batch_size : (b+1)*batch_size]
                if len(to_encode) == 0:
                    continue

                latent_actions, batch_bad_idx = self.latent_space.encode(to_encode)
                if len(batch_bad_idx) > 0:
                    clean_smi = []
                    for idx, smi in enumerate(to_encode):
                        if idx in batch_bad_idx:
                            continue
                        else:
                            clean_smi.append(smi)
                else:
                    clean_smi = to_encode

                for smi, z in zip(clean_smi, latent_actions):
                    self.latent_cache[smi] = z.detach().cpu().numpy()
                    # del z # dels were here before
                # del latent_actions
        results = []
        bad_idx = []
        for idx, smi in enumerate(smiles):
            z = self.latent_cache[smi]
            if z == "MAYBE":
                bad_idx.append(idx)
            else:
                results.append(z)
        
        self.rebalance(self.latent_cache)
        torch.cuda.empty_cache()
        return torch.from_numpy(np.stack(results, axis = 0)), bad_idx

    def react(self, A, B):
        dirty_A = []
        dirty_B = []
        set_a, set_b = set(), set()
        count = 0
        for a,b in zip(A,B):
            key = f"{a}<>{b}"
            key2 = f"{b}<>{a}"
            if key not in self.react_cache and key2 not in self.react_cache:
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
        if len(indicies) == 0:
            return np.asarray(data)
        return np.asarray([j for i, j in enumerate(data) if i not in indicies])

    def discretize(self, actions):
        return torch.Tensor([self.action_map[smi] for smi in actions])

    def scale_reward(self, r, c = 100):
        if self.scale_rewards == 0:
            return r
        elif self.scale_rewards == 1:
            return max(c*r*r+1, r)
        elif self.scale_rewards == 2:
            return max(c*r+1, r)
        elif self.scale_rewards == 3:
            return max(c*np.sqrt(r)+1, r)
        else:
            return r

    def undiscretize(self, actions):
        if type(actions) != np.ndarray:
            return [self.action_map_inv[x] for x in actions.detach().cpu().numpy()]
        else:
            return [self.action_map_inv[x] for x in actions]

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
        self.reward_cache = self.load_cache("reward_cache_test")
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
                        if epoch > 0:
                            print("cleaning last time step at start of next epoch")
                            self.clean_up_buffer(self.model_buffer_map[self.args['horizon'] -1], t)
                        self.entropy_beta = 1#100 #1 for non pre train
                        train_samples = train_buffer.sample(int(self.data_collection_samples * self.args['data_collection_per_epoch']))
                        smile_obs = train_samples['obs']
                        smile_acs = train_samples['act']
                        all_next_obs = train_samples['obs_next']
                        rewards = train_samples['rew']
                        if self.scale_rewards > 0:
                            rewards = [self.scale_reward(x) for x in rewards]

                    else:
                        if self.clean_up: # no clean up buffer TODO
                            self.clean_up_buffer(self.model_buffer_map[t-1], t)
                 
                        self.entropy_beta = 1 #0.05
                        if self.entropy_beta_decay:
                            self.entropy_beta = max(0, self.entropy_beta - epoch/20)
                        model_samples = self.model_buffer_map[t-1].sample(int(self.data_collection_samples * self.args['data_collection_per_epoch']))
                        smile_obs = model_samples['obs_next']
                        obs, bad_prodd_idx = self.encode(smile_obs)

                        if len(bad_prodd_idx) > 0:
                            smile_obs = self.prune(smile_obs, bad_prod_idx)
                        if self.use_learned_actor:
                            action_dist = self.actor(obs.to(self.device2))
                        else:
                            action_dist = self.actor(self.q1, obs.to(self.device2))

                        if self.use_eps_greedy:
                            perform_random_action = np.random.random() < self.eps or epoch < self.learning_start_epoch
                            if perform_random_action:
                                number_random = max(1, self.learning_start_epoch - epoch + 1)
                                print(f"Grabbing {number_random} random actions for each state")
                                action_samples = np.random.choice(self.num_actions, (number_random * len(smile_obs)))
                                smile_acs = self.undiscretize(action_samples)
                                smile_obs_all = []
                                for _ in range(number_random):
                                    smile_obs_all.extend(smile_obs)
                                smile_obs = smile_obs_all
                            else:
                                #TODO: finish the non eps greedy and fix the _cql_update entropy loss term
                                smile_acs = self.undiscretize(action_dist.sample())

                        # self.pdb()
                        # all_log_probs =torch.stack( [action_dist.log_prob(torch.Tensor([x]*len(smile_obs)).to(self.device2)) for x in range(self.num_actions)])
                        # amax, aidx = torch.max(all_log_probs, dim = 0)
                        # smile_acs = self.undiscretize(action_dist.sample())

                        print("Reacting...", flush = True)
                        all_next_obs = self.react(smile_obs, smile_acs)
                        print("Encoding...", flush = True)
                        _, bad_prod_idx = self.encode(all_next_obs)

                        if len(bad_prod_idx) > 0:
                            print(f"Chemical Reaction: {len(bad_prod_idx)} invalid molecules out of {len(all_next_obs)}")
                            smile_obs = self.prune(smile_obs, bad_prod_idx)
                            smile_acs = self.prune(smile_acs, bad_prod_idx)
                            all_next_obs = self.prune(all_next_obs, bad_prod_idx)

                        # REWARDS CAN BE CALCUALTED LATER FOR BETTER EFFICIENCY
                        # self.pdb()
                        print("Calculating Rewards...", flush = True)
                        rewards = self.reward(all_next_obs, k = len(smile_obs), scale = self.scale_rewards)

                    rewards = torch.Tensor(rewards)
                    mean_reward = rewards.mean()
                    print(self.save_name)
                    print('Gather Step: average reward:', mean_reward.item(), "max reward", rewards.max().item(), "std reward", rewards.std().item())
                    log["total_mean_offline_reward"].append(mean_reward.item())
                    log["total_max_offline_reward"].append(rewards.max().item())

                    if not self.no_done and t == self.args['horizon'] -1:
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
                            # "epoch": self.make_numpy([epoch]*len(smile_obs))
                        })

                    self.model_buffer_map[t].put(batch_data)
                    self.model_buffer_map[t].max_score = max(self.model_buffer_map[t].max_score, rewards.max().item())
                    log["buffer_max_score"].append(self.model_buffer_map[t].max_score)
                    self.save_cache(self.latent_cache, "latent_cache_test")
                    self.save_cache(self.react_cache, "react_cache_test")
                    self.save_cache(self.reward_cache, "reward_cache_test")
                        

                # update
                update_steps = self.args['steps_per_epoch']
                # if t > 0:
                #     self.pdb()
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
                    # if t >= 0 and self.inject:
                    #     # self.pdb() 
                    #     smi_obs = [x for x in smi_obs] + ['NC(=O)c1ccccc1Nc1nc(Cl)ncc1Cl']
                    #     # smi_obs.append('NC(=O)c1ccccc1Nc1nc(Cl)ncc1Cl')
                    #     # smi_action.append('Nc1ccc(Cl)cc1')
                    #     smi_action = [x for x in smi_action] + ['Nc1ccc(Cl)cc1']
                    #     # smi_next_obs.append('NC(=O)c1ccccc1Nc1nc(Nc2ccc(Cl)cc2)ncc1Cl')
                    #     smi_next_obs = [x for x in smi_next_obs] + ['NC(=O)c1ccccc1Nc1nc(Nc2ccc(Cl)cc2)ncc1Cl']
                    #     # batch['rew'] = torch.cat((batch['rew'], torch.Tensor([0.99])))
                    #     batch['rew'] = np.concatenate((batch['rew'], np.asarray([self.scale_reward(0.99)])))
                    #     if t == self.args['horizon'] -1:
                    #         done_inject = 1
                    #     else:
                    #         done_inject = 0
                    #     # batch['done'] = torch.cat((batch['done'], torch.Tensor([done_inject])))
                    #     batch['done'] = np.concatenate((batch['done'], np.asarray([done_inject])))

                    
                    batch['obs'], bad_obs_idx = self.encode(smi_obs) #.detach().cpu() cant cause now a tuple
                    batch['act']  = self.discretize(smi_action)
                    batch['obs_next'], bad_prod_idx  = self.encode(smi_next_obs)
                    assert(len(bad_prod_idx) == 0)
                    assert(len(bad_obs_idx) == 0)

                    
                    # batch.to_torch(device=self.device)
                    batch.to_torch(device = "cpu")
                    print(".", end = "", flush = True)
                    # self.pdb()
                    # losses = self._cql_update(batch, t)
                    losses = self._discrete_cql_update(batch, t)
                    self.update_log(losses, log)
                print("")
                self.save_cache(self.latent_cache, "latent_cache_test")
                self.save_cache(self.react_cache, "react_cache_test")
                self.save_cache(self.reward_cache, "reward_cache_test")
                print("Saving Actor...")
                torch.save(self.get_policy().state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/policy_{self.save_name}.pt")
                print("Saving Q function...")
                torch.save(self.q1.state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/q1_{self.save_name}.pt")

            self.log_list.append(LOG)
            with open(f"/home/yangk/dreidenbach/rl/molecule/logs/log_{self.save_name}.pkl", "wb") as f:
                pickle.dump(self.log_list, f)

            for key, buff in self.model_buffer_map.items():
                print(f"Model Buffer: {key} {buff.max_score}")

            print("Saving Model Buffer List...")
            with open(f"/home/yangk/dreidenbach/rl/molecule/data/model_buffer_map_{self.save_name}.pkl", "wb") as f:
                pickle.dump(self.model_buffer_map, f)
            print("Saving Actor...")
            torch.save(self.get_policy().state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/policy_{self.save_name}.pt")
            print("Saving Q function...")
            torch.save(self.q1.state_dict(), f"/home/yangk/dreidenbach/rl/molecule/model_ckpt/q1_{self.save_name}.pt")
        return self.get_policy()
        
    
    def update_log(self, losses, log):
        for k, v in losses.items():
            log[k].append(v)
        
    def pdb(self):
        import ipdb; ipdb.set_trace()
        return

    def gather(self, qa, a, dim = 1):
        return torch.gather(qa, dim, a.long().unsqueeze(dim)).squeeze(dim)

    def _discrete_cql_update(self, batch_data, t):
        obs = batch_data['obs'].to(device=self.device2)
        action = batch_data['act'].to(device=self.device2)
        next_obs = batch_data['obs_next'].to(device=self.device2)
        reward = batch_data['rew'].to(device=self.device2)
        done = batch_data['done'].to(device=self.device2)
        batch_size = obs.shape[0]
        # from_epoch = batch_data['epoch']
        '''update critic''' # max q back up in cql update propagagte info change scritic target currently we just sample one

        _qa1 = self.q1(obs)
        _qa2 = self.q2(obs)
        _q1 = self.gather(_qa1, action)
        _q2 = self.gather(_qa2, action)
        #TODO for discrete setting can do this
        # new_actor = Categofical(softmax(self.q1(obs) - mean(q1))) # can do mean Max Ent RL TODO: anikait
        if self.use_learned_actor:
            action_dist = self.actor(obs)
        else:
            action_dist = self.actor(self.q1, obs)

        new_action = action_dist.sample()
        log_prob = action_dist.log_prob(new_action)

        if self.use_learned_actor:
            next_action_dist = self.actor(next_obs)
        else:
            next_action_dist = self.actor(self.q1, next_obs)
        # next_action_dist = self.actor(next_obs)
        # self.pdb()
        if self.max_q_back_up > 1:
            with torch.no_grad():
                next_action= torch.stack([next_action_dist.sample() for _ in range(self.max_q_back_up)], dim = 0)
                next_log_pi = next_action_dist.log_prob(next_action)#.sum(dim=-1) #, keepdim=True)
                repeated_nobs = torch.repeat_interleave(next_obs.unsqueeze(0), self.max_q_back_up, 0)
                target_q_values, max_target_indices = torch.max(torch.min(self.gather(self.target_q1(repeated_nobs), next_action, dim = 2), self.gather(self.target_q2(repeated_nobs), next_action, dim = 2)),dim = 0) #-1)
                # next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
                next_log_pi = torch.gather(next_log_pi.T, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
                alpha = torch.exp(self.log_alpha)
                y = reward + self.args['discount'] * (1 - done) * (target_q_values - alpha * next_log_pi)
        else:
            with torch.no_grad():
                next_action= next_action_dist.sample()
                log_prob = next_action_dist.log_prob(next_action) # I don't think we ware supposed to sum here .sum(dim=-1, keepdim=True)
                # next_obs_action = torch.cat([next_obs, next_action], dim=-1)
                _target_q1 = self.gather(self.target_q1(next_obs), next_action)
                _target_q2 = self.gather(self.target_q2(next_obs), next_action)
                alpha = torch.exp(self.log_alpha)
                y = reward + self.args['discount'] * (1 - done) * (torch.min(_target_q1, _target_q2) - alpha * log_prob)

        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()
        critic_loss_pre = critic_loss.item()

        discrete_iterate = True
        if discrete_iterate:
            # self.pdb()
            # all_actions = torch.Tensor([acs for acs in range(self.num_actions)], dtype = torch.int64).to(action)
            # all_actions_stack = torch.stack([all_actions for _ in range(batch_size)], dim = 0).to(action)
            sampled_q1 = _qa1
            sampled_q2 = _qa2
            if self.args['with_important_sampling']:
                # perform important sampling
                # _log_prob = action_dist.log_prob(all_actions_stack).detach()
                info = [torch.tensor([x]*batch_size, dtype = torch.int64).to(self.device2) for x in range(self.num_actions)]
                _actions =torch.stack(info, dim = 0)
                _log_prob = action_dist.log_prob(_actions).detach()
                # _next_log_prob = next_action_dist.log_prob(all_actions).detach()
                # is_weight = torch.cat([_random_log_prob, _log_prob, _random_log_prob, _next_log_prob], dim=0)
                is_weight = _log_prob.T
                sampled_q1 = sampled_q1 - is_weight
                sampled_q2 = sampled_q2 - is_weight
            q1_penalty = (torch.logsumexp(sampled_q1, dim=1) - _q1) * self.args['base_beta']
            q2_penalty = (torch.logsumexp(sampled_q2, dim=1) - _q2) * self.args['base_beta']

        else:
            push_amount = 100 #self.num_actions #1000
            # OR 
            random_actions = torch.randint(self.num_actions, (push_amount, batch_size)).to(action)
            sampled_actions = torch.stack([action_dist.sample() for _ in range(push_amount)], dim=0)


            random_next_actions = torch.randint(self.num_actions, (push_amount, batch_size)).to(action)
            sampled_next_actions = torch.stack([next_action_dist.sample() for _ in range(push_amount)], dim=0)

            random_actions2 = torch.randint(self.num_actions, (push_amount, batch_size)).to(action)
            full_sampled_actions = torch.cat([random_actions2, sampled_actions], dim=0) # 2000 x 10
            # repeated_obs = torch.repeat_interleave(obs.unsqueeze(0), full_sampled_actions.shape[0], 0)
            repeated_q1 =  torch.repeat_interleave(self.q1(obs).unsqueeze(0), full_sampled_actions.shape[0], 0)
            sampled_q1 = self.gather(repeated_q1, full_sampled_actions, dim = 2) 
            repeated_q2 =  torch.repeat_interleave(self.q2(obs).unsqueeze(0), full_sampled_actions.shape[0], 0)
            sampled_q2 = self.gather(repeated_q2, full_sampled_actions, dim = 2)

            full_sampled_next_actions = torch.cat([random_next_actions, sampled_next_actions], dim=0)
            # repeated_next_obs = torch.repeat_interleave(next_obs.unsqueeze(0), full_sampled_next_actions.shape[0], 0)
            repeated_next_q1 =  torch.repeat_interleave(self.q1(next_obs).unsqueeze(0), full_sampled_next_actions.shape[0], 0)
            sampled_next_q1 = self.gather(repeated_next_q1, full_sampled_next_actions, dim = 2)
            repeated_next_q2 =  torch.repeat_interleave(self.q2(next_obs).unsqueeze(0), full_sampled_next_actions.shape[0], 0)
            sampled_next_q2 = self.gather(repeated_next_q2, full_sampled_next_actions, dim = 2)

            sampled_q1 = torch.cat([sampled_q1, sampled_next_q1], dim=0)
            sampled_q2 = torch.cat([sampled_q2, sampled_next_q2], dim=0)        
            if self.args['with_important_sampling']:
                # perform important sampling
                _random_log_prob = torch.ones(push_amount, batch_size).to(self.device2) * np.log(1/self.num_actions)
                _log_prob = action_dist.log_prob(sampled_actions).detach() #.sum(dim=-1, keepdim=True)
                _next_log_prob = next_action_dist.log_prob(sampled_next_actions).detach() #.sum(dim=-1, keepdim=True)
                is_weight = torch.cat([_random_log_prob, _log_prob, _random_log_prob, _next_log_prob], dim=0)
                sampled_q1 = sampled_q1 - is_weight
                sampled_q2 = sampled_q2 - is_weight

            q1_penalty = (torch.logsumexp(sampled_q1, dim=0) - _q1) * self.args['base_beta']
            q2_penalty = (torch.logsumexp(sampled_q2, dim=0) - _q2) * self.args['base_beta']

        if self.args['learnable_beta']:
            beta_loss = - torch.mean(torch.exp(self.log_beta) * (q1_penalty - self.args['lagrange_thresh']).detach()) - \
                torch.mean(torch.exp(self.log_beta) * (q2_penalty - self.args['lagrange_thresh']).detach())

            self.log_beta_optim.zero_grad()
            beta_loss.backward()
            self.log_beta_optim.step()

        q1_penalty = q1_penalty * torch.exp(self.log_beta)
        q2_penalty = q2_penalty * torch.exp(self.log_beta)

        critic_loss = critic_loss + torch.mean(q1_penalty) + torch.mean(q2_penalty)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # # soft target update
        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

        if self.use_learned_actor:
            '''update actor'''
            if self.args['learnable_alpha']:
                # update alpha
                alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.args['target_entropy']).detach())
                # print("alpha loss", alpha_loss.item())
                self.log_alpha_optim.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optim.step()

            #: trying BC for iteration other
            if t > 0:
                action_log_prob = log_prob
                
            else: #if t ==0 we have an experct action to BC our policy
                new_action = action
                action_log_prob = action_dist.log_prob(new_action)
            # self.pdb()
            # log_prob_check = any([x > -4 for x in action_log_prob])
            # if log_prob_check:
            #     print(f"max log prob actor update {max(action_log_prob)}")
            qv = torch.min(self.gather(self.q1(obs), new_action), self.gather(self.q2(obs), new_action))
            # self.pdb()
            check = action_dist.entropy().sum().item()
            if check < 1:
                print(check, end = "", flush=True)
            if self.entropy_beta > 0:
                # self.pdb()
                entropy_term = action_dist.entropy()
                actor_loss = (torch.exp(self.log_alpha) * action_log_prob - qv - self.entropy_beta*entropy_term).mean()
            else:
                actor_loss = (torch.exp(self.log_alpha) * action_log_prob - qv).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
        else:
            actor_loss = torch.Tensor([0])
            alpha_loss = torch.Tensor([0])
            check = action_dist.entropy().sum().item()

        return {
            "actor_loss": actor_loss.item(), 
            "alpha_loss": alpha_loss.item(), 
            "critic_loss_pre":critic_loss_pre, 
            "critic_loss":critic_loss.item(),
            "cql_update_mean_reward": reward.mean().item(),
            "entropy": check,
            # "from_epoch": np.mean(from_epoch)
        }

    def test_qvalues(self):
        # TODO: test t= 0 train buffer q functions
        return None

