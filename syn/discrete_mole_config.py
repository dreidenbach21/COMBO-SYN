import torch
from offlinerl.utils.exp import select_free_cuda

task = "Synthesis"
task_data_type = "low"
task_train_num = 99

seed = 42 
device_num = 0
# device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
device = 'cuda'+":"+str(device_num) if torch.cuda.is_available() else 'cpu'
device2 = device #'cuda:1'
obs_shape = None
act_shape = None
max_action = None

dynamics_path = None

hidden_layer_size = 6000 #400
hidden_layers = 3 #2
# transition_layers = 4

# transition_init_num = 7
# transition_select_num = 5

real_data_ratio = 0.1 #0.2

# transition_batch_size = 256
policy_batch_size = 10 #5 # transitions sampled at a time for cql update
data_collection_per_epoch = 300 #500 #50e3 # transitions grabbed per epoch
buffer_size = 120e4
steps_per_epoch = 1500 #1000 #number of cql updates
max_epoch = 500

learnable_alpha =  True
transition_lr = 1e-3
actor_lr = 1e-4
critic_lr = 3e-4
target_entropy = None
discount = 2/3 #0.99 #2/3 for n = 1/(1-gamma) and n = 3
soft_target_tau = 5e-3

num_samples = 10
learnable_beta = False
base_beta = 0.5
lagrange_thresh = 5
with_important_sampling = True

horizon = 3


oracle = "DRD2" #"JNK3" DRD2
mt_path = ""
latent_space_path = ""
data_root = "/home/yangk/dreidenbach/rl/molecule"
discrete = True
cache_size = 10000000
scale_rewards = 2
save_name = "discrete_horizon_3_max_q_10_scale_2_explore_DRD2_discount_23_iter_clean_buffer_qupdate_yes_done_entropy_decay"
data_collection_samples = 5
max_q_back_up = 9
inject = False
pretrain_policy_path = "/home/yangk/dreidenbach/rl/molecule/model_ckpt/pre_train_policy_imitation_14.pt"
use_pretrain_policy = False
use_eps_greedy = True
eps = 0.3
use_learned_actor = True
learning_start_epoch = 3
clean_up = True
entropy_beta_decay = True
no_done = False
#tune
params_tune = {
    "horzion" : {"type" : "discrete", "value": [1, 5]},
    "base_beta" : {"type" : "discrete", "value": [0.5, 1, 5]},
    "with_important_sampling" : {"type" : "discrete", "value": [True, False]},
}

#tune
grid_tune = {
    "horizon" : [1, 5],
    "base_beta" : [0.5, 1, 5],
    "with_important_sampling" : [True, False],
}