import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42 
device_num = 0
# device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
device = 'cuda'+":"+str(device_num) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

dynamics_path = None

hidden_layer_size = 400
hidden_layers = 2
transition_layers = 4

transition_init_num = 7
transition_select_num = 5

real_data_ratio = 0.2 #0.01 #0.5

transition_batch_size = 256
policy_batch_size = 5 #1000#256
data_collection_per_epoch = 500 #50e3
buffer_size = 120e4
steps_per_epoch = 1000
max_epoch = 500

learnable_alpha =  True
transition_lr = 1e-3
actor_lr = 1e-4
critic_lr = 3e-4
target_entropy = None
discount = 0.99
soft_target_tau = 5e-3

num_samples = 10
learnable_beta = False
base_beta = 0.5
lagrange_thresh = 5
with_important_sampling = False #True

horizon = 5


oracle = "JNK3"
mt_path = ""
latent_space_path = ""
data_root = "/home/yangk/dreidenbach/rl/molecule"
discrete = False
cache_size = 1000000
scale_rewards = True
save_name = "cont_scale_test"
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