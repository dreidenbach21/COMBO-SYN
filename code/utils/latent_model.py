import os
import torch
import sys
# sys.path.insert(0, '/home/yangk/dreidenbach/ChemLSO')
# sys.path.insert(0, '/home/yangk/dreidenbach/ChemLSO/hgraph2graph')
# sys.path.insert(0, os.getcwd()+'/MolecularTransformer')
import math, random
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from hgraph2graph.hgraph import *

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
#     tree_tensors = [make_tensor(x).long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
#     graph_tensors = [make_tensor(x).long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x, bad_idx = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab, clean = True)
    if len(bad_idx) == len(mol_batch):
        return None, bad_idx
    return to_numpy(x), bad_idx

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x
    
def tokenize_graph(smiles, vocab):
    return tensorize(smiles, vocab)

class LatentSpaceModel():
    
    def __init__(self, test = False):
        BASE = '/home/yangk/dreidenbach/ChemLSO'
        print("___Loading Latent Space___")
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', required=True)
        parser.add_argument('--vocab', required=True)
        parser.add_argument('--atom_vocab', default=common_atom_vocab)
        parser.add_argument('--save_dir', required=True)

        
        # if tiny:
        #     parser.add_argument('--load_model', default="{}/bb_syn/models/building_block_latent/model.ckpt.end.6800.19.557_best".format(BASE))
        # else:
        if test:
            parser.add_argument('--load_model', default="{}/bb_syn/models/mt_reactant_latent_hvae".format(BASE))
        # parser.add_argument('--load_model', default="/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space/combo_all_molecules_loss_1023")
        else:
            # parser.add_argument('--load_model', default="/home/yangk/dreidenbach/rl/molecule/code/hgraph2graph/ckpt/combo/model.ckpt.100000.1097.634")
            parser.add_argument('--load_model', default="/home/yangk/dreidenbach/rl/molecule/code/hgraph2graph/ckpt/combo_V2/model.ckpt.55000.224.691")
            

    
        parser.add_argument('--seed', type=int, default=7)

        parser.add_argument('--rnn_type', type=str, default='LSTM')
        parser.add_argument('--hidden_size', type=int, default=250)
        parser.add_argument('--embed_size', type=int, default=250)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--latent_size', type=int, default=32)
        parser.add_argument('--depthT', type=int, default=15)
        parser.add_argument('--depthG', type=int, default=15)
        parser.add_argument('--diterT', type=int, default=1)
        parser.add_argument('--diterG', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.0)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--clip_norm', type=float, default=5.0)
        parser.add_argument('--step_beta', type=float, default=0.001)
        parser.add_argument('--max_beta', type=float, default=1.0)
        parser.add_argument('--warmup', type=int, default=10000)
        parser.add_argument('--kl_anneal_iter', type=int, default=2000)

        parser.add_argument('--epoch', type=int, default=20)
        parser.add_argument('--anneal_rate', type=float, default=0.9)
        parser.add_argument('--anneal_iter', type=int, default=25000)
        parser.add_argument('--print_iter', type=int, default=50)
        parser.add_argument('--save_iter', type=int, default=5000)
        
        # if tiny:
        #     args = parser.parse_args(["--train", "{}/bb_syn/data/building_blocks.txt ".format(BASE), "--vocab", "{}/bb_syn/data/building_blocks_vocab.txt".format(BASE), "--save_dir", "{}/bb_syn/models/building_block_latent".format(BASE)])
        # else:
        if test:
            args = parser.parse_args(["--train", "{}/bb_syn/models/mt_and_reactants_molecules.txt".format(BASE), "--vocab", "{}/bb_syn/models/mt_and_reactants_vocab.txt".format(BASE), "--save_dir", "{}/bb_syn/results".format(BASE)])
        else:
            # args = parser.parse_args(["--train", "/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space/combo_all_molecules.txt", "--vocab", "/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space/combo_all_molecules_vocab.txt", "--save_dir", "/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space"])
            args = parser.parse_args(["--train", "/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space/combo_all_molecules_V2.txt", "--vocab", "/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space/combo_all_molecules_vocab_V2.txt", "--save_dir", "/home/yangk/dreidenbach/rl/molecule/model_ckpt/latent_space"])
    
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        
        vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
        args.vocab = PairVocab(vocab)

        model = HierVAE(args).cuda()
        print("HierVAE Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))


        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

        if args.load_model:
            print('continuing from checkpoint ' + args.load_model)
            model_state, optimizer_state, total_step, beta = torch.load(args.load_model)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            self.beta = beta
        self.model = model
        self.vocab = args.vocab
    
    def make_cuda(self, tensors):
        tree_tensors, graph_tensors = tensors
        make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
        tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
        graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
        return tree_tensors, graph_tensors

    # def encode(self, smiles, verbose = False):
    #     graphs, tensors, all_orders = tokenize_graph(smiles, self.vocab)
    #     tree_tensors, graph_tensors = tensors = self.make_cuda(tensors)
    #     root_vecs, tree_vecs, _, graph_vecs = self.model.encoder(tree_tensors, graph_tensors)
    #     init_root_vecs = root_vecs
    #     root_vecs, root_kl = self.model.rsample(root_vecs, self.model.R_mean, self.model.R_var, perturb = False)
    #     if verbose:
    #         return root_vecs, init_root_vecs 
    #     return root_vecs

    def encode(self, smiles):
        g_info, bad_idx = tokenize_graph(smiles, self.vocab)
        if g_info == None:
            return None, bad_idx
        _, tensors, _ = g_info
        tree_tensors, graph_tensors = tensors = self.make_cuda(tensors)
        root_vecs, _, _, _ = self.model.encoder(tree_tensors, graph_tensors)
        root_vecs = self.rsample2(root_vecs, self.model.R_mean, self.model.R_var, perturb = False)
        # del tree_tensors, graph_tensors
        return root_vecs, bad_idx

    def rsample2(self, z_vecs, W_mean, W_var, scale = 1.0, perturb=True):
        z_mean = W_mean(z_vecs)
        if not perturb:
            return z_mean
        batch_size = z_vecs.size(0)
        z_log_var = -torch.abs( W_var(z_vecs) )
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + scale*torch.exp(z_log_var / 2) * epsilon
        return z_vecs
    
    def sample(self, smiles, scale = 1.0):
        g_info, _ = tokenize_graph(smiles, self.vocab)
        graphs, tensors, all_orders = g_info
        tree_tensors, graph_tensors = tensors = self.make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.model.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = self.rsample(root_vecs, self.model.R_mean, self.model.R_var, scale, perturb = True)
        return root_vecs
    
    def rsample(self, z_vecs, W_mean, W_var, scale = 1.0, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + scale*torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss
    
    def learned_sample(self, root_vecs, noise, scale = 1.0):
        root_vecs, root_kl = self.force_sample(root_vecs, self.model.R_mean, self.model.R_var, noise, scale, perturb = True)
        return root_vecs, root_kl
    
    def combo_sample(self, z_mean, noise, scale = 1):
        return z_mean +  scale*noise 

    def check_encodable(self, smile):
        try:
            _, tensors, _ = tokenize_graph([smile], self.vocab)
            return True
        except:
            return False
            
    # def force_sample(self, z_vecs, W_mean, W_var, noise, scale = 1.0, perturb=True):
    #     batch_size = z_vecs.size(0)
    #     print(z_vecs.shape, z_vecs)
    #     print(W_mean)
    #     z_mean = W_mean(z_vecs) # this has error with torch 1.9.0 on gpu
    #     print(z_mean.shape, z_mean)
    #     z_log_var = -torch.abs( W_var(z_vecs) )
    #     kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
    #     epsilon = torch.randn_like(z_mean).cuda()
    #     if noise is None:
    #         z_vecs = z_mean + scale*torch.exp(z_log_var / 2) * epsilon 
    #     else:
    #         z_vecs = z_mean +  scale*noise # +scale*torch.exp(z_log_var / 2) * epsilon +
    #     return z_vecs, kl_loss

    
        
#     def sample_with_logprobs(self, smiles, noise = None, scale = 1.0):
#         graphs, tensors, orders = tokenize_graph(smiles, self.vocab)
#         tree_tensors, graph_tensors = tensors = make_cuda(tensors)

#         root_vecs, tree_vecs, _, graph_vecs = self.model.encoder(tree_tensors, graph_tensors)

#         root_vecs, root_kl = self.learned_sample(root_vecs,noise, scale)
#         kl_div = root_kl
        
#         reconstruction = self.decode(root_vecs)
        
#         loss, pieces, accs = self.model.decoder.forward_pass((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
#         return reconstruction, loss, kl_div.item(), accs
    
    def logprobs(self, all_smiles, all_root_vecs):
        # works only for batch size of 1 currently
        losses = []
        for smiles, root_vecs in zip(all_smiles, all_root_vecs):
            root_vecs = root_vecs.unsqueeze(0)
            smiles = [smiles]
            graphs, tensors, orders = tokenize_graph(smiles, self.vocab)
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            loss, pieces, accs = self.model.decoder.forward_pass((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
            losses.append(loss)
        return losses

    def decode(self, z):
        z = z.cuda()
        return self.model.decoder.decode((z, z, z), greedy=True, max_decode_step=150)


if __name__ == "__main__":
    D = LatentSpaceModel()
    print("Loaded")
    