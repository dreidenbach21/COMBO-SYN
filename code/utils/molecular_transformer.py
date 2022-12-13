import os
import torch
import sys
# sys.path.insert(0, '/home/yangk/dreidenbach/ChemLSO')
# sys.path.insert(0, '/home/yangk/dreidenbach/ChemLSO/MolecularTransformer')
import argparse
import re
import numpy as np
from collections import OrderedDict
try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.error')
except:
    print('Warning: molecule dependencies not installed; install if running molecule exps')
# import chemprop # had to manually update to 1.4.0
from MolecularTransformer.translate import * # from MolecularTransformer
import torch.multiprocessing as tmp

def detokenize(x):
        return ''.join(x.strip().split()).split('.')

def smiles_to_rdkit_mol(smiles):
  """Convert smiles into rdkit's mol (molecule) format. 
  Args: 
    smiles: str, SMILES string. 
  Returns:
    mol: rdkit.Chem.rdchem.Mol
  """
  mol = Chem.MolFromSmiles(smiles)
  #  Sanitization check (detects invalid valence)
  if mol is not None:
    try:
        Chem.SanitizeMol(mol)
        return 1
    except ValueError:
        return None
    
def sift(rxn_tuple):
        ab, Cs = rxn_tuple
#         smiles = [detokenize(pro)[0] for pro in Cs if pro != '' and pro != None]
        smiles = []
        for pro in Cs:
            if pro != '' and pro != None:
                opts = detokenize(pro)
                if len(opts) > 1:
                    lg = [len(x) for x in opts]
                    smiles.append(opts[np.argmax(lg)])
                else:
                    smiles.append(opts[0])
        good_smiles = [smi for smi in smiles if smiles_to_rdkit_mol(smi) != None]
        if len(good_smiles) == 0:
            good_smiles = detokenize(ab.split(' . ')[0])
            good_smiles.append("NO RXN")
            print("No RXN for stuck = ", good_smiles)
        return ab, good_smiles
    
class MolecularTransformer():
    def __init__(self, property_name = 'JNK3',
                 n_best = 1,
                 weight_path = "/home/yangk/dreidenbach/rl/molecule/model_ckpt/mt/dags_mt_weights.pt",
                 cache_size = 10240,
                 num_threads = 32,
                 parallelize = True):
        
#         assert property_name in {'JNK3', 'QED', 'SA', 'DRD2', 'GSK3B', 'LOGP'}
        self.num_threads = num_threads
        self.cache_size = cache_size
        
        parser = argparse.ArgumentParser()
        onmt.opts.add_md_help_argument(parser)
        onmt.opts.translate_opts(parser)
        
        model_path = weight_path
        opt = parser.parse_args(args=["-model", model_path, "-src", None, "-batch_size", "300","-replace_unk", "-max_length", "500", "-fast", "-gpu", "1", "-n_best", "{}".format(n_best)])
        self.model = build_translator(opt, report_score=False, out_file = None)
        # self.cache = OrderedDict()
        self.parallelize = parallelize
        
    def tokenize(self, smi):
        """
        Tokenize a SMILES molecule or reaction. From Molecular Transformer
        """
#         import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        try:
            assert smi == ''.join(tokens)
        except:
            print("\n\n\n TOKENIZATION ERROR: ", smi, tokens, "\n\n\n")
        return ' '.join(tokens)
    
    def detokenize(self, x):
        return ''.join(x.strip().split()).split('.')
    
#     def pair_rxn(self, arg):
#         a,b = arg
#         return "{} . {}".format(self.tokenize(a), self.tokenize(b))
    def all_detokenize(self, x):
        return [self.detokenize(y)[0] for y in x]
    
    def react(self, states, actions):
        A = states
        B = actions
        pair_rxn = lambda a,b: "{} . {}".format(self.tokenize(a), self.tokenize(b))
        reaction_feed = []
        
        for stuck, reactant in zip(A, B):
            paired = pair_rxn(stuck, reactant)
            reaction_feed.append(paired)
            
        try:
            mt_scores, products = self.model.translate(src_data_iter=reaction_feed, batch_size = 500)
            return mt_scores, [self.all_detokenize(x) for x in products]
        except Exception as e:
            print("Failed to React: returning stucks ERROR:", e)
            # return None, [[x] for x in A]
            assert(1 == 0)
            return None, [['DOG'] for x in A]
        
         
#         rxn_tuples = list(zip(reaction_feed, products))
#         if self.parallelize:
#             with tmp.Pool(self.num_threads) as pool:
#                 sift_results = pool.map(sift, rxn_tuples)
#         else:
#             sift_results = [sift(x) for x in rxn_tuples]
        
#         bad_apples = []

#         final_products = []
#         for ab, good_smiles in sift_results:
#             if good_smiles[-1] == "NO RXN":
#                 bad_apples.append(ab)
#                 good_smiles = good_smiles[:-1]
#             final_products.append(good_smiles)

#         return final_products

if __name__ == "__main__":
    D = MolecularTransformer()
    print("Loaded")