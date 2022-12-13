import cma

class MoleculeEnv:
    def __init__(self, stuck, mean, iterations = 100, gamma = 1, std = 1):
        self.stuck = stuck
        assert self.stuck != None
        print(self.stuck)
        self._seed = 0
        self.max_episode_steps = 1
        self.reset()
        self.told = None
        
        self.num_evals = 0
        self.iterations = iterations

        self.gamma = gamma
        self.cmaes_sigma_mult = std
        self.es = cma.CMAEvolutionStrategy(mean, self.cmaes_sigma_mult, {'maxfevals': self.iterations})
        
        self.all_samples = []
        self.all_fX = []
        self.all_reactants = []
        self.all_products = []
        
    def stick(self, smi):
        self.stuck = smi
        
    def seed(self, s):
        self._seed = s # unused though
        return
    
    def reset(self, seed=None):
        # seed not used
        self.t = 0

    def _get_obs(self):
        return None
        