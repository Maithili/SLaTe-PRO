import random
from copy import deepcopy
import numpy as np
import torch


class Baseline():
    def __init__(self):
        pass

    def extract(self, batch):
        self.dynamic_edges = batch['dynamic_edges_mask']
        self.evaluate_node = self.dynamic_edges.sum(-1) > 0
        self.static_nodes_mask = (torch.unsqueeze(self.dynamic_edges.sum(-1) == 0, dim=-1)).expand(-1,-1,self.evaluate_node.size(1))
        self.edges = batch['edges']
        self.nodes = batch['nodes']
        self.gt = batch['y_edges']
        self.time = batch['time']
    
    def step(self, batch, prev_context=None):
        self.extract(batch)
        result = deepcopy(self.run())
        result[self.dynamic_edges == 0] = self.edges[self.dynamic_edges == 0]
        result = result/(result.sum(dim=-1).unsqueeze(-1).repeat(1,1,self.edges.size()[-1])+1e-8)
        assert result.squeeze(-1).argmax(-1).size() == self.edges.squeeze(-1).argmax(-1).size(), f"{result.size()} == {self.edges.size()}"
        details = {'input':{'class':self.nodes.argmax(-1), 'location':self.edges.squeeze(-1).argmax(-1)}, 
            'output_probs':{'location': result}, 
            'gt':{'class':self.nodes.argmax(-1), 'location': self.gt.squeeze(-1).argmax(-1)}, 
            'losses':{'location': None}, 
            'output':{'class':self.nodes.argmax(-1), 'location': result.squeeze(-1).argmax(-1)}, 
            'evaluate_node':self.evaluate_node}
        return None, details, None



class StateTimeConditionedBaseline(Baseline):
    def __init__(self):
        super().__init__()

class TimeConditionedBaseline(Baseline):
    def __init__(self):
        super().__init__()


class LastSeen(StateTimeConditionedBaseline):
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        return self.edges

class StaticSemantic(TimeConditionedBaseline):
    def __init__(self, cooccurence_freq) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq

    def run(self):
        return self.cooccurence_freq.unsqueeze(0).repeat(self.edges.size()[0],1,1)

class LastSeenAndStaticSemantic(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.4) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def run(self):
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges

class LastSeenAndStaticSemantic2(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.6) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def run(self):
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges

class LastSeenAndStaticSemantic3(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.7) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def run(self):
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges

class LastSeenAndStaticSemantic4(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.8) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def run(self):
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges


class LastSeenAndStaticSemantic5(StateTimeConditionedBaseline):
    def __init__(self, cooccurence_freq, prob_change = 0.9) -> None:
        super().__init__()
        self.cooccurence_freq = cooccurence_freq
        self.prob_change = prob_change

    def run(self):
        next_edges = self.edges * (1-self.prob_change)
        next_edges += self.prob_change/next_edges.size()[-1]
        return self.cooccurence_freq * next_edges


class Fremen(TimeConditionedBaseline):
    def __init__(self, spectral_components):
        super().__init__()
        self.spectral_components = spectral_components
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        return prior.unsqueeze(0).repeat(self.edges.size()[0],1,1)

class FremenStateConditioned(StateTimeConditionedBaseline):
    def __init__(self, spectral_components, dt, time_decay=25):
        super().__init__()
        self.spectral_components = spectral_components
        self.decay_exponent = np.exp(-dt / time_decay)
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        posterior = prior + (self.edges-prior) * self.decay_exponent
        return posterior

class FremenStateConditioned1(StateTimeConditionedBaseline):
    def __init__(self, spectral_components, dt, time_decay=20):
        super().__init__()
        self.spectral_components = spectral_components
        self.decay_exponent = np.exp(-dt / time_decay)
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        posterior = prior + (self.edges-prior) * self.decay_exponent
        return posterior

class FremenStateConditioned2(StateTimeConditionedBaseline):
    def __init__(self, spectral_components, dt, time_decay=15):
        super().__init__()
        self.spectral_components = spectral_components
        self.decay_exponent = np.exp(-dt / time_decay)
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        posterior = prior + (self.edges-prior) * self.decay_exponent
        return posterior

class FremenStateConditioned3(StateTimeConditionedBaseline):
    def __init__(self, spectral_components, dt, time_decay=10):
        super().__init__()
        self.spectral_components = spectral_components
        self.decay_exponent = np.exp(-dt / time_decay)
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        posterior = prior + (self.edges-prior) * self.decay_exponent
        return posterior

class FremenStateConditioned4(StateTimeConditionedBaseline):
    def __init__(self, spectral_components, dt, time_decay=5):
        super().__init__()
        self.spectral_components = spectral_components
        self.decay_exponent = np.exp(-dt / time_decay)
    
    def run(self):
        prior = sum([2*spec['amplitude']*np.cos(2*np.pi*self.time/spec['period'] - spec['phase']) for spec in self.spectral_components])
        posterior = prior + (self.edges-prior) * self.decay_exponent
        return posterior


