import sys
print(sys.path)
import basis
from src.cost_function import UnitaryCostFunction
from src.optimizer import TemplateOptimizer
from src.sampler import SampleFunction
from multiprocessing import Pool

class Experiment():
    def __init__(self, basis: basis.VariationalTemplate, cost: UnitaryCostFunction, optimizer: TemplateOptimizer, sampler: SampleFunction):
        self.basis = basis
        self.cost = cost
        self.optimizer = optimizer
        self.sampler = sampler
    
    def cost_run_parallel(self):
        results = []
        with Pool() as p:
            results = p.map(self.optimizer.approximate_target_U, self.sampler)

    def approximate_run_parallel(self):
        with Pool() as p:
            results = p.map(self.optimizer.cost_target_U, self.sampler)

#test case to show whether Pool is working
if __name__ == "__main__":
    def f(x):
        return x+1
    with Pool() as p:
        print(p.map(f, [1,2,3]))
