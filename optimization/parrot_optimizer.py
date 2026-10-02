import numpy as np
import math
import concurrent.futures
from tqdm import tqdm
from optimization.fitness import fitness, compute_raw_fitness, repair_solution

class ParrotOptimizer:
    def __init__(self, config, dist_matrix, labels, num_classes):
        self.config = config.po
        self.fitness_config = config.fitness
        self.dist_matrix = dist_matrix
        self.labels = labels
        self.num_classes = num_classes
        self.N = dist_matrix.shape[0]
        self.beta = self.config.levy_beta
        
        num = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        self.sigma = (num / den) ** (1 / self.beta)
        
    def levy_flight(self, dim):
        u = np.random.normal(0, self.sigma, dim)
        v = np.random.normal(0, 1, dim)
        v = np.where(np.abs(v) < 1e-8, 1e-8, v)
        return u / (np.abs(v) ** (1 / self.beta))
        
    def evaluate(self, individual):
        return fitness(
            individual, self.dist_matrix, self.labels, 
            self.num_classes, self.fitness_config
        )
        
    def v_shaped_transfer(self, delta_x):
        return np.abs(np.tanh(delta_x))
        
    def binarize(self, x_current, x_new):
        delta_x = x_new - x_current
        v_prob = self.v_shaped_transfer(delta_x)
        mask = (np.random.rand(self.N) < v_prob).astype(int)
        x_binary = np.where(mask, 1 - x_current, x_current)
        return repair_solution(x_binary)

    def generate_candidates(self, population, gbest_pos, t, T_max):
        mean_pos = np.mean(population, axis=0)
        candidates = []
        pop_size = len(population)
        
        for i in range(pop_size):
            x_current = population[i].copy()
            St = np.random.randint(1, 5)
            
            if St == 1:
                levy = self.levy_flight(self.N)
                rand_val = np.random.rand()
                term2 = rand_val * mean_pos * ((1 - t / T_max) ** (2 * t / T_max))
                x_new_continuous = (x_current - gbest_pos) * levy + term2
            elif St == 2:
                levy = self.levy_flight(self.N)
                randn_val = np.random.randn()
                term2 = randn_val * (1 - t / T_max) * np.ones(self.N)
                x_new_continuous = x_current + gbest_pos * levy + term2
            elif St == 3:
                alpha = np.random.rand() / 5
                H = np.random.rand()
                if H < 0.5:
                    x_new_continuous = x_current + alpha * (1 - t / T_max) * (x_current - mean_pos)
                else:
                    j = np.random.randint(1, pop_size + 1)
                    rand_val = max(np.random.rand(), 1e-8)
                    term2 = np.exp(-j / (rand_val * T_max))
                    x_new_continuous = x_current + alpha * (1 - t / T_max) * term2
            else:
                rand_val = np.random.rand()
                theta = np.random.rand() * np.pi
                term1 = rand_val * np.cos(np.pi * t / (2 * T_max)) * (gbest_pos - x_current)
                term2 = np.cos(theta) * ((t / T_max) ** (2 / T_max)) * (x_current - gbest_pos)
                x_new_continuous = x_current + term1 - term2
            
            x_new_binary = self.binarize(x_current, x_new_continuous)
            candidates.append(x_new_binary)
            
        return candidates

    def run(self, initial_population, initial_fitness=None, steps=None, start_iter=0, total_iters=None, disable_early_stop=False):
        n_steps = steps if steps is not None else self.config.iterations
        T_max = max(1, total_iters if total_iters is not None else n_steps)
        
        population = initial_population.copy()
        if initial_fitness is not None:
            fitness_scores = initial_fitness.copy()
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                fitness_scores = np.array(list(executor.map(self.evaluate, population)))
        
        best_idx = np.argmax(fitness_scores)
        gbest_pos = population[best_idx].copy()
        gbest_fitness = fitness_scores[best_idx]
        
        history = []
        no_improve_count = 0
        
        for step in tqdm(range(1, n_steps + 1), desc="PO Iterations", leave=False):
            t = min(T_max, start_iter + step)
            candidates = self.generate_candidates(population, gbest_pos, t, T_max)
                
            with concurrent.futures.ThreadPoolExecutor() as executor:
                new_fitness_scores = list(executor.map(self.evaluate, candidates))
                
            for i in range(len(population)):
                if new_fitness_scores[i] > fitness_scores[i]:
                    population[i] = candidates[i]
                    fitness_scores[i] = new_fitness_scores[i]
                    
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > gbest_fitness + 1e-6:
                gbest_pos = population[current_best_idx].copy()
                gbest_fitness = fitness_scores[current_best_idx]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            raw_fit = compute_raw_fitness(gbest_pos, self.dist_matrix, self.labels, self.num_classes)
            history.append({
                'total': float(gbest_fitness),
                'div': float(raw_fit[0]),
                'cov': float(raw_fit[1]),
                'bal': float(raw_fit[2]),
                'com': float(raw_fit[3])
            })
            
            if not disable_early_stop and no_improve_count >= self.config.early_stopping_patience:
                break
                
        sorted_idx = np.argsort(fitness_scores)[::-1]
        self.population = population[sorted_idx].copy()
        self.fitness_scores = fitness_scores[sorted_idx].copy()
        
        return gbest_pos, gbest_fitness, history
