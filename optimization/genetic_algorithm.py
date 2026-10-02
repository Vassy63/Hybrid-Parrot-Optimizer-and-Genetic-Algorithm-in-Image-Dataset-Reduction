import numpy as np
import concurrent.futures
from tqdm import tqdm
from optimization.fitness import fitness, compute_raw_fitness, repair_solution

class GeneticAlgorithm:
    def __init__(self, config, dist_matrix, labels, num_classes):
        self.config = config.ga
        self.fitness_config = config.fitness
        self.dist_matrix = dist_matrix
        self.labels = labels
        self.num_classes = num_classes
        self.N = dist_matrix.shape[0]
        
    def _repair(self, ind):
        return repair_solution(ind)

    def initialize_population(self):
        raw_pop = (np.random.rand(self.config.population_size, self.N) < self.config.init_ratio).astype(int)
        return np.array([self._repair(ind) for ind in raw_pop])
        
    def evaluate_population(self, population):
        return np.array([
            fitness(ind, self.dist_matrix, self.labels, self.num_classes, self.fitness_config)
            for ind in population
        ])
        
    def tournament_selection(self, population, fitness_scores):
        selected_indices = np.random.choice(
            self.config.population_size, 
            size=self.config.tournament_size, 
            replace=False
        )
        best_idx = selected_indices[np.argmax(fitness_scores[selected_indices])]
        return population[best_idx].copy()
        
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.config.crossover_rate:
            mask = (np.random.rand(self.N) < 0.5).astype(int)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()
        
    def mutate(self, individual, current_gen, total_gens=None):
        p_max = self.config.mutation_rate_max
        p_min = self.config.mutation_rate_min
        t_gens = max(1, total_gens if total_gens is not None else self.config.generations)
        p_mut = max(p_min, p_max - (p_max - p_min) * (current_gen / t_gens))
        
        r = float(np.mean(individual))
        prob = p_mut * np.where(individual == 1, 1.0 - r, r)
        mask = (np.random.rand(self.N) < prob).astype(int)
        individual = np.where(mask, 1 - individual, individual)
        return self._repair(individual)
        
    def run(self, initial_population=None, initial_fitness=None, steps=None, start_gen=0, total_gens=None, disable_early_stop=False):
        n_steps = steps if steps is not None else self.config.generations
        t_gens = total_gens if total_gens is not None else n_steps
        if initial_population is not None:
            population = initial_population.copy()
            fitness_scores = initial_fitness.copy() if initial_fitness is not None else self.evaluate_population(population)
        else:
            population = self.initialize_population()
            fitness_scores = self.evaluate_population(population)
        
        history = []
        best_overall_fitness = float(np.max(fitness_scores))
        stagnation_count = 0
        
        for step in tqdm(range(n_steps), desc="GA Generations", leave=False):
            gen = start_gen + step
            new_population = []
            
            sorted_idx = np.argsort(fitness_scores)[::-1]
            for i in range(self.config.elitism_count):
                new_population.append(population[sorted_idx[i]].copy())
                
            while len(new_population) < self.config.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, gen, t_gens)
                child2 = self.mutate(child2, gen, t_gens)
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
                    
            population = np.array(new_population)
            fitness_scores = self.evaluate_population(population)
            
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_sol = population[best_idx]
            raw_fit = compute_raw_fitness(best_sol, self.dist_matrix, self.labels, self.num_classes)
            
            history.append({
                'total': float(best_fitness),
                'div': float(raw_fit[0]),
                'cov': float(raw_fit[1]),
                'bal': float(raw_fit[2]),
                'com': float(raw_fit[3])
            })
            
            if best_fitness > best_overall_fitness + 1e-6:
                best_overall_fitness = best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            if not disable_early_stop and stagnation_count >= getattr(self.config, 'early_stopping_patience', 15):
                break
            
        sorted_idx = np.argsort(fitness_scores)[::-1]
        population = population[sorted_idx]
        fitness_scores = fitness_scores[sorted_idx]
        
        return population, fitness_scores, history
