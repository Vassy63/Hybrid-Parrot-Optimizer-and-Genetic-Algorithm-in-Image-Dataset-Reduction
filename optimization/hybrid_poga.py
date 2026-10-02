import numpy as np
from optimization.genetic_algorithm import GeneticAlgorithm
from optimization.parrot_optimizer import ParrotOptimizer

def _init_population(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    ga = GeneticAlgorithm(config, dist_matrix, labels, num_classes)
    po = ParrotOptimizer(config, dist_matrix, labels, num_classes)
    if initial_population is not None:
        pop = initial_population.copy()
        fits = initial_fitness.copy() if initial_fitness is not None else ga.evaluate_population(pop)
    else:
        pop = ga.initialize_population()
        fits = ga.evaluate_population(pop)
    return ga, po, pop, fits

# =====================================================================
# 1. SEQUENTIAL HYBRID (S1, S2, S3, S4)
# =====================================================================

def run_s1_alternating_ga_po(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    """S1: Alternating GA -> PO -> GA -> PO ..."""
    print("--- [S1] Alternating Sequential: GA -> PO -> GA -> PO ---")
    ga, po, pop, fits = _init_population(config, dist_matrix, labels, num_classes, initial_population, initial_fitness)
    total_iters = config.total_iterations
    phase = max(1, getattr(config, 'phase_size', 10))
    history = []
    cur = 0
    turn_ga = True
    while cur < total_iters:
        steps = min(phase, total_iters - cur)
        if turn_ga:
            pop, fits, h = ga.run(pop, fits, steps=steps, start_gen=cur, total_gens=total_iters, disable_early_stop=True)
        else:
            _, _, h = po.run(pop, fits, steps=steps, start_iter=cur, total_iters=total_iters, disable_early_stop=True)
            pop, fits = po.population, po.fitness_scores
        history.extend(h)
        cur += steps
        turn_ga = not turn_ga
    best_idx = int(np.argmax(fits))
    print(f"Hoàn thành S1 | Best Fitness: {fits[best_idx]:.4f}")
    return pop[best_idx].copy(), float(fits[best_idx]), history

def run_s2_alternating_po_ga(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    """S2: Alternating PO -> GA -> PO -> GA ..."""
    print("--- [S2] Alternating Sequential: PO -> GA -> PO -> GA ---")
    ga, po, pop, fits = _init_population(config, dist_matrix, labels, num_classes, initial_population, initial_fitness)
    total_iters = config.total_iterations
    phase = max(1, getattr(config, 'phase_size', 10))
    history = []
    cur = 0
    turn_po = True
    while cur < total_iters:
        steps = min(phase, total_iters - cur)
        if turn_po:
            _, _, h = po.run(pop, fits, steps=steps, start_iter=cur, total_iters=total_iters, disable_early_stop=True)
            pop, fits = po.population, po.fitness_scores
        else:
            pop, fits, h = ga.run(pop, fits, steps=steps, start_gen=cur, total_gens=total_iters, disable_early_stop=True)
        history.extend(h)
        cur += steps
        turn_po = not turn_po
    best_idx = int(np.argmax(fits))
    print(f"Hoàn thành S2 | Best Fitness: {fits[best_idx]:.4f}")
    return pop[best_idx].copy(), float(fits[best_idx]), history

def run_s3_block_ga_po(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    """S3: Block Sequential GA(50%) -> PO(50%)"""
    print("--- [S3] Block Sequential: GA(50%) -> PO(50%) ---")
    ga, po, pop, fits = _init_population(config, dist_matrix, labels, num_classes, initial_population, initial_fitness)
    total_iters = config.total_iterations
    ga_steps = int(total_iters * getattr(config, 'split_ratio', 0.5))
    po_steps = total_iters - ga_steps
    pop, fits, ga_h = ga.run(pop, fits, steps=ga_steps, start_gen=0, total_gens=ga_steps, disable_early_stop=True)
    best_pos, best_fit, po_h = po.run(pop, fits, steps=po_steps, start_iter=0, total_iters=po_steps, disable_early_stop=True)
    print(f"Hoàn thành S3 | Best Fitness: {best_fit:.4f}")
    return best_pos, float(best_fit), ga_h + po_h

def run_s4_block_po_ga(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    """S4: Block Sequential PO(50%) -> GA(50%)"""
    print("--- [S4] Block Sequential: PO(50%) -> GA(50%) ---")
    ga, po, pop, fits = _init_population(config, dist_matrix, labels, num_classes, initial_population, initial_fitness)
    total_iters = config.total_iterations
    po_steps = int(total_iters * (1.0 - getattr(config, 'split_ratio', 0.5)))
    ga_steps = total_iters - po_steps
    _, _, po_h = po.run(pop, fits, steps=po_steps, start_iter=0, total_iters=po_steps, disable_early_stop=True)
    pop, fits, ga_h = ga.run(po.population, po.fitness_scores, steps=ga_steps, start_gen=0, total_gens=ga_steps, disable_early_stop=True)
    print(f"Hoàn thành S4 | Best Fitness: {fits[0]:.4f}")
    return pop[0].copy(), float(fits[0]), po_h + ga_h

# =====================================================================
# 2. PARALLEL HYBRID (P1: Independent Parallel + Merge)
# =====================================================================

def run_p1_parallel_merge(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    """P1: Independent Parallel (GA || PO) -> Merge -> Co-Refinement"""
    print("--- [P1] Parallel Hybrid: Independent GA(50%) || PO(50%) -> Merge ---")
    ga, po, pop, fits = _init_population(config, dist_matrix, labels, num_classes, initial_population, initial_fitness)
    total_iters = config.total_iterations
    par_steps = max(1, int(total_iters * 0.25))
    ref_steps = max(1, (total_iters - 2 * par_steps) // 2)

    pop_ga, fits_ga, h_ga = ga.run(pop.copy(), fits.copy(), steps=par_steps, start_gen=0, total_gens=par_steps, disable_early_stop=True)
    _, _, h_po = po.run(pop.copy(), fits.copy(), steps=par_steps, start_iter=0, total_iters=par_steps, disable_early_stop=True)
    pop_po, fits_po = po.population, po.fitness_scores

    merged_pop = np.vstack([pop_ga, pop_po])
    merged_fits = np.concatenate([fits_ga, fits_po])
    sorted_idx = np.argsort(merged_fits)[::-1][:len(pop)]
    final_pop = merged_pop[sorted_idx].copy()
    final_fits = merged_fits[sorted_idx].copy()

    final_pop, final_fits, h_ref_ga = ga.run(final_pop, final_fits, steps=ref_steps, start_gen=0, total_gens=ref_steps, disable_early_stop=True)
    best_pos, best_fit, h_ref_po = po.run(final_pop, final_fits, steps=ref_steps, start_iter=0, total_iters=ref_steps, disable_early_stop=True)

    history = []
    n_steps = min(len(h_ga), len(h_po))
    for i in range(n_steps):
        best_entry = h_ga[i] if h_ga[i]['total'] >= h_po[i]['total'] else h_po[i]
        history.append(best_entry)
        history.append(best_entry)
    history.extend(h_ref_ga)
    history.extend(h_ref_po)
    while len(history) < total_iters and history:
        history.append(history[-1])
    history = history[:total_iters]

    print(f"Hoàn thành P1 | Best Fitness sau Merge: {best_fit:.4f}")
    return best_pos.copy(), float(best_fit), history

# =====================================================================
# 3. COOPERATIVE HYBRID (C1: Bidirectional Elite Exchange)
# =====================================================================

def run_c1_cooperative_exchange(config, dist_matrix, labels, num_classes, initial_population=None, initial_fitness=None):
    """C1: Cooperative Bidirectional Elite Exchange giữa GA và PO"""
    print("--- [C1] Cooperative Hybrid: GA <-> PO Bidirectional Elite Exchange ---")
    ga, po, pop, fits = _init_population(config, dist_matrix, labels, num_classes, initial_population, initial_fitness)
    total_iters = config.total_iterations
    half_steps = max(1, total_iters // 2)
    interval = max(1, getattr(config, 'phase_size', 10) // 2)

    pop_ga, fits_ga = pop.copy(), fits.copy()
    pop_po, fits_po = pop.copy(), fits.copy()

    history = []
    cur = 0
    while cur < half_steps:
        steps = min(interval, half_steps - cur)
        pop_ga, fits_ga, h_ga = ga.run(pop_ga, fits_ga, steps=steps, start_gen=cur, total_gens=half_steps, disable_early_stop=True)
        _, _, h_po = po.run(pop_po, fits_po, steps=steps, start_iter=cur, total_iters=half_steps, disable_early_stop=True)
        pop_po, fits_po = po.population, po.fitness_scores

        for i in range(min(len(h_ga), len(h_po))):
            best_entry = h_ga[i] if h_ga[i]['total'] >= h_po[i]['total'] else h_po[i]
            history.append(best_entry)
            history.append(best_entry)

        # Bidirectional Elite Exchange có kiểm soát trùng lặp (Duplicate Control)
        merged_pop = np.vstack([pop_ga, pop_po])
        merged_fits = np.concatenate([fits_ga, fits_po])
        unique_pop, unique_idx = np.unique(merged_pop, axis=0, return_index=True)
        unique_fits = merged_fits[unique_idx]
        s_idx = np.argsort(unique_fits)[::-1]
        top_pop = unique_pop[s_idx[:len(pop)]].copy()
        top_fits = unique_fits[s_idx[:len(pop)]].copy()
        while len(top_pop) < len(pop):
            top_pop = np.vstack([top_pop, top_pop[0]])
            top_fits = np.append(top_fits, top_fits[0])
        pop_ga, fits_ga = top_pop.copy(), top_fits.copy()
        pop_po, fits_po = top_pop.copy(), top_fits.copy()

        cur += steps

    while len(history) < total_iters and history:
        history.append(history[-1])
    history = history[:total_iters]

    best_pos, best_fit = pop_po[0].copy(), float(fits_po[0])
    print(f"Hoàn thành C1 | Best Fitness: {best_fit:.4f}")
    return best_pos, best_fit, history

# =====================================================================
# 4. BASELINES (B1: Random Selection, B2: Stratified Random)
# =====================================================================

def random_selection(N: int, target_size: int, seed: int = None) -> np.ndarray:
    """B1: Chọn ngẫu nhiên đúng target_size mẫu."""
    rng = np.random.RandomState(seed)
    target_size = max(2, min(N, int(target_size)))
    sol = np.zeros(N, dtype=int)
    sol[rng.choice(N, size=target_size, replace=False)] = 1
    return sol

def stratified_random(labels: np.ndarray, target_size: int, seed: int = None) -> np.ndarray:
    """B2: Chọn ngẫu nhiên phân tầng bảo toàn tỷ lệ từng lớp."""
    rng = np.random.RandomState(seed)
    N = len(labels)
    target_size = max(2, min(N, int(target_size)))
    sol = np.zeros(N, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    selected = []
    for c, cnt in zip(classes, counts):
        c_idx = np.where(labels == c)[0]
        n_c = max(1, int(round(target_size * (cnt / N))))
        n_c = min(n_c, len(c_idx))
        selected.extend(rng.choice(c_idx, size=n_c, replace=False).tolist())
    if len(selected) > target_size:
        selected = rng.choice(selected, size=target_size, replace=False).tolist()
    elif len(selected) < target_size:
        rem = list(set(range(N)) - set(selected))
        selected.extend(rng.choice(rem, size=target_size - len(selected), replace=False).tolist())
    sol[selected] = 1
    return sol
