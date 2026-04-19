"""Population-based evolutionary policy tuner for fuzzy controller weights."""
import random
from config import (
    EVOLUTION_POPULATION_SIZE,
    EVOLUTION_ELITE_COUNT,
    EVOLUTION_MUTATION_SIGMA,
    EVOLUTION_MUTATION_PROB,
    EVOLUTION_WEIGHT_MIN,
    EVOLUTION_WEIGHT_MAX,
)


class EvolutionaryPolicyTuner:
    def __init__(self):
        self.keys = ["eat", "sleep", "socialize", "work", "flee", "guard"]
        self.population_size = max(4, int(EVOLUTION_POPULATION_SIZE))
        self.elite_count = max(1, min(int(EVOLUTION_ELITE_COUNT), self.population_size - 1))
        self.mutation_sigma = float(EVOLUTION_MUTATION_SIGMA)
        self.mutation_prob = float(EVOLUTION_MUTATION_PROB)
        self.weight_min = float(EVOLUTION_WEIGHT_MIN)
        self.weight_max = float(EVOLUTION_WEIGHT_MAX)

        base = {k: 1.0 for k in self.keys}
        self.population = [self._mutate(base, heavy=False) for _ in range(self.population_size)]
        self.population[0] = dict(base)
        self.fitness = [None] * self.population_size
        self.eval_index = 0
        self.generation = 0
        self.best_weights = dict(base)
        self.best_score = float("-inf")

    def step(self, score_or_metrics, controller):
        score = self._to_score(score_or_metrics)

        # Record fitness for the currently tested individual.
        if 0 <= self.eval_index < len(self.population):
            self.fitness[self.eval_index] = score
            if score > self.best_score:
                self.best_score = score
                self.best_weights = dict(self.population[self.eval_index])

        # Move evaluation pointer.
        self.eval_index += 1

        # Finished evaluating this generation -> evolve population.
        if self.eval_index >= self.population_size:
            self._evolve_population()
            self.eval_index = 0

        # Apply next candidate policy to controller.
        controller.set_weights(self.population[self.eval_index])

    def _to_score(self, score_or_metrics):
        if isinstance(score_or_metrics, dict):
            m = score_or_metrics
            social = float(m.get("social_stability", 0.0))
            trust = float(m.get("avg_trust", 0.0))
            mood = float(m.get("avg_mood", 0.0))
            conflict = float(m.get("conflict_rate", 0.0))
            latency = float(m.get("chat_latency", 0.0))
            latency_term = 1.0 - min(1.0, latency / 6.0)
            return (
                0.36 * social
                + 0.24 * trust
                + 0.20 * mood
                + 0.12 * (1.0 - conflict)
                + 0.08 * latency_term
            )
        return float(score_or_metrics)

    def _evolve_population(self):
        scored = []
        for idx, weights in enumerate(self.population):
            fit = self.fitness[idx]
            if fit is None:
                fit = float("-inf")
            scored.append((fit, weights))
        scored.sort(key=lambda item: item[0], reverse=True)

        elites = [dict(w) for _, w in scored[:self.elite_count]]
        parent_pool = [dict(w) for _, w in scored[: max(2, self.population_size // 2)]]

        new_population = elites[:]
        while len(new_population) < self.population_size:
            p1 = random.choice(parent_pool)
            p2 = random.choice(parent_pool)
            child = self._crossover(p1, p2)
            child = self._mutate(child, heavy=False)
            new_population.append(child)

        # Keep best known policy reachable for stability.
        if self.best_weights:
            new_population[-1] = self._mutate(self.best_weights, heavy=False)

        self.population = new_population
        self.fitness = [None] * self.population_size
        self.generation += 1

    def _crossover(self, a, b):
        child = {}
        for k in self.keys:
            if random.random() < 0.5:
                child[k] = float(a[k])
            else:
                child[k] = float(b[k])
            if random.random() < 0.25:
                child[k] = (float(a[k]) + float(b[k])) / 2.0
        return child

    def _mutate(self, base, heavy=False):
        cand = dict(base)
        sigma = self.mutation_sigma * (1.8 if heavy else 1.0)
        for k in self.keys:
            if random.random() < self.mutation_prob:
                cand[k] = float(cand[k]) + random.gauss(0.0, sigma)
            cand[k] = max(self.weight_min, min(self.weight_max, float(cand[k])))
        return cand
