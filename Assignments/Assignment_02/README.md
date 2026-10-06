# Assignment 2 — Simulated Driving Agent Behavior

**Tata Technologies TechPulse Applied AI/ML Laboratory**

## 1. Objective
Build an autonomous self-driving agent that learns to navigate a 1-D multi-lane environment and avoid obstacles using a **Genetic Algorithm (GA)** in Python.

## 2. Environment & Concept
- **Lane Width:** 5 parallel lanes (positions 0, 1, 2, 3, 4).
- **Episode Length:** 40 timesteps.
- **Sensor Range:** 3 timesteps lookahead in current lane (8 binary obstacle states: 0 to 7).
- **Agent Actions:** `-1` (Move Left), `0` (Stay in Lane), `+1` (Move Right).

## 3. Genome Representation
- **Lookup Table Policy:** Maps `(lane_position, sensed_obstacle_pattern) -> action`.
- **Total Genes:** 5 lane positions × 8 sensor patterns = **40 genes**.

## 4. Training & Fitness Configuration
- **Fitness Metric:** Average survival time across a suite of 20 fixed training seeds (`TRAINING_SEEDS = list(range(100, 120))`).
- **Population Size:** 80 agents.
- **Generations:** 40 generations.
- **Elitism:** Top 12 agents preserved per generation.
- **Crossover:** Single-point crossover between elite parents.
- **Mutation Rate:** ~3% gene mutation probability.

## 5. Unseen Test Evaluation
- Evaluated on a separate set of 5 unseen obstacle seeds (`[777, 888, 999, 1234, 5678]`).

## 6. How to Run
1. Open terminal and navigate to `Assignments/Assignment_02/`:
   ```bash
   cd Assignments/Assignment_02
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook TTL_Assignment_02.ipynb
   ```
3. Run all cells sequentially (**Cell -> Run All**).

## 7. Expected Output
- Generational progress output showing average survival time across 20 training seeds.
- Line chart of **Best Average Survival Time vs. Generation**.
- Unseen test evaluation report listing survival steps per unseen seed.

## 8. Conclusion
The Genetic Algorithm successfully evolves driving actions over 40 generations. Evaluating on unseen seeds provides an honest assessment of generalization limits for small lookup-table policies.
