
# Scenario Generation in URBAN-SIM

We provide two distinct pipelines for scenario generation in URBAN-SIM, each with different strengths suited for various training and evaluation goals.

---

## 1. Random Object Placement

This pipeline places objects such as static obstacles, buildings, and pedestrians at random locations within a predefined region. It supports parameters like object density and random seeds to ensure controlled variability.

### Characteristics:
- Lightweight and fast
- Suitable for large-scale multi-environment training
- High randomness; less spatial consistency

<div align="center">
  <div style="display: inline-block; text-align: center; margin: 0 1%">
    <img src="./assets/random_large.png" width="95%"><br>
  </div>
</div>

---

## 2. Procedural Generation (PG)

The PG pipeline generates structured environments using rule-based or programmatic layout generation. This allows for replicable scenes, progressive difficulty, and layout logic for urban simulations.

### Characteristics:
- Structured and repeatable
- Supports curriculum learning
- Better for generalization and benchmarking

<div align="center">
  <div style="display: inline-block; text-align: center; margin: 0 1%">
    <img src="./assets/pg_large.png" width="95%"><br>
  </div>
</div>

---

## Comparisons

| Feature                  | Random Placement      | Procedural Generation (PG) |
|--------------------------|-----------------------|-----------------------------|
| Speed                    | ✅ Fast               | ⚠️ Moderate                 |
| Spatial Structure        | ❌ None               | ✅ Present                  |
| Curriculum-friendly      | ✅ Yes                 | ✅ Yes                      |
| Use case                 | Pretraining | Benchmarking, Finetuning   |

---

## Examples
### 1. Random Object Placement
```bash
python urbansim/envs/separate_envs/random_env.py --enable_cameras --num_envs 16 --use_async
```

<div align="center">
  <div style="display: inline-block; text-align: center; margin: 0 1%">
    <img src="./assets/random_detail.png" width="100%"><br>
    <p>Random Placement</p>
  </div>
</div>

### 2. Procedural  Generation
```bash
python urbansim/envs/separate_envs/pg_env.py --enable_cameras --num_envs 16 --use_async
```
<div align="center">
  <div style="display: inline-block; text-align: center; margin: 0 1%">
    <img src="./assets/pg_detail.png" width="100%"><br>
    <p>Procedural Generation</p>
  </div>
</div>

