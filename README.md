# CLAQ

Research code for `Cost- and Leakage-Constrained Adaptive Concept Querying`.

Current notebooks:

- `notebooks/bias_in_bios_dataset_creation.ipynb`
- `notebooks/bias_in_bios_concept_qa_training.ipynb`
- `notebooks/bias_in_bios_leakage_experiment.ipynb`
- `notebooks/bias_in_bios_cost_experiment.ipynb`
- `notebooks/bias_in_bios_joint_experiment.ipynb`
- `notebooks/celeba_leakage_experiment.ipynb`
- `notebooks/celeba_cost_experiment.ipynb`
- `notebooks/celeba_joint_experiment.ipynb`
- `notebooks/cifar10_experiment.ipynb`

Install with:

```bash
pip install -e .
```

For Conda, create an environment with Python 3.10+ first, then run the same install command inside it.

## Experiment artifacts

- Plotting and tables require only the relevant files in `artifacts/runs/`.
- Evaluation without retraining additionally requires the per-run policy checkpoints in
  `artifacts/models/`.
- Full retraining does not require policy checkpoints. It requires the dataset, the
  Concept-QA checkpoint, and the code.
- Deterministic hard Concept-QA outputs are cached under `artifacts/concept_answers/`.
  Copying these small caches avoids repeating CLIP and Concept-QA inference.

Policy checkpoints are stored separately by condition and seed. Completed runs are
loaded automatically unless a notebook's `force_retrain` or `FORCE_RETRAIN` option is
enabled.
