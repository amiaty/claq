# CLAQ

**CLAQ: Adaptive Concept Querying Under Cost and Leakage Constraints**

CLAQ is a reference research implementation for adaptive concept acquisition.
At each round, an acquisition policy selects one feasible concept query from
the current transcript. The implementation supports target prediction,
query-cost accounting, transcript-leakage regularization, pathwise empirical
constraints, proxy screening, optional outcome-fairness penalties, and
held-out leakage evaluation.

This repository is intended for scientific evaluation and reproducibility. It
does not provide a formal privacy certificate or a production deployment
framework.

## Method overview

For transcript \(h_k\) and feasible query \(q\), the reference acquisition
score is

\[
\widehat U_k(q;h_k)
=
\widehat{\mathsf I}_Y(q;h_k)
-
\lambda_s\widehat{\mathsf I}_S(q;h_k)
-
\lambda_c\mathsf C(q),
\]

where

- \(\widehat{\mathsf I}_Y(q;h_k)\) estimates the label information associated
  with executing \(q\) after \(h_k\);
- \(\widehat{\mathsf I}_S(q;h_k)\) estimates conditional sensitive information
  given the task label;
- \(\mathsf C(q)\) is the query cost;
- \(\lambda_s,\lambda_c\ge 0\) are weighting coefficients.

The implementation distinguishes four quantities that should not be
interchanged:

1. **Terminal-transcript leakage:** the population quantity
   \(I(S;\mathcal H_T\mid Y)\).
2. **Pathwise empirical leakage account:** a sum of estimated one-step
   conditional-information terms used for operational masking.
3. **Designated-query rate:** the fraction of executed queries belonging to a
   pre-specified query set.
4. **Probe leakage:** a probe-class-dependent held-out diagnostic.

Only the first quantity is the population leakage criterion in the paper.
The remaining quantities are empirical diagnostics or operational
constraints.

## Installation

Python 3.10 or later is required. A fresh virtual environment is recommended.

### Core package

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### Dataset-specific dependencies

CelebA and CIFAR-10 exporters:

```bash
python -m pip install -e ".[vision]"
```

Bias in Bios exporter:

```bash
python -m pip install -e ".[text]"
```

All optional dependencies and test tools:

```bash
python -m pip install -e ".[all,dev]"
```

Verify the installation:

```bash
python scripts/check_installation.py
python -m pytest -q
```

The equivalent Make targets are:

```bash
make check
make test
```

## Quickstart

Run the complete pipeline on a deterministic synthetic dataset:

```bash
python scripts/quickstart.py
```

The command creates a canonical response archive, trains a small model,
evaluates the selected checkpoint, and writes a summary to

```text
/tmp/claq_quickstart/run/summary.json
```

Specify a different output directory with:

```bash
python scripts/quickstart.py \
  --work-dir artifacts/quickstart
```

## Repository structure

```text
claq/
  constraints.py          conditional-information scorer and feasibility masks
  objectives.py           adversarial, cost, query-set, and fairness objectives
  precomputed.py          canonical response archive and rollout evaluation
  evaluation.py           held-out conditional leakage probe
  training/
    tensor_claq.py         reference unrolled training implementation

scripts/
  export_celeba_responses.py
  export_bias_in_bios_responses.py
  export_cifar10_responses.py
  validate_precomputed.py
  make_synthetic_dataset.py
  quickstart.py
  release_check.py
  run_precomputed_experiment.py
  run_ablation_suite.py
  plot_summary.py

configs/
  paper_protocol.json
```

See [`MANIFEST.md`](MANIFEST.md) for the complete package map and
[`docs/PAPER_CODE_ALIGNMENT.md`](docs/PAPER_CODE_ALIGNMENT.md) for the
correspondence between the mathematical formulation and the implementation.

## Canonical response archive

Experiments use a validated `.npz` archive with fixed operational responses.

| Field | Shape | Description |
|---|---:|---|
| `responses_train` | `[N_train, Q]` | Training responses |
| `responses_validation` | `[N_validation, Q]` | Validation responses |
| `responses_test` | `[N_test, Q]` | Test responses |
| `y_train`, `y_validation`, `y_test` | `[N]` | Task labels |
| `s_train`, `s_validation`, `s_test` | `[N]` | Sensitive targets |
| `query_names` | `[Q]` | Unique query names |
| `query_set_mask` | `[Q]` | Pre-specified query-composition mask |
| `query_costs` | `[Q]` | Finite nonnegative query costs |
| `admissible_mask` | `[Q]` | Base admissibility mask |
| `justification_mask` | `[Q]` | Proxy-screening override mask |
| `metadata_json` | scalar | Versioned JSON metadata |

The loader enforces the following conditions:

- every split is nonempty;
- responses are binary and encoded as `{0,1}` or `{-1,+1}`;
- training task and sensitive class indices are contiguous and start at zero;
- validation and test contain no class absent from training;
- query names are nonempty and unique;
- query costs are finite and nonnegative;
- all query-level vectors have length `Q`.

Responses are converted to `{-1,+1}`. The value `0` is reserved for an
unqueried transcript coordinate.

`query_set_mask` defines a query-composition statistic and, when requested, a
query-set penalty. It is not the sensitive random variable and does not certify
transcript leakage.

Current archives contain no NumPy object arrays and are loaded with pickling
disabled. A trusted archive produced by an earlier package version should be
re-exported. The explicit `--allow-legacy-pickle` option is available only for
controlled migration.

Validate an archive before training:

```bash
python scripts/validate_precomputed.py \
  --data artifacts/precomputed/celeba_attractive.npz
```

Use `--json` to obtain machine-readable validation output.

## Preparing operational responses

### CelebA

CelebA uses dataset-provided binary attributes as exact operational responses.
The target attribute is removed from the query vocabulary. The `Male`
attribute is used as the sensitive target and may remain an executable query.

```bash
python scripts/export_celeba_responses.py \
  --root data \
  --target Attractive \
  --sensitive Male \
  --output artifacts/precomputed/celeba_attractive.npz
```

The exporter records the default pre-specified gender-associated query set in
the archive metadata. Use `--associated` to supply a different set. Use
`--download` only when automated download is permitted by the dataset terms.

### Bias in Bios

The exporter loads a validation-selected `ConceptAnswererMLP`, applies it to
the fixed train, validation, and test biographies, converts the response
probabilities to hard binary responses using the stored threshold, and records
the dataset-provided profession and gender labels.

```bash
python scripts/export_bias_in_bios_responses.py \
  --data-dir artifacts/data/bias_in_bios \
  --concepts assets/concepts/bias_in_bios.csv \
  --checkpoint artifacts/models/bias_in_bios_concept_answerer.pt \
  --output artifacts/precomputed/bias_in_bios.npz
```

The exporter records hashes of the concept specification and response-model
checkpoint. The response model must be selected without test data. Its
held-out response quality should be reported separately from acquisition-policy
performance.

### CIFAR-10

CIFAR-10 does not provide a protected attribute. The exporter constructs a
binary sensitive target from CLIP similarities to the pre-specified concept
set. This setting is a controlled concept-query experiment and should not be
interpreted as a demographic fairness benchmark.

```bash
python scripts/export_cifar10_responses.py \
  --root data \
  --concepts assets/concepts/cifar10.txt \
  --concept-qa-checkpoint artifacts/models/cifar10_concept_qa.pt \
  --output artifacts/precomputed/cifar10.npz \
  --validation-fraction 0.1 \
  --split-seed 0 \
  --sensitive-tau 0.7 \
  --sensitive-topk 3
```

The training subset is exported without shuffling to make archive generation
deterministic. The sensitive-target construction, split seed, threshold,
top-\(k\) value, and source hashes are stored in the metadata. Equalized-odds
reporting is disabled by default for this constructed target.

## Reference multi-seed protocol

The reference runner performs the following operations:

- fits the conditional-information scorer on the training split only;
- trains the acquisition policy from the empty transcript using an unrolled
  horizon;
- conditions the sensitive adversary on the ground-truth task label;
- selects checkpoints using the validation split only;
- evaluates each selected checkpoint once on the test split;
- reports hard executed-query frequencies rather than actor probability mass;
- fits an independent conditional probe on training rollouts, selects probe
  regularization on validation rollouts, and evaluates the probe on test
  rollouts;
- records the command, software environment, dataset hash, query
  specification, masks, costs, and model configuration.

The settings in [`configs/paper_protocol.json`](configs/paper_protocol.json)
are explicit reference templates. They are not asserted to be optimal
hyperparameters. Every modification should be preserved in the run manifest
and reported with the corresponding results.

### CelebA

```bash
python scripts/run_precomputed_experiment.py \
  --data artifacts/precomputed/celeba_attractive.npz \
  --output artifacts/runs/celeba_attractive \
  --variant-name CLAQ \
  --lambdas 0,0.2,0.4 \
  --seeds 0,1,2,3,4 \
  --budget 16 \
  --epochs 5 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --deterministic
```

### Bias in Bios

```bash
python scripts/run_precomputed_experiment.py \
  --data artifacts/precomputed/bias_in_bios.npz \
  --output artifacts/runs/bias_in_bios \
  --variant-name CLAQ \
  --lambdas 0,0.4 \
  --seeds 0,1,2,3,4 \
  --budget 20 \
  --epochs 50 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --deterministic
```

### CIFAR-10

```bash
python scripts/run_precomputed_experiment.py \
  --data artifacts/precomputed/cifar10.npz \
  --output artifacts/runs/cifar10 \
  --variant-name CLAQ \
  --lambdas 0,0.4 \
  --seeds 0,1,2,3,4 \
  --budget 20 \
  --epochs 5 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --deterministic
```

The runner does not write into a nonempty output directory unless
`--overwrite` is supplied.

## Pathwise constraints and optional extensions

### Unit query cost

The primary specification uses

\[
\mathsf C(q)=1
\qquad
\text{for every }q\in\mathcal Q.
\]

When all stored costs equal one and `--cost-budget` is omitted, the runner sets
the pathwise cost budget equal to the rollout horizon. This setting makes the
cost accounting explicit but does not further restrict a fixed-horizon
rollout. Use `--disable-auto-unit-cost-budget` to leave the pathwise cost
budget inactive.

For heterogeneous costs, store a nonconstant `query_costs` vector and specify
the cost coefficient and budget:

```bash
python scripts/run_precomputed_experiment.py \
  ... \
  --lambda-cost 0.1 \
  --cost-budget 12.0
```

### Pathwise empirical leakage account

```bash
python scripts/run_precomputed_experiment.py \
  ... \
  --leakage-budget 0.25
```

This option enforces a pathwise budget using the training-reference plug-in
information scorer. It does not establish the population condition

\[
I(S;\mathcal H_T\mid Y)\le\varepsilon.
\]

### Static and dynamic proxy screening

```bash
python scripts/run_precomputed_experiment.py \
  ... \
  --proxy-leakage-threshold 0.10 \
  --proxy-label-threshold 0.01 \
  --dynamic-proxy-screen
```

Static screening is applied before policy training when both thresholds are
specified. Dynamic screening additionally masks candidates at each transcript.
The justification mask exempts a query from the proxy-screening rule only; it
does not override the base admissibility mask.

### Confidence-based stopping

```bash
python scripts/run_precomputed_experiment.py \
  ... \
  --confidence-threshold 0.95 \
  --minimum-rollout-steps 3
```

The confidence condition is evaluated before selecting the next query and only
after the specified minimum number of queries has been executed.

### Optional outcome-fairness extension

```bash
python scripts/run_precomputed_experiment.py \
  ... \
  --lambda-fairness 0.1 \
  --fairness-min-group-count 2 \
  --fairness-reduction max
```

A dual update may be enabled with a pre-specified empirical tolerance:

```bash
python scripts/run_precomputed_experiment.py \
  ... \
  --fairness-tolerance 0.10 \
  --fairness-dual-lr 0.01 \
  --fairness-lambda-max 10
```

This extension optimizes an empirical equalized-odds criterion. It does not
provide a population equalized-odds certificate.

## Ablation protocol

The ablation runner evaluates six variants with common data splits, seeds,
architectures, and checkpoint-selection rules:

```text
CLAQ-Base
CLAQ-Cost
CLAQ-Leak
CLAQ-LeakCost
CLAQ-Fair
CLAQ-Full
```

Example:

```bash
python scripts/run_ablation_suite.py \
  --data artifacts/precomputed/celeba_attractive.npz \
  --output artifacts/runs/celeba_ablation \
  --seeds 0,1,2,3,4 \
  --budget 16 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --lambda-sensitive 0.4 \
  --lambda-cost 0.1 \
  --lambda-fairness 0.1 \
  --lambda-query-set 0.1 \
  --cost-budget 12 \
  --leakage-budget 0.25 \
  --proxy-leakage-threshold 0.10 \
  --proxy-label-threshold 0.01 \
  --deterministic
```

`CLAQ-Full` enables the designated-query penalty and the dynamic proxy screen.
Static proxy screening, when requested, is held fixed across variants. The
runner writes one run directory per variant and a combined
`ablation_summary.json`.

## Output structure

A run directory contains:

```text
run_manifest.json
summary.json
checkpoints/<variant>_lambda_<value>_seed_<seed>.pt
rollouts/<variant>_lambda_<value>_seed_<seed>_test.npz
<variant>_lambda_<value>_seed_<seed>_history.json
<variant>_lambda_<value>_seed_<seed>_metrics.json
```

Each checkpoint contains:

- CPU model state dictionaries;
- architecture and temperature configuration;
- objective and constraint parameters;
- the dataset SHA-256 digest;
- query names, costs, masks, and metadata;
- the validation-selected epoch and score.

Held-out metrics include, when applicable:

- accuracy and macro-F1;
- empirical multiclass equalized-odds violation;
- minimum included equalized-odds cell count;
- supported equalized-odds group-cell and group-pair counts;
- mean number of executed queries;
- mean acquisition cost;
- mean pathwise empirical leakage account;
- designated-query rate;
- stopping-reason frequencies;
- independent-probe accuracy;
- probe leakage in bits.

Generate a utility versus query-composition plot with:

```bash
python scripts/plot_summary.py \
  --summary artifacts/runs/celeba_attractive/summary.json \
  --output artifacts/figures/celeba_tradeoff.pdf
```

## Interpretation of reported quantities

### Target-conditioned adversary

The sensitive head predicts \(S\) from \((\mathcal H_T,Y)\). Gradient reversal
is applied to the transcript representation and not to the one-hot task label.
The resulting objective is a restricted adversarial surrogate related to
conditional transcript leakage. It is not exact mutual-information
minimization.

### Empirical conditional-information scorer

`EmpiricalConditionalMIScorer` estimates

\[
\widehat I(Y;R_q\mid \mathcal H=h)
\qquad\text{and}\qquad
\widehat I(S;R_q\mid Y,\mathcal H=h)
\]

from a training-only reference response matrix. It retains rows consistent
with the current binary transcript. Histories with support below the configured
minimum use global scores. The estimator is finite-sample,
support-dependent, and not differentiated through.

### Designated-query rate

The designated-query rate is

\[
\frac{
\text{number of executed queries in the pre-specified set}
}{
\text{total number of executed queries}
}.
\]

This statistic describes the composition of the selected query sequence. It
is not an estimate of terminal-transcript mutual information.

### Pathwise empirical leakage account

The account sums the estimated one-step conditional leakage assigned to each
executed query at the preceding transcript. It is neither a realized
information density nor a certified estimator or upper bound for
\(I(S;\mathcal H_T\mid Y)\).

### Held-out conditional probe

The independent probe reports

\[
\max\!\left\{
0,\,
\widehat H(S\mid Y)
-
\operatorname{CE}_{\mathrm{probe}}(S\mid\mathcal H_T,Y)
\right\}
\]

in bits. This quantity is finite-sample and probe-class dependent. It should be
reported as **probe leakage**, not as the exact conditional mutual information.

## Reproducibility and validation

Run the complete local validation suite with:

```bash
make release-check
```

The release check verifies package import, source compilation, automated tests,
the synthetic quickstart, archive validation, and repository hygiene.

For deterministic execution, use `--deterministic`. Exact numerical
reproducibility may still depend on the operating system, accelerator,
dependency versions, and hardware-specific kernels. The run manifest records
the relevant software and experiment configuration.

Historical results produced by earlier implementations are documented in
[`docs/LEGACY_RESULTS.md`](docs/LEGACY_RESULTS.md). They should not be
attributed to the reference protocol in this repository without rerunning the
experiments.

## Data and model artifacts

Datasets, pretrained models, response-model checkpoints, and generated
experiment outputs are not included in this repository. Their licenses,
redistribution terms, provenance requirements, and expected directory
structure are documented in [`DATA_AND_MODELS.md`](DATA_AND_MODELS.md).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub users
may also use the repository's **Cite this repository** interface.

When reporting experimental results, cite both the CLAQ paper and the software
release used to produce the results.

## License

This repository is distributed under the
[`CLAQ Academic Research License 1.0`](LICENSE). The license permits specified
non-commercial academic research uses and is not an OSI-approved open-source
license. No patent license is granted.

Third-party datasets, model weights, and dependencies remain subject to their
respective licenses and terms.
