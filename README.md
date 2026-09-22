# HMNCE: Hard-Sample and Missing-Aware Numerical Contrastive Embedding

**HMNCE** is a general-purpose PyTorch model for numerical reasoning over knowledge graphs. It integrates missing-aware numerical encoding, relation-aware representation learning, stable multi-expert fusion, and hard-sample contrastive learning into a unified link-prediction model.

This repository provides tools for model training and evaluation, ablation studies, structural scoring-function analysis, numerical-attribute experiments, semantic-relation experiments, and geographic case studies.

---

## 1. Overview

HMNCE combines entities, relations, and numerical attributes to learn representations for numerical relation prediction across different knowledge graph domains.

The model has three main components:

- **Missing-aware numerical encoding (ME)**: uses attribute-specific representations to distinguish observed numerical values from missing states.
- **Stable relation-aware mixture (SRM)**: combines relation-conditioned expert representations using numerically stable mixture weights.
- **Hard-sample contrastive learning (HCL)**: selects low-scoring positive candidates and high-scoring negative candidates to improve discriminative representations.

The model is evaluated through filtered tail-entity prediction using MR, MRR, Hits@1, Hits@3, and Hits@10.

---

## 2. Features

- **Unified HMNCE training and evaluation**
- **Missing-aware numerical attribute encoding**
- **Relation-aware multi-head attention**
- **Stable multi-expert representation learning**
- **Top-k hard-sample selection and contrastive learning**
- **Multiple structural scoring functions**:
  - TransE
  - ComplEx
  - ConvE
  - TuckER
- **Built-in baseline models**:
  - TransE
  - DistMult
  - ConvE
  - HAKE
  - CompGCN
  - R-GCN
  - KBLN
  - Literal-gated model
  - DRAE
  - RAKGE
- **Named configurations for ablation studies**
- **Training-set-size and semantic-relation experiments**
- **City-type reconstruction, numerical comparison, and city-ranking analysis**
- **GeoJSON export for geographic visualization**
- **Checkpoint and dataset auditing**

---

## 3. Project Structure

```text
HMNCE/
├── train.py                       # Main training and evaluation entry point
├── experiments.py                 # Ablation and scoring-function experiments
├── preprocess_kg_num_lit.py        # Numerical-literal preprocessing
├── model/
│   ├── hmnce_model.py              # HMNCE model
│   ├── drae_baseline.py            # DRAE baseline
│   ├── rakge_model.py              # RAKGE baseline
│   ├── kge_score.py                # Structural scoring functions
│   └── ...                        # KGE, literal, and graph-convolution models
├── data/                          # Dataset loading
├── utils/                         # Training and evaluation utilities
├── inference_app/
│   ├── cli.py                     # Case-study analysis and export commands
│   ├── analysis.py                # Reconstruction, comparison, and ranking metrics
│   ├── core.py                    # Dataset and checkpoint loading
│   ├── geojson/                   # Case-study administrative boundaries
│   └── model_registry.json        # Optional user-defined aliases (empty by default)
├── tests/                         # HMNCE and baseline smoke tests
├── datasets/                      # Dataset directories
├── checkpoints/                   # Saved model checkpoints
├── logs/                          # Training logs
├── results/                       # Evaluation and geographic analysis outputs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 4. Installation


### 4.1 Research Server Environment

```text
Python       3.10.11
PyTorch      2.2.1+cu121
DGL          2.2.1+cu121
NumPy        1.26.4
Pandas       2.3.3
SciPy        1.15.3
RDFLib       7.6.0
OpenPyXL     3.1.5
CUDA         12.1
```

### 4.2 Requirements

- Python ≥3.8
- Mutually compatible CUDA-enabled PyTorch and DGL
- numpy, pandas, scipy, rdflib, openpyxl, tqdm, packaging

### 4.3 Install Dependencies

First install compatible CUDA-enabled PyTorch and DGL builds for the research environment, then install the remaining dependencies:

```bash
pip install numpy pandas scipy rdflib openpyxl tqdm packaging
```

---

## 5. Dataset Preparation

### 5.1 Dataset Directories

```text
datasets/
├── credit/
├── spotify/
└── Geo-NR/
    ├── Geo-NR-5K/
    ├── Geo-NR-5K-noTypes/
    ├── Geo-NR-10K/
    ├── Geo-NR-15K/
    ├── Geo-NR-45K/
    └── Geo-NR-50K/
```

`Geo-NR-5K-noTypes` is the control dataset without semantic relations.

### 5.2 Input Format

```text
datasets/<dataset>/
├── train.txt
├── valid.txt
├── test.txt
├── neg.txt
└── literals/
    └── numerical_literals.txt
```

Text files are tab-separated and have no header. Triple fields are ordered as head entity, relation, and tail entity; numerical-attribute fields are ordered as entity, attribute, and value.

### 5.3 Run Preprocessing

For Geo-NR-5K:

```bash
python preprocess_kg_num_lit.py --dataset Geo-NR-5K --data-root ./datasets/Geo-NR
```

Use `--data-root ./datasets` for Credit and Spotify, and `--data-root ./datasets/Geo-NR` for the Geo-NR variants.

### 5.4 Preprocessed Format

```text
datasets/Geo-NR/Geo-NR-5K/
├── entities.dict
├── relations.dict
├── train.txt
├── valid.txt
├── test.txt
├── neg.txt
├── train.npy
└── literals/
    ├── numerical_literals.txt
    └── numerical_literals.npy
```

---

## 6. Key Arguments

| Argument | Description | Geo-NR setting |
|---|---|---:|
| `--model` | Model name | `hmnce` |
| `--data` | Dataset directory name | Experiment dependent |
| `--data-root` | Dataset root directory | `./datasets/Geo-NR` |
| `--output-root` | Output root for logs and checkpoints | `.` |
| `--gpu` | CUDA device; use `-1` for CPU | `0` |
| `--epoch` | Maximum number of training epochs | `1000` |
| `--batch` | Batch size | `128` |
| `--lr` | Learning rate | `0.001` |
| `--seed` | Random seed | `12345` |
| `--init_dim` | Entity and relation dimension | `200` |
| `--att_dim` | Numerical attention dimension | `200` |
| `--head_num` | Number of attention heads | `5` |
| `--num_mixture` | Number of mixture experts | `5` |
| `--drop` | HMNCE dropout rate | `0.7` |
| `--gamma` | TransE margin | `9.0` |
| `--order` | Numerical order-score weight | `0.25` |
| `--scale` | Contrastive-loss weight | `0.25` |
| `--top_k` | Number of selected hard samples | `5` |
| `--numeric_density` | Proportion of numerical values retained | `0.8` |
| `--ft` | Numerical input type | `numeric` |
| `--hmnce-score-func` | HMNCE structural scoring function | `transe` |

Ablation switches for the three main HMNCE modules:

| Argument | Description |
|---|---|
| `--disable_me` | Disable missing-aware numerical encoding |
| `--disable_hcl` | Disable hard-sample contrastive learning |
| `--disable_srm` | Disable stable relation-aware mixture |

---

## 7. Usage Examples

### 7.1 Train HMNCE on Geo-NR-5K

```bash
python train.py --model hmnce --name hmnce_geo5k_full \
  --data Geo-NR-5K --data-root ./datasets/Geo-NR --output-root . \
  --gpu 0 --batch 128 --epoch 1000 --seed 12345 --lr 0.001 \
  --init_dim 200 --att_dim 200 --head_num 5 --num_mixture 5 \
  --drop 0.7 --gamma 9.0 --order 0.25 --scale 0.25 \
  --top_k 5 --numeric_density 0.8 --hmnce-score-func transe
```

### 7.2 Evaluate a Saved Model

```bash
python train.py --model hmnce --eval-only \
  --checkpoint ./checkpoints/hmnce_geo5k_full.pt \
  --data Geo-NR-5K --data-root ./datasets/Geo-NR --gpu 0
```

### 7.3 Optional HMNCE CPU Check

Set the GPU argument to `-1`:

```bash
python train.py --model hmnce --data Geo-NR-5K \
  --data-root ./datasets/Geo-NR --gpu -1 --epoch 10 \
  --init_dim 200 --att_dim 200
```

---

## 8. Output Files

For a run named `hmnce_geo5k_full`, the main outputs are:

```text
checkpoints/hmnce_geo5k_full.pt
logs/hmnce_geo5k_full.log
```

The checkpoint contains model parameters and structural configuration. The log records:

- training loss;
- validation MRR;
- the selected best epoch;
- test MR and MRR;
- Hits@1, Hits@3, and Hits@10;
- training and evaluation time.

---

## 9. Extending the Framework

This section is intended for developers adding new functionality. These changes are not required to run the supplied experiments.

### 9.1 Add a New Baseline Model

1. Implement the model under `model/`.
2. Register it in the `--model` choices in `train.py`.
3. Add model construction in `Runner.get_model()`.
4. If needed, add a named experiment configuration in `experiments.py`.

### 9.2 Add a New HMNCE Structural Scorer

1. Implement the scorer in `model/kge_score.py`.
2. Register it in `model/hmnce_model.py`.
3. Add it to the `--hmnce-score-func` choices in `train.py`.
4. Add a corresponding configuration in `experiments.py`.

### 9.3 Add an Evaluation or Geographic Export

1. Implement the metric or analysis in `inference_app/analysis.py`.
2. Add a command in `inference_app/cli.py`.
3. Write the generated files under `results/`.

---

## 10. Experimental Procedures

The experiments below use the settings in Section 6; arguments omitted from the commands use their code defaults. Training logs are saved under `logs/`, and checkpoints under `checkpoints/`. Use a distinct `--name` for each experiment.

### Experiment 1: Main Link-Prediction Comparison (Table 2)

**Objective**: Compare the link-prediction performance of HMNCE and the baseline models.

1. Prepare Credit, Spotify, and Geo-NR-5K as described in Section 5.
2. Train each model using its corresponding arguments. For example, train HMNCE on Credit:

```bash
python train.py --model hmnce --name table2_hmnce_credit --data credit --data-root ./datasets --batch 128 --init_dim 200
```

3. Replace the model arguments using the table below and update `--name`. Change `--data` to select another dataset; use `--data-root ./datasets/Geo-NR` for Geo-NR.
4. Extract MR, MRR, and Hits@1/3/10 from the logs and compile Table 2.

| Model | Model arguments |
|---|---|
| ConvE | `--model conve --n_layer 0` |
| DistMult | `--model distmult --n_layer 0` |
| HAKE | `--model hake --n_layer 0` |
| CompGCN-TransE | `--model transe --n_layer 1 --encoder compgcn --score_func transe --gcn_dim 200 --embed_dim 200` |
| R-GCN | `--model transe --n_layer 1 --encoder rgcn --score_func transe --gcn_dim 200 --embed_dim 200` |
| Numerical-literal baseline | `--model kbln --literal` |
| LiteralE | `--model literal_gate --literal` |
| DRAE | `--model drae --literal --n_layer 0` |
| RAKGE | `--model rakge --literal --n_layer 0` |
| HMNCE | `--model hmnce --literal --n_layer 0` |

Use these arguments to replace the model-selection arguments in the example, keeping the other training parameters unchanged.

### Experiment 2: Ablation Study (Table 3)

**Objective**: Evaluate the contributions of ME, HCL, and SRM.

1. Run the ablation group on Geo-NR-5K:

```bash
python experiments.py ablation --data Geo-NR-5K --data-root ./datasets/Geo-NR --batch 128 --extra --init_dim 200
```

2. The command runs the full model, three single-module ablations, and three two-module ablations in sequence.
3. Extract the test metrics from each log and compile Table 3.

### Experiment 3: Numerical Attribute Representation (Table 4)

**Objective**: Compare continuous numerical attributes with binary attributes.

1. Train with `numeric` and `binary` inputs on Geo-NR-5K. For HMNCE:

```bash
python train.py --model hmnce --name table4_hmnce_numeric --data Geo-NR-5K --data-root ./datasets/Geo-NR --batch 128 --init_dim 200 --ft numeric
python train.py --model hmnce --name table4_hmnce_binary --data Geo-NR-5K --data-root ./datasets/Geo-NR --batch 128 --init_dim 200 --ft binary
```

2. Replace `--model hmnce` with `--model literal_gate`, `--model drae`, `--model CompGCN-TransE`, and `--model rakge` in turn. Update `--name` and keep all other parameters unchanged.
3. Extract the evaluation metrics for both settings from the logs and compile Table 4.


### Experiment 4: Semantic Relations (Table 5)

**Objective**: Assess the effect of semantic relations.

1. Train on Geo-NR-5K and Geo-NR-5K-noTypes separately. For HMNCE:

```bash
python train.py --model hmnce --name table5_hmnce_types --data Geo-NR-5K --data-root ./datasets/Geo-NR --batch 128 --init_dim 200
python train.py --model hmnce --name table5_hmnce_noTypes --data Geo-NR-5K-noTypes --data-root ./datasets/Geo-NR --batch 128 --init_dim 200
```

2. Repeat these paired runs for CompGCN-TransE, LiteralE, DRAE, and RAKGE using the model arguments in Experiment 1, and update `--name`.
3. Keep the random seed, other hyperparameters, and numerical-comparison splits unchanged, and compile the results in Table 5.

### Experiment 5: Structural Scoring Functions (Table 6)

**Objective**: Compare TransE, ComplEx, ConvE, and TuckER within HMNCE.

1. Run the scoring-function group on Geo-NR-5K:

```bash
python experiments.py scores --data Geo-NR-5K --data-root ./datasets/Geo-NR --batch 128 --extra --init_dim 200
```

2. Extract the evaluation metrics from the four logs and compile Table 6.

### Experiment 6: Training-Set Size (Table 7)

**Objective**: Evaluate model performance at different training-set sizes.

1. Prepare Geo-NR-5K, Geo-NR-10K, Geo-NR-15K, Geo-NR-45K, and Geo-NR-50K.
2. Train HMNCE on each dataset. For Geo-NR-50K:

```bash
python train.py --model hmnce --name table7_hmnce_geo50k --data Geo-NR-50K --data-root ./datasets/Geo-NR --batch 128 --init_dim 200
```

3. Change `--data` and `--name` to repeat training at the other sizes.
4. Extract the evaluation metrics from the logs and compile Table 7.

### Experiment 7: Geographic Case Study (Figures 4–6 and Table 8)

**Objective**: Present geographic case-study results for numerical comparison, city ranking, and city-type relation reconstruction.

1. Train HMNCE on Geo-NR-50K:

```bash
python train.py --model hmnce --name hmnce_geo50k_full --data Geo-NR-50K --data-root ./datasets/Geo-NR --batch 128 --init_dim 200
```

After training completes, use the generated `checkpoints/hmnce_geo50k_full.pt` for the following analyses.

Case-study inference uses the full HMNCE model with TransE scoring and numerical inputs. Use the dataset paired with the checkpoint.

2. Use the checkpoint generated by training to export the map data and compute type-reconstruction and numerical-comparison metrics for the full study period:

```bash
python inference_app/cli.py export-web-bundle --dataset ./datasets/Geo-NR/Geo-NR-50K --checkpoint ./checkpoints/hmnce_geo50k_full.pt --device cuda:0 --boundaries ./inference_app/geojson/hubei_prefecture_cities.geojson --years {2003..2023} --output ./results/maps_all_years
python inference_app/cli.py reconstruct-types --dataset ./datasets/Geo-NR/Geo-NR-50K --checkpoint ./checkpoints/hmnce_geo50k_full.pt --device cuda:0 --relations gdp_tier industry_type --output ./results/table8_types.json
python inference_app/cli.py evaluate-comparisons --dataset ./datasets/Geo-NR/Geo-NR-50K --checkpoint ./checkpoints/hmnce_geo50k_full.pt --device cuda:0 --threshold 0.5 --output ./results/table8_comparisons.json
```

Select the following results for 2022 from the exported files. The GeoJSON files contain map data; render and arrange them to produce the final map panels:

| Figure | Results to use |
|---|---|
| Figure 4 | Reference comparison relations and model scores for Xianning's primary-industry share of GDP |
| Figure 5 | Reference ranks, predicted ranks, and rank errors for GDP and primary-industry share of GDP |
| Figure 6 | Reference categories and reconstructed categories for GDP tier and industry type |

Take the arithmetic mean of each ranking metric in `metadata.summary` across the 273 generated `ranking_*.geojson` files. Combine these averages with the type-reconstruction and numerical-comparison results to compile Table 8.

Repository: [Bingbing-web/HMNCE](https://github.com/Bingbing-web/HMNCE)
