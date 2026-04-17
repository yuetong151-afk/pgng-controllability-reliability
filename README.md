# Test-Retest Reliability of a Pavlovian Go/No-Go Task with a Controllability Manipulation
 
Analysis code for BSc dissertation, King's College London (2026).
 
## Study Overview
 
This study examined how environmental controllability modulates Pavlovian-instrumental arbitration, and evaluated the test-retest reliability of behavioural and computational measures extracted from a modified Pavlovian Go/No-Go (PGNG) task.
 
## Repository Structure
 
```
├── 00_ParameterRecovery.ipynb
├── 01_Screening.ipynb
├── 02_Descriptive.ipynb
├── 03_Modeling.ipynb
├── stan_models/
│   ├── pgng_m1.stan         # M1: baseline model
│   ├── pgng_m1_sh.stan      # M1: split-half variant
│   ├── pgng_m2.stan         # M2: + controllability modulation
│   ├── pgng_m2_sh.stan
│   ├── pgng_m3.stan         # M3: + lapse rate
│   ├── pgng_m3_sh.stan
│   ├── pgng_m4.stan         # M4: full model (winning)
│   └── pgng_m4_sh.stan
├── scripts/
│   ├── fit_pgng_conpit.py
│   ├── fit_pgng_ppc_conpit.py
│   ├── fit_pgng_recovery.py
│   ├── fit_pgng_splithalf_conpit.py
│   ├── fit_pgng_trt_conpit.py
│   ├── collate_pgng_conpit_full.py
│   ├── reliability_conpit.py
│   ├── simulate_pgng.py
│   └── psis.py
└── README.md
```
 
## Notebooks
 
- `00_ParameterRecovery.ipynb`: Parameter recovery analyses and model identifiability checks for all four models
- `01_Screening.ipynb`: Participant screening and exclusion criteria — generates `data/reject.csv`
- `02_Descriptive.ipynb`: Behavioural analyses, model-free reliability (ICC), and Pavlovian-instrumental dissociation (LPM + GEE)
- `03_Modeling.ipynb`: Computational model comparison, group-level parameter estimation, convergent validity, and parameter reliability
## Computational Models
 
Four reinforcement learning models of increasing complexity were compared using PSIS-LOO-CV. All models implement Rescorla-Wagner updating with separate reward and punishment learning rates, and were estimated using hierarchical Bayesian estimation in Stan.
 
| Model | Parameters | Description |
|-------|------------|-------------|
| M1 | β+, β−, τ+, τ−, η+, η− | Baseline: outcome sensitivity, Pavlovian bias, learning rates |
| M2 | M1 + δ+, δ− | Adds controllability modulation of Pavlovian bias |
| M3 | M1 + ξ | Adds lapse rate |
| M4 | M1 + δ+, δ− + ξ | Full model — selected via LOO-CV |
 
**Parameter definitions:**
- β+/β−: Outcome sensitivity (reward/punishment)
- τ+/τ−: Pavlovian approach/avoidance bias (reward/punishment)
- η+/η−: Learning rate (reward/punishment)
- δ+/δ−: Controllability modulation of Pavlovian bias (reward/punishment)
- ξ: Lapse rate (stimulus-independent random responding)
`_sh` variants are used for split-half reliability estimation.
 
## Scripts
 
- `fit_pgng_conpit.py`: Fits Stan models to empirical data
- `fit_pgng_ppc_conpit.py`: Generates posterior predictive checks
- `fit_pgng_recovery.py`: Fits models to simulated data for parameter recovery
- `fit_pgng_splithalf_conpit.py`: Fits models to odd/even trial halves for split-half reliability
- `fit_pgng_trt_conpit.py`: Fits joint model for test-retest reliability estimation
- `collate_pgng_conpit_full.py`: Collates Stan output files
- `reliability_conpit.py`: Computes split-half and test-retest reliability
- `simulate_pgng.py`: Simulates choice data for parameter recovery
- `psis.py`: PSIS-LOO-CV implementation
## Model Fitting
 
Model fitting procedures were adapted from Zorowitz et al. (2025), with code available at https://github.com/nivlab/RobotFactory. Pre-fitted Stan results are not included due to file size. To reproduce model fitting, run `fit_pgng_conpit.py` for each model and session.
 
## Requirements
 
- Python 3.13.5
- CmdStanPy 2.38.0
- pandas, numpy, matplotlib, seaborn, pingouin, statsmodels, scipy, scikit-learn
```
pip install -r requirements.txt
```
 
## Data Availability
 
Raw data are not included due to participant confidentiality. Data are available upon reasonable request.
 
