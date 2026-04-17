import os
import sys
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


# PATHS
ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR     = os.path.join(ANALYSIS_DIR, 'data')
STAN_DIR     = os.path.join(ANALYSIS_DIR, 'stan_models')
RESULTS_DIR  = os.path.join(ANALYSIS_DIR, 'stan_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

## I/O parameters.
stan_model = sys.argv[1]
session    = sys.argv[2]

## Sampling parameters
iter_warmup      = 5000   
iter_sampling    = 1250    
chains           = 4
thin             = 1
parallel_chains  = 4

# Load and prepare data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print(f"Loading data for session: {session}")

## Load pgng data
data = pd.read_csv(os.path.join(DATA_DIR, 'pgng.csv'))

## Map session name to number
session_map = {'s1': 1, 's2': 2}
session_num = session_map[session]
data = data[data['session'] == session_num].reset_index(drop=True)

## Load reject list and filter
reject = pd.read_csv(os.path.join(DATA_DIR, 'reject.csv'))
keep = reject.query('reject == 0')['subject'].values
data = data[data['subject'].isin(keep)].reset_index(drop=True)

print(f"Participants retained: {data['subject'].nunique()}")
print(f"Total trials: {len(data)}")

## Format valence: win=1, lose=0
data['valence_num'] = (data['valence'] == 'win').astype(int)

## Outcome is already encoded as 1 (favourable) or 0 (unfavourable)
## from collate_pgng_conpit.py

## Controllability: controllable=0, uncontrollable=1
## (already in data as 'controllable' column)

# Assemble data for Stan
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Define metadata.
N = len(data)
J = np.unique(data['subject'].values, return_inverse=True)[-1] + 1
K = np.unique(data['stimulus'].values, return_inverse=True)[-1] + 1

Y = data['choice'].values.astype(int)
R = data['outcome'].values.astype(int)
V = data['valence_num'].values.astype(int)
C = (1 - data['controllable'].values).astype(int)  # for M2 and M4

print(f"\nStan data dimensions:")
print(f"  N (trials):       {N}")
print(f"  J (subjects):     {J.max()}")
print(f"  K (stimuli):      {K.max()}")
print(f"  Y unique:         {np.unique(Y)}")
print(f"  R unique:         {np.unique(R)}")
print(f"  V unique:         {np.unique(V)}")
print(f"  C unique:         {np.unique(C)}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Assemble Stan data dictionary
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
if stan_model in ['pgng_m1', 'pgng_m3']:      ## M3: M1 + lapse rate (no controllability)
    dd = dict(N=N, J=J, K=K, Y=Y, R=R, V=V)
elif stan_model in ['pgng_m2', 'pgng_m4']:    ## M4: M2 + lapse rate (controllability + lapse)
    dd = dict(N=N, J=J, K=K, Y=Y, R=R, V=V, C=C)
else:
    raise ValueError(f"Unknown model: {stan_model}")

# Fit Stan model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
stan_file = os.path.join(STAN_DIR, f'{stan_model}.stan')
print(f"\nFitting model: {stan_model}")
print(f"Stan file: {stan_file}")

StanModel = CmdStanModel(stan_file=stan_file)

StanFit = StanModel.sample(
    data=dd,
    chains=chains,
    iter_warmup=iter_warmup,
    iter_sampling=iter_sampling,
    thin=thin,
    parallel_chains=parallel_chains,
    seed=0,
    show_progress=True
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Check diagnostics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("\nDiagnostics:")
print(f"  Divergences: {StanFit.divergences.sum()}")
summary = StanFit.summary(percentiles=(2.5, 50, 97.5), sig_figs=3)
rhat_issues = summary[summary['R_hat'] >= 1.01]
print(f"  Parameters with R_hat >= 1.01: {len(rhat_issues)}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Save results
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("\nSaving results...")

session_results_dir = os.path.join(RESULTS_DIR, session)
os.makedirs(session_results_dir, exist_ok=True)

## Define fout.
fout = os.path.join(session_results_dir, stan_model)

## Save samples
samples = StanFit.draws_pd()

cols = np.concatenate([
    samples.filter(regex='__').columns,
    samples.filter(regex='[a,b,c,d][0-9]_mu').columns,
    samples.filter(regex='sigma').columns,
    samples.filter(regex=r'[a,b,c,d][0-9]\[').columns,
])

samples[cols].to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
summary.to_csv(f'{fout}_summary.tsv', sep='\t')

print(f"Saved samples to: {fout}.tsv.gz")
print(f"Saved summary to: {fout}_summary.tsv")

## Print group-level parameter estimates
print("\nGroup-level parameter estimates:")
group_params = summary[summary.index.str.endswith('_mu')]
print(group_params[['Mean', '2.5%', '97.5%', 'R_hat']].to_string())
