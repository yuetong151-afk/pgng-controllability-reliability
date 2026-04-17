import os
import sys
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR     = os.path.join(ANALYSIS_DIR, 'data')
STAN_DIR     = os.path.join(ANALYSIS_DIR, 'stan_models')
SIM_DIR      = os.path.join(DATA_DIR, 'simulated')
RECOVERY_DIR = os.path.join(ANALYSIS_DIR, 'stan_results', 'recovery')
os.makedirs(RECOVERY_DIR, exist_ok=True)

MODEL = sys.argv[1]   # pgng_m1, pgng_m2, pgng_m3, pgng_m4

has_controllability = MODEL in ['pgng_m2', 'pgng_m4']

iter_warmup     = 5000
iter_sampling   = 1250
chains          = 4
thin            = 1
parallel_chains = 4



print(f'Fitting recovery model: {MODEL}...')

# ── Load simulated data ───────────────────────────────────────────────────────
data = pd.read_csv(os.path.join(SIM_DIR, f'pgng_sim_{MODEL}.csv'))
data['valence_num'] = (data['valence'] == 'win').astype(int)
print(f'  Participants: {data.subject.nunique()}, Trials: {len(data)}')

# ── Assemble Stan data ────────────────────────────────────────────────────────
N = len(data)
J = np.unique(data.subject.values, return_inverse=True)[-1] + 1
K = np.unique(data.stimulus.values, return_inverse=True)[-1] + 1
Y = data.choice.values.astype(int)
R = data.outcome.values.astype(int)
V = data.valence_num.values.astype(int)

dd = dict(N=N, J=J, K=K, Y=Y, R=R, V=V)
if has_controllability:
    C = (1 - data.controllable.values).astype(int)
    dd['C'] = C

print(f'  N={N}, J={J.max()}, K={K.max()}')

# ── Fit Stan model ────────────────────────────────────────────────────────────
stan_file = os.path.join(STAN_DIR, f'{MODEL}.stan')
StanModel = CmdStanModel(stan_file=stan_file)

StanFit = StanModel.sample(
    data=dd,
    chains=chains,
    iter_warmup=iter_warmup,
    iter_sampling=iter_sampling,
    parallel_chains=parallel_chains,
    seed=0,
    show_progress=True
)

# ── Diagnostics ───────────────────────────────────────────────────────────────
print(f'\nDivergences: {StanFit.divergences.sum()}')
summary = StanFit.summary(percentiles=(2.5, 50, 97.5), sig_figs=3)
print(f'R_hat >= 1.01: {len(summary[summary["R_hat"] >= 1.01])}')

# ── Save ──────────────────────────────────────────────────────────────────────
samples = StanFit.draws_pd()
cols = np.concatenate([
    samples.filter(regex='__').columns,
    samples.filter(regex='[a,b,c,d][0-9]_mu').columns,
    samples.filter(regex='sigma').columns,
    samples.filter(regex=r'[a,b,c,d][0-9]\[').columns,
])

fout = os.path.join(RECOVERY_DIR, MODEL)
samples[cols].to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
summary.to_csv(f'{fout}_summary.tsv', sep='\t')

print(f'\nSaved: {fout}.tsv.gz')
print(f'Saved: {fout}_summary.tsv')

print('\nGroup-level parameter estimates:')
group_params = summary[summary.index.str.endswith('_mu')]
print(group_params[['Mean', '2.5%', '97.5%', 'R_hat']].to_string())