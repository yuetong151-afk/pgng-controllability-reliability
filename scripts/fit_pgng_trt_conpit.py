import os
import sys
import numpy as np
import pandas as pd
import atexit, shutil, tempfile
from cmdstanpy import CmdStanModel

ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR     = os.path.join(ANALYSIS_DIR, 'data')
STAN_DIR     = os.path.join(ANALYSIS_DIR, 'stan_models')
RESULTS_DIR  = os.path.join(ANALYSIS_DIR, 'stan_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Run as: python fit_pgng_trt_conpit.py pgng_m1_sh
#      or: python fit_pgng_trt_conpit.py pgng_m2_sh
#      or: python fit_pgng_trt_conpit.py pgng_m3_sh
stan_model = sys.argv[1]

iter_warmup     = 5000
iter_sampling   = 1250
chains          = 4
thin            = 1
parallel_chains = 4

print(f'Fitting: {stan_model}')

data   = pd.read_csv(os.path.join(DATA_DIR, 'pgng_reliability.csv'))
reject = pd.read_csv(os.path.join(DATA_DIR, 'reject.csv'))
data   = data[data['subject'].isin(reject.query('reject==0')['subject'].values)]
data   = data.reset_index(drop=True)

print(f'Participants: {data["subject"].nunique()}, Trials: {len(data)}')

data['valence_num'] = (data['valence'] == 'win').astype(int)

N = len(data)
J = np.unique(data['subject'].values, return_inverse=True)[-1] + 1
K = np.unique(data['stimulus'].values, return_inverse=True)[-1] + 1
M = data['session'].values.astype(int)
Y = data['choice'].values.astype(int)
R = data['outcome'].values.astype(int)
V = data['valence_num'].values.astype(int)
C = (1 - data['controllable'].values).astype(int)

print(f'N={N}, J={J.max()}, K={K.max()}')

if stan_model == 'pgng_m1_sh':
    dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V)
elif stan_model == 'pgng_m2_sh':
    dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V, C=C)
elif stan_model == 'pgng_m3_sh':
    dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V)
elif stan_model == 'pgng_m4_sh':
    dd = dict(N=N, J=J, K=K, M=M, Y=Y, R=R, V=V, C=C)
else:
    raise ValueError(f'Unknown model: {stan_model}')

stan_file    = os.path.join(STAN_DIR, f'{stan_model}.stan')
_stan_build  = tempfile.mkdtemp(prefix='cmdstan_pgng_trt_')
atexit.register(lambda d=_stan_build: shutil.rmtree(d, ignore_errors=True))
_stan_safe   = os.path.join(_stan_build, f'{stan_model}.stan')
shutil.copy2(stan_file, _stan_safe)
StanModel    = CmdStanModel(stan_file=_stan_safe)

StanFit   = StanModel.sample(
    data=dd, chains=chains,
    iter_warmup=iter_warmup, iter_sampling=iter_sampling,
    thin=thin, parallel_chains=parallel_chains,
    seed=42, show_progress=True
)

print(f'\nDivergences: {StanFit.divergences.sum()}')
summary = StanFit.summary(percentiles=(2.5, 50, 97.5), sig_figs=3)
print(f'R_hat >= 1.01: {len(summary[summary["R_hat"] >= 1.01])}')

fout    = os.path.join(RESULTS_DIR, stan_model)
samples = StanFit.draws_pd()
cols = np.concatenate([
    samples.filter(regex='__').columns,
    samples.filter(regex='[a,b,c,d][0-9]_mu').columns,
    samples.filter(regex='sigma').columns,
    samples.filter(regex=r'[a,b,c,d][0-9]\[').columns,
])
samples[cols].to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
summary.to_csv(f'{fout}_summary.tsv', sep='\t')
print(f'Saved: {fout}.tsv.gz')
