import os, sys
import numpy as np
from pandas import read_csv
from tqdm import tqdm

# ============================================================
# Copy psis.py to analysis_full folder first!
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from psis import psisloo

ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~#

## I/O parameters.
## Run as: python fit_pgng_ppc_conpit.py pgng_m1 s1
stan_model = sys.argv[1]
session    = sys.argv[2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
data = read_csv(os.path.join(ANALYSIS_DIR, 'data', 'pgng.csv'))

## Map session name to number
session_map = {'s1': 1, 's2': 2}
session_num = session_map[session]
data = data[data['session'] == session_num].reset_index(drop=True)

## Restrict participants using reject.csv
reject = read_csv(os.path.join(ANALYSIS_DIR, 'data', 'reject.csv'))
data   = data[data.subject.isin(reject.query('reject==0').subject)].reset_index(drop=True)

## Format valence
data['valence_num'] = (data['valence'] == 'win').astype(int)

## Outcome already encoded as 0/1 in collate script

print(f'Session {session}: {data.subject.nunique()} participants, {len(data)} trials')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

N = len(data)
J = np.unique(data.subject, return_inverse=True)[-1]
K = np.unique(data.stimulus, return_inverse=True)[-1]

Y = data.choice.values.astype(int)
R = data.outcome.values.astype(int)
V = data.valence_num.values.astype(int)
C = (1 - data.controllable.values).astype(int)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Extract parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

StanFit = read_csv(os.path.join(ANALYSIS_DIR, 'stan_results', session, f'{stan_model}.tsv.gz'),
                   sep='\t', compression='gzip')

## Extract subject-level parameters
b1 = StanFit.filter(regex=r'b1\[').values
b2 = StanFit.filter(regex=r'b2\[').values
b3 = StanFit.filter(regex=r'b3\[').values
b4 = StanFit.filter(regex=r'b4\[').values
a1 = StanFit.filter(regex=r'a1\[').values
a2 = StanFit.filter(regex=r'a2\[').values
d1 = StanFit.filter(regex=r'd1\[').values   # controllability reward
d2 = StanFit.filter(regex=r'd2\[').values   # controllability punishment
c1 = StanFit.filter(regex=r'c1\[').values   # lapse rate

## Handle missing parameters (for M1 which has no d1/d2/c1)
if not np.any(b2): b2 = b1.copy()
if not np.any(b3): b3 = np.zeros_like(b1)
if not np.any(b4): b4 = b3.copy()
if not np.any(a2): a2 = a1.copy()
if not np.any(d1): d1 = np.zeros_like(b1)   # no controllability modulation
if not np.any(d2): d2 = np.zeros_like(b1)
if not np.any(c1): c1 = np.zeros_like(b1)   # no lapse rate

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Posterior predictive check.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
np.random.seed(47404)

def inv_logit(x):
    return 1. / (1 + np.exp(-x))

## Initialize Q-values
n_samp = len(StanFit)
Q = np.ones((n_samp, J.max()+1, K.max()+1, 2)) * 0.5

## Preallocate
Y_hat, Y_pred = np.zeros((2, N))
cll = np.zeros((n_samp, N))

## Main loop
for n in tqdm(range(N)):

    ## Assign trial-level parameters
    beta    = b1[:, J[n]] if V[n] else b2[:, J[n]]
    tau     = b3[:, J[n]] if V[n] else b4[:, J[n]]
    eta     = a1[:, J[n]] if V[n] else a2[:, J[n]]
    delta_c = d1[:, J[n]] if V[n] else d2[:, J[n]]
    xi      = c1[:, J[n]]

    ## Compute linear predictor with controllability modulation
    mu = beta * (Q[:, J[n], K[n], 1] - Q[:, J[n], K[n], 0]) + tau + delta_c * C[n]

    ## Apply lapse rate
    p = (0.5 * xi) + (1 - xi) * inv_logit(mu)

    ## Simulate choice
    Y_hat[n] = np.random.binomial(1, p).mean(axis=0)

    ## Compute choice likelihood
    cll[:, n] = np.where(Y[n], p, 1 - p)
    Y_pred[n] = cll[:, n].mean(axis=0)

    ## Compute prediction error
    delta = R[n] - Q[:, J[n], K[n], Y[n]]

    ## Update Q-values
    Q[:, J[n], K[n], Y[n]] += eta * delta

## Store posterior predictive variables
data['Y_hat']  = Y_hat.round(6)
data['Y_pred'] = Y_pred.round(6)

## Compute p_waic
pwaic = cll.var(axis=0)

## Compute PSIS-LOO (proper version from Vehtari et al.)
loo, loos, ku = psisloo(cll)
data['pwaic'] = pwaic.round(6)
data['loo']   = loos.round(6)
data['k_u']   = ku.round(6)

print(f'\nLOO: {loo:.1f}')
print(f'Bad k (>0.7): {np.sum(ku > 0.7)} / {len(ku)}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Columns to save (adapted from Zorowitz - no runsheet, added controllability)
cols = ['subject', 'session', 'block', 'trial', 'exposure', 'stimulus',
        'valence', 'action', 'robot_type', 'choice', 'accuracy',
        'outcome', 'sham', 'controllability', 'controllable',
        'Y_hat', 'Y_pred', 'pwaic', 'k_u', 'loo']
data = data[cols]

## Sort DataFrame.
data = data.sort_values(['subject','session','block','trial']).reset_index(drop=True)

## Save
f = os.path.join(ANALYSIS_DIR, 'stan_results', session, f'{stan_model}_ppc.csv')
data.to_csv(f, index=False)
print(f'Saved: {f}')
