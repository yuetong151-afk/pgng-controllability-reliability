import os, sys
import numpy as np
from pandas import DataFrame, read_csv
from tqdm import tqdm

ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR  = os.path.join(ANALYSIS_DIR, 'stan_results')
np.random.seed(47404)

stan_model = sys.argv[1]   # e.g. pgng_m4
sessions   = ['s1', 's2']
pairings   = [1]            # only one pairing: s1 vs s2

# params per model
if stan_model == 'pgng_m1':
    params = ['b1', 'b2', 'b3', 'b4', 'a1', 'a2']
elif stan_model == 'pgng_m2':
    params = ['b1', 'b2', 'b3', 'b4', 'a1', 'a2', 'd1', 'd2']
elif stan_model == 'pgng_m3':
    params = ['b1', 'b2', 'b3', 'b4', 'a1', 'a2', 'c1']
elif stan_model == 'pgng_m4':
    params = ['b1', 'b2', 'b3', 'b4', 'a1', 'a2', 'd1', 'd2', 'c1']

n_shuffle = 5000
bounds    = [2.5, 97.5]

reliability = []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Split-half reliability
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
for param in tqdm(params):

    rho      = []
    rho_null = []

    for session in sessions:

        f = os.path.join(RESULTS_DIR, session, f'{stan_model}_sh_summary.tsv')
        if not os.path.exists(f):
            print(f'WARNING: {f} not found, skipping')
            continue
        summary = read_csv(f, sep='\t', index_col=0)

        arr = summary.T.filter(regex=f'{param}\\[').T['Mean'].values.reshape(-1, 2)
        if not np.any(arr): continue

        rho.append(np.corrcoef(arr.T)[0, 1])

        null = []
        for _ in range(n_shuffle):
            ix = np.random.choice(np.arange(len(arr)), len(arr), replace=True)
            null.append(np.corrcoef(arr[ix].T)[0, 1])
        rho_null.append(null)

    if not np.any(rho): continue

    lbs, ubs = np.percentile(rho_null, bounds, axis=1)
    mu       = np.mean(rho)
    lb, ub   = np.percentile(np.mean(rho_null, axis=0), bounds)

    reliability.append({'Param': param, 'Type': 'sh', 'Group': 0,
                        'Mean': mu, '2.5%': lb, '97.5%': ub})
    for i, (mu_i, lb_i, ub_i) in enumerate(zip(rho, lbs, ubs)):
        reliability.append({'Param': param, 'Type': 'sh', 'Group': i+1,
                            'Mean': mu_i, '2.5%': lb_i, '97.5%': ub_i})

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Test-retest reliability
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
for param in tqdm(params):

    rho      = []
    rho_null = []

    for pair in pairings:

        f = os.path.join(RESULTS_DIR, f'{stan_model}_sh_summary.tsv')
        if not os.path.exists(f):
            print(f'WARNING: {f} not found, skipping')
            continue
        summary = read_csv(f, sep='\t', index_col=0)

        arr = summary.T.filter(regex=f'{param}\\[').T['Mean'].values.reshape(-1, 2)
        if not np.any(arr): continue

        rho.append(np.corrcoef(arr.T)[0, 1])

        null = []
        for _ in range(n_shuffle):
            ix = np.random.choice(np.arange(len(arr)), len(arr), replace=True)
            null.append(np.corrcoef(arr[ix].T)[0, 1])
        rho_null.append(null)

    if not np.any(rho): continue

    lbs, ubs = np.percentile(rho_null, bounds, axis=1)
    mu       = np.mean(rho)
    lb, ub   = np.percentile(np.mean(rho_null, axis=0), bounds)

    reliability.append({'Param': param, 'Type': 'trt', 'Group': 0,
                        'Mean': mu, '2.5%': lb, '97.5%': ub})
    for i, (mu_i, lb_i, ub_i) in enumerate(zip(rho, lbs, ubs)):
        reliability.append({'Param': param, 'Type': 'trt', 'Group': i+1,
                            'Mean': mu_i, '2.5%': lb_i, '97.5%': ub_i})

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
reliability = DataFrame(reliability)
cols = reliability.columns[np.isin(reliability.dtypes,
                                   [np.float16, np.float32, np.float64])]
reliability[cols] = reliability[cols].round(6)

fout = os.path.join(RESULTS_DIR, f'{stan_model}_reliability.csv')
reliability.to_csv(fout, index=False)
print(f'\nSaved: {fout}')
print(reliability.to_string(index=False))