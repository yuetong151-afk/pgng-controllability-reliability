import os
import sys
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR     = os.path.join(ANALYSIS_DIR, 'data')
SIM_DIR      = os.path.join(ANALYSIS_DIR, 'data', 'simulated')
os.makedirs(SIM_DIR, exist_ok=True)

MODEL = sys.argv[1]   # pgng_m1, pgng_m2, pgng_m3, pgng_m4
np.random.seed(33567)

# ── Model config ──────────────────────────────────────────────────────────────
has_controllability = MODEL in ['pgng_m2', 'pgng_m4']
has_lapse           = MODEL in ['pgng_m3', 'pgng_m4']

# ── Load task structure ───────────────────────────────────────────────────────
print(f'Loading task structure for {MODEL}...')
data   = pd.read_csv(os.path.join(DATA_DIR, 'pgng.csv'))
reject = pd.read_csv(os.path.join(DATA_DIR, 'reject.csv'))
data   = data[data['session'] == 1].reset_index(drop=True)
data   = data[data.subject.isin(reject.query('reject==0').subject)].reset_index(drop=True)
data['valence_num'] = (data['valence'] == 'win').astype(int)

# ── Expand to 100 simulated subjects ─────────────────────────────────────────
n_subj_sim      = 100
trials_per_subj = 360
n_repeat        = n_subj_sim // data.subject.nunique() + 1
data_sim        = pd.concat([data] * n_repeat, ignore_index=True)
data_sim        = data_sim.iloc[:n_subj_sim * trials_per_subj].reset_index(drop=True)
data_sim['subject'] = np.repeat(np.arange(n_subj_sim), trials_per_subj)

# shuffle the sequences of trials within each subject
shuffled = []
for subj in range(n_subj_sim):
    subj_data = data_sim[data_sim['subject'] == subj].copy()
    subj_data = subj_data.sample(frac=1, random_state=subj).reset_index(drop=True)
    subj_data['subject'] = subj
    shuffled.append(subj_data)
data_sim = pd.concat(shuffled, ignore_index=True)


N      = len(data_sim)
J      = np.unique(data_sim.subject.values, return_inverse=True)[-1]
K      = np.unique(data_sim.stimulus.values, return_inverse=True)[-1]
V      = data_sim.valence_num.values.astype(int)
C      = (1 - data_sim.controllable.values).astype(int)
R      = data_sim.outcome.values.astype(int)
n_subj = data_sim.subject.nunique()

print(f'  Simulated participants: {n_subj}, Trials: {N}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def inv_logit(x):
    return 1. / (1 + np.exp(-x))

def phi_approx(x):
    return inv_logit(0.07056 * x**3 + 1.5976 * x)

# ── Sample true parameters from prior ────────────────────────────────────────
n_params = 6
if has_controllability: n_params += 2
if has_lapse:           n_params += 1

theta_mu = np.random.normal(0, 1, n_params)
sigma    = np.abs(np.random.standard_t(3, size=n_params))
theta_pr = np.random.normal(0, 1, (n_params, n_subj))
theta    = theta_mu[:, None] + sigma[:, None] * theta_pr

b1_true = np.abs(theta[0] * 10)      
b2_true = np.abs(theta[1] * 10)      
b3_true = theta[2] * 5
b4_true = theta[3] * 5
a1_true = phi_approx(theta[4] - 1.5) 
a2_true = phi_approx(theta[5] - 1.5) 

idx = 6
if has_controllability:
    d1_true = theta[idx] * 5;   idx += 1
    d2_true = theta[idx] * 5;   idx += 1
else:
    d1_true = np.zeros(n_subj)
    d2_true = np.zeros(n_subj)

if has_lapse:
    c1_true = phi_approx(-2.0 + 0.5 * theta[idx])
else:
    c1_true = np.zeros(n_subj)

print(f'  b1 mean: {b1_true.mean():.3f}')
print(f'  a1 mean: {a1_true.mean():.3f}')
if has_controllability: print(f'  d1 mean: {d1_true.mean():.3f}')
if has_lapse:           print(f'  c1 mean: {c1_true.mean():.3f}')

# ── Simulate choices ──────────────────────────────────────────────────────────
print('Simulating choices...')
Q = np.ones((n_subj, K.max()+1, 2)) * 0.5
sim_choices = np.zeros(N, dtype=int)

for n in range(N):
    j = J[n]

    beta    = b1_true[j] if V[n] else b2_true[j]
    tau     = b3_true[j] if V[n] else b4_true[j]
    eta     = a1_true[j] if V[n] else a2_true[j]
    delta_c = d1_true[j] if V[n] else d2_true[j]
    xi      = c1_true[j]

    mu = beta * (Q[j, K[n], 1] - Q[j, K[n], 0]) + tau
    if has_controllability:
        mu += delta_c * C[n]

    if has_lapse:
        p = 0.5 * xi + (1 - xi) * inv_logit(mu)
    else:
        p = inv_logit(mu)

    p = np.clip(p, 1e-6, 1-1e-6)
    sim_choices[n] = np.random.binomial(1, p)

    delta = R[n] - Q[j, K[n], sim_choices[n]]
    Q[j, K[n], sim_choices[n]] += eta * delta

# ── Save ──────────────────────────────────────────────────────────────────────
sim_data = data_sim.copy()
sim_data['choice']   = sim_choices
sim_data['accuracy'] = (sim_choices == (data_sim['action'] == 'go').astype(int)).astype(int)

true_params = {
    'subject': range(n_subj),
    'b1': b1_true, 'b2': b2_true,
    'b3': b3_true, 'b4': b4_true,
    'a1': a1_true, 'a2': a2_true,
}
if has_controllability:
    true_params['d1'] = d1_true
    true_params['d2'] = d2_true
if has_lapse:
    true_params['c1'] = c1_true

sim_data.to_csv(os.path.join(SIM_DIR, f'pgng_sim_{MODEL}.csv'), index=False)
pd.DataFrame(true_params).to_csv(os.path.join(SIM_DIR, f'true_params_{MODEL}.csv'), index=False)

print(f'\nSaved: pgng_sim_{MODEL}.csv')
print(f'Saved: true_params_{MODEL}.csv')
print(f'Simulated choice rate: {sim_choices.mean():.3f}')