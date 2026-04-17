import os
import re
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = "/Users/hai/Library/CloudStorage/OneDrive-King'sCollegeLondon/Desktop/Dissertation/analysis_pilot_20"
RAW_DIR      = os.path.join(ANALYSIS_DIR, 'data', 'raw')
OUTPUT_DIR   = os.path.join(ANALYSIS_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def encode_outcome(row):
    """Encode outcome as favourable=1, unfavourable=0."""
    if row['valence'] == 'win':
        return 1 if row['outcome'] > 5 else 0
    else:
        return 1 if row['outcome'] > -5 else 0

# ── Main loop ─────────────────────────────────────────────────────────────────
all_data = []

for session_num, folder_name in [(1, 's1'), (2, 's2')]:
    folder = os.path.join(RAW_DIR, folder_name)
    files  = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])

    print(f"\nProcessing session {session_num} ({len(files)} files)...")

    for fname in files:
        fpath = os.path.join(folder, fname)
        df    = pd.read_csv(fpath)

        # Get participant ID
        pid = df['participant_ID'].dropna().unique()
        if len(pid) == 0 or pid[0] in ['{participant_ID}']:
            print(f"  WARNING: No valid participant ID in {fname}, skipping")
            continue

        subject = str(pid[0])

        # Keep only real task trials (block > 0)
        data = df[df['trial_type'] == 'pit-trial'].copy()
        data = data[data['block'] > 0].reset_index(drop=True)

        if len(data) == 0:
            print(f"  WARNING: No task trials in {fname}, skipping")
            continue

        print(f"  {subject}: {len(data)} trials")

        # Select and clean columns
        cols = ['block', 'trial', 'valence', 'action', 'robot',
                'choice', 'accuracy', 'outcome', 'sham',
                'controllability', 'mini_block_index', 'rune']
        data = data[cols].copy()

        # Standardise choice
        data['choice'] = data['choice'].replace({' ': 1, '-1': 0})
        data['choice'] = pd.to_numeric(data['choice'], errors='coerce').fillna(0).astype(int)

        # Encode outcome
        data['outcome']     = pd.to_numeric(data['outcome'], errors='coerce')
        data['outcome_raw'] = data['outcome'].copy()
        data['outcome']     = data.apply(encode_outcome, axis=1)

        # Standardise robot
        data['robot']      = data['robot'].astype(int)
        data['robot_type'] = data['robot'].replace({1:'gw', 2:'ngw', 3:'gal', 4:'ngal'})

        # Controllability as binary
        data['controllable'] = (data['controllability'] == 'controllable').astype(int)

        # Convert types
        data['block']            = data['block'].astype(int)
        data['trial']            = data['trial'].astype(int)
        data['sham']             = data['sham'].astype(int)
        data['mini_block_index'] = data['mini_block_index'].astype(int)
        data['accuracy']         = data['accuracy'].astype(int)

        # Stimulus ID
        data['stimulus'] = data.groupby(['mini_block_index', 'robot', 'rune']).ngroup() + 1

        # Exposure
        data['exposure'] = data.groupby(['mini_block_index', 'robot', 'rune']).cumcount() + 1

        # Add subject and session
        data.insert(0, 'subject', subject)
        data.insert(1, 'session', session_num)

        all_data.append(data)

# ── Combine ───────────────────────────────────────────────────────────────────
all_data = pd.concat(all_data, ignore_index=True)
all_data = all_data.sort_values(['subject', 'session', 'block', 'trial'])
all_data = all_data.reset_index(drop=True)

# ── Summary ───────────────────────────────────────────────────────────────────
s1_subs   = set(all_data[all_data['session']==1]['subject'].unique())
s2_subs   = set(all_data[all_data['session']==2]['subject'].unique())
both_subs = s1_subs & s2_subs

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"Total trials:                {len(all_data)}")
print(f"Unique subjects:             {all_data['subject'].nunique()}")
print(f"Session 1 subjects:          {len(s1_subs)}")
print(f"Session 2 subjects:          {len(s2_subs)}")
print(f"Subjects with both sessions: {len(both_subs)}")

# Sanity check
trial_counts = all_data.groupby(['subject','session']).size()
if (trial_counts != 360).any():
    print("WARNING: Some subjects do not have 360 trials!")
    print(trial_counts[trial_counts != 360])
else:
    print("All subjects have 360 trials. ✓")

# ── Save ──────────────────────────────────────────────────────────────────────
all_data.to_csv(os.path.join(OUTPUT_DIR, 'pgng.csv'), index=False)
all_data[all_data['session']==1].to_csv(os.path.join(OUTPUT_DIR, 'pgng_s1.csv'), index=False)
all_data[all_data['session']==2].to_csv(os.path.join(OUTPUT_DIR, 'pgng_s2.csv'), index=False)
all_data[all_data['subject'].isin(both_subs)].to_csv(
    os.path.join(OUTPUT_DIR, 'pgng_reliability.csv'), index=False)

print(f"\nSaved to: {OUTPUT_DIR}")
print(f"  pgng.csv             ({len(all_data)} trials)")
print(f"  pgng_s1.csv          ({len(all_data[all_data['session']==1])} trials)")
print(f"  pgng_s2.csv          ({len(all_data[all_data['session']==2])} trials)")
print(f"  pgng_reliability.csv ({len(all_data[all_data['subject'].isin(both_subs)])} trials)")

# ── Verification ──────────────────────────────────────────────────────────────
data_check = pd.read_csv(os.path.join(OUTPUT_DIR, 'pgng.csv'))

session_check = data_check.groupby('subject')['session'].unique().reset_index()
session_check['n_sessions'] = session_check['session'].apply(len)
session_check['sessions']   = session_check['session'].apply(sorted)

print("\nSubject-Session mapping:")
print(session_check.to_string(index=False))

print(f"\nOnly S1: {session_check[session_check['n_sessions']==1]['subject'].tolist()}")
print(f"Both sessions: {session_check[session_check['n_sessions']==2]['subject'].tolist()}")

trial_counts = data_check.groupby(['subject','session']).size().reset_index(name='n_trials')
wrong = trial_counts[trial_counts['n_trials'] != 360]
if len(wrong):
    print(f"\nWARNING: Incorrect trial counts:")
    print(wrong.to_string(index=False))
else:
    print(f"\nAll participants have 360 trials ✓")