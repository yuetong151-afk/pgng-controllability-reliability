python3 -c "
import pandas as pd
summary = pd.read_csv('/Users/hai/Documents/25-26 KCL-3rd/Final Project/analysis_full/stan_results/pgng_m3_sh_summary.tsv', sep='\t', index_col=0)
print(summary.filter(regex='b1\[').head(10))
print()
print(summary.shape)
print()
print(summary.index[:20])
"

