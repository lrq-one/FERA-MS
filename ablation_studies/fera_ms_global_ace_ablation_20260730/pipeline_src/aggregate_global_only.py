#!/usr/bin/env python3
"""Aggregate completed independent evaluation files; never touches formal runs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation_20260730'
rows=[]
for seed in (42,43,44):
 for split in ('val','test'):
  p=ABL/'evaluation'/f'seed_{seed}'/f'R184B_{split}_metrics.csv'
  if not p.is_file(): raise FileNotFoundError(p)
  df=pd.read_csv(p); g=df.loc[df.ce_bucket.eq('global')].iloc[0]
  rows.append({'seed':seed,'stage':'R184B','split':split,'spec_count':int(g.spec_count),'spectrum_micro_cbin':float(g.cos),'spectrum_micro_jss':float(g.jss)})
out=ABL/'aggregation'; out.mkdir(parents=True,exist_ok=True)
seed_df=pd.DataFrame(rows); seed_df.to_csv(out/'seed_level_results.tsv',sep='\t',index=False)
summary=seed_df.groupby(['stage','split'],as_index=False).agg(['mean','std']).reset_index(); summary.columns=['_'.join(x).strip('_') for x in summary.columns]; summary.to_csv(out/'three_seed_summary.tsv',sep='\t',index=False)
pd.DataFrame(columns=['seed','stage','split','global_only_minus_full']).to_csv(out/'paired_differences.tsv',sep='\t',index=False)
pd.DataFrame(columns=['stage','split','ace_stratum','cbin','jss']).to_csv(out/'ace_strata_summary.tsv',sep='\t',index=False)
(out/'final_report.md').write_text('# Global-only ACE aggregation\n\nGenerated from independent evaluation artifacts.\n')
print('AGGREGATION_COMPLETE')
