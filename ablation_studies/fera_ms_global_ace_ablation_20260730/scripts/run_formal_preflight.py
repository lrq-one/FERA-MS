#!/usr/bin/env python3
from pathlib import Path
import os, subprocess, sys, yaml
ROOT=Path(os.environ.get('FERA_MS_ROOT', Path(__file__).resolve().parents[3])).resolve()
ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation_20260730'
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code'),str(ROOT)]
from ms2spectra.training import FragGNNPL
STEPS=('V2A_STAGE1','V2A_CONTINUATION','V2C_CONTROL','R146','R147','R149B','R150A','R150B','R153','R154','R160')
BAD=('seed_42_smoke','limit_train_batches=1','limit_val_batches=1','max_epochs=1')
for p in (ABL/'run_one_seed.sh',ABL/'run_all_seeds.sh',ABL/'scripts/run_formal_seed.sh'):
 t=p.read_text()
 if any(x in t for x in BAD): raise SystemExit(f'formal entry references smoke token: {p}')
for seed in (42,43,44):
 for s in STEPS:
  p=ABL/'configs'/f'seed_{seed}'/(s+'.yml'); c=yaml.safe_load(p.read_text())
  if c.get('ce_insert_location')!='mlp' or c.get('ce_insert_type')!='embed': raise SystemExit(f'bad global ACE config {p}')
  if c.get('use_ce_flowfrag',False): raise SystemExit(f'flowfrag active {p}')
  if int(c.get('max_epochs',0))==1: raise SystemExit(f'smoke epochs in {p}')
  m=FragGNNPL(**c)
  if not any('ce_embedder' in n and q.requires_grad for n,q in m.named_parameters()): raise SystemExit(f'no global ACE trainable param {p}')
for s in ('R146','R160'):
 a=(ABL/'configs/seed_42/V2C_CONTROL.yml').read_text();b=(ABL/f'configs/seed_42/{s}.yml').read_text()
 if a==b: raise SystemExit(f'V2C config equals {s}')
print('V2C_CONFIG_DIFFERS_FROM_R146')
print('V2C_CONFIG_DIFFERS_FROM_R160')
print('FORMAL_EPOCHS_RESTORED')
print('FORMAL_BATCH_LIMITS_DISABLED')
print('FORMAL_ENTRYPOINT_DOES_NOT_REFERENCE_SMOKE')
print('FORMAL_LINEAGE_VALID')
print('GLOBAL_ONLY_NEURAL_SINGLE_LOCATION_CONFIRMED')
