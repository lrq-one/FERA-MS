#!/usr/bin/env python3
from pathlib import Path
import os, subprocess, sys, yaml
ROOT=Path(os.environ.get('FERA_MS_ROOT', Path(__file__).resolve().parents[3])).resolve()
ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation'
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code'),str(ROOT)]
sys.path.insert(0,str(ABL/'pipeline_src'))
from config_store import load_locked_config
from ms2spectra.training import FragGNNPL
STEPS=('backbone_training','retained_control_continuation','global_ace_control','formula_composition_refinement','collision_energy_response_refinement','neural_refinement','peak_distillation_warmup','peak_distillation_continuation','fragment_representation_refinement','bounded_residual_flow_refinement','final_peak_distillation')
BAD=('seed_42_smoke','limit_train_batches=1','limit_val_batches=1','max_epochs=1')
for p in (ABL/'run_one_seed.sh',ABL/'run_all_seeds.sh',ABL/'scripts/run_formal_seed.sh'):
 t=p.read_text()
 if any(x in t for x in BAD): raise SystemExit(f'formal entry references smoke token: {p}')
for seed in (42,43,44):
 for s in STEPS:
  c=load_locked_config(ABL,seed,s)
  label=f'seed={seed}, stage={s}'
  if c.get('ce_insert_location')!='mlp' or c.get('ce_insert_type')!='embed': raise SystemExit(f'bad global ACE config {label}')
  if c.get('use_ce_flowfrag',False): raise SystemExit(f'flowfrag active {label}')
  if int(c.get('max_epochs',0))==1: raise SystemExit(f'smoke epochs in {label}')
  m=FragGNNPL(**c)
  if not any('ce_embedder' in n and q.requires_grad for n,q in m.named_parameters()): raise SystemExit(f'no global ACE trainable param {label}')
for s in ('formula_composition_refinement','final_peak_distillation'):
 a=load_locked_config(ABL,42,'global_ace_control');b=load_locked_config(ABL,42,s)
 if a==b: raise SystemExit(f'global_ace_control config equals {s}')
print('global_ace_control_CONFIG_DIFFERS_FROM_formula_composition_refinement')
print('global_ace_control_CONFIG_DIFFERS_FROM_final_peak_distillation')
print('FORMAL_EPOCHS_RESTORED')
print('FORMAL_BATCH_LIMITS_DISABLED')
print('FORMAL_ENTRYPOINT_DOES_NOT_REFERENCE_SMOKE')
print('FORMAL_LINEAGE_VALID')
print('GLOBAL_ONLY_NEURAL_SINGLE_LOCATION_CONFIRMED')
