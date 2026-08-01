#!/usr/bin/env python3
"""Only formal global-only control flow; smoke is an explicit runtime mode."""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from pathlib import Path
ROOT=Path(os.environ.get('FERA_MS_ROOT', Path(__file__).resolve().parents[3])).resolve(); ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation_20260730'
STEPS=('V2A_STAGE1','V2A_CONTINUATION','V2C_CONTROL','R146','R147','R149B','R150A','R150B','R153','R154','R160','R172D','R184B','VALIDATION_EVALUATION','TEST_EVALUATION')
def manifest(seed, smoke):
 out=ABL/('runs/formal_full_chain_smoke_seed42' if smoke else f'runs/seed_{seed}')
 cfg=lambda s: ABL/'configs'/f'seed_{seed}'/(s+'.yml')
 p=[]; prev='RANDOM_INITIALIZATION'
 for s in STEPS[:11]:
  ck=out/s/'model_last.pt';p.append((s,'internal_neural_stage',str(cfg(s)),prev,str(ck),'false'));prev=str(ck)
 r172=out/'R172D/r170_regressor.pkl';p.append(('R172D','pipeline_src/refinement_steps/candidate_reranker.py',str(cfg('R172D')),prev,str(r172),'false'))
 r184=out/'R184B/r184_allocator_best.pt';p.append(('R184B','pipeline_src/refinement_steps/spectrum_allocator.py',str(cfg('R184B')),prev+' + '+str(r172),str(r184),'false'))
 p += [('VALIDATION_EVALUATION','evaluate_global_only_validation',str(cfg('FINAL_EVALUATION')),str(r184),str(ABL/f'evaluation/seed_{seed}'),'false'),('TEST_EVALUATION','evaluate_global_only_test',str(cfg('FINAL_EVALUATION')),str(r184),str(ABL/f'evaluation/seed_{seed}'),'true')]
 return out,p
def run(seed,smoke,dry):
 out,p=manifest(seed,smoke)
 for row in p: print('\t'.join(row))
 if dry:return
 # Delegates actual neural stage execution to the established isolated runner.
 cmd=[sys.executable,str(ABL/'pipeline_src/neural_formal_stages.py'),'--seed',str(seed),'--out',str(out)]
 if smoke:cmd+=['--smoke']
 subprocess.run(cmd,check=True)
 env=os.environ.copy();env['PYTHONPATH']=f'{ROOT}/code/src:{ROOT}/code:{ROOT}'
 r160=out/'R160/model_last.pt'; r172=out/'R172D';r184=out/'R184B';r172.mkdir(parents=True,exist_ok=True);r184.mkdir(parents=True,exist_ok=True)
 if not (r172/'DONE').is_file():
  args=[sys.executable,str(ABL/'pipeline_src/refinement_steps/candidate_reranker.py'),'-c',str(ABL/'configs'/f'seed_{seed}/R172D.yml'),'--ckpt_path',str(r160),'--out_dir',str(r172),'--seed',str(seed)]
  if smoke:args += ['--backend','sklearn','--max_train_rows','256','--max_eval_batches','1','--alpha_grid','0','--hgb_iter','1','--num_workers','1']
  subprocess.run(args,check=True,env=env)
  if not (r172/'r170_regressor.pkl').is_file(): raise RuntimeError('R172D did not produce its model')
  (r172/'DONE').write_text('DONE\n')
 if not (r184/'DONE').is_file():
  args=[sys.executable,str(ABL/'pipeline_src/refinement_steps/spectrum_allocator.py'),'-c',str(ABL/'configs'/f'seed_{seed}/R184B.yml'),'--ckpt_path',str(r160),'--regressor_path',str(r172/'r170_regressor.pkl'),'--out_dir',str(r184),'--seed',str(seed)]
  if smoke:args += ['--epochs','1','--max_train_batches','1','--max_eval_batches','1']
  subprocess.run(args,check=True,env=env)
  if not (r184/'r184_allocator_best.pt').is_file(): raise RuntimeError('R184B did not produce its allocator')
  (r184/'DONE').write_text('DONE\n')
 # The real evaluator is called after R184B; smoke is validation-only by rule.
 evdir=(out/'FINAL_EVALUATION') if smoke else (ABL/f'evaluation/seed_{seed}')
 for stage in ('R160','R172D','R184B'):
  evargs=[sys.executable,str(ABL/'pipeline_src/evaluate_global_only.py'),'--seed',str(seed),'--run-root',str(out),'--stage',stage,'--split','val','--out-dir',str(evdir)]
  if smoke: evargs += ['--limit-batches','1']
  subprocess.run(evargs,check=True,env=env)
 ev=evdir; (ev/'VALIDATION_EVALUATION.DONE').write_text('DONE\n')
 if not smoke:
  for stage in ('R160','R172D','R184B'):
   subprocess.run([sys.executable,str(ABL/'pipeline_src/evaluate_global_only.py'),'--seed',str(seed),'--run-root',str(out),'--stage',stage,'--split','test','--out-dir',str(ev)],check=True,env=env)
  (ev/'TEST_EVALUATION.DONE').write_text('DONE\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--smoke-mode',action='store_true');ap.add_argument('--dry-run',action='store_true');ap.add_argument('--aggregate',action='store_true');a=ap.parse_args()
 if a.aggregate:
  subprocess.run([sys.executable,str(ABL/'pipeline_src/aggregate_global_only.py')],check=True)
  return
 run(a.seed,a.smoke_mode,a.dry_run)
if __name__=='__main__':main()
