#!/usr/bin/env python3
"""Only formal global-only control flow; smoke is an explicit runtime mode."""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from pathlib import Path
ROOT=Path(os.environ.get('FERA_MS_ROOT', Path(__file__).resolve().parents[3])).resolve(); ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation'
from config_store import write_effective_config
STEPS=('backbone_training','retained_control_continuation','global_ace_control','formula_composition_refinement','collision_energy_response_refinement','neural_refinement','peak_distillation_warmup','peak_distillation_continuation','fragment_representation_refinement','bounded_residual_flow_refinement','final_peak_distillation','candidate_reranking','spectrum_allocation','validation_evaluation','test_evaluation')
def manifest(seed, smoke):
 out=ABL/(f'runs/smoke/seed_{seed}' if smoke else f'runs/seed_{seed}')
 cfg=lambda s: f"{ABL/'configs/pipeline.yml'}#seed={seed},stage={s}"
 p=[]; prev='RANDOM_INITIALIZATION'
 for s in STEPS[:11]:
  ck=out/s/'model_last.pt';p.append((s,'internal_neural_stage',str(cfg(s)),prev,str(ck),'false'));prev=str(ck)
 candidate_reranker=out/'candidate_reranking/candidate_reranker_regressor.pkl';p.append(('candidate_reranking','pipeline_src/refinement_steps/candidate_reranker.py',str(cfg('candidate_reranking')),prev,str(candidate_reranker),'false'))
 spectrum_allocator=out/'spectrum_allocation/spectrum_allocator.pt';p.append(('spectrum_allocation','pipeline_src/refinement_steps/spectrum_allocator.py',str(cfg('spectrum_allocation')),prev+' + '+str(candidate_reranker),str(spectrum_allocator),'false'))
 p += [('validation_evaluation','evaluate_global_only_validation',str(cfg('final_evaluation')),str(spectrum_allocator),str(ABL/f'evaluation/seed_{seed}'),'false'),('test_evaluation','evaluate_global_only_test',str(cfg('final_evaluation')),str(spectrum_allocator),str(ABL/f'evaluation/seed_{seed}'),'true')]
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
 final_peak_distillation=out/'final_peak_distillation/model_last.pt'; candidate_reranker=out/'candidate_reranking';spectrum_allocator=out/'spectrum_allocation';candidate_reranker.mkdir(parents=True,exist_ok=True);spectrum_allocator.mkdir(parents=True,exist_ok=True)
 if not (candidate_reranker/'DONE').is_file():
  config_path=write_effective_config(ABL,seed,'candidate_reranking',candidate_reranker/'input_config.yml')
  args=[sys.executable,str(ABL/'pipeline_src/refinement_steps/candidate_reranker.py'),'-c',str(config_path),'--ckpt_path',str(final_peak_distillation),'--out_dir',str(candidate_reranker),'--seed',str(seed)]
  if smoke:args += ['--backend','sklearn','--max_train_rows','256','--max_eval_batches','1','--alpha_grid','0','--hgb_iter','1','--num_workers','1']
  subprocess.run(args,check=True,env=env)
  if not (candidate_reranker/'candidate_reranker_regressor.pkl').is_file(): raise RuntimeError('candidate_reranking did not produce its model')
  (candidate_reranker/'DONE').write_text('DONE\n')
 if not (spectrum_allocator/'DONE').is_file():
  config_path=write_effective_config(ABL,seed,'spectrum_allocation',spectrum_allocator/'input_config.yml')
  args=[sys.executable,str(ABL/'pipeline_src/refinement_steps/spectrum_allocator.py'),'-c',str(config_path),'--ckpt_path',str(final_peak_distillation),'--regressor_path',str(candidate_reranker/'candidate_reranker_regressor.pkl'),'--out_dir',str(spectrum_allocator),'--seed',str(seed)]
  if smoke:args += ['--epochs','1','--max_train_batches','1','--max_eval_batches','1']
  subprocess.run(args,check=True,env=env)
  if not (spectrum_allocator/'spectrum_allocator.pt').is_file(): raise RuntimeError('spectrum_allocation did not produce its allocator')
  (spectrum_allocator/'DONE').write_text('DONE\n')
 # The real evaluator is called after spectrum_allocation; smoke is validation-only by rule.
 evdir=(out/'final_evaluation') if smoke else (ABL/f'evaluation/seed_{seed}')
 for stage in ('final_peak_distillation','candidate_reranking','spectrum_allocation'):
  evargs=[sys.executable,str(ABL/'pipeline_src/evaluate_global_only.py'),'--seed',str(seed),'--run-root',str(out),'--stage',stage,'--split','val','--out-dir',str(evdir)]
  if smoke: evargs += ['--limit-batches','1']
  subprocess.run(evargs,check=True,env=env)
 ev=evdir; (ev/'validation_evaluation.DONE').write_text('DONE\n')
 if not smoke:
  for stage in ('final_peak_distillation','candidate_reranking','spectrum_allocation'):
   subprocess.run([sys.executable,str(ABL/'pipeline_src/evaluate_global_only.py'),'--seed',str(seed),'--run-root',str(out),'--stage',stage,'--split','test','--out-dir',str(ev)],check=True,env=env)
  (ev/'test_evaluation.DONE').write_text('DONE\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--smoke-mode',action='store_true');ap.add_argument('--dry-run',action='store_true');ap.add_argument('--aggregate',action='store_true');a=ap.parse_args()
 if a.aggregate:
  subprocess.run([sys.executable,str(ABL/'pipeline_src/aggregate_global_only.py')],check=True)
  return
 run(a.seed,a.smoke_mode,a.dry_run)
if __name__=='__main__':main()
