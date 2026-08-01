#!/usr/bin/env python3
"""Shared neural-stage executor used by the formal pipeline and its smoke mode."""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import torch, yaml
try: import lightning.pytorch as pl
except ModuleNotFoundError: import pytorch_lightning as pl
try:
 from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
except ModuleNotFoundError:
 from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback
ROOT=Path(os.environ.get('FERA_MS_ROOT', Path(__file__).resolve().parents[3])).resolve(); ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation_20260730'
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code'),str(ROOT)]
from ms2spectra.training import FragGNNPL
from ms2spectra import workflow
STEPS=('V2A_STAGE1','V2A_CONTINUATION','V2C_CONTROL','R146','R147','R149B','R150A','R150B','R153','R154','R160')
class EpochSummary(Callback):
 def __init__(self, step): self.step=step
 def on_validation_end(self, trainer, pl_module):
  if trainer.sanity_checking: return
  means=getattr(pl_module,'val_mean_metrics',{})
  values=[]
  for key in sorted(means):
   value=means[key]
   if hasattr(value, 'detach'):
    value=value.detach().cpu()
    # FragGNNPL stores an epoch-history vector; report this epoch only.
    if getattr(value, 'ndim', 0) > 0:
     value=value[int(trainer.current_epoch)]
   values.append(f'val_{key}={float(value):.6f}')
  print(f'[EPOCH_SUMMARY] step={self.step} epoch={trainer.current_epoch + 1} ' + ' '.join(values), flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 prev=None; rows=[]; data=None
 for step in STEPS:
  d=a.out/step
  prior=d/'model_last.pt'
  # Resuming a stopped independent run is safe only after its explicit marker
  # and checkpoint have both been written.  Never overwrite a completed stage.
  if (d/'DONE').is_file() and prior.is_file():
   prev=prior; rows.append([step,str(prev),'RESUMED_DONE','[]']); continue
  resume_ckpt=None
  if d.exists():
   resume_ckpt=d/'last.ckpt'
   if not resume_ckpt.is_file(): raise RuntimeError(f'incomplete stage has no resumable last.ckpt: {d}')
  c=yaml.safe_load((ABL/'configs'/f'seed_{a.seed}'/(step+'.yml')).read_text())
  c.update({'accelerator':'gpu','devices':1,'num_workers':0,'eval_test_split':False})
  if a.smoke:c.update({'max_epochs':1,'min_epochs':1})
  if data is None or step=='R146':
   tr,va=workflow.init_dataset(c,splits=('train','val'));data=(workflow.init_dataloader(tr,c),workflow.init_dataloader(va,c))
  m=FragGNNPL(**c); missing=[];unexpected=[]
  if prev:
   inc=m.load_state_dict(torch.load(prev,map_location='cpu',weights_only=False)['state_dict'],strict=False);missing=list(inc.missing_keys);unexpected=list(inc.unexpected_keys)
   allowed=('model.formula_comp_residual_head',) if step=='R146' else ()
   if unexpected or any(not k.startswith(allowed) for k in missing):raise RuntimeError(f'{step} lineage mismatch {missing} {unexpected}')
  if not d.exists():
   d.mkdir(parents=True,exist_ok=False);(d/'effective_config.yml').write_text(yaml.safe_dump(c,sort_keys=False))
  monitor=str(c.get('checkpoint_metric','val_mean_loss_epoch/mean')); mode=str(c.get('checkpoint_metric_mode','max'))
  checkpoint=ModelCheckpoint(dirpath=d,filename='best',monitor=monitor,mode=mode,save_top_k=1,save_last=True,save_on_train_epoch_end=False)
  t=pl.Trainer(accelerator='gpu',devices=1,max_epochs=int(c['max_epochs']),limit_train_batches=1 if a.smoke else 1.0,limit_val_batches=1 if a.smoke else 1.0,num_sanity_val_steps=0,logger=False,callbacks=[TQDMProgressBar(refresh_rate=20),EpochSummary(step),checkpoint],enable_checkpointing=True,enable_progress_bar=True,default_root_dir=str(d),log_every_n_steps=max(1,int(c.get('log_every_n_steps',50))))
  t.fit(m,train_dataloaders=data[0],val_dataloaders=data[1],ckpt_path=str(resume_ckpt) if resume_ckpt else None);prev=d/'model_last.pt';torch.save({'state_dict':m.state_dict(),'config':c},prev);(d/'DONE').write_text(f'DONE\nbest_checkpoint={checkpoint.best_model_path}\nlast_checkpoint={checkpoint.last_model_path}\n');rows.append([step,str(prev),str(missing),str(unexpected)])
 with (a.out/'neural_lineage.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['step','checkpoint','missing','unexpected']);w.writerows(rows)
if __name__=='__main__':main()
