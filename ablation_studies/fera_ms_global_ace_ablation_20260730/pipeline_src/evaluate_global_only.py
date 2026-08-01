#!/usr/bin/env python3
"""Independent, read-only evaluation of a completed global-only pipeline."""
from __future__ import annotations
import argparse, importlib.util, os, pickle, sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch

ROOT=Path(os.environ.get('FERA_MS_ROOT', Path(__file__).resolve().parents[3])).resolve()
RUNS_ROOT=Path(os.environ.get('FERA_MS_RUNS_DIR', ROOT/'runs')).resolve()
ABL=ROOT/'ablation_studies/fera_ms_global_ace_ablation_20260730'
sys.path[:0]=[str(ROOT/'code/src'),str(ROOT/'code'),str(ROOT)]
from ms2spectra.workflow import load_config, init_dataset, init_dataloader
from ms2spectra.training import FragGNNPL

def module(path, name):
    spec=importlib.util.spec_from_file_location(name, path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--stage',choices=('R160','R172D','R184B'),required=True)
    ap.add_argument('--split',choices=('val','test'),required=True); ap.add_argument('--limit-batches',type=int,default=0); ap.add_argument('--out-dir',type=Path,default=None)
    a=ap.parse_args()
    if a.split=='test' and 'smoke' in str(a.run_root): raise RuntimeError('smoke evaluator must not access test')
    r170=module(ABL/'pipeline_src/refinement_steps/candidate_reranker.py','global_r170')
    r184=module(ABL/'pipeline_src/refinement_steps/spectrum_allocator.py','global_r184')
    cfg=load_config(RUNS_ROOT/'_config/template.yml', ABL/'configs'/f'seed_{a.seed}/R160.yml')
    cfg=r170.force_r160_arch(cfg)
    data=init_dataset(cfg,splits=(a.split,))[0]; dl=init_dataloader(data,cfg)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('formal evaluator requires CUDA')
    base=FragGNNPL(**cfg)
    state=r170.load_state_dict_any(a.run_root/'R160/model_last.pt')
    missing, unexpected=base.load_state_dict(state,strict=False)
    if unexpected: raise RuntimeError(f'R160 unexpected checkpoint keys: {unexpected}')
    base.to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)
    if a.stage == 'R160':
        rows=[]
        for bi,batch in enumerate(dl):
            batch=r170.move_to_device(batch,device); res=base._common_step(batch,split=a.split,log=False)
            bs=int(res['unique_id'].numel())
            true=r170.dense_by_round_bins(res['true_mzs'],res['true_logprobs'].exp(),res['true_batch_idxs'],bs,float(base.hparams.mz_max),.01)
            pred=r170.dense_by_round_bins(res['pred_mzs'],res['pred_logprobs'].exp(),res['pred_batch_idxs'],bs,float(base.hparams.mz_max),.01)
            cos=r170.cosine_dense(true,pred); jss=r170.jss_dense(true,pred); ce,_=r170.find_ce(batch); ce=ce.detach().cpu().reshape(-1); buckets=r170.ce_bucket_names(ce)
            for i,sid in enumerate(res['unique_id'].detach().cpu().reshape(-1).numpy().astype(int)): rows.append({'spec_id':int(sid),'ce':float(ce[i]),'ce_bucket':buckets[i],'cos':float(cos[i].cpu()),'jss':float(jss[i].cpu())})
            if a.limit_batches and bi+1>=a.limit_batches: break
        detail=pd.DataFrame(rows); table=r184.summarize_rows(rows)[0]
    else:
        with (a.run_root/'R172D/r170_regressor.pkl').open('rb') as f: r170pack=pickle.load(f)
        if a.stage == 'R172D':
            args=SimpleNamespace(max_eval_batches=int(a.limit_batches), max_extra_dims=32, local_bin_res=.01, eval_bin_res=.01, score_clip=6.0)
            alpha=float(pd.read_csv(a.run_root/'R172D/r170_alpha_val.csv').sort_values('val_cos',ascending=False).iloc[0]['alpha'])
            table,detail=r170.eval_split(base,r170pack['model'],r170pack.get('extra_schema',[]),dl,device,args,a.split,alpha,return_detail=True)
        else:
            allocpack=torch.load(a.run_root/'R184B/r184_allocator_best.pt',map_location=device,weights_only=False)
            args=SimpleNamespace(**dict(allocpack['args'])); args.max_eval_batches=int(a.limit_batches)
            extra=allocpack.get('extra_schema',r170pack.get('extra_schema',[]))
            allocator=r184.ResidualAllocator(int(allocpack['input_dim']),int(args.hidden),int(args.layers),float(args.dropout),float(args.score_clip)).to(device)
            allocator.load_state_dict(allocpack['model'],strict=True); allocator.eval()
            table, detail=r184.eval_split(base,allocator,r170pack['model'],extra,dl,device,r170,args,a.split)
    out=a.out_dir or (ABL/'evaluation'/f'seed_{a.seed}'); out.mkdir(parents=True,exist_ok=True)
    tag=f'{a.stage}_{a.split}'
    table.to_csv(out/f'{tag}_metrics.csv',index=False); detail.to_csv(out/f'{tag}_per_spectrum.csv',index=False)
    (out/f'{tag}.DONE').write_text('DONE\n')
    print(f'EVALUATION_DONE\t{tag}\t{len(detail)}')
if __name__=='__main__': main()
