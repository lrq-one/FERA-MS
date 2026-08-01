# Rendering-component ablations

Two validated three-seed controls are retained:

- `without_mz_offset_expansion`: disables local m/z-offset peak expansion.
- `without_rendered_peak_gate`: disables the rendered-entry drop gate.

Run both controls for seeds 42/43/44 with:

```bash
bash ablation_studies/fera_ms_panelb_ablation/scripts/run_remaining_panelb_ablation.sh
```

The verified seed-42 templates define the single intended switch; the runner applies only that switch plus the paired seed. Logs, checkpoints, and result tables are written below ignored local directories.
