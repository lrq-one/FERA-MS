#!/usr/bin/env python3
"""
Stage A: Unified base model -> retained control-style low-LR all-parameter transfer.

"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"YAML顶层必须是字典: {path}")
    return data


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path.resolve()

    attempted = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(
        f"找不到{label}，已检查：\n{attempted}"
    )


def rotate_directory(path: Path) -> Path | None:
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{timestamp}")
    path.rename(backup)
    return backup


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = (
            getattr(streams[0], "encoding", None)
            or "utf-8"
        )
        self.errors = (
            getattr(streams[0], "errors", None)
            or "strict"
        )

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


def parse_base_model_validation(
    base_model_log: Path,
    output_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "log": str(base_model_log),
        "found": False,
        "best": None,
        "last": None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not base_model_log.is_file():
        text = (
            f"base model validation log not found:\n"
            f"{base_model_log}\n"
        )
        output_path.write_text(text, encoding="utf-8")
        return result

    log_text = base_model_log.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    patterns = [
        re.compile(
            r"Epoch\s+(\d+),\s*step\s+(\d+)--\s*"
            r"(val_[^:\n]*cos_sim[^:\n]*):\s*"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        ),
        re.compile(
            r"(?:Epoch|epoch)[ =:]+(\d+).*?"
            r"(val_[^\s:=,]*cos_sim[^\s:=,]*).*?"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        ),
    ]

    rows: list[dict[str, Any]] = []

    for pattern_index, pattern in enumerate(patterns):
        matches = list(pattern.finditer(log_text))
        if not matches:
            continue

        for match in matches:
            groups = match.groups()

            if pattern_index == 0:
                epoch = int(groups[0])
                step = int(groups[1])
                metric = groups[2]
                value = float(groups[3])
            else:
                epoch = int(groups[0])
                step = -1
                metric = groups[1]
                value = float(groups[2])

            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "metric": metric,
                    "value": value,
                }
            )

        if rows:
            break

    if rows:
        best = max(rows, key=lambda x: x["value"])
        last = max(rows, key=lambda x: (x["epoch"], x["step"]))

        result["found"] = True
        result["best"] = best
        result["last"] = last

        text = (
            f"base model LOG: {base_model_log}\n"
            f"base model BEST VAL: {best}\n"
            f"base model LAST VAL: {last}\n"
            f"NUM MATCHES: {len(rows)}\n"
        )
    else:
        text = (
            f"base model LOG: {base_model_log}\n"
            "base model validation cosine not found by supported patterns\n"
        )

    output_path.write_text(text, encoding="utf-8")
    return result



def materialize_training_config(
    bundle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bundle_path = Path(bundle_path)
    output_dir = Path(output_dir)

    bundle = load_yaml(bundle_path)

    names = {
        "template": "template.yml",
        "base_stage": "base_stage.yml",
        "continuation_stage": "continuation_stage.yml",
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result: dict[str, Path] = {}

    for key, filename in names.items():
        value = bundle.get(key)

        if not isinstance(value, dict):
            raise KeyError(
                f"统一配置缺少字典节：{key}"
            )

        path = output_dir / filename

        path.write_text(
            yaml.safe_dump(
                value,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        result[key] = path

    return result

def prepare_effective_config(
    template_cfg: dict[str, Any],
    base_model_custom: dict[str, Any],
    retained_control_custom: dict[str, Any],
    epochs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_base_config = deep_merge(template_cfg, base_model_custom)
    effective_retained_control_config = deep_merge(template_cfg, retained_control_custom)

    transfer = copy.deepcopy(effective_base_config)
    copied: dict[str, Any] = {}

    exact_training_keys = {
        "optimizer",
        "lr",
        "weight_decay",
        "lr_schedule",
        "lr_decay_rate",
        "lr_warmup_steps",
        "lr_decay_steps",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accumulate_grad_batches",
        "precision",
        "use_tensor_float32",
        "deterministic",
        "automatic_optimization",
    }

    for key in sorted(exact_training_keys):
        if key in effective_retained_control_config:
            transfer[key] = copy.deepcopy(effective_retained_control_config[key])
            copied[key] = copy.deepcopy(effective_retained_control_config[key])

    for key, value in effective_retained_control_config.items():
        key_lower = key.lower()

        is_scheduler_key = (
            "scheduler" in key_lower
            or key_lower.startswith("lr_")
            or key_lower.endswith("_lr")
            or key_lower.startswith("optimizer_")
        )

        if is_scheduler_key:
            transfer[key] = copy.deepcopy(value)
            copied[key] = copy.deepcopy(value)

    transfer["min_epochs"] = 1
    transfer["max_epochs"] = int(epochs)

    if "spectrum_refiner_train_scope" in transfer:
        transfer["spectrum_refiner_train_scope"] = "all"
        copied["spectrum_refiner_train_scope"] = "all"

    if "train_scope" in transfer:
        transfer["train_scope"] = "all"
        copied["train_scope"] = "all"

    if "use_support_oracle_support_oracle_reweight_loss" in transfer:
        transfer["use_support_oracle_support_oracle_reweight_loss"] = False
        copied["use_support_oracle_support_oracle_reweight_loss"] = False

    if "support_oracle_support_oracle_weight" in transfer:
        transfer["support_oracle_support_oracle_weight"] = 0.0
        copied["support_oracle_support_oracle_weight"] = 0.0

    transfer["eval_test_split"] = True
    transfer["disable_checkpoints"] = False
    transfer["delete_checkpoints"] = False
    transfer["upload_checkpoints"] = False
    transfer["checkpoint_save_last"] = True
    transfer["checkpoint_metric_mode"] = "max"

    checkpoint_metric = transfer.get(
        "checkpoint_metric",
        "val_cos_sim_0.01_epoch/mean",
    )

    if "cos_sim" not in str(checkpoint_metric):
        checkpoint_metric = "val_cos_sim_0.01_epoch/mean"

    transfer["checkpoint_metric"] = checkpoint_metric

    copied.update(
        {
            "min_epochs": transfer["min_epochs"],
            "max_epochs": transfer["max_epochs"],
            "eval_test_split": True,
            "checkpoint_metric": transfer["checkpoint_metric"],
            "checkpoint_metric_mode": "max",
            "checkpoint_save_last": True,
        }
    )

    assert not bool(
        transfer.get(
            "use_support_oracle_support_oracle_reweight_loss",
            False,
        )
    )
    assert float(
        transfer.get("support_oracle_support_oracle_weight", 0.0)
    ) == 0.0

    return transfer, copied


def create_runtime_links(root: Path, run_dir: Path) -> None:
    link_names = [
        "data",
        "code",
        "config",
        "stages",    ]

    for name in link_names:
        target = root / name
        link = run_dir / name

        if not target.exists() or link.exists():
            continue

        link.symlink_to(
            target,
            target_is_directory=target.is_dir(),
        )


def install_local_pythonpath(root: Path) -> None:
    candidates = [
        root / "code" / "src",
        root / "code",
        root,
        root / "src",
    ]

    additions = [
        str(path)
        for path in candidates
        if path.exists()
    ]

    current = os.environ.get("PYTHONPATH", "")
    if current:
        additions.append(current)

    os.environ["PYTHONPATH"] = os.pathsep.join(additions)

    for path in reversed(additions):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="retained control-style transfer epochs; first run should use 3",
    )
    parser.add_argument(
        "--run-name",
        default="base_model_retained_control_reproduce",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="validation cosine early-stop patience",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1.0e-4,
        help="minimum validation cosine improvement",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"工作目录不存在: {root}")

    base_model_cfg_path = root / (
        "runs/_config/base_stage.yml"
    )
    retained_control_cfg_path = root / (
        "runs/_config/continuation_stage.yml"
    )
    base_model_ckpt_path = root / (
        "runs/base_model/model_epoch39.ckpt"
    )
    base_model_log_path = root / (
        "runs/base_model/training.log"
    )

    template_path = first_existing(
        [
            root / "runs/_config/template.yml",
            root / "runs/_config/template.yml",
        ],
        "本地 template.yml",
    )

    required = {
        "base model配置": base_model_cfg_path,
        "retained control配置": retained_control_cfg_path,
        "base model checkpoint": base_model_ckpt_path,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "以下本地文件不存在：\n" + "\n".join(missing)
        )

    exact_val_path = (
        root
        / "artifacts"
        / "base_model_exact_val.txt"
    )

    base_model_val_result = parse_base_model_validation(
        base_model_log_path,
        exact_val_path,
    )

    run_dir = root / "runs" / args.run_name
    backup_dir = rotate_directory(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "training.log"
    log_file = log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print("=" * 78)
        print("LOCAL base model -> retained control TRANSFER")
        print("=" * 78)
        print(f"ROOT          : {root}")
        print(f"base model CONFIG     : {base_model_cfg_path}")
        print(f"retained control CONFIG   : {retained_control_cfg_path}")
        print(f"base model CHECKPOINT : {base_model_ckpt_path}")
        print(f"TEMPLATE      : {template_path}")
        print(f"RUN DIR       : {run_dir}")
        print(f"EPOCHS        : {args.epochs}")

        if backup_dir is not None:
            print(f"OLD RUN BACKUP: {backup_dir}")

        template_cfg = load_yaml(template_path)
        base_model_custom = load_yaml(base_model_cfg_path)
        retained_control_custom = load_yaml(retained_control_cfg_path)

        transfer_cfg, copied_training_settings = (
            prepare_effective_config(
                template_cfg=template_cfg,
                base_model_custom=base_model_custom,
                retained_control_custom=retained_control_custom,
                epochs=args.epochs,
            )
        )

        transfer_cfg["wandb_name"] = args.run_name
        transfer_cfg["wandb_group"] = (
            "FERA_MS_BASE_CONFIG"
        )
        transfer_cfg["seed"] = 42

        if "early_stopping" in transfer_cfg:
            transfer_cfg["early_stopping"] = False

        resolved_cfg_path = (
            run_dir / "config.yml"
        )

        with resolved_cfg_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                transfer_cfg,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

        create_runtime_links(root, run_dir)
        install_local_pythonpath(root)

        manifest = {
            "stage": "A",
            "name": args.run_name,
            "root": str(root),
            "random_initialization": False,
            "weight_initialization": "Unified base model epoch39",
            "optimizer_state_restored": False,
            "epoch_state_restored": False,
            "base_model_config": str(base_model_cfg_path),
            "retained_control_config": str(retained_control_cfg_path),
            "template": str(template_path),
            "base_model_checkpoint": str(base_model_ckpt_path),
            "base_model_checkpoint_sha256": sha256_file(
                base_model_ckpt_path
            ),
            "epochs": args.epochs,
            "early_stopping": {
                "enabled": True,
                "monitor": transfer_cfg["checkpoint_metric"],
                "mode": "max",
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "active_stage": "retained control low-LR transfer",
            },
            "copied_retained_control_training_settings": (
                copied_training_settings
            ),
            "base_model_validation_parse": base_model_val_result,
            "selection_metric": transfer_cfg[
                "checkpoint_metric"
            ],
            "test_policy": (
                "test once after fit using best validation checkpoint"
            ),
        }

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n[retained control TRANSFER SETTINGS]")
        print(
            json.dumps(
                copied_training_settings,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n[base model VALIDATION AUDIT]")
        print(
            exact_val_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        )

        print("\n[LOAD LOCAL RUNNER]")

        import ms2spectra.workflow as runner

        try:
            from lightning.pytorch.callbacks import (
                EarlyStopping as LocalEarlyStopping,
            )
        except ModuleNotFoundError:
            from pytorch_lightning.callbacks import (
                EarlyStopping as LocalEarlyStopping,
            )

        original_trainer_cls = runner.pl.Trainer

        def trainer_with_local_early_stop(
            *trainer_args,
            callbacks=None,
            **trainer_kwargs,
        ):
            callback_list = list(callbacks or [])

            callback_list.append(
                LocalEarlyStopping(
                    monitor=transfer_cfg["checkpoint_metric"],
                    mode="max",
                    patience=int(args.patience),
                    min_delta=float(args.min_delta),
                    verbose=True,
                    strict=True,
                )
            )

            print(
                "[LOCAL EARLY STOP] "
                f"monitor={transfer_cfg['checkpoint_metric']}, "
                f"mode=max, "
                f"patience={args.patience}, "
                f"min_delta={args.min_delta}"
            )

            return original_trainer_cls(
                *trainer_args,
                callbacks=callback_list,
                **trainer_kwargs,
            )

        runner.pl.Trainer = trainer_with_local_early_stop

        if not hasattr(runner, "FragGNNPL"):
            raise RuntimeError(
                "本地 ms2spectra.workflow 中没有 FragGNNPL，"
                "拒绝猜测其他模型类。"
            )

        original_init = runner.FragGNNPL.__init__
        load_audit: dict[str, Any] = {}

        def patched_init(self, *model_args, **model_kwargs):
            original_init(
                self,
                *model_args,
                **model_kwargs,
            )

            print(
                "\n[base model WEIGHT LOAD] "
                f"loading strictly from {base_model_ckpt_path}"
            )

            checkpoint = torch.load(
                base_model_ckpt_path,
                map_location="cpu",
            )

            if (
                isinstance(checkpoint, dict)
                and "state_dict" in checkpoint
            ):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            if not isinstance(state_dict, dict):
                raise TypeError(
                    "base model checkpoint 中没有可用的 state_dict"
                )

            self.load_state_dict(
                state_dict,
                strict=True,
            )

            trainable = sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )
            total = sum(
                p.numel()
                for p in self.parameters()
            )

            load_audit.update(
                {
                    "strict": True,
                    "state_key_count": len(state_dict),
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                }
            )

            print(
                "[base model WEIGHT LOAD] strict=True, "
                f"state_keys={len(state_dict)}, "
                f"trainable={trainable}, total={total}"
            )

        runner.FragGNNPL.__init__ = patched_init

        old_cwd = Path.cwd()
        os.chdir(run_dir)

        try:
            print("\n[START LOCAL TRAINING]")
            print(
                "注意：这是新的 optimizer 状态，"
                "不是从 base model optimizer/epoch 恢复。"
            )

            runner.init_run(
                str(template_path),
                str(resolved_cfg_path),
                "disabled",
                None,
            )
        finally:
            os.chdir(old_cwd)
            runner.FragGNNPL.__init__ = original_init
            runner.pl.Trainer = original_trainer_cls

        tmp_ckpt_dir = run_dir / "tmp_ckpt"
        ckpt_files = sorted(
            tmp_ckpt_dir.glob("*.ckpt")
        )

        best_candidates = [
            path
            for path in ckpt_files
            if path.name != "last.ckpt"
        ]

        canonical_best = None
        if best_candidates:
            source_best = max(
                best_candidates,
                key=lambda p: p.stat().st_mtime,
            )
            canonical_best = (
                run_dir / "model_best.ckpt"
            )
            shutil.copy2(
                source_best,
                canonical_best,
            )

            print(
                f"\n[BEST CHECKPOINT] {source_best}"
            )
            print(
                f"[CANONICAL COPY]  {canonical_best}"
            )

        last_ckpt = tmp_ckpt_dir / "last.ckpt"
        canonical_last = None

        if last_ckpt.is_file():
            canonical_last = (
                run_dir / "model_last.ckpt"
            )
            shutil.copy2(
                last_ckpt,
                canonical_last,
            )
            print(
                f"[LAST CHECKPOINT]  {canonical_last}"
            )

        manifest.update(
            {
                "weight_load_audit": load_audit,
                "completed": True,
                "checkpoint_files": [
                    str(p) for p in ckpt_files
                ],
                "canonical_best": (
                    str(canonical_best)
                    if canonical_best
                    else None
                ),
                "canonical_last": (
                    str(canonical_last)
                    if canonical_last
                    else None
                ),
            }
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("[SUCCESS] 阶段A本地迁移训练完成")
        print(f"日志     : {log_path}")
        print(f"配置     : {resolved_cfg_path}")
        print(f"manifest : {manifest_path}")
        print(f"base model审计   : {exact_val_path}")
        print("=" * 78)

        return 0

    except Exception:
        print("\n[FATAL] 阶段A失败，已停止，不会静默降级：")
        traceback.print_exc()

        failure_path = run_dir / "FAILED.txt"
        failure_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        return 1

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
