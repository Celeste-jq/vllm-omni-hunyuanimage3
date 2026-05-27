#!/usr/bin/env python3
"""Compare HunyuanImage3 AR ViT DP on/off with multi-request batching and profiling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEPLOY = REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_ar.yaml"


def parse_profiler_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--profiler-config must be valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("--profiler-config must be a JSON object")
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HunyuanImage3 img2text AR-only comparison with ViT DP on/off and torch profiling."
    )
    parser.add_argument("--model", required=True, help="Model path or HF ID.")
    parser.add_argument("--image-path", required=True, help="Input image path. Comma-separated paths also supported.")
    parser.add_argument(
        "--deploy-config",
        default=str(DEFAULT_DEPLOY),
        help="Base AR deploy config. The script derives DP-on/off temp configs from it.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Prompt reused across the whole request batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of requests submitted in one generate() call.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup generate() calls per mode before timing/profiling.",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Measured generate() calls per mode. Profiling stays active across these runs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for logs, summaries, temporary deploy configs, and profiler artifacts.",
    )
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help=(
            "JSON profiler config forwarded to Omni, for example "
            '\'{"profiler":"torch","torch_profiler_dir":"./perf","torch_profiler_record_shapes":true}\''
        ),
    )
    parser.add_argument(
        "--profiler-record-shapes",
        action="store_true",
        help="Convenience flag for the default torch profiler config.",
    )
    parser.add_argument(
        "--profiler-with-stack",
        action="store_true",
        help="Convenience flag for the default torch profiler config.",
    )
    parser.add_argument(
        "--profiler-with-memory",
        action="store_true",
        help="Convenience flag for the default torch profiler config.",
    )
    parser.add_argument(
        "--profiler-use-gzip",
        action="store_true",
        help="Convenience flag for the default torch profiler config.",
    )
    parser.add_argument(
        "--profiler-with-flops",
        action="store_true",
        help="Convenience flag for the default torch profiler config.",
    )
    parser.add_argument(
        "--profiler-dump-cuda-time-total",
        action="store_true",
        help="Convenience flag for the default torch profiler config.",
    )
    parser.add_argument(
        "--enable-ar-profiler",
        action="store_true",
        default=True,
        help="Enable AR stage timing aggregation in Omni outputs.",
    )
    parser.add_argument(
        "--disable-ar-profiler",
        dest="enable_ar_profiler",
        action="store_false",
        help="Disable AR stage timing aggregation in Omni outputs.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Pass log_stats=True to Omni.",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=600,
        help="Initialization timeout in seconds.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Disable torch.compile for deterministic profiling.",
    )
    parser.add_argument(
        "--disable-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow compiled mode if your environment wants it.",
    )

    # Internal child-runner flags.
    parser.add_argument("--_mode-run", choices=["vit_dp_on", "vit_dp_off"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_resolved-output-root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_resolved-profiler-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_resolved-deploy-config", default=None, help=argparse.SUPPRESS)
    return parser


def default_profiler_config(args: argparse.Namespace, profiler_dir: Path) -> dict[str, Any]:
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(profiler_dir),
        "torch_profiler_record_shapes": bool(args.profiler_record_shapes),
        "torch_profiler_with_stack": bool(args.profiler_with_stack),
        "torch_profiler_with_memory": bool(args.profiler_with_memory),
        "torch_profiler_use_gzip": bool(args.profiler_use_gzip),
        "torch_profiler_with_flops": bool(args.profiler_with_flops),
        "torch_profiler_dump_cuda_time_total": bool(args.profiler_dump_cuda_time_total),
    }


def ensure_path_exists(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def make_output_root(requested: str | None) -> Path:
    if requested:
        root = Path(requested).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = (REPO_ROOT / "test_outputs" / f"hunyuan_image3_vit_dp_profile_compare_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_temp_deploy_configs(base_deploy_path: Path, output_root: Path) -> dict[str, Path]:
    base_text = base_deploy_path.read_text(encoding="utf-8")
    if "mm_encoder_tp_mode: data" not in base_text:
        raise ValueError(
            f"Base deploy config does not contain 'mm_encoder_tp_mode: data': {base_deploy_path}"
        )

    deploy_dir = output_root / "deploy_configs"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    on_path = deploy_dir / "hunyuan_image3_ar_vit_dp_on.yaml"
    off_path = deploy_dir / "hunyuan_image3_ar_vit_dp_off.yaml"

    off_lines: list[str] = []
    for line in base_text.splitlines():
        if "mm_encoder_tp_mode:" in line:
            indent = line[: len(line) - len(line.lstrip())]
            off_lines.append(f"{indent}# mm_encoder_tp_mode removed for ViT TP baseline")
            continue
        off_lines.append(line)
    off_text = "\n".join(off_lines) + ("\n" if base_text.endswith("\n") else "")

    on_path.write_text(base_text, encoding="utf-8")
    off_path.write_text(off_text, encoding="utf-8")
    return {"vit_dp_on": on_path, "vit_dp_off": off_path}


def launch_child(args: argparse.Namespace, output_root: Path, mode: str, deploy_config: Path) -> tuple[int, Path]:
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = mode_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    log_file = mode_dir / f"{mode}.log"

    child_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--image-path",
        args.image_path,
        "--deploy-config",
        str(args.deploy_config),
        "--prompt",
        args.prompt,
        "--batch-size",
        str(args.batch_size),
        "--warmup-runs",
        str(args.warmup_runs),
        "--profile-runs",
        str(args.profile_runs),
        "--init-timeout",
        str(args.init_timeout),
        "--_mode-run",
        mode,
        "--_resolved-output-root",
        str(output_root),
        "--_resolved-profiler-dir",
        str(profiler_dir),
        "--_resolved-deploy-config",
        str(deploy_config),
    ]

    if args.profiler_config is not None:
        child_cmd.extend(["--profiler-config", json.dumps(args.profiler_config)])
    if args.profiler_record_shapes:
        child_cmd.append("--profiler-record-shapes")
    if args.profiler_with_stack:
        child_cmd.append("--profiler-with-stack")
    if args.profiler_with_memory:
        child_cmd.append("--profiler-with-memory")
    if args.profiler_use_gzip:
        child_cmd.append("--profiler-use-gzip")
    if args.profiler_with_flops:
        child_cmd.append("--profiler-with-flops")
    if args.profiler_dump_cuda_time_total:
        child_cmd.append("--profiler-dump-cuda-time-total")
    if args.enable_ar_profiler:
        child_cmd.append("--enable-ar-profiler")
    else:
        child_cmd.append("--disable-ar-profiler")
    if args.log_stats:
        child_cmd.append("--log-stats")
    if args.enforce_eager:
        child_cmd.append("--enforce-eager")
    else:
        child_cmd.append("--disable-enforce-eager")

    print(f"[parent] Launch {mode}")
    print(f"[parent]   deploy_config={deploy_config}")
    print(f"[parent]   log_file={log_file}")
    print(f"[parent]   profiler_dir={profiler_dir}")

    with log_file.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            child_cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fp.write(line)
        return_code = proc.wait()

    return return_code, log_file


def build_formatted_prompts(
    *,
    model: str,
    prompt: str,
    image_path: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    from PIL import Image
    from transformers import AutoTokenizer

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
    )
    from vllm_omni.inputs.data import OmniPromptType

    del OmniPromptType  # imported for side-effect typing parity with the existing example

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    token_stop_ids = resolve_stop_token_ids(task="i2t", bot_task=None, tokenizer=tokenizer)

    image_paths = [p.strip() for p in image_path.split(",") if p.strip()]
    if not image_paths:
        raise ValueError(f"--image-path produced no usable paths: {image_path!r}")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    mm_payload = images[0] if len(images) == 1 else images

    formatted_prompts: list[dict[str, Any]] = []
    for _ in range(batch_size):
        result = build_prompt_tokens(prompt, tokenizer, task="i2t", bot_task=None, sys_type=None, num_images=len(images))
        formatted_prompts.append(
            {
                "prompt_token_ids": result.token_ids,
                "prompt": prompt,
                "use_system_prompt": None,
                "modalities": ["text"],
                "multi_modal_data": {"image": mm_payload},
                "stop_token_ids": token_stop_ids,
            }
        )
    return formatted_prompts, images


def run_mode(args: argparse.Namespace) -> int:
    from vllm_omni.entrypoints.omni import Omni

    mode = args._mode_run
    assert mode in {"vit_dp_on", "vit_dp_off"}
    output_root = Path(args._resolved_output_root).resolve()
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = Path(args._resolved_profiler_dir).resolve()
    profiler_dir.mkdir(parents=True, exist_ok=True)
    deploy_config = Path(args._resolved_deploy_config).resolve()

    image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
    for image_path in image_paths:
        ensure_path_exists(image_path, "image path")
    ensure_path_exists(deploy_config, "resolved deploy config")

    profiler_config = dict(args.profiler_config) if args.profiler_config is not None else default_profiler_config(
        args, profiler_dir
    )
    profiler_config["torch_profiler_dir"] = str(profiler_dir)

    print("=" * 72)
    print(f"[mode] {mode}")
    print(f"[mode] model={args.model}")
    print(f"[mode] deploy_config={deploy_config}")
    print(f"[mode] image_path={args.image_path}")
    print(f"[mode] prompt={args.prompt}")
    print(f"[mode] batch_size={args.batch_size}")
    print(f"[mode] warmup_runs={args.warmup_runs}")
    print(f"[mode] profile_runs={args.profile_runs}")
    print(f"[mode] profiler_config={profiler_config}")
    print("=" * 72)

    prompts, images = build_formatted_prompts(
        model=args.model,
        prompt=args.prompt,
        image_path=args.image_path,
        batch_size=args.batch_size,
    )

    omni = Omni(
        model=args.model,
        mode="image-to-text",
        deploy_config=str(deploy_config),
        profiler_config=profiler_config,
        enable_ar_profiler=args.enable_ar_profiler,
        enforce_eager=args.enforce_eager,
        log_stats=args.log_stats,
        init_timeout=args.init_timeout,
    )

    params_list = list(omni.default_sampling_params_list)
    for sp in params_list:
        if hasattr(sp, "stop_token_ids"):
            # Prompt builder already resolved i2t stop ids into the prompt dict, but
            # the AR stage still expects stop_token_ids on its sampling params path.
            stop_ids = prompts[0].get("stop_token_ids")
            sp.stop_token_ids = stop_ids

    for warmup_idx in range(args.warmup_runs):
        print(f"[warmup] {warmup_idx + 1}/{args.warmup_runs}")
        list(omni.generate(prompts=prompts, sampling_params_list=params_list))

    print("[profile] start")
    omni.start_profile(profile_prefix=mode)

    run_summaries: list[dict[str, Any]] = []
    outputs_path = mode_dir / "outputs.json"
    summary_path = mode_dir / "summary.json"
    for run_idx in range(args.profile_runs):
        print(f"[run] {run_idx + 1}/{args.profile_runs}")
        start = time.perf_counter()
        outputs = list(omni.generate(prompts=prompts, sampling_params_list=params_list))
        elapsed_s = time.perf_counter() - start

        serialized_outputs: list[dict[str, Any]] = []
        for req_idx, req_output in enumerate(outputs):
            ro = getattr(req_output, "request_output", None)
            text = ""
            if ro and getattr(ro, "outputs", None):
                text = "".join(getattr(item, "text", "") or "" for item in ro.outputs)
            if not text:
                ar_text = getattr(req_output, "custom_output", {}).get("ar_generated_text")
                if isinstance(ar_text, list):
                    text = "\n".join(part for part in ar_text if part)
                elif isinstance(ar_text, str):
                    text = ar_text

            stage_durations = getattr(req_output, "stage_durations", {}) or {}
            serialized_outputs.append(
                {
                    "request_index": req_idx,
                    "text": text,
                    "stage_durations": stage_durations,
                }
            )

        run_summary = {
            "run_index": run_idx,
            "elapsed_s": elapsed_s,
            "num_requests": len(outputs),
            "num_images_per_request": len(images),
            "outputs": serialized_outputs,
        }
        run_summaries.append(run_summary)
        print(f"[run] elapsed_s={elapsed_s:.4f} num_requests={len(outputs)}")

    print("[profile] stop")
    profile_results = omni.stop_profile()

    outputs_path.write_text(json.dumps(run_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "mode": mode,
        "model": args.model,
        "deploy_config": str(deploy_config),
        "image_path": args.image_path,
        "prompt": args.prompt,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup_runs,
        "profile_runs": args.profile_runs,
        "enforce_eager": args.enforce_eager,
        "enable_ar_profiler": args.enable_ar_profiler,
        "profiler_config": profiler_config,
        "profiler_dir": str(profiler_dir),
        "outputs_json": str(outputs_path),
        "runs": run_summaries,
        "profile_results": profile_results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[done] summary={summary_path}")
    print(f"[done] outputs={outputs_path}")
    print(f"[done] profiler_dir={profiler_dir}")
    return 0


def run_parent(args: argparse.Namespace) -> int:
    base_deploy = ensure_path_exists(args.deploy_config, "deploy config")
    output_root = make_output_root(args.output_root)
    deploy_configs = write_temp_deploy_configs(base_deploy, output_root)

    results: dict[str, Any] = {"output_root": str(output_root), "modes": {}}
    for mode in ("vit_dp_on", "vit_dp_off"):
        return_code, log_file = launch_child(args, output_root, mode, deploy_configs[mode])
        mode_dir = output_root / mode
        summary_path = mode_dir / "summary.json"
        results["modes"][mode] = {
            "return_code": return_code,
            "log_file": str(log_file),
            "summary_json": str(summary_path),
            "deploy_config": str(deploy_configs[mode]),
        }
        if return_code != 0:
            results_path = output_root / "compare_summary.json"
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[error] mode {mode} failed, see {log_file}", file=sys.stderr)
            print(f"[error] partial summary written to {results_path}", file=sys.stderr)
            return return_code

    results_path = output_root / "compare_summary.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("[compare] finished")
    print(f"[compare] output_root={output_root}")
    print(f"[compare] compare_summary={results_path}")
    print(f"[compare] vit_dp_on log={results['modes']['vit_dp_on']['log_file']}")
    print(f"[compare] vit_dp_off log={results['modes']['vit_dp_off']['log_file']}")
    print("=" * 72)
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.profile_runs <= 0:
        parser.error("--profile-runs must be > 0")
    if args._mode_run is not None:
        return run_mode(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
