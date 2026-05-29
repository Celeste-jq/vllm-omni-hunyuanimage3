#!/usr/bin/env python3
"""Run HunyuanImage3 AR img2text batches with an explicit deploy yaml."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
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
        description="Run HunyuanImage3 img2text AR-only batching with an explicit deploy yaml and optional profiling."
    )
    parser.add_argument("--model", required=True, help="Model path or HF ID.")
    parser.add_argument("--image-path", required=True, help="Input image path. Comma-separated paths also supported.")
    parser.add_argument(
        "--deploy-config",
        default=str(DEFAULT_DEPLOY),
        help="AR deploy config to use as-is for this run.",
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
        help="Root directory for logs, summaries, and profiler artifacts.",
    )
    parser.add_argument(
        "--mode-label",
        default=None,
        help="Optional output subdirectory label. Defaults to the deploy config stem.",
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
        "--disable-profiler",
        dest="enable_profiler",
        action="store_false",
        default=True,
        help="Disable torch profiler collection and skip start_profile/stop_profile.",
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


def resolve_profiler_config(args: argparse.Namespace, profiler_dir: Path) -> dict[str, Any] | None:
    if not getattr(args, "enable_profiler", True):
        return None
    return dict(args.profiler_config) if args.profiler_config is not None else default_profiler_config(args, profiler_dir)


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
        root = (REPO_ROOT / "test_outputs" / f"hunyuan_image3_vit_dp_profile_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_mode_label(args: argparse.Namespace) -> str:
    if args.mode_label:
        return args.mode_label
    return Path(args.deploy_config).stem


def resolve_effective_stage_preview(
    deploy_config: Path,
    *,
    platform: str = "npu",
    stage_id: int = 0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort preview of the final stage config after platform overrides."""
    try:
        from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config
    except Exception as exc:
        return None, f"unable to import stage config helpers: {exc}"

    try:
        deploy = load_deploy_config(deploy_config)
        deploy = _apply_platform_overrides(deploy, platform)
        stage = next((s for s in deploy.stages if s.stage_id == stage_id), None)
        if stage is None:
            return None, f"stage_id={stage_id} not found"
        keys = (
            "max_num_seqs",
            "max_num_batched_tokens",
            "devices",
            "tensor_parallel_size",
            "gpu_memory_utilization",
        )
        preview = {key: getattr(stage, key, None) for key in keys}
        preview.update(getattr(stage, "engine_extras", {}) or {})
        return preview, None
    except Exception as exc:
        return None, f"unable to resolve deploy config: {exc}"


def print_deploy_preflight(
    *,
    args: argparse.Namespace,
    deploy_config: Path,
    prompts: list[dict[str, Any]],
) -> None:
    preview, error = resolve_effective_stage_preview(deploy_config)
    if error:
        print(f"[preflight] effective_config=unavailable ({error})")
    else:
        print(f"[preflight] effective_npu_stage0={preview}")
        max_num_seqs = preview.get("max_num_seqs")
        if isinstance(max_num_seqs, int) and max_num_seqs < args.batch_size:
            print(
                "[preflight][warning] max_num_seqs is smaller than batch_size; "
                f"scheduler cannot admit all requests together ({max_num_seqs} < {args.batch_size})."
            )

        max_num_batched_tokens = preview.get("max_num_batched_tokens")
        # HunyuanImage3 AR image requests are large; recent NPU logs for this
        # model show about 12.8k tokens/request. Keep this as a warning only.
        recommended_tokens = 12800 * args.batch_size
        if isinstance(max_num_batched_tokens, int) and max_num_batched_tokens < recommended_tokens:
            print(
                "[preflight][warning] max_num_batched_tokens may be too small for "
                f"batch_size={args.batch_size}; observed HunyuanImage3 image runs need "
                f"about >= {recommended_tokens} to batch {args.batch_size} requests "
                f"(current={max_num_batched_tokens})."
            )

    prompt_lens = [len(p.get("prompt_token_ids") or []) for p in prompts]
    if prompt_lens:
        print(
            "[preflight] prompt_token_lens="
            f"min={min(prompt_lens)} max={max(prompt_lens)} batch_size={len(prompt_lens)}"
        )


def build_request_mm_uuids(req_idx: int, num_images: int, batch_id: str | None = None) -> dict[str, list[str]]:
    prefix = f"{batch_id}-" if batch_id else ""
    return {"image": [f"{prefix}req-{req_idx}-image-{image_idx}" for image_idx in range(num_images)]}


def count_prompt_images(prompt: dict[str, Any]) -> int:
    image_payload = prompt.get("multi_modal_data", {}).get("image")
    if image_payload is None:
        return 0
    if isinstance(image_payload, (list, tuple)):
        return len(image_payload)
    return 1


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
    for req_idx in range(batch_size):
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


def generate_with_batch_admission(omni: Any, prompts: list[dict[str, Any]], sampling_params_list: list[Any]) -> list[Any]:
    """Preprocess a full batch before enqueueing, so the scheduler can see it together."""
    from tqdm.auto import tqdm

    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.entrypoints.client_request_state import ClientRequestState
    from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics

    sampling_params_list = list(omni.resolve_sampling_params_list(sampling_params_list))
    sampling_params_list = omni._set_final_only_for_llm_stages(sampling_params_list)

    request_ids = [f"{i}_{uuid.uuid4()}" for i in range(len(prompts))]
    wall_start_ts = time.time()
    req_start_ts: dict[str, float] = {}
    req_final_stage_ids: dict[str, int] = {}
    pending_msgs: list[tuple[str, Any]] = []

    try:
        batch_id = uuid.uuid4().hex
        for req_idx, (req_id, prompt) in enumerate(zip(request_ids, prompts)):
            # Use a fresh multimodal cache key for every generation call so
            # warmup and measured runs both exercise runtime ViT encoding.
            prompt["multi_modal_uuids"] = build_request_mm_uuids(req_idx, count_prompt_images(prompt), batch_id)
            prompt_modalities = prompt.get("modalities", None)
            final_stage_id = omni._compute_final_stage_id(prompt_modalities)
            req_final_stage_ids[req_id] = final_stage_id

            metrics = OrchestratorMetrics(
                omni.num_stages,
                omni.log_stats,
                wall_start_ts,
                final_stage_id,
            )
            req_state = ClientRequestState(req_id)
            req_state.metrics = metrics
            omni.request_states[req_id] = req_state

            req_sp_list = list(sampling_params_list)
            pd_pair = omni._get_pd_separation_pair()
            if pd_pair is not None:
                p_id = pd_pair[0]
                req_sp_list[p_id] = omni._prepare_prefill_sampling_params(req_id, req_sp_list[p_id])

            msg = omni.engine._build_add_request_message(
                request_id=req_id,
                prompt=prompt,
                sampling_params_list=req_sp_list,
                final_stage_id=final_stage_id,
            )
            pending_msgs.append((req_id, msg))

        enqueue_start = time.time()
        for req_id, msg in pending_msgs:
            omni.engine.request_queue.sync_q.put_nowait(msg)
            req_state = omni.request_states[req_id]
            if req_state.metrics is not None:
                req_state.metrics.stage_first_ts[0] = enqueue_start
            req_start_ts[req_id] = enqueue_start

        active_reqs = set(request_ids)
        outputs: list[Any] = []
        pbar = tqdm(total=len(request_ids), desc="Processed prompts", dynamic_ncols=True)
        try:
            while active_reqs:
                msg = omni.engine.try_get_output()
                should_continue, req_id, stage_id, req_state = omni._handle_output_message(msg)
                if should_continue:
                    continue

                if req_id not in active_reqs:
                    continue

                omni._check_engine_output_error(msg, req_id, stage_id)
                if req_state.metrics is None:
                    continue

                output = omni._process_single_result(
                    result=msg,
                    stage_id=stage_id,
                    metrics=req_state.metrics,
                    req_start_ts=req_start_ts,
                    wall_start_ts=wall_start_ts,
                    final_stage_id_for_e2e=req_final_stage_ids[req_id],
                )
                if output is not None:
                    outputs.append(output)

                if isinstance(msg, OutputMessage) and msg.finished:
                    active_reqs.discard(req_id)
                    pbar.update(1)
                    omni._log_summary_and_cleanup(req_id)
        finally:
            pbar.close()

        return outputs
    except Exception:
        if request_ids:
            omni.abort(request_ids)
        raise


def run_mode(args: argparse.Namespace) -> int:
    from vllm_omni.entrypoints.omni import Omni

    mode = resolve_mode_label(args)
    output_root = make_output_root(args.output_root)
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = mode_dir / "profiler"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    deploy_config = ensure_path_exists(args.deploy_config, "deploy config")

    image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
    for image_path in image_paths:
        ensure_path_exists(image_path, "image path")

    profiler_config = resolve_profiler_config(args, profiler_dir)
    if profiler_config is not None:
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
    print_deploy_preflight(args=args, deploy_config=deploy_config, prompts=prompts)

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
        generate_with_batch_admission(omni, prompts, params_list)

    if profiler_config is not None:
        print("[profile] start")
        omni.start_profile(profile_prefix=mode)

    run_summaries: list[dict[str, Any]] = []
    outputs_path = mode_dir / "outputs.json"
    summary_path = mode_dir / "summary.json"
    for run_idx in range(args.profile_runs):
        print(f"[run] {run_idx + 1}/{args.profile_runs}")
        start = time.perf_counter()
        outputs = generate_with_batch_admission(omni, prompts, params_list)
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

    profile_results = None
    if profiler_config is not None:
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


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.profile_runs <= 0:
        parser.error("--profile-runs must be > 0")
    return run_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
