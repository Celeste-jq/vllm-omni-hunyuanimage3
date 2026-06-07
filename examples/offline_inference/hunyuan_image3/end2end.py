"""
HunyuanImage-3.0-Instruct unified end-to-end inference script.
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    MAX_IMAGES_PER_REQUEST,
    build_prompt_tokens,
    resolve_stop_token_ids,
    resolve_sys_type,
)
from vllm_omni.entrypoints.omni import Omni

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image_3_moe.yaml")
_DEFAULT_AR_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3_ar.yaml")

_MODALITY_TASK_MAP: dict[str, tuple[str, str | None]] = {
    "text2img": ("t2i", "think"),
    "img2img": ("it2i", "think"),
    "img2text": ("i2t", None),
    "text2text": ("t2t", None),
}

_MODALITY_DEFAULT_DEPLOY_CONFIG = {
    "text2img": _DEFAULT_DEPLOY_CONFIG,
    "img2img": _DEFAULT_DEPLOY_CONFIG,
    "img2text": _DEFAULT_AR_DEPLOY_CONFIG,
    "text2text": _DEFAULT_AR_DEPLOY_CONFIG,
}

_MODALITY_MODE = {
    "text2img": "text-to-image",
    "img2img": "image-editing",
    "img2text": "image-to-text",
    "text2text": "text-to-text",
}


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0-Instruct end-to-end inference.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct", help="Model name or local path.")
    parser.add_argument(
        "--modality",
        default="text2img",
        choices=list(_MODALITY_TASK_MAP),
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Input image path(s) for img2img/img2text. Comma-separated for multi-image (up to 3).",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory to save results.")
    parser.add_argument(
        "--batch-admission",
        action="store_true",
        help="Enqueue the whole request batch before polling outputs, so the scheduler sees it in one tick.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional request count override for batch-admission mode. If set, a single prompt is repeated to this size.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of warmup batch-admission runs to execute before the measured run.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--height", type=int, default=None, help="Output image height.")
    parser.add_argument("--width", type=int, default=None, help="Output image width.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument(
        "--bot-task",
        type=str,
        default=None,
        choices=["none", "think", "recaption", "think_recaption", "vanilla"],
        help="Override prompt mode. Default: auto from --modality.",
    )
    parser.add_argument("--sys-type", type=str, default=None, help="Override system prompt type.")
    parser.add_argument("--deploy-config", type=str, default=None, help="Custom deploy YAML path.")
    parser.add_argument("--stage-configs-path", type=str, default=None, help="Custom legacy stage config YAML path.")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--init-timeout", type=int, default=300, help="Initialization timeout in seconds.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--diffusion-kv-cache-dtype",
        type=str,
        default=None,
        help="Diffusion attention KV cache dtype, for example 'fp8'. Separate from vLLM --kv-cache-dtype.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-skip-steps",
        type=str,
        default=None,
        help="Denoising step selector to keep diffusion KV cache in native dtype, for example '0,1,4-6'.",
    )
    parser.add_argument(
        "--diffusion-kv-cache-skip-layers",
        type=str,
        default=None,
        help="Transformer layer selector to keep diffusion KV cache in native dtype, for example '0-2,10'.",
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help=(
            "JSON object forwarded to Omni/additional_config, for example "
            '\'{"torchair_graph_config":{"enabled":true}}\'. Different platforms may support different '
            "configs. Make sure the configs are valid for the platform you are using. "
            "Contents must be hashable."
        ),
    )

    return parser.parse_args()


def parse_additional_config(raw_value: str | None) -> dict | None:
    """Parse a JSON string into an additional_config mapping."""
    if raw_value is None:
        return None

    try:
        additional_config = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --additional-config JSON: {exc}") from exc

    if additional_config is None:
        return None
    if not isinstance(additional_config, dict):
        raise ValueError(f"--additional-config must decode to a JSON object, got {type(additional_config).__name__}")
    return additional_config


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
    prompts: list[str],
    task: str,
    bot_task: str | None,
    sys_type: str | None,
    modality: str,
    tokenizer: Any,
    input_images: list[Any],
) -> list[dict[str, Any]]:
    mm_image_payload = (input_images[0] if len(input_images) == 1 else input_images) if input_images else None
    formatted_prompts: list[dict[str, Any]] = []
    for prompt in prompts:
        build_kwargs: dict[str, Any] = {"task": task, "bot_task": bot_task, "sys_type": sys_type}
        if input_images:
            build_kwargs["num_images"] = len(input_images)
        result = build_prompt_tokens(prompt, tokenizer, **build_kwargs)
        token_ids = result.token_ids
        effective_sys_type = sys_type or resolve_sys_type(bot_task)

        prompt_dict: dict[str, Any] = {
            "prompt_token_ids": token_ids,
            "prompt": prompt,
            "use_system_prompt": effective_sys_type,
        }
        if modality == "text2img":
            prompt_dict["modalities"] = ["image"]
        elif modality == "img2img":
            prompt_dict["modalities"] = ["image"]
            prompt_dict["multi_modal_data"] = {"image": mm_image_payload}
            prompt_dict["height"] = input_images[0].height
            prompt_dict["width"] = input_images[0].width
        elif modality == "img2text":
            prompt_dict["modalities"] = ["text"]
            prompt_dict["multi_modal_data"] = {"image": mm_image_payload}
        else:
            prompt_dict["modalities"] = ["text"]
        formatted_prompts.append(prompt_dict)
    return formatted_prompts


def run_batch_admission(
    omni: Omni,
    *,
    prompts: list[dict[str, Any]],
    sampling_params_list: list[Any],
    log_prefix: str = "[batch-admission]",
) -> list[Any]:
    """Submit a batch of requests before polling outputs.

    This mirrors the profile-oriented harness used in the vit-dp sharded
    branch: all requests are enqueued first so the scheduler can see the
    entire batch together on the same tick.
    """
    from vllm_omni.entrypoints.client_request_state import ClientRequestState
    from vllm_omni.engine.messages import OutputMessage
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
            print(f"{log_prefix} enqueued {req_id}")

        active_reqs = set(request_ids)
        outputs: list[Any] = []
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
                omni._log_summary_and_cleanup(req_id)

        return outputs
    except Exception:
        if request_ids:
            omni.abort(request_ids)
        raise


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    additional_config = parse_additional_config(args.additional_config)

    task, default_bot_task = _MODALITY_TASK_MAP[args.modality]
    if args.bot_task is None:
        bot_task: str | None = default_bot_task
    elif args.bot_task == "none":
        bot_task = None
    else:
        bot_task = args.bot_task

    if args.deploy_config is not None and args.stage_configs_path is not None:
        raise ValueError("--deploy-config and --stage-configs-path are mutually exclusive.")

    deploy_config = args.deploy_config
    stage_configs_path = args.stage_configs_path
    if deploy_config is None and stage_configs_path is None:
        deploy_config = _MODALITY_DEFAULT_DEPLOY_CONFIG[args.modality]

    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
        "mode": _MODALITY_MODE[args.modality],
        "diffusion_kv_cache_dtype": args.diffusion_kv_cache_dtype,
        "diffusion_kv_cache_skip_steps": args.diffusion_kv_cache_skip_steps,
        "diffusion_kv_cache_skip_layers": args.diffusion_kv_cache_skip_layers,
    }

    if additional_config is not None:
        omni_kwargs["additional_config"] = additional_config
    if deploy_config is not None:
        omni_kwargs["deploy_config"] = deploy_config
    else:
        omni_kwargs["stage_configs_path"] = stage_configs_path

    omni = Omni(**omni_kwargs)

    prompts = args.prompts or ["A cute cat"]
    input_images: list = []
    if args.modality in ("img2img", "img2text"):
        if not args.image_path:
            raise ValueError(f"--image-path required for {args.modality}, got: {args.image_path}")
        from PIL import Image

        image_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
        if len(image_paths) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"--image-path accepts at most {MAX_IMAGES_PER_REQUEST} images for "
                f"HunyuanImage-3.0 IT2I, got {len(image_paths)}: {args.image_path}"
            )
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise ValueError(f"Image path does not exist: {image_path}")
            input_images.append(Image.open(image_path).convert("RGB"))
        if not input_images:
            raise ValueError(f"--image-path produced no usable paths: {args.image_path!r}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
        if len(prompts) == 1:
            prompts = prompts * args.batch_size
        elif len(prompts) != args.batch_size:
            raise ValueError(
                f"--batch-size={args.batch_size} requires either one prompt to replicate or "
                f"exactly {args.batch_size} prompts, got {len(prompts)}"
            )

    formatted_prompts = build_formatted_prompts(
        prompts=prompts,
        task=task,
        bot_task=bot_task,
        sys_type=args.sys_type,
        modality=args.modality,
        tokenizer=tokenizer,
        input_images=input_images,
    )

    params_list = list(omni.default_sampling_params_list)

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must both be specified or both omitted.")
    user_specified_size = args.height is not None and args.width is not None
    if args.modality in ("img2text", "text2text"):
        ar_image_size = "auto"
    elif user_specified_size:
        ar_image_size = f"{args.width}x{args.height}"
    else:
        ar_image_size = None
    ar_stop_token_ids = resolve_stop_token_ids(
        task=task, bot_task=bot_task, tokenizer=tokenizer, image_size=ar_image_size
    )
    print(
        f"[AR Config] task={task}, bot_task={bot_task}, image_size={ar_image_size}, stop_token_ids={ar_stop_token_ids}"
    )
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            if args.seed is not None:
                sp.seed = args.seed
            if args.modality == "text2img":
                sp.height = args.height
                sp.width = args.width
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = ar_stop_token_ids

    print(f"\n{'=' * 60}")
    print("HunyuanImage-3.0 Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Modality: {args.modality}")
    print(f"  Prompt task: {task}")
    print(f"  Bot task: {bot_task}")
    if deploy_config is not None:
        print(f"  Deploy config: {deploy_config}")
    else:
        print(f"  Stage config: {stage_configs_path}")
    print(f"  Num stages: {omni.num_stages}")
    if args.modality in ("text2img", "img2img"):
        print(f"  Inference steps: {args.steps}")
        print(f"  Guidance scale: {args.guidance_scale}")
        print(f"  Seed: {args.seed}")
        print(f"  diffusion_kv_cache_dtype: {args.diffusion_kv_cache_dtype}")
        print(f"  diffusion_kv_cache_skip_steps: {args.diffusion_kv_cache_skip_steps}")
        print(f"  diffusion_kv_cache_skip_layers: {args.diffusion_kv_cache_skip_layers}")
    if args.modality == "text2img":
        print(f"  Output size: {args.width}x{args.height}")
    if args.image_path:
        print(f"  Input image: {args.image_path}")
    if additional_config is not None:
        print(f"  Additional config: {additional_config}")
    print(f"  Prompts: {prompts}")
    print(f"{'=' * 60}\n")

    if args.batch_admission:
        if args.warmup_runs < 0:
            raise ValueError(f"--warmup-runs must be non-negative, got {args.warmup_runs}")
        for warmup_idx in range(args.warmup_runs):
            print(f"[warmup] {warmup_idx + 1}/{args.warmup_runs}")
            run_batch_admission(omni, prompts=formatted_prompts, sampling_params_list=params_list, log_prefix="[warmup]")
        omni_outputs = run_batch_admission(omni, prompts=formatted_prompts, sampling_params_list=params_list)
    else:
        omni_outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))
    img_idx = 0
    for req_output in omni_outputs:
        ro = getattr(req_output, "request_output", None)
        txt = ""
        if ro and getattr(ro, "outputs", None):
            txt = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
        if not txt:
            ar_text = getattr(req_output, "custom_output", {}).get("ar_generated_text")
            if isinstance(ar_text, list):
                txt = "\n".join(text for text in ar_text if text)
            else:
                txt = ar_text or ""
        if txt:
            print(f"[Output] Text:\n{txt}")

        images = getattr(req_output, "images", None)
        if not images and ro and hasattr(ro, "images"):
            images = ro.images
        if images:
            for j, img in enumerate(images):
                save_path = os.path.join(args.output, f"output_{img_idx}_{j}.png")
                img.save(save_path)
                print(f"[Output] Saved image to {save_path}")
            img_idx += 1


if __name__ == "__main__":
    main()
