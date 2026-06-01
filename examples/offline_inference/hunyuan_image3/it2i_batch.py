"""
Batch IT2I smoke test for HunyuanImage-3.0-Instruct.

This script focuses on AR-stage batching for image-to-image requests. It
submits multiple img2img prompts in one Omni.generate() call so scheduler-side
runtime batching can expose multiple image items to model-side preprocessing.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    MAX_IMAGES_PER_REQUEST,
    build_prompt_tokens,
    resolve_stop_token_ids,
    resolve_sys_type,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEPLOY_CONFIG = str(_REPO_ROOT / "vllm_omni" / "deploy" / "hunyuan_image3.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 IT2I batch inference test.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct", help="Model name or local path.")
    parser.add_argument("--deploy-config", type=str, default=_DEFAULT_DEPLOY_CONFIG, help="Deploy YAML path.")
    parser.add_argument("--output", type=str, default="./it2i_batch_outputs", help="Output directory.")
    parser.add_argument("--prompts", nargs="+", default=None, help="One or more prompts.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text file with one prompt per line.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of requests to submit. If larger than prompt count, prompts/images are cycled.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Comma-separated image paths shared by all requests.",
    )
    parser.add_argument(
        "--image-paths",
        nargs="+",
        default=None,
        help=(
            "Per-request image groups. Each group is comma-separated, for example "
            "'a.png,b.png' 'c.png,d.png'. Cycled when --batch-size is larger."
        ),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bot-task", default="think_recaption", choices=["think", "recaption", "think_recaption"])
    parser.add_argument("--sys-type", type=str, default="en_unified")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument(
        "--additional-config",
        type=str,
        default=None,
        help="JSON object forwarded to Omni/additional_config.",
    )

    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def parse_additional_config(raw_value: str | None) -> dict | None:
    if raw_value is None:
        return None
    value = json.loads(raw_value)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"--additional-config must be a JSON object, got {type(value).__name__}")
    return value


def load_prompts(args) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if args.prompts:
        prompts.extend(args.prompts)
    if not prompts:
        raise ValueError("Provide --prompts or --prompt-file.")
    return prompts


def split_image_group(raw_group: str) -> list[str]:
    paths = [p.strip() for p in raw_group.split(",") if p.strip()]
    if not paths:
        raise ValueError(f"Image group produced no paths: {raw_group!r}")
    if len(paths) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"Each IT2I request supports at most {MAX_IMAGES_PER_REQUEST} images, got {len(paths)}: {raw_group}"
        )
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"Image path does not exist: {path}")
    return paths


def load_image_payload(paths: list[str]):
    images = [Image.open(path).convert("RGB") for path in paths]
    return images[0] if len(images) == 1 else images


def build_request_plan(args) -> list[tuple[str, list[str]]]:
    prompts = load_prompts(args)
    if args.image_paths:
        image_groups = [split_image_group(group) for group in args.image_paths]
    elif args.image_path:
        image_groups = [split_image_group(args.image_path)]
    else:
        raise ValueError("Provide --image-path or --image-paths for IT2I.")

    request_count = args.batch_size or max(len(prompts), len(image_groups))
    if request_count <= 0:
        raise ValueError("--batch-size must be positive.")

    prompt_iter = itertools.cycle(prompts)
    image_iter = itertools.cycle(image_groups)
    return [(next(prompt_iter), next(image_iter)) for _ in range(request_count)]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    additional_config = parse_additional_config(args.additional_config)
    omni_kwargs = {
        "model": args.model,
        "vae_use_tiling": args.vae_use_tiling,
        "log_stats": args.log_stats,
        "init_timeout": args.init_timeout,
        "enforce_eager": args.enforce_eager,
        "mode": "image-editing",
        "deploy_config": args.deploy_config,
    }
    if additional_config is not None:
        omni_kwargs["additional_config"] = additional_config

    plan = build_request_plan(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    omni = Omni(**omni_kwargs)

    formatted_prompts: list[OmniPromptType] = []
    for req_idx, (prompt, image_paths) in enumerate(plan):
        image_payload = load_image_payload(image_paths)
        num_images = len(image_paths)
        result = build_prompt_tokens(
            prompt,
            tokenizer,
            task="it2i",
            bot_task=args.bot_task,
            sys_type=args.sys_type,
            num_images=num_images,
        )
        formatted_prompts.append(
            {
                "prompt_token_ids": result.token_ids,
                "prompt": prompt,
                "use_system_prompt": args.sys_type or resolve_sys_type(args.bot_task),
                "modalities": ["image"],
                "multi_modal_data": {"image": image_payload},
                "height": image_payload[0].height if isinstance(image_payload, list) else image_payload.height,
                "width": image_payload[0].width if isinstance(image_payload, list) else image_payload.width,
                "request_index": req_idx,
            }
        )

    params_list = list(omni.default_sampling_params_list)
    ar_stop_token_ids = resolve_stop_token_ids(task="it2i", bot_task=args.bot_task, tokenizer=tokenizer)
    for sp in params_list:
        if isinstance(sp, OmniDiffusionSamplingParams):
            sp.num_inference_steps = args.steps
            sp.guidance_scale = args.guidance_scale
            sp.guidance_scale_provided = True
            sp.seed = args.seed
        elif hasattr(sp, "stop_token_ids"):
            sp.stop_token_ids = ar_stop_token_ids

    print("=" * 72)
    print("HunyuanImage-3.0 IT2I Batch Test")
    print(f"  Model: {args.model}")
    print(f"  Deploy config: {args.deploy_config}")
    print(f"  Requests: {len(formatted_prompts)}")
    print(f"  Images per request: {[len(paths) for _, paths in plan]}")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")
    print("=" * 72)

    outputs = list(omni.generate(prompts=formatted_prompts, sampling_params_list=params_list))
    for req_idx, req_output in enumerate(outputs):
        ro = getattr(req_output, "request_output", None)
        text = ""
        if ro and getattr(ro, "outputs", None):
            text = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
        if not text:
            custom_output = getattr(req_output, "custom_output", {}) or {}
            ar_text = custom_output.get("ar_generated_text")
            text = "\n".join(ar_text) if isinstance(ar_text, list) else (ar_text or "")
        if text:
            text_path = os.path.join(args.output, f"output_{req_idx}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[Output] Saved text to {text_path}")

        images = getattr(req_output, "images", None)
        if not images and ro and hasattr(ro, "images"):
            images = ro.images
        for img_idx, image in enumerate(images or []):
            save_path = os.path.join(args.output, f"output_{req_idx}_{img_idx}.png")
            image.save(save_path)
            print(f"[Output] Saved image to {save_path}")


if __name__ == "__main__":
    main()
