import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "compare_hunyuan_image3_vit_dp_profile.py"
    spec = importlib.util.spec_from_file_location("compare_hunyuan_image3_vit_dp_profile", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_temp_deploy_configs_updates_max_num_seqs(tmp_path: Path):
    module = _load_module()
    base_deploy = tmp_path / "base.yaml"
    base_deploy.write_text(
        "\n".join(
            [
                "pipeline: hunyuan_image3_ar",
                "stages:",
                "  - stage_id: 0",
                "    max_num_seqs: 1",
                "    mm_encoder_tp_mode: data",
                "",
            ]
        ),
        encoding="utf-8",
    )

    configs = module.write_temp_deploy_configs(base_deploy, tmp_path, batch_size=8)

    on_text = configs["vit_dp_on"].read_text(encoding="utf-8")
    off_text = configs["vit_dp_off"].read_text(encoding="utf-8")
    assert "max_num_seqs: 8" in on_text
    assert "max_num_seqs: 8" in off_text
    assert "mm_encoder_tp_mode: data" in on_text
    assert "mm_encoder_tp_mode: data" not in off_text


def test_resolve_profiler_config_returns_none_when_disabled(tmp_path: Path):
    module = _load_module()
    args = Namespace(
        enable_profiler=False,
        profiler_config=None,
        profiler_record_shapes=False,
        profiler_with_stack=False,
        profiler_with_memory=False,
        profiler_use_gzip=False,
        profiler_with_flops=False,
        profiler_dump_cuda_time_total=False,
    )

    assert module.resolve_profiler_config(args, tmp_path) is None
