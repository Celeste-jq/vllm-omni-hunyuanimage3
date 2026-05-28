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


def test_resolve_mode_label_uses_explicit_value(tmp_path: Path):
    module = _load_module()
    args = Namespace(mode_label="manual_label", deploy_config=str(tmp_path / "deploy.yaml"))

    assert module.resolve_mode_label(args) == "manual_label"


def test_resolve_mode_label_falls_back_to_deploy_stem(tmp_path: Path):
    module = _load_module()
    args = Namespace(mode_label=None, deploy_config=str(tmp_path / "hunyuan_image3_ar_custom.yaml"))

    assert module.resolve_mode_label(args) == "hunyuan_image3_ar_custom"


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


def test_build_request_mm_uuids_uses_request_index():
    module = _load_module()

    assert module.build_request_mm_uuids(0, 1) == {"image": ["req-0-image-0"]}
    assert module.build_request_mm_uuids(3, 2) == {"image": ["req-3-image-0", "req-3-image-1"]}
