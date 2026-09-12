"""Config loader enforces the frozen 6-class + data_version contract."""

import pytest
import yaml

from src.config import DATA_VERSION, STANDARD_CLASSES, load_config, PROJECT_ROOT


def test_all_shipped_configs_load_and_match_standard(tmp_path):
    for name in ("baseline_6class_mobilenetv3.yaml",
                 "baseline_6class_resnet18.yaml",
                 "baseline_6class_mobilevit_xxs.yaml"):
        cfg = load_config(PROJECT_ROOT / "configs" / name)
        assert list(cfg.dataset.class_names) == STANDARD_CLASSES
        assert cfg.dataset.data_version == DATA_VERSION


def test_wrong_class_list_rejected(tmp_path):
    bad = {
        "dataset": {"class_names": ["palm", "fist"], "data_version": DATA_VERSION,
                    "hf_repo_id": "x", "hf_filename": ["a"]},
        "preprocessing": {"image_size": 224, "min_pixel_std": 1.0},
        "model": {"architecture": "resnet18", "pretrained": False},
        "training": {"batch_size": 2, "max_epochs": 1},
        "output": {"checkpoints_dir": "outputs/_t/checkpoints",
                   "metrics_dir": "outputs/_t/metrics",
                   "plots_dir": "outputs/_t/plots"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="STANDARD_CLASSES"):
        load_config(p)
