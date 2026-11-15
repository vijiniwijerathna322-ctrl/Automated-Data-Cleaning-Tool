from pathlib import Path

import pytest

from datacleaner import PipelineConfig
from datacleaner.core.errors import InvalidConfigError


def test_defaults() -> None:
    config = PipelineConfig()
    assert config.handle_missing is True
    assert config.target is None
    assert config.outlier_iqr_multiplier == 1.5


def test_unknown_keys_rejected() -> None:
    with pytest.raises(InvalidConfigError, match="bogus"):
        PipelineConfig.from_dict({"bogus": 1})


def test_range_validation() -> None:
    with pytest.raises(InvalidConfigError, match="outlier_iqr_multiplier"):
        PipelineConfig.from_dict({"outlier_iqr_multiplier": 0})


def test_from_yaml_and_json(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text("target: score\nhandle_missing: false\n", encoding="utf-8")
    config = PipelineConfig.from_file(yml)
    assert config.target == "score"
    assert config.handle_missing is False

    js = tmp_path / "c.json"
    js.write_text('{"feature_source": "x"}', encoding="utf-8")
    assert PipelineConfig.from_file(js).feature_source == "x"


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    yml = tmp_path / "empty.yaml"
    yml.write_text("", encoding="utf-8")
    assert PipelineConfig.from_file(yml) == PipelineConfig()


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing.yaml", None, "not found"),
        ("bad.yaml", "a: [", "Could not parse"),
        ("bad.json", "{", "Could not parse"),
        ("list.yaml", "- 1\n- 2\n", "mapping"),
        ("c.toml", "a = 1", "Unsupported"),
    ],
)
def test_bad_config_files(tmp_path: Path, name: str, content: str | None, message: str) -> None:
    path = tmp_path / name

    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(InvalidConfigError, match=message):
        PipelineConfig.from_file(path)


def test_default_config_file_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    PipelineConfig.from_file(root / "configs" / "default.yaml")
