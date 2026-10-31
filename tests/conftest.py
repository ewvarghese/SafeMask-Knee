"""Session-scoped invented release; never downloads or reads competition data."""

from __future__ import annotations

from pathlib import Path
import sys

# The `pytest` executable (unlike `python -m pytest`) may omit the repository
# root from sys.path; include it so the synthetic generator can be imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import yaml

from safemask_knee.config import DEFAULT_CONFIG, load_config
from scripts.make_synthetic import generate_release


@pytest.fixture(scope="session")
def release_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("invented_release")
    result = generate_release(path)
    assert result == {"studies": 160, "pairs": 1760, "expert_studies": 58,
                      "expert_pairs": 638, "expert_accepted": 383,
                      "accepted_errors": 11}
    return path


@pytest.fixture(scope="session")
def config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("config") / "test.yaml"
    cfg = load_config()
    cfg["bootstrap_repetitions"] = 160
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    assert set(DEFAULT_CONFIG) == set(load_config(path))
    return path
