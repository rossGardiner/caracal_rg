"""Exercise the actual coverage-gate CLI using tiny temporary source trees."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def gate_project(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "test").mkdir()
    for name in ["covered.py", "untested.py"]:
        (root / "src" / name).write_text("value = 1\n", encoding="utf-8")
    checker = root / "test/check_coverage.py"
    shutil.copyfile(Path(__file__).with_name("check_coverage.py"), checker)
    settings = {
        "schema_version": 1,
        "focused_modules": ["src/covered.py"],
        "profiles": {
            "core": {"application": {"statements": 50, "branches": 50},
                     "focused": {"statements": 100, "branches": 100}},
            "combined": {"application": {"statements": 75, "branches": 75},
                         "focused": {"statements": 100, "branches": 100}},
        },
    }
    report = {
        "meta": {"branch_coverage": True},
        "files": {
            "src/covered.py": {"summary": {"covered_lines": 1, "num_statements": 1,
                                              "covered_branches": 2, "num_branches": 2}},
            "src/untested.py": {"summary": {"covered_lines": 0, "num_statements": 1,
                                               "covered_branches": 0, "num_branches": 2}},
        },
        # Deliberately misleading summary: the checker must aggregate files.
        "totals": {"percent_covered": 100},
    }

    def run(*, data=None, config=None, profile="core"):
        report_path = root / "coverage.json"
        report_path.write_text(json.dumps(report if data is None else data), encoding="utf-8")
        (root / "test/coverage_thresholds.json").write_text(
            json.dumps(settings if config is None else config), encoding="utf-8"
        )
        return subprocess.run(
            [sys.executable, str(checker), str(report_path), "--profile", profile],
            cwd=tmp_path, capture_output=True, text=True, check=False,
        )

    return root, report, settings, run


def test_complete_report_passes_at_exact_floors_from_another_directory(gate_project):
    _, _, _, run = gate_project
    result = run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("PASS") == 4
    assert "application statements: 1/2 = 50.00%" in result.stdout
    assert "application branches: 2/4 = 50.00%" in result.stdout


def test_combined_profile_cannot_be_satisfied_by_core_coverage(gate_project):
    _, _, _, run = gate_project
    result = run(profile="combined")
    assert result.returncode == 1
    assert "FAIL combined application statements" in result.stdout
    assert "FAIL combined application branches" in result.stdout


@pytest.mark.parametrize("scope, metric", [
    ("application", "statements"), ("application", "branches"),
    ("focused", "statements"), ("focused", "branches"),
])
def test_each_floor_is_enforced_independently(gate_project, scope, metric):
    _, report, settings, run = gate_project
    data, config = deepcopy(report), deepcopy(settings)
    for thresholds in config["profiles"]["core"].values():
        thresholds.update(statements=0, branches=0)
    config["profiles"]["core"][scope][metric] = 100
    key = "covered_lines" if metric == "statements" else "covered_branches"
    data["files"]["src/covered.py"]["summary"][key] = 0
    result = run(data=data, config=config)
    assert result.returncode == 1
    assert result.stdout.count("FAIL") == 1
    assert f"FAIL core {scope} {metric}" in result.stdout


def test_percentages_are_compared_before_display_rounding(gate_project):
    _, report, settings, run = gate_project
    data, config = deepcopy(report), deepcopy(settings)
    data["files"]["src/covered.py"]["summary"].update(covered_lines=99999, num_statements=100000)
    config["profiles"]["core"]["focused"]["statements"] = 100
    result = run(data=data, config=config)
    assert result.returncode == 1
    assert "FAIL core focused statements" in result.stdout
    assert "100.00%" in result.stdout


def test_omitting_an_untested_module_cannot_improve_coverage(gate_project):
    _, report, _, run = gate_project
    data = deepcopy(report)
    del data["files"]["src/untested.py"]
    result = run(data=data)
    assert result.returncode == 2
    assert "missing=['src/untested.py']" in result.stderr


def test_new_source_module_requires_a_regenerated_report(gate_project):
    root, _, _, run = gate_project
    (root / "src/new.py").write_text("new_value = 2\n", encoding="utf-8")
    result = run()
    assert result.returncode == 2
    assert "src/new.py" in result.stderr


def test_extra_non_source_file_is_rejected(gate_project):
    _, report, _, run = gate_project
    data = deepcopy(report)
    data["files"]["test/something.py"] = data["files"]["src/covered.py"]
    result = run(data=data)
    assert result.returncode == 2
    assert "extra=['test/something.py']" in result.stderr


def test_statement_only_report_is_rejected(gate_project):
    _, report, _, run = gate_project
    data = deepcopy(report)
    data["meta"]["branch_coverage"] = False
    result = run(data=data)
    assert result.returncode == 2
    assert "must include branch coverage" in result.stderr


@pytest.mark.parametrize("field, value", [
    ("covered_lines", -1), ("num_statements", True),
    ("covered_branches", 3), ("num_branches", 1),
])
def test_invalid_or_impossible_counts_are_rejected(gate_project, field, value):
    _, report, _, run = gate_project
    data = deepcopy(report)
    data["files"]["src/covered.py"]["summary"][field] = value
    result = run(data=data)
    assert result.returncode == 2
    assert "Coverage gate error" in result.stderr


@pytest.mark.parametrize("threshold", [-1, 101, float("nan"), float("inf"), True])
def test_invalid_thresholds_are_rejected(gate_project, threshold):
    _, _, settings, run = gate_project
    config = deepcopy(settings)
    config["profiles"]["core"]["application"]["statements"] = threshold
    result = run(config=config)
    assert result.returncode == 2
    assert "Invalid threshold" in result.stderr


def test_missing_summary_counts_fail_instead_of_using_defaults(gate_project):
    _, report, _, run = gate_project
    data = deepcopy(report)
    del data["files"]["src/covered.py"]["summary"]["num_branches"]
    result = run(data=data)
    assert result.returncode == 2
    assert "num_branches" in result.stderr
