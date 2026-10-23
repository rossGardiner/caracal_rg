"""Check separate statement/branch floors against a complete source report."""

import argparse
import json
import math
from pathlib import Path
import sys


COUNT_FIELDS = ("covered_lines", "num_statements", "covered_branches", "num_branches")


def evaluate(report, settings, root, profile):
    if settings.get("schema_version") != 1:
        raise ValueError("Unsupported coverage threshold schema")
    thresholds = settings["profiles"][profile]
    if report.get("meta", {}).get("branch_coverage") is not True:
        raise ValueError("Coverage report must include branch coverage")

    root = root.resolve()
    expected = {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py")}
    if not expected:
        raise ValueError("No Python source files found under src")
    files = {}
    for name, entry in report["files"].items():
        path = (root / Path(name.replace("\\", "/"))).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Coverage file is outside the project: {name}") from exc
        if relative in files:
            raise ValueError(f"Duplicate coverage file: {relative}")
        summary = entry["summary"]
        for field in COUNT_FIELDS:
            value = summary[field]
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {field} for {relative}")
        if (summary["covered_lines"] > summary["num_statements"]
                or summary["covered_branches"] > summary["num_branches"]):
            raise ValueError(f"Covered counts exceed totals for {relative}")
        files[relative] = summary

    missing, extra = expected - files.keys(), files.keys() - expected
    if missing or extra:
        raise ValueError(
            f"Source inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    focused = settings["focused_modules"]
    if not isinstance(focused, list) or not focused or len(set(focused)) != len(focused):
        raise ValueError("Focused modules must be a nonempty unique list")
    if not set(focused) <= expected:
        raise ValueError("Focused modules must exist in the source inventory")

    results = []
    for scope, names in [("application", sorted(expected)), ("focused", focused)]:
        counts = {field: sum(files[name][field] for name in names) for field in COUNT_FIELDS}
        for metric, covered_key, total_key in [
            ("statements", "covered_lines", "num_statements"),
            ("branches", "covered_branches", "num_branches"),
        ]:
            minimum = thresholds[scope][metric]
            if (type(minimum) not in (int, float) or not math.isfinite(minimum)
                    or not 0 <= minimum <= 100):
                raise ValueError(f"Invalid threshold for {scope} {metric}")
            covered, total = counts[covered_key], counts[total_key]
            percentage = 100 * covered / total if total else 100.0
            # Compare unrounded values; display precision must not hide a failure.
            passed = percentage >= minimum
            results.append((scope, metric, covered, total, percentage, minimum, passed))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage.py JSON report for this suite")
    parser.add_argument("--profile", choices=("core", "combined"), required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        settings = json.loads((root / "test/coverage_thresholds.json").read_text(encoding="utf-8"))
        results = evaluate(report, settings, root, args.profile)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"Coverage gate error: {exc}", file=sys.stderr)
        return 2
    for scope, metric, covered, total, percentage, minimum, passed in results:
        print(f"{'PASS' if passed else 'FAIL'} {args.profile} {scope} {metric}: "
              f"{covered}/{total} = {percentage:.2f}% (floor {minimum:.2f}%)")
    return 0 if all(result[-1] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
