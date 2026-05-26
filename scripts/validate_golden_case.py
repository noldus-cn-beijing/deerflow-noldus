#!/usr/bin/env python3
"""Golden-case schema validator.

Usage:
    python scripts/validate_golden_case.py [CASE_DIR ...]

If no CASE_DIR given, validates all directories under golden-cases/ that
match the case-XXX pattern (skips TEMPLATE/).

Exit codes:
    0  all cases pass
    1  one or more cases fail schema validation
    2  one or more cases have warnings (only when run with --strict)

Checks performed:
    Schema (hard fail):
        - metadata.yaml exists and has all required fields
        - expected-analysis.yaml exists and has all required fields
        - all enum values are valid
        - YAML syntax is correct

    Content (warning):
        - subjects in expected_metrics are defined in groups
        - required_keywords lists are non-empty
        - notes.md has all 6 required headings
        - raw-data/ directory is not empty
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "golden-cases"

VALID_PARADIGMS = {
    # v0.1 supported (catalog/<paradigm>.yaml exists)
    "epm",
    "open_field",
    "fst",
    "o_maze",
    "light_dark",
    # planned / paradigm knowledge exists but code-layer not implemented in v0.1
    "mwm",
    "y_maze",
    "barnes",
    "nor",
    "three_chamber",
    "social_interaction",
    "novel_suppressed_feeding",
    "footprint",
    "fine_behavior",
    "phenotyper",
}

VALID_SEVERITIES = {"low", "moderate", "high", "critical"}
VALID_FINDING_TYPES = {
    "outlier_detection",
    "counterfactual_analysis",
    "confound_note",
    "statistical_conclusion",
    "data_quality_warning",
    "phenotype_indication",
}

METADATA_REQUIRED = {
    "case_id",
    "paradigm",
    "species",
    "n_subjects",
    "n_groups",
    "groups",
    "experimental_condition",
    "source",
    "raw_data_files",
    "created",
    "annotator",
}

ANALYSIS_REQUIRED = {
    "paradigm",
    "species",
    "n_subjects",
    "groups",
    "expected_findings",
}

NOTES_REQUIRED_HEADINGS = [
    "## 1. 数据背景",
    "## 2. 初看数据的第一印象",
    "## 3. 逐个指标的判断过程",
    "## 4. 异常识别与辨别",
    "## 5. 最终结论",
    "## 6. 参考文献",
]

FINDING_REQUIRED_FIELDS = {"type", "reasoning"}

FINDING_CONDITIONAL_FIELDS = {
    "outlier_detection": {"subject", "metrics", "severity"},
    "counterfactual_analysis": {"subject", "claim"},
    "confound_note": {"claim"},
    "statistical_conclusion": {"claim"},
}


class CaseResult:
    def __init__(self, path: Path):
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"  YAML syntax error in {path}: {e}")
        return None


def validate_metadata(r: CaseResult):
    meta_path = r.path / "metadata.yaml"
    data = _load_yaml(meta_path)
    if data is None:
        r.error(f"metadata.yaml missing or unreadable")
        return

    missing = METADATA_REQUIRED - set(data.keys())
    if missing:
        r.error(f"metadata.yaml missing required fields: {sorted(missing)}")

    if "paradigm" in data and data["paradigm"] not in VALID_PARADIGMS:
        r.error(f"metadata.yaml: invalid paradigm '{data['paradigm']}'")

    if "groups" in data and isinstance(data["groups"], dict):
        all_subjects = set()
        for group_name, subjects in data["groups"].items():
            if not isinstance(subjects, list):
                r.error(f"metadata.yaml: groups.{group_name} should be a list")
            else:
                for s in subjects:
                    all_subjects.add(str(s))
        if "n_subjects" in data and data["n_subjects"] != len(all_subjects):
            r.error(
                f"metadata.yaml: n_subjects={data['n_subjects']} but "
                f"groups define {len(all_subjects)} subjects"
            )

    if "difficulty" in data and data["difficulty"] not in {"easy", "moderate", "hard", ""}:
        r.error(f"metadata.yaml: invalid difficulty '{data['difficulty']}'")

    raw_dir = r.path / "raw-data"
    if raw_dir.exists():
        txt_files = list(raw_dir.glob("*.txt"))
        xlsx_files = list(raw_dir.glob("*.xlsx"))
        csv_files = list(raw_dir.glob("*.csv"))
        total = len(txt_files) + len(xlsx_files) + len(csv_files)
        if "raw_data_files" in data and data["raw_data_files"] != total:
            r.error(
                f"metadata.yaml: raw_data_files={data['raw_data_files']} "
                f"but raw-data/ contains {total} data files"
            )
        if total == 0:
            r.warn("raw-data/ directory is empty")


def validate_analysis(r: CaseResult):
    analysis_path = r.path / "expected-analysis.yaml"
    data = _load_yaml(analysis_path)
    if data is None:
        r.error("expected-analysis.yaml missing or unreadable")
        return

    missing = ANALYSIS_REQUIRED - set(data.keys())
    if missing:
        r.error(f"expected-analysis.yaml missing required fields: {sorted(missing)}")

    if "paradigm" in data and data["paradigm"] not in VALID_PARADIGMS:
        r.error(f"expected-analysis.yaml: invalid paradigm '{data['paradigm']}'")

    findings = data.get("expected_findings", [])
    if not findings:
        r.error("expected-analysis.yaml: expected_findings is empty (need at least 1)")
        return

    has_stat_conclusion = False
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            r.error(f"expected-analysis.yaml: finding[{i}] is not a dict")
            continue

        ftype = f.get("type", "")
        if ftype not in VALID_FINDING_TYPES:
            r.error(f"expected-analysis.yaml: finding[{i}] invalid type '{ftype}'")

        fmissing = FINDING_REQUIRED_FIELDS - set(f.keys())
        if fmissing:
            r.error(f"expected-analysis.yaml: finding[{i}] missing {sorted(fmissing)}")

        if ftype in FINDING_CONDITIONAL_FIELDS:
            cond_missing = FINDING_CONDITIONAL_FIELDS[ftype] - set(f.keys())
            if cond_missing:
                r.error(
                    f"expected-analysis.yaml: finding[{i}] type={ftype} "
                    f"missing conditional fields: {sorted(cond_missing)}"
                )

        if "severity" in f and f["severity"] not in VALID_SEVERITIES:
            r.error(f"expected-analysis.yaml: finding[{i}] invalid severity '{f['severity']}'")

        kw = f.get("required_keywords", [])
        if isinstance(kw, list) and len(kw) == 0:
            r.warn(f"expected-analysis.yaml: finding[{i}] has empty required_keywords")

        if ftype == "statistical_conclusion":
            has_stat_conclusion = True

    if not has_stat_conclusion:
        r.warn("expected-analysis.yaml: no finding of type 'statistical_conclusion'")

    metrics = data.get("expected_metrics", [])
    meta = _load_yaml(r.path / "metadata.yaml")
    if meta and "groups" in meta and isinstance(meta["groups"], dict):
        all_subjects = set()
        for subjects in meta["groups"].values():
            for s in subjects:
                all_subjects.add(str(s))
        for i, m in enumerate(metrics):
            if isinstance(m, dict) and "subject" in m:
                if str(m["subject"]) not in all_subjects:
                    r.warn(
                        f"expected-analysis.yaml: metrics[{i}] subject "
                        f"'{m['subject']}' not defined in groups"
                    )


def validate_notes(r: CaseResult):
    notes_path = r.path / "notes.md"
    if not notes_path.exists():
        r.error("notes.md missing")
        return

    content = notes_path.read_text(encoding="utf-8")
    for heading in NOTES_REQUIRED_HEADINGS:
        if heading not in content:
            r.warn(f"notes.md: missing required heading '{heading}'")


def validate_case(case_dir: Path) -> CaseResult:
    r = CaseResult(case_dir)
    validate_metadata(r)
    validate_analysis(r)
    validate_notes(r)
    return r


def find_cases() -> list[Path]:
    if not GOLDEN_DIR.exists():
        return []
    return sorted(
        p for p in GOLDEN_DIR.iterdir()
        if p.is_dir() and p.name.startswith("case-") and p.name != "TEMPLATE"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate golden-case directories")
    parser.add_argument("cases", nargs="*", help="Case directories to validate")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()

    if args.cases:
        case_dirs = [Path(c).resolve() for c in args.cases]
    else:
        case_dirs = find_cases()

    if not case_dirs:
        print("No case directories found.")
        sys.exit(0)

    all_ok = True
    total_errors = 0
    total_warnings = 0

    for case_dir in case_dirs:
        print(f"\nValidating {case_dir.name}...")
        result = validate_case(case_dir)

        for e in result.errors:
            print(f"  ERROR: {e}")
        for w in result.warnings:
            print(f"  WARN:  {w}")

        total_errors += len(result.errors)
        total_warnings += len(result.warnings)

        if result.errors:
            print(f"  FAIL ({len(result.errors)} errors, {len(result.warnings)} warnings)")
            all_ok = False
        elif args.strict and result.warnings:
            print(f"  FAIL (--strict: {len(result.warnings)} warnings treated as errors)")
            all_ok = False
        else:
            status = "PASS"
            if result.warnings:
                status += f" ({len(result.warnings)} warnings)"
            print(f"  {status}")

    print(f"\n{'='*60}")
    print(f"Total: {len(case_dirs)} cases, {total_errors} errors, {total_warnings} warnings")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
