"""Reporting and dashboard regression checks using existing review outputs."""
from pathlib import Path
import json
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.v2.config import Config
from src.v2.reporting import create_figures, write_report
from src.v2.evaluation import aggregate

ROOT = Path(__file__).resolve().parents[1]


def test_reports_render_from_validation_without_touching_holdout(tmp_path):
    reports = ROOT/"reports"
    metrics = pd.read_csv(reports/"validation_metrics.csv")
    events = pd.read_csv(reports/"validation_events.csv")
    types = pd.read_csv(reports/"validation_types.csv")
    comparison = pd.read_csv(reports/"model_comparison.csv")
    flight = pd.read_csv(ROOT/"data/v2/validation/307.csv")
    summary = aggregate(metrics)
    (tmp_path/"reports").mkdir()
    create_figures(tmp_path/"figures", summary, metrics, types, comparison, flight,
                   events[events.flight_id.eq(307)], Config(), "307 (test fixture)")
    manifest = {"config": Config().to_dict(), "frozen_at_utc": "test fixture", "selection": {}}
    write_report(tmp_path, manifest, summary, metrics, events, types)
    assert len(list((tmp_path/"figures").glob("*.png"))) == 6
    assert (tmp_path/"reports/technical_report.md").stat().st_size > 1000


def test_dashboard_split_event_and_evaluation_controls():
    app = AppTest.from_file(str(ROOT/"app.py")).run(timeout=30)
    assert not app.exception
    for split in ["Validation", "Development (in-sample profiles)", "Historical V1 final (already inspected)"]:
        app.sidebar.selectbox[0].select(split).run()
        assert not app.exception
        focus = app.selectbox[0]
        if len(focus.options) > 1:
            focus.select(focus.options[1]).run()
            assert not app.exception
        app.sidebar.checkbox[0].uncheck().run()
        assert not app.exception
        assert len(app.metric) == 3
        app.sidebar.checkbox[0].check().run()
        assert not app.exception


def test_existing_freeze_receipt_consistent_if_present():
    # Read the receipt only; do not generate/re-evaluate final data in unit tests.
    path = ROOT/"reports/freeze_manifest.json"
    if path.exists():
        from src.v2.protocol import source_hashes
        freeze = json.loads(path.read_text())
        receipt = json.loads((ROOT/"reports/benchmark_receipt.json").read_text())
        assert freeze["source_sha256"] == source_hashes(ROOT)
        assert freeze["frozen_at_utc"] < receipt["first_evaluated_at_utc"]
