"""The static browser demo (deploy/browser/build.py) must show exactly the stored reference results."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from app.dataset import store

    if not store.is_available():
        pytest.skip("SMAP/MSL dataset not installed")
    spec = importlib.util.spec_from_file_location("browser_build", ROOT / "deploy" / "browser" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("browser-demo")
    mod.build(out)
    return out


def test_summary_matches_reference_metrics(built):
    summary = json.loads((built / "data" / "summary.json").read_text())
    assert len(summary["methods"]) == 7 and len(summary["channels"]) == 81
    for m in summary["methods"]:
        ref = json.loads((ROOT / "reference" / "experiments" / f"{m['key']}.json").read_text())["metrics"]["overall"]
        for k, v in m["overall"]["event"].items():
            assert v == ref["event"][k], (m["key"], k)
        assert m["overall"]["mean_average_precision"] == ref["mean_average_precision"]
    assert sum(c["n_labels"] for c in summary["channels"]) == 104


def test_channel_arrays_are_the_stored_arrays(built):
    meta = json.loads((built / "data" / "channels" / "A-1.json").read_text())
    floats = np.frombuffer((built / "data" / "channels" / "A-1.bin").read_bytes(), dtype="<f4")
    run = meta["methods"]["lstm_forecaster_standard"]
    with np.load(ROOT / "reference" / "arrays" / "lstm_forecaster_standard" / "A-1.npz") as a:
        for name, off in run["arrays"].items():
            np.testing.assert_array_equal(floats[off:off + meta["n"]], a[name].astype("<f4"))
        assert run["eval_start"] == int(a["eval_start"])
    assert len(run["detections"]) == 1 and run["metrics"]["event"]["tp"] == 1
    # The telemetry itself is not part of the demo build.
    assert not any(p.suffix == ".npy" for p in built.rglob("*"))
