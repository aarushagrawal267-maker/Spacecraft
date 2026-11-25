import time

import numpy as np
import pytest

from tests.integration.conftest import wait_experiment

pytestmark = pytest.mark.dataset


def test_health_system_methods(client):
    assert client.get("/api/health").json()["status"] == "ok"
    sysinfo = client.get("/api/system").json()
    assert "torch" in sysinfo and sysinfo["cpu_count"] >= 1
    m = client.get("/api/methods").json()
    keys = {x["key"] for x in m["methods"]}
    assert {"lstm_forecaster", "lstm_autoencoder", "persistence", "random"} <= keys
    assert m["eval_start"] == 250


def test_dataset_registered_from_real_files(client):
    st = client.get("/api/dataset/status").json()
    assert st["available"] is True
    s = st["snapshot"]["summary"]
    assert s["label_rows"] == 82 and s["unique_channels"] == 81
    assert s["label_sequences_raw"] == 105 and s["label_sequences_after_merge"] == 104
    assert s["by_spacecraft"]["SMAP"]["channels"] == 54 and s["by_spacecraft"]["MSL"]["channels"] == 27
    assert s["by_spacecraft"]["SMAP"]["features"] == [25] and s["by_spacecraft"]["MSL"]["features"] == [55]
    chans = client.get("/api/channels").json()
    assert len(chans) == 81
    p2 = client.get("/api/channels/P-2").json()
    assert p2["label_rows"] == 2
    assert [(i["start"], i["end"]) for i in p2["intervals"]] == [(5300, 6575)]
    assert p2["intervals"][0]["merged_from"]


def test_telemetry_endpoint(client):
    t = client.get("/api/telemetry/T-9?split=test").json()
    assert t["n"] == 1096 and len(t["values"]) == 1096 and t["spacecraft"] == "MSL"
    assert t["labels"] and all(0 <= lab["start"] <= lab["end"] < 1096 for lab in t["labels"])
    d = client.get("/api/telemetry/P-1?split=test&max_points=400").json()
    assert len(d["values"]) <= 400 and d["n"] == 8505
    assert client.get("/api/telemetry/NOPE-1").status_code == 404
    tr = client.get("/api/telemetry/T-9?split=train").json()
    assert tr["labels"] == []  # labels describe the test split only


def test_experiment_validation(client):
    assert client.post("/api/experiments", json={"method": "nope", "channels": ["T-9"]}).status_code == 422
    r = client.post("/api/experiments", json={"method": "lstm_forecaster", "channels": ["T-9"],
                                              "config": {"window": 400}})
    assert r.status_code == 422
    assert client.post("/api/experiments", json={"method": "persistence", "channels": ["ZZ-1"]}).status_code == 422


@pytest.fixture(scope="module")
def lstm_run(client):
    r = client.post("/api/experiments", json={
        "method": "lstm_forecaster", "profile": "quick", "channels": ["T-9", "C-2"],
        "config": {"epochs": 3}, "name": "integration quick LSTM",
    })
    assert r.status_code == 201, r.text
    exp = wait_experiment(client, r.json()["id"])
    assert exp["status"] == "completed", exp
    return exp


def test_training_job_produces_real_results(client, lstm_run):
    runs = {r["chan_id"]: r for r in lstm_run["runs"]}
    assert set(runs) == {"T-9", "C-2"}
    for r in runs.values():
        assert r["status"] == "completed" and r["has_model"] and r["epochs_run"] >= 1
    detail = client.get(f"/api/runs/{runs['T-9']['id']}").json()
    assert len(detail["epochs"]) == detail["epochs_run"]
    assert all(e["train_loss"] > 0 for e in detail["epochs"])
    assert detail["model_available"]
    m = lstm_run["metrics"]["overall"]
    assert m["n_channels"] == 2 and m["event"]["n_labels"] == 4  # T-9: 2 sequences, C-2: 2


def test_series_and_live_inference_agree(client, lstm_run):
    run_id = next(r["id"] for r in lstm_run["runs"] if r["chan_id"] == "T-9")
    stored = client.get(f"/api/runs/{run_id}/series").json()
    assert stored["n"] == 1096 and stored["eval_start"] == 250
    assert stored["prediction"][99] is None and stored["prediction"][100] is not None  # quick window = 100
    live = client.post(f"/api/runs/{run_id}/infer", json={}).json()
    assert live["source"] == "live_inference"
    np.testing.assert_allclose(
        np.array(live["prediction"][100:], float), np.array(stored["prediction"][100:], float), atol=1e-4
    )
    assert [(d["start"], d["end"]) for d in live["detections"]] == [(d["start"], d["end"]) for d in stored["detections"]]
    alt = client.post(f"/api/runs/{run_id}/infer", json={"threshold": {"mode": "static", "static_k": 3}}).json()
    assert alt["threshold_config"]["mode"] == "static"


def test_inference_rejects_bad_threshold_overrides(client, lstm_run):
    run_id = next(r["id"] for r in lstm_run["runs"] if r["chan_id"] == "T-9")
    for bad in ({"p": None}, {"mode": "bogus"}, {"batch_size": 0}, {"mode": "static", "static_k": None}):
        r = client.post(f"/api/runs/{run_id}/infer", json={"threshold": bad})
        assert r.status_code == 422, (bad, r.text)
        assert "threshold" in r.json()["detail"]
    r = client.post("/api/experiments", json={"method": "persistence", "channels": ["T-9"],
                                              "threshold": {"error_buffer": -3}})
    assert r.status_code == 422 and "error_buffer" in r.json()["detail"]


def test_corrupt_checkpoint_gives_clear_error(client, lstm_run):
    from app.core.config import resolve_path
    from app.db.models import ChannelRun
    from app.db.session import session_scope

    run = next(r for r in lstm_run["runs"] if r["chan_id"] == "C-2")
    with session_scope() as db:
        path = resolve_path(db.get(ChannelRun, run["id"]).model_path)
    original = path.read_bytes()
    try:
        path.write_bytes(b"not a checkpoint")
        for r in (client.post(f"/api/runs/{run['id']}/infer", json={}),
                  client.post("/api/replay", json={"run_id": run["id"]})):
            assert r.status_code == 409, r.text
            assert "could not be loaded" in r.json()["detail"]
    finally:
        path.write_bytes(original)
    assert client.post(f"/api/runs/{run['id']}/infer", json={}).status_code == 200


def test_missing_dataset_gives_clear_error(client, lstm_run, monkeypatch):
    from app.dataset import store

    run_id = lstm_run["runs"][0]["id"]
    monkeypatch.setattr(store, "is_available", lambda: False)
    for r in (client.post(f"/api/runs/{run_id}/infer", json={}), client.get(f"/api/runs/{run_id}/series")):
        assert r.status_code == 503, r.text
        assert "setup.sh" in r.json()["detail"]


def test_replay_matches_offline(client, lstm_run):
    run_id = next(r["id"] for r in lstm_run["runs"] if r["chan_id"] == "C-2")
    stored = client.get(f"/api/runs/{run_id}/series").json()
    s = client.post("/api/replay", json={"run_id": run_id}).json()
    assert "not a live spacecraft connection" in s["notice"]
    sid, chunks, last = s["session_id"], [], None
    rng = np.random.default_rng(0)
    while True:
        last = client.post(f"/api/replay/{sid}/advance", json={"n": int(rng.integers(5, 300))}).json()
        chunks.append(last)
        if last["done"]:
            break
    pred = np.concatenate([np.array(c["prediction"], float) for c in chunks])
    np.testing.assert_allclose(pred[100:], np.array(stored["prediction"][100:], float), atol=1e-4)
    assert [(i["start"], i["end"]) for i in last["intervals"]] == [(d["start"], d["end"]) for d in stored["detections"]]
    # verdicts cover exactly the evaluated region
    covered = sum(c["verdict"]["to"] - c["verdict"]["from"] for c in chunks)
    assert covered == stored["n"] - stored["eval_start"]
    client.delete(f"/api/replay/{sid}")


def test_detections_and_exports(client, lstm_run):
    dets = client.get(f"/api/detections?experiment_id={lstm_run['id']}").json()
    for d in dets:
        detail = client.get(f"/api/detections/{d['id']}").json()
        assert detail["start"] == d["start"] and detail["context"]["actual"]
        assert "not a calibrated probability" in detail["confidence_note"]
    j = client.get(f"/api/export/experiments/{lstm_run['id']}.json")
    assert j.status_code == 200 and j.json()["id"] == lstm_run["id"]
    c = client.get(f"/api/export/experiments/{lstm_run['id']}/channels.csv")
    assert c.status_code == 200 and c.text.startswith("chan_id")
    assert client.get(f"/api/export/experiments/{lstm_run['id']}/detections.csv").status_code == 200


def test_overview_counts_come_from_database(client, lstm_run):
    o = client.get("/api/overview").json()
    assert o["dataset"]["available"]
    assert o["spacecraft"]["SMAP"]["channels"] == 54 and o["spacecraft"]["MSL"]["label_sequences"] == 36
    assert o["completed_channel_runs"] >= 2
    assert any(e["id"] == lstm_run["id"] for e in o["recent_experiments"])


def test_cancel_running_experiment(client):
    r = client.post("/api/experiments", json={"method": "lstm_forecaster", "profile": "paper", "channels": ["P-10"]})
    exp_id = r.json()["id"]
    t0 = time.time()
    while client.get(f"/api/experiments/{exp_id}").json()["status"] != "running":
        assert time.time() - t0 < 60
        time.sleep(0.3)
    assert client.post(f"/api/experiments/{exp_id}/cancel").json()["result"] == "cancelling"
    final = wait_experiment(client, exp_id, timeout=90)
    assert final["status"] == "cancelled"
    assert client.delete(f"/api/experiments/{exp_id}").status_code == 200
