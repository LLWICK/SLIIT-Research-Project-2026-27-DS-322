from pathlib import Path

from fastapi.testclient import TestClient

from member3.api import build_app


def payload(**overrides):
    base = {
        "district": "Badulla",
        "market": "colombo",
        "crop": "carrot",
        "season": "Maha",
        "proposed_area_ha": 20.0,
        "horizon_weeks": 4,
        "suitability": {"carrot": 0.8, "leek": 0.7, "tomato": 0.6},
    }
    base.update(overrides)
    return base


def test_assessment_does_not_create_a_commitment(tmp_path: Path):
    client = TestClient(build_app(tmp_path / "test.sqlite3"))
    response = client.post("/recommendations/assess", json=payload())
    assert response.status_code == 200
    assert client.get("/market-state/Badulla/colombo/carrot/Maha").json()["confirmed_area_ha"] == 0


def test_confirm_update_and_cancel_adjust_aggregate(tmp_path: Path):
    client = TestClient(build_app(tmp_path / "test.sqlite3"))
    commitment = payload(
        idempotency_key="commitment-001",
        farmer_reference="demo-farmer",
        planting_date="2026-10-01",
        harvest_date="2027-01-15",
        confirmed=True,
    )
    created = client.post("/commitments", json=commitment)
    assert created.status_code == 201
    commitment_id = created.json()["id"]
    repeated = client.post("/commitments", json=commitment)
    assert repeated.status_code == 201
    assert repeated.json()["id"] == commitment_id
    assert client.get("/market-state/Badulla/colombo/carrot/Maha").json()["confirmed_area_ha"] == 20
    assert client.patch(f"/commitments/{commitment_id}", json={"area_ha": 25}).status_code == 200
    assert client.get("/market-state/Badulla/colombo/carrot/Maha").json()["confirmed_area_ha"] == 25
    assert client.delete(f"/commitments/{commitment_id}").status_code == 200
    assert client.get("/market-state/Badulla/colombo/carrot/Maha").json()["confirmed_area_ha"] == 0


def test_unconfirmed_proposal_is_rejected(tmp_path: Path):
    client = TestClient(build_app(tmp_path / "test.sqlite3"))
    candidate = payload(
        idempotency_key="commitment-002",
        farmer_reference="demo-farmer",
        planting_date="2026-10-01",
        harvest_date="2027-01-15",
        confirmed=False,
    )
    assert client.post("/commitments", json=candidate).status_code == 400
