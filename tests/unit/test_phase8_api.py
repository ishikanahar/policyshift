"""Phase 8 API smoke (requires fastapi)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from policyshift.api.app import app

pytestmark = pytest.mark.phase8


def test_api_health() -> None:
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_experiments_list() -> None:
    client = TestClient(app)
    res = client.get("/api/experiments")
    assert res.status_code == 200
    assert "experiments" in res.json()
