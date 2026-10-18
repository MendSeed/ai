"""
Unit Tests for MendSeed AI Security and Authentication.

Tests:
1. Missing X-MendSeed-Key header returns 401 Unauthorized.
2. Invalid X-MendSeed-Key returns 403 Forbidden.
3. Valid X-MendSeed-Key allows access to protected endpoints.
4. Unconfigured MENDSEED_AI_SECRET_KEY safely fails with 500 Internal Server Error (never defaults to an insecure production key).
5. Public health check (/health) reports readiness without leaking secret configuration details.
6. CORS middleware prevents credentialed wildcard access.
"""

import os
import pytest
from fastapi.testclient import TestClient

from server.main import app

TEST_SECRET = "test_mendseed_secret_key_abcdef123456"


@pytest.fixture(autouse=True)
def setup_env():
    """Ensure test secret is active by default for standard tests."""
    original = os.environ.get("MENDSEED_AI_SECRET_KEY")
    os.environ["MENDSEED_AI_SECRET_KEY"] = TEST_SECRET
    yield
    if original is not None:
        os.environ["MENDSEED_AI_SECRET_KEY"] = original
    else:
        os.environ.pop("MENDSEED_AI_SECRET_KEY", None)


def test_missing_api_key_returns_401():
    client = TestClient(app)
    resp = client.post("/v1/analyze", json={"text": "Road has deep potholes near the school gate."})
    assert resp.status_code == 401
    assert "Missing required header" in resp.json()["detail"]


def test_invalid_api_key_returns_403():
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze",
        headers={"X-MendSeed-Key": "wrong_invalid_key_xyz"},
        json={"text": "Road has deep potholes near the school gate."}
    )
    assert resp.status_code == 403
    assert "Invalid MendSeed API key" in resp.json()["detail"]


def test_valid_api_key_succeeds():
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze",
        headers={"X-MendSeed-Key": TEST_SECRET},
        json={"text": "Road has deep potholes near the school gate."}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "category" in data
    assert "summary" in data


def test_unconfigured_secret_fails_safely(monkeypatch):
    """
    When MENDSEED_AI_SECRET_KEY is empty or missing, protected endpoints
    must reject requests with 500 rather than falling back to an insecure default key.
    """
    monkeypatch.delenv("MENDSEED_AI_SECRET_KEY", raising=False)
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze",
        headers={"X-MendSeed-Key": "any_attempted_key"},
        json={"text": "Road has deep potholes near the school gate."}
    )
    assert resp.status_code == 500
    assert "unconfigured" in resp.json()["detail"].lower()


def test_public_health_check_does_not_leak_secrets():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert data["gateway_ready"] is True

    # Security verification: Ensure secret states and API tokens are NEVER exposed
    resp_str = resp.text.lower()
    assert "secret" not in resp_str
    assert "token" not in resp_str
    assert "hf_token" not in resp_str
    assert "mendseed_ai_secret_key" not in resp_str
    assert "env_secret_active" not in resp_str


def test_cors_prevents_wildcard_credentials():
    """
    Verify that CORS headers do NOT allow wildcard origin together with credentials.
    """
    client = TestClient(app)
    resp = client.options(
        "/v1/analyze",
        headers={
            "Origin": "https://unauthorized-evil-site.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-MendSeed-Key, Content-Type"
        }
    )
    # If Access-Control-Allow-Origin is '*', Access-Control-Allow-Credentials must NOT be 'true'
    allow_origin = resp.headers.get("access-control-allow-origin")
    allow_cred = resp.headers.get("access-control-allow-credentials")

    if allow_origin == "*":
        assert allow_cred != "true", "Wildcard origin must never have allow_credentials=True"
