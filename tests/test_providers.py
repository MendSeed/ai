"""
Unit Tests for MendSeed AI Inference Providers and Model Configuration.

Tests:
1. ModelConfig validation (fails when adapter is required but not configured).
2. MockInferenceProvider returns valid typed ProviderResult.
3. MockInferenceProvider simulated error states:
   - Timeout
   - Model loading (503 with estimated wait time)
   - Malformed JSON output
   - Provider failure
"""

import json
import pytest

from server.provider import (
    BaseInferenceProvider,
    HuggingFaceInferenceProvider,
    MockInferenceProvider,
    ModelConfig,
    ProviderResult,
)


def test_model_config_requires_adapter_validation():
    # Valid without required adapter
    cfg_no_adapter = ModelConfig(
        provider_type="huggingface",
        endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen3-4B-Instruct-2507",
        require_adapter=False
    )
    valid, msg = cfg_no_adapter.validate_configuration()
    assert valid is True

    # Invalid when adapter is required but empty
    cfg_missing_adapter = ModelConfig(
        provider_type="huggingface",
        endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen3-4B-Instruct-2507",
        adapter_id=None,
        require_adapter=True
    )
    valid, msg = cfg_missing_adapter.validate_configuration()
    assert valid is False
    assert "require_adapter=true" in msg.lower()

    # Missing endpoint url
    cfg_missing_endpoint = ModelConfig(
        provider_type="huggingface",
        endpoint_url=None
    )
    valid, msg = cfg_missing_endpoint.validate_configuration()
    assert valid is False
    assert "no inference_endpoint_url" in msg.lower()


@pytest.mark.asyncio
async def test_mock_provider_success_response():
    provider = MockInferenceProvider()
    res = await provider.generate(
        system_prompt="Return GrievanceAnalysis schema",
        user_prompt="Farmers in Kolar lose tomato crops due to lack of cold storage."
    )
    assert isinstance(res, ProviderResult)
    assert res.success is True
    assert res.status == "success"
    assert res.is_fallback is False
    assert res.text is not None

    parsed = json.loads(res.text)
    assert "category" in parsed
    assert "normalized_title" in parsed


@pytest.mark.asyncio
async def test_mock_provider_simulated_timeout():
    provider = MockInferenceProvider()
    provider.simulate_mode = "timeout"

    res = await provider.generate("sys", "user")
    assert res.success is False
    assert res.status == "timeout"
    assert res.is_fallback is True
    assert "timeout" in res.error_detail.lower()


@pytest.mark.asyncio
async def test_mock_provider_simulated_model_loading():
    provider = MockInferenceProvider()
    provider.simulate_mode = "model_loading"

    res = await provider.generate("sys", "user")
    assert res.success is False
    assert res.status == "model_loading"
    assert res.is_fallback is True
    assert res.estimated_wait_time == 15.0


@pytest.mark.asyncio
async def test_mock_provider_simulated_malformed_json():
    provider = MockInferenceProvider()
    provider.simulate_mode = "malformed"

    res = await provider.generate("sys", "user")
    assert res.success is True  # Raw call returned text
    assert res.text is not None
    # Verify that text is indeed malformed JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.text)


@pytest.mark.asyncio
async def test_mock_provider_canned_response():
    provider = MockInferenceProvider()
    canned = json.dumps({"custom": "special_test_case_result"})
    provider.set_canned_response("specific_unique_key", canned)

    res = await provider.generate("system prompt", "contains specific_unique_key here")
    assert res.success is True
    assert res.text == canned
