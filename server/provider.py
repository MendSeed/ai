"""
Inference Provider Interface and Implementations for MendSeed AI.

Provides unified abstractions across:
- Production Hugging Face Inference API / Dedicated Endpoints / vLLM
- Mock Provider for deterministic tests, CI, and error-case simulation
- Heuristic fallback mechanisms

All providers return typed ProviderResult objects and never silently fall back
to unauthorized or unconfigured models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional
import httpx

logger = logging.getLogger("mendseed_ai.provider")


@dataclass
class ModelConfig:
    """Explicit configuration for inference model and endpoint."""
    provider_type: Literal["huggingface", "custom_endpoint", "mock", "none"] = "huggingface"
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_id: Optional[str] = None
    adapter_revision: Optional[str] = None
    endpoint_url: Optional[str] = None
    api_token: Optional[str] = None
    temperature: float = 0.1
    max_new_tokens: int = 512
    timeout_seconds: float = 25.0
    require_adapter: bool = False

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Builds explicit model configuration from environment variables."""
        provider = os.getenv("MODEL_PROVIDER", "huggingface").lower()
        base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        adapter_id = os.getenv("ADAPTER_ID") or os.getenv("ADAPTER_REPO")
        adapter_revision = os.getenv("ADAPTER_REVISION")
        endpoint_url = os.getenv("HF_ENDPOINT_URL") or os.getenv("INFERENCE_ENDPOINT_URL")
        api_token = os.getenv("HF_TOKEN")
        require_adapter = os.getenv("REQUIRE_ADAPTER", "false").lower() in ("true", "1", "yes")

        try:
            temperature = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
        except ValueError:
            temperature = 0.1

        try:
            max_new_tokens = int(os.getenv("MODEL_MAX_NEW_TOKENS", "512"))
        except ValueError:
            max_new_tokens = 512

        try:
            timeout_seconds = float(os.getenv("MODEL_TIMEOUT_SECONDS", "25.0"))
        except ValueError:
            timeout_seconds = 25.0

        return cls(
            provider_type=provider if provider in ["huggingface", "custom_endpoint", "mock", "none"] else "huggingface",
            base_model=base_model,
            adapter_id=adapter_id if adapter_id else None,
            adapter_revision=adapter_revision if adapter_revision else None,
            endpoint_url=endpoint_url if endpoint_url else None,
            api_token=api_token if api_token else None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
            require_adapter=require_adapter,
        )

    def validate_configuration(self) -> tuple[bool, str]:
        """
        Validates that required configuration is present.
        Never silently selects public base model if adapter is required.
        """
        if self.provider_type == "none":
            return False, "Model provider explicitly disabled ('none')"
        if self.provider_type == "mock":
            return True, "Mock provider active"
        if self.require_adapter and not self.adapter_id:
            return False, "Adapter is marked as required (REQUIRE_ADAPTER=true), but no ADAPTER_ID was configured"
        if not self.endpoint_url:
            return False, "No INFERENCE_ENDPOINT_URL or HF_ENDPOINT_URL configured"
        return True, "Configuration valid"



@dataclass
class ProviderResult:
    """Standardized response from any inference provider call."""
    success: bool
    text: Optional[str] = None
    status: Literal[
        "success",
        "timeout",
        "model_loading",
        "auth_error",
        "provider_error",
        "unconfigured",
        "malformed_output"
    ] = "success"
    is_fallback: bool = False
    error_detail: Optional[str] = None
    estimated_wait_time: Optional[float] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseInferenceProvider(ABC):
    """Abstract interface for all model inference providers."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        """Generates text from system and user prompt strings."""
        pass


class HuggingFaceInferenceProvider(BaseInferenceProvider):
    """
    HTTP client for remote Hugging Face Inference API and dedicated Endpoints.
    Handles timeouts, 503 model-loading (with estimated_time), and auth errors explicitly.
    """

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        is_valid, reason = self.config.validate_configuration()
        if not is_valid:
            logger.warning(f"Inference called but provider configuration is invalid: {reason}")
            return ProviderResult(
                success=False,
                status="unconfigured",
                is_fallback=True,
                error_detail=reason,
                model_metadata=self._get_meta()
            )

        headers = {
            "Content-Type": "application/json"
        }
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": False,
                "return_full_text": False
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                resp = await client.post(
                    self.config.endpoint_url,
                    headers=headers,
                    json=payload
                )

                # Case 1: Success (200 OK)
                if resp.status_code == 200:
                    data = resp.json()
                    raw = ""
                    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                        raw = data[0]["generated_text"].strip()
                    elif isinstance(data, dict) and "generated_text" in data:
                        raw = data["generated_text"].strip()
                    elif isinstance(data, str):
                        raw = data.strip()
                    else:
                        raw = str(data)

                    # Strip outer code fences if any
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
                    return ProviderResult(
                        success=True,
                        text=cleaned,
                        status="success",
                        is_fallback=False,
                        model_metadata=self._get_meta()
                    )

                # Case 2: Model is currently cold/loading (503 Service Unavailable)
                elif resp.status_code == 503:
                    try:
                        err_body = resp.json()
                        estimated_time = err_body.get("estimated_time", 20.0)
                        err_msg = err_body.get("error", "Model currently loading")
                    except Exception:
                        estimated_time = 20.0
                        err_msg = resp.text
                    logger.info(f"Model loading (503), estimated time: {estimated_time}s")
                    return ProviderResult(
                        success=False,
                        status="model_loading",
                        is_fallback=True,
                        error_detail=f"Model loading: {err_msg}",
                        estimated_wait_time=float(estimated_time),
                        model_metadata=self._get_meta()
                    )

                # Case 3: Auth Error (401 or 403)
                elif resp.status_code in (401, 403):
                    logger.error(f"Upstream provider authentication failed: HTTP {resp.status_code}")
                    return ProviderResult(
                        success=False,
                        status="auth_error",
                        is_fallback=True,
                        error_detail=f"Upstream provider authentication error: {resp.status_code}",
                        model_metadata=self._get_meta()
                    )

                # Case 4: Other HTTP errors (400, 404, 500, etc.)
                else:
                    logger.warning(f"Upstream provider returned HTTP {resp.status_code}: {resp.text[:200]}")
                    return ProviderResult(
                        success=False,
                        status="provider_error",
                        is_fallback=True,
                        error_detail=f"Provider returned HTTP {resp.status_code}: {resp.text[:150]}",
                        model_metadata=self._get_meta()
                    )

        except httpx.TimeoutException:
            logger.warning(f"Inference timeout after {self.config.timeout_seconds}s")
            return ProviderResult(
                success=False,
                status="timeout",
                is_fallback=True,
                error_detail=f"Inference request timed out after {self.config.timeout_seconds}s",
                model_metadata=self._get_meta()
            )
        except Exception as e:
            logger.warning(f"Unexpected inference provider error: {str(e)}")
            return ProviderResult(
                success=False,
                status="provider_error",
                is_fallback=True,
                error_detail=str(e),
                model_metadata=self._get_meta()
            )

    def _get_meta(self) -> Dict[str, Any]:
        return {
            "provider": "huggingface",
            "base_model": self.config.base_model,
            "adapter_id": self.config.adapter_id,
            "adapter_revision": self.config.adapter_revision,
            "endpoint_configured": bool(self.config.endpoint_url),
            "temperature": self.config.temperature,
            "max_new_tokens": self.config.max_new_tokens,
        }


class MockInferenceProvider(BaseInferenceProvider):
    """
    Test and offline mock inference provider.
    Enables unit tests, regression suites, and CI to run without network or GPU access.
    Supports programming specific canned responses, simulated timeouts, loading states,
    and malformed outputs.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config or ModelConfig(provider_type="mock"))
        self.canned_responses: Dict[str, str] = {}
        self.simulate_mode: Literal["success", "timeout", "model_loading", "malformed", "provider_error"] = "success"
        self.call_count: int = 0
        self.last_prompt: Optional[str] = None

    def set_canned_response(self, prompt_substring: str, response_json_str: str) -> None:
        """Sets a canned output for any prompt containing the given substring."""
        self.canned_responses[prompt_substring] = response_json_str

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        self.call_count += 1
        self.last_prompt = f"{system_prompt}\n---\n{user_prompt}"
        meta = {
            "provider": "mock",
            "base_model": self.config.base_model,
            "adapter_id": self.config.adapter_id,
            "call_count": self.call_count
        }

        if self.simulate_mode == "timeout":
            return ProviderResult(
                success=False,
                status="timeout",
                is_fallback=True,
                error_detail="Simulated mock timeout",
                model_metadata=meta
            )

        if self.simulate_mode == "model_loading":
            return ProviderResult(
                success=False,
                status="model_loading",
                is_fallback=True,
                error_detail="Simulated model cold-start loading",
                estimated_wait_time=15.0,
                model_metadata=meta
            )

        if self.simulate_mode == "malformed":
            return ProviderResult(
                success=True,
                text="This is not valid JSON! {incomplete...",
                status="success",
                is_fallback=False,
                model_metadata=meta
            )

        if self.simulate_mode == "provider_error":
            return ProviderResult(
                success=False,
                status="provider_error",
                is_fallback=True,
                error_detail="Simulated provider failure",
                model_metadata=meta
            )

        # Check canned responses
        combined = f"{system_prompt} {user_prompt}"
        for sub, resp in self.canned_responses.items():
            if sub in combined:
                return ProviderResult(
                    success=True,
                    text=resp,
                    status="success",
                    is_fallback=False,
                    model_metadata=meta
                )

        # Default synthetic schema outputs for each task if no specific match
        if "GrievanceAnalysis" in system_prompt:
            first_sentence = user_prompt.split(".")[0].replace("Process citizen grievance:\n", "").strip()[:80]
            default_resp = json.dumps({
                "category": "Agriculture & Rural Development" if "farm" in user_prompt.lower() else "Civic & Public Infrastructure",
                "normalized_title": first_sentence or "Mock Structured Citizen Grievance",
                "summary": user_prompt[:180].strip(),
                "affected_sector": "Agriculture & Rural Development" if "farm" in user_prompt.lower() else "Civic & Public Infrastructure",
                "location_or_demographic": "Mock Region Ward 10",
                "observed_friction": user_prompt[:120],
                "current_workarounds": "Informal community coping mechanisms",
                "missing_information": ["GPS coordinates", "Administrative block ID"],
                "confidence": 0.95
            })
            return ProviderResult(success=True, text=default_resp, status="success", is_fallback=False, model_metadata=meta)

        if "GrievanceComparison" in system_prompt:
            default_resp = json.dumps({
                "relationship": "Same underlying problem",
                "evidence": ["Both reports address localized infrastructural service failures."],
                "matching_score": 0.88,
                "meaningful_differences": ["Different geographical wards and administrative zones."],
                "recommended_action": "cluster_under_canonical"
            })
            return ProviderResult(success=True, text=default_resp, status="success", is_fallback=False, model_metadata=meta)

        if "CommunitySummary" in system_prompt:
            default_resp = json.dumps({
                "main_problems": ["Recurring infrastructure maintenance failure across ward reports."],
                "recurring_themes": ["Slow administrative response", "Hazard to commuters and school children"],
                "disagreements": [],
                "evidence_backed_feedback": ["Multiple verified citizen submissions reporting unresolved friction."]
            })
            return ProviderResult(success=True, text=default_resp, status="success", is_fallback=False, model_metadata=meta)

        if "BuilderBrief" in system_prompt:
            default_resp = json.dumps({
                "problem_statement": "Actionable community challenge extracted from citizen evidence.",
                "target_beneficiaries": "Local affected citizens and municipal ward residents",
                "current_ground_reality": "Current municipal coping mechanisms are manual and slow.",
                "impact": "Causes recurring daily delays and localized economic loss.",
                "solution_opportunities": [
                    "Develop low-cost community reporting tool",
                    "Build open-data tracking dashboard"
                ],
                "unresolved_questions": ["What is the mobile smartphone penetration in the ward?"]
            })
            return ProviderResult(success=True, text=default_resp, status="success", is_fallback=False, model_metadata=meta)

        if "CanonicalClusterProposal" in system_prompt or "cluster" in system_prompt.lower():
            cand_id = "PROB-001"
            match = re.search(r'[\'"]id[\'"]:\s*[\'"]([^\'"]+)[\'"]', user_prompt)
            if match:
                cand_id = match.group(1)
            default_resp = json.dumps({
                "candidate_canonical_problem_id": cand_id,
                "proposed_grievance_links": [cand_id],
                "supporting_evidence": "Ground symptoms align with existing canonical problem pattern.",
                "uncertainty_level": "Low",
                "human_review_required": True
            })
            return ProviderResult(success=True, text=default_resp, status="success", is_fallback=False, model_metadata=meta)

        # Generic valid fallback
        return ProviderResult(
            success=True,
            text=json.dumps({"status": "mock_response"}),
            status="success",
            is_fallback=False,
            model_metadata=meta
        )
