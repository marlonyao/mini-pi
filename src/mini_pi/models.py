"""
Multi-model provider registry for mini-pi.

Supports configuring multiple providers (Kimi, 智谱, DeepSeek, OpenAI, etc.)
and switching between models at runtime via /model command.

Configuration: ~/.mini-pi/models.json or .mini-pi/models.json

Format:
{
  "providers": {
    "kimi": {
      "api_key_env": "MOONSHOT_API_KEY",
      "base_url": "https://api.moonshot.cn/v1",
      "models": {
        "kimi-k2.6": {
          "max_context_tokens": 131072,
          "thinking": true
        },
        "kimi-k2.5": {
          "max_context_tokens": 131072,
          "thinking": true
        },
        "kimi-k2-turbo-preview": {
          "max_context_tokens": 131072
        }
      }
    },
    "zhipu": {
      "api_key_env": "ZHIPU_API_KEY",
      "base_url": "https://open.bigmodel.cn/api/paas/v4/",
      "models": {
        "glm-5.1": {
          "max_context_tokens": 200000,
          "thinking": true,
          "thinking_type": "enabled",
          "temperature": 1.0
        },
        "glm-5": {
          "max_context_tokens": 200000,
          "thinking": true,
          "thinking_type": "enabled",
          "temperature": 1.0
        },
        "glm-5-turbo": {
          "max_context_tokens": 200000,
          "thinking": true,
          "thinking_type": "enabled",
          "temperature": 1.0
        },
        "glm-4.7": {
          "max_context_tokens": 200000,
          "thinking": true,
          "thinking_type": "enabled"
        },
        "glm-4.7-flashx": {
          "max_context_tokens": 200000
        }
      }
    },
    "deepseek": {
      "api_key_env": "DEEPSEEK_API_KEY",
      "base_url": "https://api.deepseek.com",
      "models": {
        "deepseek-chat": {
          "max_context_tokens": 128000
        },
        "deepseek-reasoner": {
          "max_context_tokens": 128000,
          "thinking": true
        }
      }
    },
    "openai": {
      "api_key_env": "OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "models": {
        "gpt-4o": {
          "max_context_tokens": 128000
        },
        "gpt-4o-mini": {
          "max_context_tokens": 128000
        }
      }
    }
  },
  "default": "zhipu/glm-5.1"
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm.base import LLMBase
from .llm.openai_llm import OpenAILLM


# ── Defaults (bundled) ──────────────────────────────────────────────

BUILTIN_MODELS: dict[str, Any] = {
    "providers": {
        "kimi": {
            "api_key_env": "MOONSHOT_API_KEY",
            "base_url": "https://api.moonshot.cn/v1",
            "models": {
                "kimi-k2.6": {
                    "max_context_tokens": 131072,
                    "thinking": True,
                },
                "kimi-k2.5": {
                    "max_context_tokens": 131072,
                    "thinking": True,
                },
                "kimi-k2-turbo-preview": {
                    "max_context_tokens": 131072,
                },
            },
        },
        "kimi-coding": {
            "api_key_env": "KIMI_API_KEY",
            "base_url": "https://api.kimi.com/coding/v1",
            "api_type": "anthropic",
            "headers": {"User-Agent": "KimiCLI/1.5"},
            "models": {
                "kimi-for-coding": {
                    "max_context_tokens": 262144,
                    "thinking": True,
                },
                "kimi-k2-thinking": {
                    "max_context_tokens": 262144,
                    "thinking": True,
                },
            },
        },
        "zhipu": {
            "api_key_env": "ZHIPU_API_KEY",
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "models": {
                "glm-5.1": {
                    "max_context_tokens": 200000,
                    "thinking": True,
                    "thinking_type": "enabled",
                    "temperature": 1.0,
                },
                "glm-5": {
                    "max_context_tokens": 200000,
                    "thinking": True,
                    "thinking_type": "enabled",
                    "temperature": 1.0,
                },
                "glm-5-turbo": {
                    "max_context_tokens": 200000,
                    "thinking": True,
                    "thinking_type": "enabled",
                    "temperature": 1.0,
                },
                "glm-4.7": {
                    "max_context_tokens": 200000,
                    "thinking": True,
                    "thinking_type": "enabled",
                },
                "glm-4.7-flashx": {
                    "max_context_tokens": 200000,
                },
            },
        },
        "zai": {
            "api_key_env": "ZAI_API_KEY",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "models": {
                "glm-5.1": {
                    "max_context_tokens": 200000,
                    "thinking": True,
                    "thinking_type": "zai",
                    "zai_tool_stream": True,
                },
                "glm-5-turbo": {
                    "max_context_tokens": 200000,
                    "thinking": True,
                    "thinking_type": "zai",
                    "zai_tool_stream": True,
                },
                "glm-4.7": {
                    "max_context_tokens": 204800,
                    "thinking": True,
                    "thinking_type": "zai",
                    "zai_tool_stream": True,
                },
                "glm-4.5-air": {
                    "max_context_tokens": 131072,
                    "thinking": True,
                    "thinking_type": "zai",
                },
            },
        },
        "deepseek": {
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com",
            "models": {
                "deepseek-chat": {
                    "max_context_tokens": 128000,
                },
                "deepseek-reasoner": {
                    "max_context_tokens": 128000,
                    "thinking": True,
                },
            },
        },
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "models": {
                "gpt-4o": {
                    "max_context_tokens": 128000,
                },
                "gpt-4o-mini": {
                    "max_context_tokens": 128000,
                },
            },
        },
    },
    "default": None,
}


# ── Model Config ────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    """Resolved info for a specific model."""

    provider: str  # e.g. "zhipu", "kimi", "zai", "kimi-coding"
    model: str  # e.g. "glm-5.1", "kimi-k2.6"
    api_key: str
    base_url: str
    api_type: str = "openai"  # "openai" or "anthropic"
    max_context_tokens: int = 128000
    thinking: bool = False
    thinking_type: str | None = None  # "enabled" for zhipu, "zai" for z.ai
    temperature: float | None = None
    zai_tool_stream: bool = False  # z.ai specific: tool_stream=true
    headers: dict[str, str] | None = None  # Custom headers (e.g. User-Agent for kimi-coding)  # Some models require specific temperature


@dataclass
class ModelEntry:
    """A model entry from the config (not yet resolved with API key)."""

    provider: str
    model: str
    max_context_tokens: int = 128000
    thinking: bool = False
    thinking_type: str | None = None
    temperature: float | None = None


class ModelRegistry:
    """
    Manages available models and providers.

    Loads from:
    1. Built-in defaults (always available)
    2. ~/.mini-pi/models.json (user overrides)
    3. .mini-pi/models.json (project overrides)

    Later sources merge into earlier ones.
    """

    def __init__(self):
        self._providers: dict[str, dict[str, Any]] = {}
        self._load_defaults()
        self._load_user_config()

    def _load_defaults(self) -> None:
        """Load built-in model definitions."""
        self._providers = json.loads(json.dumps(BUILTIN_MODELS["providers"]))

    def _load_user_config(self) -> None:
        """Load user/project overrides."""
        config_paths = [
            Path.home() / ".mini-pi" / "models.json",
            Path.cwd() / ".mini-pi" / "models.json",
        ]
        for path in config_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._merge_config(data)
                except (json.JSONDecodeError, KeyError):
                    pass

    def _merge_config(self, data: dict[str, Any]) -> None:
        """Merge a config dict into current providers."""
        providers = data.get("providers", {})
        for prov_name, prov_data in providers.items():
            if prov_name not in self._providers:
                self._providers[prov_name] = prov_data
            else:
                existing = self._providers[prov_name]
                # Override provider-level settings
                if "api_key_env" in prov_data:
                    existing["api_key_env"] = prov_data["api_key_env"]
                if "base_url" in prov_data:
                    existing["base_url"] = prov_data["base_url"]
                # Merge models
                if "models" in prov_data:
                    existing.setdefault("models", {}).update(prov_data["models"])

    def resolve(self, model_spec: str) -> ModelInfo | None:
        """
        Resolve a model spec to ModelInfo.

        Specs:
        - "glm-5.1"        → search all providers for this model
        - "zhipu/glm-5.1"  → explicit provider
        - "glm-5.1-flashx" → search all providers (fuzzy match)

        Returns None if not found or API key unavailable.
        """
        # Parse provider/model
        if "/" in model_spec:
            provider, model = model_spec.split("/", 1)
            return self._resolve_explicit(provider, model)
        else:
            return self._resolve_fuzzy(model_spec)

    def _resolve_explicit(self, provider: str, model: str) -> ModelInfo | None:
        """Resolve with explicit provider name."""
        prov = self._providers.get(provider)
        if not prov:
            return None
        model_data = prov.get("models", {}).get(model)
        if not model_data:
            return None
        return self._build_model_info(provider, model, prov, model_data)

    def _resolve_fuzzy(self, model: str) -> ModelInfo | None:
        """Resolve by searching all providers for the model name."""
        # Exact match first
        for prov_name, prov_data in self._providers.items():
            if model in prov_data.get("models", {}):
                return self._build_model_info(
                    prov_name, model, prov_data, prov_data["models"][model]
                )
        return None

    def _build_model_info(
        self,
        provider: str,
        model: str,
        prov_data: dict[str, Any],
        model_data: dict[str, Any],
    ) -> ModelInfo:
        """Build ModelInfo from config data."""
        api_key_env = prov_data.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env, "")
        base_url = prov_data.get("base_url", "https://api.openai.com/v1")

        return ModelInfo(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_type=prov_data.get("api_type", "openai"),
            max_context_tokens=model_data.get("max_context_tokens", 128000),
            thinking=model_data.get("thinking", False),
            thinking_type=model_data.get("thinking_type"),
            temperature=model_data.get("temperature"),
            zai_tool_stream=model_data.get("zai_tool_stream", False),
            headers=prov_data.get("headers"),
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models with their provider and key status."""
        result = []
        for prov_name, prov_data in sorted(self._providers.items()):
            api_key_env = prov_data.get("api_key_env", "OPENAI_API_KEY")
            api_key_set = bool(os.getenv(api_key_env))
            base_url = prov_data.get("base_url", "")

            for model_name, model_data in sorted(prov_data.get("models", {}).items()):
                result.append({
                    "provider": prov_name,
                    "model": model_name,
                    "spec": f"{prov_name}/{model_name}",
                    "base_url": base_url,
                    "max_context_tokens": model_data.get("max_context_tokens", 128000),
                    "thinking": model_data.get("thinking", False),
                    "api_key_env": api_key_env,
                    "api_key_set": api_key_set,
                })
        return result

    def list_available(self) -> list[dict[str, Any]]:
        """List only models that have API keys configured."""
        return [m for m in self.list_models() if m["api_key_set"]]

    def get_default_spec(self) -> str | None:
        """Get the default model spec from config."""
        # Check user config for default
        for path in [
            Path.home() / ".mini-pi" / "models.json",
            Path.cwd() / ".mini-pi" / "models.json",
        ]:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if "default" in data and data["default"]:
                        return data["default"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return None


def create_llm(model_info: ModelInfo) -> LLMBase:
    """
    Create an LLM instance from ModelInfo.

    All providers use OpenAI-compatible API, so we always use OpenAILLM.
    Provider-specific quirks are handled via extra_body kwargs in the agent loop.
    """
    return OpenAILLM(
        api_key=model_info.api_key,
        base_url=model_info.base_url,
        model=model_info.model,
        default_headers=model_info.headers,
    )



def get_model_extra_kwargs(model_info: ModelInfo) -> dict[str, Any]:
    """
    Get extra kwargs to pass to the LLM call for provider-specific features.

    Provider-specific handling:
    - zhipu: extra_body={"thinking": {"type": "enabled"}}, temperature=1.0
    - zai: extra_body={"enable_thinking": True}, tool_stream=True
    - kimi: extra_body={"thinking": {"type": "enabled"}}
    - kimi-coding: (anthropic format, handled separately)
    - deepseek: reasoning_content in streaming (auto-detected)
    """
    kwargs: dict[str, Any] = {}

    # Temperature override
    if model_info.temperature is not None:
        kwargs["temperature"] = model_info.temperature
    else:
        kwargs["temperature"] = 0

    # Thinking mode
    if model_info.thinking:
        if model_info.provider == "zai":
            # 智谱 Coding Plan (z.ai): enable_thinking + tool_stream
            kwargs.setdefault("extra_body", {})["enable_thinking"] = True
            if model_info.zai_tool_stream:
                kwargs.setdefault("extra_body", {})["tool_stream"] = True
        elif model_info.provider == "zhipu":
            # 智谱开放平台: thinking as extra_body
            kwargs.setdefault("extra_body", {})["thinking"] = {
                "type": model_info.thinking_type or "enabled"
            }
        elif model_info.provider == "kimi":
            # Kimi 开放平台: thinking as extra_body
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        # kimi-coding uses Anthropic format, handled by AnthropicLLM

    return kwargs
