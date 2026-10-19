from abc import ABC, abstractmethod
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ModelClient(ABC):
    model: str

    @abstractmethod
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


class OpenAICompatibleModelClient(ModelClient):
    """Chat-completions wire adapter; also usable with a compatible Runpod endpoint."""
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("invalid LLM_BASE_URL")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("use HTTPS for non-local model endpoints")
        if parsed.query or parsed.fragment:
            raise ValueError("base URL cannot contain query or fragment")
        if not api_key or not model:
            raise ValueError("LLM_API_KEY and LLM_MODEL are required for LLM mode")
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    @classmethod
    def from_env(cls):
        return cls(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                   os.getenv("LLM_API_KEY", ""), os.getenv("LLM_MODEL", ""))

    def complete(self, messages, tools):
        payload = dict(model=self.model, messages=messages, tools=tools, tool_choice="auto")
        req = Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise RuntimeError("model response exceeds size limit")
            return json.loads(raw)["choices"][0]["message"]
        except HTTPError as exc:
            raise RuntimeError(f"model endpoint HTTP {exc.code}") from None
        except (URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid or unreachable model endpoint ({type(exc).__name__})") from None
