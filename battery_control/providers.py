"""
LLM backend abstraction — pluggable providers for LLMController.

Each provider knows its own request/response format and its own token-limit
parameter name (num_predict for Ollama, max_tokens for OpenAI/Anthropic).
API keys are read from environment variables ONLY, never hardcoded:

    OPENAI_API_KEY       required for --provider openai
    ANTHROPIC_API_KEY    required for --provider anthropic

Ollama needs no key (local server). A missing key raises ProviderError at
CONSTRUCTION time, not on the first request — a run that fails auth on every
single call would otherwise fall back to the RBC action every step and look
like a clean (if boring) result, exactly the kind of silent failure this
codebase has been burned by before (see README: num_predict truncation).

Adding a new provider: add one class here, register it in PROVIDERS and
DEFAULT_MODELS.

Every provider's request() returns (text: str, elapsed_seconds: float) or
raises — LLMController's existing retry/timeout/fallback logic in _ask_llm
is provider-agnostic and untouched by this abstraction.
"""
import json
import os
import time
import urllib.request

DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    # best-effort defaults for the cloud providers -- these move faster than
    # local models; verify against current provider docs and override with
    # --model / model= if these have been superseded.
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
}


class ProviderError(RuntimeError):
    """Config-time failure (unknown provider, missing API key)."""


class Provider:
    name = "base"

    def __init__(self, model, timeout=300):
        self.model = model
        self.timeout = timeout

    def request(self, prompt, token_limit):
        """Returns (text, elapsed_seconds). Raises on error/timeout."""
        raise NotImplementedError


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model, timeout=300, host="http://localhost:11434"):
        super().__init__(model, timeout)
        self.host = host

    def request(self, prompt, token_limit):
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0,
                                      "num_predict": token_limit}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.loads(r.read())["response"]
        return text, time.perf_counter() - t0


class OpenAIProvider(Provider):
    name = "openai"
    ENV_KEY = "OPENAI_API_KEY"
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model, timeout=300, host=None):
        super().__init__(model, timeout)
        self.api_key = os.environ.get(self.ENV_KEY)
        if not self.api_key:
            raise ProviderError(f"{self.ENV_KEY} is not set. "
                                f"export {self.ENV_KEY}=... before using --provider openai")

    def request(self, prompt, token_limit):
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": token_limit}).encode()
        req = urllib.request.Request(self.URL, data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"]
        return text, time.perf_counter() - t0


class AnthropicProvider(Provider):
    name = "anthropic"
    ENV_KEY = "ANTHROPIC_API_KEY"
    URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model, timeout=300, host=None):
        super().__init__(model, timeout)
        self.api_key = os.environ.get(self.ENV_KEY)
        if not self.api_key:
            raise ProviderError(f"{self.ENV_KEY} is not set. "
                                f"export {self.ENV_KEY}=... before using --provider anthropic")

    def request(self, prompt, token_limit):
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": token_limit}).encode()
        req = urllib.request.Request(self.URL, data=body,
                                     headers={"Content-Type": "application/json",
                                              "x-api-key": self.api_key,
                                              "anthropic-version": self.API_VERSION})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        text = data["content"][0]["text"]
        return text, time.perf_counter() - t0


PROVIDERS = {"ollama": OllamaProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}


def make_provider(provider, model=None, timeout=300, host="http://localhost:11434"):
    if provider not in PROVIDERS:
        raise ProviderError(f"unknown provider {provider!r} — choose from {list(PROVIDERS)}")
    resolved_model = model or DEFAULT_MODELS[provider]
    return PROVIDERS[provider](resolved_model, timeout=timeout, host=host)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("--- unknown provider rejected ---")
    try:
        make_provider("does-not-exist")
        print("[FAIL] should have raised ProviderError")
    except ProviderError as e:
        print(f"[PASS] {e}")

    print("\n--- ollama needs no key, resolves default model ---")
    p = make_provider("ollama")
    good = p.model == "qwen3:8b" and isinstance(p, OllamaProvider)
    print(f"[{'PASS' if good else 'FAIL'}] model={p.model}")

    print("\n--- cloud providers require their API key ---")
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    try:
        for name in ("openai", "anthropic"):
            try:
                make_provider(name)
                print(f"[FAIL] {name}: should have raised ProviderError with no key set")
            except ProviderError as e:
                print(f"[PASS] {name}: {e}")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    print("\n--- explicit model overrides the per-provider default ---")
    os.environ["OPENAI_API_KEY"] = "test-key-not-real"
    try:
        p = make_provider("openai", model="gpt-4.1")
        good = p.model == "gpt-4.1"
        print(f"[{'PASS' if good else 'FAIL'}] model={p.model}")
    finally:
        del os.environ["OPENAI_API_KEY"]
