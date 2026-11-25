import hashlib
import math
import re
import time
from itertools import pairwise

import httpx

from app.config import get_settings

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_'-]*")
HTTP_CLIENT = httpx.Client(
    timeout=httpx.Timeout(60, connect=10),
    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def local_embedding(text: str, dimensions: int) -> list[float]:
    """Deterministic feature-hash embedding used only when no model key is configured."""
    vector = [0.0] * dimensions
    words = _tokens(text)
    features = words + [f"{a}_{b}" for a, b in pairwise(words)]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dimensions
        sign = 1.0 if (value >> 11) & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.openai_api_key:
            return [local_embedding(text, self.settings.embedding_dimensions) for text in texts]

        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
            "dimensions": self.settings.embedding_dimensions,
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        response = None
        for attempt in range(4):
            try:
                response = HTTP_CLIENT.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/embeddings",
                    headers=headers,
                    json=payload,
                )
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 3:
                    response.raise_for_status()
                    break
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 8.0) if retry_after and retry_after.replace(".", "", 1).isdigit() else 0.5 * (2**attempt)
            except httpx.TransportError:
                if attempt == 3:
                    raise
                delay = 0.5 * (2**attempt)
            time.sleep(delay)
        if response is None:
            raise RuntimeError("Embedding request did not return a response")
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]
