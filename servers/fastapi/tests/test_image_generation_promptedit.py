"""PromptEdit MCP integration without spending credits or contacting PromptEdit."""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from services import promptedit_image_service as provider


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def list_tools(self):
        return [
            SimpleNamespace(name="generate_asset_tool"),
            SimpleNamespace(name="get_generation_tool"),
        ]

    async def call_tool(self, name, arguments, **_kwargs):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            is_error=False,
            structured_content=next(self.responses),
            content=[],
        )


def test_promptedit_generation_polls_and_saves_published_asset(monkeypatch, tmp_path):
    client = FakeClient(
        [
            {"asset_id": "asset-123"},
            {"status": "processing"},
            {"status": "published", "asset_url": "https://storage.contentcreatortemplates.com/test/image.png"},
        ]
    )
    transport = []
    monkeypatch.setattr(provider, "get_promptedit_api_key_env", lambda: "token-secret")
    monkeypatch.setattr(provider, "StreamableHttpTransport", lambda **kwargs: transport.append(kwargs))
    monkeypatch.setattr(provider, "Client", lambda _transport: client)
    monkeypatch.setattr(provider, "_download_image", AsyncMock(return_value=str(tmp_path / "result.png")))

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(provider.asyncio, "sleep", no_sleep)
    result = asyncio.run(provider.generate_promptedit_image("a slide illustration", str(tmp_path)))

    assert result == str(tmp_path / "result.png")
    assert transport == [{
        "url": "https://promptedit.com/mcp",
        "headers": {"Authorization": "Bearer token-secret"},
    }]
    assert client.calls == [
        (
            "generate_asset_tool",
            {
                "model": "gemini-3-pro-image",
                "params": {
                    "prompt": "a slide illustration",
                    "model": "gemini-3-pro-image",
                    "aspect_ratio": "16:9",
                    "image_size": "2K",
                    "number_of_generations": 1,
                },
            },
        ),
        ("get_generation_tool", {"asset_id": "asset-123"}),
        ("get_generation_tool", {"asset_id": "asset-123"}),
    ]


def test_promptedit_requires_token_before_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "get_promptedit_api_key_env", lambda: "")
    with pytest.raises(ValueError, match="PROMPTEDIT_API_KEY"):
        asyncio.run(provider.generate_promptedit_image("prompt", str(tmp_path)))


def test_promptedit_failed_generation_is_not_downloaded(monkeypatch, tmp_path):
    client = FakeClient([{"asset_id": "asset-123"}, {"status": "failed"}])
    monkeypatch.setattr(provider, "get_promptedit_api_key_env", lambda: "token-secret")
    monkeypatch.setattr(provider, "StreamableHttpTransport", lambda **_kwargs: None)
    monkeypatch.setattr(provider, "Client", lambda _transport: client)
    download = AsyncMock()
    monkeypatch.setattr(provider, "_download_image", download)

    with pytest.raises(RuntimeError, match="asset-123 ended as failed"):
        asyncio.run(provider.generate_promptedit_image("prompt", str(tmp_path)))
    download.assert_not_awaited()


def test_promptedit_rejects_private_download_url(tmp_path):
    with pytest.raises(ValueError, match="unsupported image URL"):
        asyncio.run(provider._download_image("http://127.0.0.1/private", str(tmp_path)))


def test_promptedit_download_stores_a_valid_image_without_forwarding_token(monkeypatch, tmp_path):
    image_data = BytesIO()
    Image.new("RGB", (12, 8), color="blue").save(image_data, format="JPEG")
    requested = []

    class FakeResponse:
        status = 200
        content = SimpleNamespace(read=AsyncMock(return_value=image_data.getvalue()))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def get(self, url, **kwargs):
            requested.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr(provider.aiohttp, "ClientSession", lambda **_kwargs: FakeSession())
    url = "https://storage.contentcreatortemplates.com/images/result.jpg"
    saved = asyncio.run(provider._download_image(url, str(tmp_path)))

    with Image.open(saved) as image:
        assert image.format == "PNG"
        assert image.size == (12, 8)
    assert requested[0][0] == url
    assert requested[0][1]["allow_redirects"] is False
    assert "headers" not in requested[0][1]
