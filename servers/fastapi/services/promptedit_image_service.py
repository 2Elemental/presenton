"""Generate a locally stored Presenton image through PromptEdit's MCP server."""

import asyncio
from io import BytesIO
import json
import os
import time
from urllib.parse import urlparse
import uuid

import aiohttp
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from PIL import Image, UnidentifiedImageError

from utils.get_env import get_promptedit_api_key_env


PROMPTEDIT_MCP_URL = "https://promptedit.com/mcp"
PROMPTEDIT_MODEL = "gemini-3-pro-image"
POLL_INTERVAL_SECONDS = 3
MAX_GENERATION_SECONDS = 300
MAX_IMAGE_BYTES = 25 * 1024 * 1024
# PromptEdit currently publishes images on this host. Keep downloads away from
# private network addresses even if a compromised tool response supplies a URL.
IMAGE_HOSTS = {"storage.contentcreatortemplates.com", "promptedit.com"}


def _tool_payload(result) -> dict:
    if result.is_error:
        raise RuntimeError("PromptEdit MCP tool returned an error")
    payload = result.structured_content
    if payload is None:
        for block in result.content:
            if hasattr(block, "text"):
                payload = json.loads(block.text)
                break
    if not isinstance(payload, dict):
        raise ValueError("PromptEdit returned an unexpected MCP result")
    if isinstance(payload.get("result"), dict):
        return payload["result"]
    if isinstance(payload.get("result"), str):
        try:
            nested = json.loads(payload["result"])
        except json.JSONDecodeError:
            nested = None
        if isinstance(nested, dict):
            return nested
    return payload


def _tool_name(tools: list, name: str) -> str:
    # Some PromptEdit clients expose tool names with a trailing '_tool'.
    names = {tool.name for tool in tools}
    for candidate in (name, f"{name}_tool"):
        if candidate in names:
            return candidate
    raise RuntimeError(f"PromptEdit MCP does not expose {name}")


async def _download_image(url: str, output_directory: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in IMAGE_HOSTS or parsed.username:
        raise ValueError("PromptEdit returned an unsupported image URL")

    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=90),
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"PromptEdit image download returned {response.status}")
            content = await response.content.read(MAX_IMAGE_BYTES + 1)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError("PromptEdit image exceeds the download limit")

    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("PromptEdit returned an unsupported image format")
            os.makedirs(output_directory, exist_ok=True)
            image_path = os.path.join(output_directory, f"{uuid.uuid4()}.png")
            image.save(image_path, format="PNG")
            return image_path
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("PromptEdit returned invalid image data") from exc


async def generate_promptedit_image(prompt: str, output_directory: str) -> str:
    token = (get_promptedit_api_key_env() or "").strip()
    if not token or token == "__configured__":
        raise ValueError("PROMPTEDIT_API_KEY is required for PromptEdit image generation")

    transport = StreamableHttpTransport(
        url=PROMPTEDIT_MCP_URL,
        headers={"Authorization": f"Bearer {token}"},
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        generate = _tool_name(tools, "generate_asset")
        get_generation = _tool_name(tools, "get_generation")
        generated = _tool_payload(
            await client.call_tool(
                generate,
                {
                    "model": PROMPTEDIT_MODEL,
                    "params": {
                        "prompt": prompt[:5000],
                        "model": PROMPTEDIT_MODEL,
                        "aspect_ratio": "16:9",
                        "image_size": "2K",
                        "number_of_generations": 1,
                    },
                },
                timeout=60,
            )
        )
        asset_id = generated.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("PromptEdit did not return a generation ID")

        deadline = time.monotonic() + MAX_GENERATION_SECONDS
        while True:
            result = _tool_payload(
                await client.call_tool(
                    get_generation, {"asset_id": asset_id}, timeout=45
                )
            )
            status = result.get("status")
            if status == "published":
                url = result.get("asset_url")
                if not isinstance(url, str) or not url:
                    raise ValueError("PromptEdit published a generation without an image URL")
                return await _download_image(url, output_directory)
            if status in {"failed", "error", "cancelled", "canceled", "rejected"}:
                raise RuntimeError(f"PromptEdit generation {asset_id} ended as {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"PromptEdit generation {asset_id} did not finish in time")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
