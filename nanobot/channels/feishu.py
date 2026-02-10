"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import io
import json
import os
import re
import threading
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateImageRequest,
        CreateImageRequestBody,
        CreateFileRequest,
        CreateFileRequestBody,
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.
    
    Uses WebSocket to receive events - no public IP or webhook required.
    
    Requires:
    - App ID and App Secret from Feishu Open Platform
    - Bot capability enabled
    - Event subscription enabled (im.message.receive_v1)
    """
    
    name = "feishu"
    
    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()  # Ordered dedup cache
        self._loop: asyncio.AbstractEventLoop | None = None
    
    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return
        
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        
        self._running = True
        self._loop = asyncio.get_running_loop()
        
        # Create Lark client for sending messages
        self._client = lark.Client.builder() \
            .app_id(self.config.app_id) \
            .app_secret(self.config.app_secret) \
            .log_level(lark.LogLevel.INFO) \
            .build()
        
        # Create event handler (only register message receive, ignore other events)
        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(
            self._on_message_sync
        ).build()
        
        # Create WebSocket client for long connection
        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO
        )
        
        # Start WebSocket client in a separate thread
        def run_ws():
            try:
                self._ws_client.start()
            except Exception as e:
                logger.error(f"Feishu WebSocket error: {e}")
        
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        logger.info("Feishu bot started with WebSocket long connection")
        logger.info("No public IP required - using WebSocket to receive events")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Feishu bot."""
        self._running = False
        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Error stopping WebSocket client: {e}")
        logger.info("Feishu bot stopped")
    
    def _upload_image_from_url(self, url: str) -> str | None:
        """
        Download an image from URL and upload to Feishu to get image_key.
        
        The returned image_key can be used in image messages.
        """
        if not self._client:
            return None
        
        try:
            with urlopen(url) as resp:
                data = resp.read()
        except Exception as e:
            logger.error(f"Failed to download image from URL {url}: {e}")
            return None
        
        return self._upload_image_bytes(data)

    def _upload_image_file(self, path: str) -> str | None:
        """
        Upload a local image file to Feishu and return image_key.
        """
        if not self._client:
            return None
        
        try:
            # Expand user (~) and resolve relative paths
            resolved = os.path.expanduser(path)
            with open(resolved, "rb") as f:
                data = f.read()
        except Exception as e:
            logger.error(f"Failed to read image file {path}: {e}")
            return None
        
        return self._upload_image_bytes(data)

    def _upload_image_bytes(self, data: bytes) -> str | None:
        """Upload raw image bytes to Feishu and return image_key."""
        if not self._client:
            return None
        
        try:
            image_io = io.BytesIO(data)
            req = CreateImageRequest.builder() \
                .request_body(
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(image_io)
                    .build()
                ).build()
            
            resp = self._client.im.v1.image.create(req)
            if not resp.success():
                logger.error(
                    f"Failed to upload image to Feishu: code={resp.code}, "
                    f"msg={resp.msg}, log_id={resp.get_log_id()}"
                )
                return None
            
            image_key = getattr(resp.data, "image_key", None)
            if not image_key:
                logger.error("Feishu image upload succeeded but no image_key returned")
            return image_key
        except Exception as e:
            logger.error(f"Error uploading image to Feishu: {e}")
            return None

    def _upload_file_from_path(self, path: str) -> str | None:
        """
        Upload a local file to Feishu and return file_key.
        """
        if not self._client:
            return None
        
        resolved = os.path.expanduser(path)
        file_name = os.path.basename(resolved) or "file"
        
        try:
            with open(resolved, "rb") as f:
                req = CreateFileRequest.builder() \
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type("stream")  # generic file type
                        .file_name(file_name)
                        .file(f)
                        .build()
                    ).build()
                
                resp = self._client.im.v1.file.create(req)
        except Exception as e:
            logger.error(f"Failed to upload file {path} to Feishu: {e}")
            return None
        
        if not resp.success():
            logger.error(
                f"Failed to upload file to Feishu: code={resp.code}, "
                f"msg={resp.msg}, log_id={resp.get_log_id()}"
            )
            return None
        
        file_key = getattr(resp.data, "file_key", None)
        if not file_key:
            logger.error("Feishu file upload succeeded but no file_key returned")
        return file_key

    def _upload_file_from_url(self, url: str) -> str | None:
        """
        Download a file from URL and upload to Feishu, returning file_key.
        """
        if not self._client:
            return None
        
        try:
            with urlopen(url) as resp:
                data = resp.read()
        except Exception as e:
            logger.error(f"Failed to download file from URL {url}: {e}")
            return None
        
        parsed = urlparse(url)
        file_name = os.path.basename(parsed.path) or "file"
        
        try:
            file_io = io.BytesIO(data)
            req = CreateFileRequest.builder() \
                .request_body(
                    CreateFileRequestBody.builder()
                    .file_type("stream")
                    .file_name(file_name)
                    .file(file_io)
                    .build()
                ).build()
            
            resp = self._client.im.v1.file.create(req)
        except Exception as e:
            logger.error(f"Failed to upload file from URL {url} to Feishu: {e}")
            return None
        
        if not resp.success():
            logger.error(
                f"Failed to upload URL file to Feishu: code={resp.code}, "
                f"msg={resp.msg}, log_id={resp.get_log_id()}"
            )
            return None
        
        file_key = getattr(resp.data, "file_key", None)
        if not file_key:
            logger.error("Feishu URL file upload succeeded but no file_key returned")
        return file_key
    
    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        """Sync helper for adding reaction (runs in thread pool)."""
        try:
            request = CreateMessageReactionRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                ).build()
            
            response = self._client.im.v1.message_reaction.create(request)
            
            if not response.success():
                logger.warning(f"Failed to add reaction: code={response.code}, msg={response.msg}")
            else:
                logger.debug(f"Added {emoji_type} reaction to message {message_id}")
        except Exception as e:
            logger.warning(f"Error adding reaction: {e}")

    async def _add_reaction(self, message_id: str, emoji_type: str = "THINKING") -> None:
        """
        Add a reaction emoji to a message (non-blocking).
        
        Common emoji types: THUMBSUP, OK, EYES, DONE, OnIt, HEART,TYPING,THINKING
        """
        if not self._client or not Emoji:
            return
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji_type)
    
    # Regex to match markdown tables (header + separator + data rows)
    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown table into a Feishu table element."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            return None
        split = lambda l: [c.strip() for c in l.strip("|").split("|")]
        headers = split(lines[0])
        rows = [split(l) for l in lines[2:]]
        columns = [{"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
                   for i, h in enumerate(headers)]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": columns,
            "rows": [{f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))} for r in rows],
        }

    def _build_card_elements(self, content: str) -> list[dict]:
        """Split content into markdown + table elements for Feishu card."""
        elements, last_end = [], 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last_end:m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            elements.append(self._parse_md_table(m.group(1)) or {"tag": "markdown", "content": m.group(1)})
            last_end = m.end()
        remaining = content[last_end:].strip()
        if remaining:
            elements.append({"tag": "markdown", "content": remaining})
        return elements or [{"tag": "markdown", "content": content}]

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Feishu."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return
        
        try:
            # Determine receive_id_type based on chat_id format
            # open_id starts with "ou_", chat_id starts with "oc_"
            if msg.chat_id.startswith("oc_"):
                receive_id_type = "chat_id"
            else:
                receive_id_type = "open_id"
            
            # 1) Send text as interactive card (if any content)
            content_text = (msg.content or "").strip()
            if content_text:
                elements = self._build_card_elements(content_text)
                card = {
                    "config": {"wide_screen_mode": True},
                    "elements": elements,
                }
                content = json.dumps(card, ensure_ascii=False)
                
                request = CreateMessageRequest.builder() \
                    .receive_id_type(receive_id_type) \
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(msg.chat_id)
                        .msg_type("interactive")
                        .content(content)
                        .build()
                    ).build()
                
                response = self._client.im.v1.message.create(request)
                
                if not response.success():
                    logger.error(
                        f"Failed to send Feishu message: code={response.code}, "
                        f"msg={response.msg}, log_id={response.get_log_id()}"
                    )
                else:
                    logger.debug(f"Feishu message sent to {msg.chat_id}")
            
            # 2) Normalize media: URL / local path -> upload to Feishu
            #    Heuristic:
            #    - Known image extensions -> image upload
            #    - Others -> file upload
            image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
            image_keys: list[str] = []
            file_keys: list[str] = []
            for item in msg.media or []:
                if not item:
                    continue
                if isinstance(item, str):
                    # Simple heuristic: http/https URL -> treat as remote image
                    if item.startswith("http://") or item.startswith("https://"):
                        parsed = urlparse(item)
                        ext = os.path.splitext(parsed.path)[1].lower()
                        if ext in image_exts:
                            key = self._upload_image_from_url(item)
                            if key:
                                image_keys.append(key)
                        else:
                            key = self._upload_file_from_url(item)
                            if key:
                                file_keys.append(key)
                        continue
                    # Local file path
                    local_candidate = os.path.expanduser(item)
                    if os.path.isfile(local_candidate):
                        ext = os.path.splitext(local_candidate)[1].lower()
                        if ext in image_exts:
                            key = self._upload_image_file(local_candidate)
                            if key:
                                image_keys.append(key)
                            else:
                                logger.error(f"Failed to upload local image file: {item}")
                        else:
                            key = self._upload_file_from_path(local_candidate)
                            if key:
                                file_keys.append(key)
                            else:
                                logger.error(f"Failed to upload local file: {item}")
                        continue
                    # Raw Feishu keys:
                    # - file_xxx... -> file_key
                    # - others -> image_key (backwards compatible)
                    if item.startswith("file_"):
                        logger.debug(f"Treating media item as Feishu file_key: {item}")
                        file_keys.append(item)
                    else:
                        logger.debug(f"Treating media item as Feishu image_key: {item}")
                        image_keys.append(item)
                else:
                    # Non-string, fallback to image_key string
                    logger.debug(f"Treating non-string media item as Feishu image_key: {item}")
                    image_keys.append(str(item))
            
            # 3) Send images if present
            for image_key in image_keys:
                if not image_key:
                    continue
                try:
                    image_content = json.dumps({"image_key": image_key}, ensure_ascii=False)
                    img_request = CreateMessageRequest.builder() \
                        .receive_id_type(receive_id_type) \
                        .request_body(
                            CreateMessageRequestBody.builder()
                            .receive_id(msg.chat_id)
                            .msg_type("image")
                            .content(image_content)
                            .build()
                        ).build()
                    
                    img_response = self._client.im.v1.message.create(img_request)
                    if not img_response.success():
                        logger.error(
                            f"Failed to send Feishu image: code={img_response.code}, "
                            f"msg={img_response.msg}, log_id={img_response.get_log_id()}"
                        )
                    else:
                        logger.debug(f"Feishu image sent to {msg.chat_id}")
                except Exception as e:
                    logger.error(f"Error sending Feishu image message: {e}")

            # 4) Send files if present
            for file_key in file_keys:
                if not file_key:
                    continue
                try:
                    file_content = json.dumps({"file_key": file_key}, ensure_ascii=False)
                    file_request = CreateMessageRequest.builder() \
                        .receive_id_type(receive_id_type) \
                        .request_body(
                            CreateMessageRequestBody.builder()
                            .receive_id(msg.chat_id)
                            .msg_type("file")
                            .content(file_content)
                            .build()
                        ).build()
                    
                    file_response = self._client.im.v1.message.create(file_request)
                    if not file_response.success():
                        logger.error(
                            f"Failed to send Feishu file: code={file_response.code}, "
                            f"msg={file_response.msg}, log_id={file_response.get_log_id()}"
                        )
                    else:
                        logger.debug(f"Feishu file sent to {msg.chat_id}")
                except Exception as e:
                    logger.error(f"Error sending Feishu file message: {e}")
                    
        except Exception as e:
            logger.error(f"Error sending Feishu message: {e}")
    
    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """
        Sync handler for incoming messages (called from WebSocket thread).
        Schedules async handling in the main event loop.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)
    
    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        """Handle incoming message from Feishu."""
        try:
            event = data.event
            message = event.message
            sender = event.sender
            
            # Deduplication check
            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            
            # Trim cache: keep most recent 500 when exceeds 1000
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)
            
            # Skip bot messages
            sender_type = sender.sender_type
            if sender_type == "bot":
                return
            
            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" or "group"
            msg_type = message.message_type
            
            # Add reaction to indicate "seen"
            await self._add_reaction(message_id, "THINKING")
            
            # Parse message content and media
            media: list[str] = []
            if msg_type == "text":
                try:
                    content = json.loads(message.content).get("text", "")
                except json.JSONDecodeError:
                    content = message.content or ""
            elif msg_type == "image":
                # Feishu image message content is JSON like: {"image_key": "..."}
                try:
                    data = json.loads(message.content) if message.content else {}
                except json.JSONDecodeError:
                    data = {}
                image_key = data.get("image_key")
                if image_key:
                    media.append(image_key)
                content = MSG_TYPE_MAP.get("image", "[image]")
            elif msg_type == "file":
                # Feishu file message content is JSON like: {"file_key": "...", "file_name": "...", ...}
                try:
                    data = json.loads(message.content) if message.content else {}
                except json.JSONDecodeError:
                    data = {}
                file_key = data.get("file_key")
                if file_key:
                    media.append(file_key)
                content = MSG_TYPE_MAP.get("file", "[file]")
            else:
                content = MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")
            
            if not content and not media:
                return
            
            # Forward to message bus
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content or "",
                media=media,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    # Channel-specific payload for consumers that need raw image keys, etc.
                    "raw_content": message.content,
                    "image_keys": media if msg_type == "image" and media else None,
                    "file_keys": media if msg_type == "file" and media else None,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Feishu message: {e}")
