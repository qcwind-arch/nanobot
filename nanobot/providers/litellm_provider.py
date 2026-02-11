"""LiteLLM provider implementation for multi-provider support."""

import json
import os
from typing import Any, List, Dict

import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway

import tiktoken


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    DEFAULT_MODEL = "anthropic/claude-opus-4-5"

    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = (default_model or "").strip() or self.DEFAULT_MODEL
        self.extra_headers = extra_headers or {}
        
<<<<<<< HEAD
        # Detect gateway / local deployment from api_key and api_base
        self._gateway = find_gateway(api_key, api_base)
        
        # Backwards-compatible flags (used by tests and possibly external code)
        self.is_openrouter = bool(self._gateway and self._gateway.name == "openrouter")
        self.is_aihubmix = bool(self._gateway and self._gateway.name == "aihubmix")
        self.is_vllm = bool(self._gateway and self._gateway.is_local)

        try:
            # 初始化token编码器（适配Qwen3-8b/OpenAI格式）
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # 兜底：如果cl100k_base不存在，用gpt2编码器
            self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            # print(f"编码器初始化警告：{e}，已切换为gpt-3.5-turbo编码器")
=======
        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)
>>>>>>> upstream/main
        
        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"
        
        return model
    
    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    def _count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """修正后的token计算（更精准，适配Qwen3-8b）"""
        total_tokens = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Qwen3-8b兼容的OpenAI token计算规则
            total_tokens += 4  # 每条消息的固定分隔符
            total_tokens += len(self.encoder.encode(role))
            total_tokens += len(self.encoder.encode(content))
        total_tokens += 2  # 最终回复的分隔符
        return total_tokens

    def _truncate_long_messages(self, messages: List[Dict[str, str]], max_total_tokens: int = 4096) -> List[Dict[str, str]]:
        """
        严格截断：确保最终token数 ≤ max_total_tokens - 预留生成空间（512）
        max_total_tokens: 模型总上下文窗口（默认4096）
        """
        # 关键：预留512 token给模型生成回复，输入token上限=4096-512=3584
        INPUT_TOKEN_LIMIT = max_total_tokens - 512
        final_messages = []
        total_tokens = 0

        # 步骤1：优先保留系统提示（如果有，且不超限）
        system_msg = None
        user_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                user_msgs.append(msg)
        
        # 计算系统提示的token数，超限则截断系统提示本身
        system_tokens = 0
        if system_msg:
            system_tokens = self._count_messages_tokens([system_msg])
            if system_tokens > INPUT_TOKEN_LIMIT:
                # 系统提示超长，直接截断内容到INPUT_TOKEN_LIMIT-100
                content = system_msg["content"]
                encoded = self.encoder.encode(content)
                truncated_encoded = encoded[:INPUT_TOKEN_LIMIT - 100]  # 预留100token冗余
                system_msg["content"] = self.encoder.decode(truncated_encoded)
                system_tokens = self._count_messages_tokens([system_msg])
            final_messages.append(system_msg)
            total_tokens += system_tokens

        # 步骤2：从后往前添加用户消息，直到接近INPUT_TOKEN_LIMIT
        # 倒序遍历用户消息（保留最新的）
        for msg in reversed(user_msgs):
            msg_tokens = self._count_messages_tokens([msg])
            # 新增：校验添加后是否超限，超限则跳过
            if total_tokens + msg_tokens > INPUT_TOKEN_LIMIT:
                # 最后一条消息也超限，截断其内容
                if total_tokens < INPUT_TOKEN_LIMIT:
                    remaining = INPUT_TOKEN_LIMIT - total_tokens - 20  # 预留20token分隔符
                    content_encoded = self.encoder.encode(msg["content"])
                    truncated_content = self.encoder.decode(content_encoded[:remaining])
                    truncated_msg = {"role": msg["role"], "content": truncated_content}
                    final_messages.append(truncated_msg)
                    total_tokens += self._count_messages_tokens([truncated_msg])
                break
            final_messages.append(msg)
            total_tokens += msg_tokens

        # 步骤3：最终强制校验，确保token≤INPUT_TOKEN_LIMIT（兜底）
        final_tokens = self._count_messages_tokens(final_messages)
        if final_tokens > INPUT_TOKEN_LIMIT:
            # 极端情况：只保留最后一条消息的核心内容
            last_msg = final_messages[-1]
            last_encoded = self.encoder.encode(last_msg["content"])
            last_truncated = self.encoder.decode(last_encoded[:INPUT_TOKEN_LIMIT - 50])
            final_messages = [{"role": last_msg["role"], "content": last_truncated}]
            final_tokens = self._count_messages_tokens(final_messages)

        # 打印精准日志
        orig_tokens = self._count_messages_tokens(messages)
        # print(f"【Messages截断】原长度{len(messages)}条 → 新长度{len(final_messages)}条")
        # print(f"【Token变化】原token{orig_tokens} → 新token{final_tokens}（上限{INPUT_TOKEN_LIMIT}）")
        return final_messages
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = (model or self.default_model or "").strip() or self.DEFAULT_MODEL
        model = self._resolve_model(model)
        if not model:
            model = self._resolve_model(self.DEFAULT_MODEL)
        
         # 计算当前messages的token数
        messages_tokens = self._count_messages_tokens(messages)
        # 如果token数超过n_ctx，截断
        # if messages_tokens > max_tokens:
            # messages = self._truncate_long_messages(messages, max_tokens)

        # print(f"n_ctx: {max_tokens}, messages_tokens: {messages_tokens}, condation: { messages_tokens > max_tokens }")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None)
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
