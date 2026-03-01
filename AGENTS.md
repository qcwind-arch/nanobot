# Agent Development Guide

This file contains build/lint/test commands and code style guidelines for working with nanobot.

## Commands

### Testing
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_tool_validation.py

# Run a specific test function
pytest tests/test_tool_validation.py::test_validate_params_missing_required

# Run with verbose output
pytest -v

# Run async tests only
pytest -m asyncio
```

### Linting & Type Checking
```bash
# Run ruff linter
ruff check nanobot/ tests/

# Auto-fix ruff issues
ruff check --fix nanobot/ tests/

# Check code formatting
ruff format --check nanobot/ tests/

# Auto-format code
ruff format nanobot/ tests/
```

### Build
```bash
# Build package
pip install build
python -m build

# Install from source
pip install -e .
```

## Code Style Guidelines

### Imports
- Order: stdlib → third-party → local (alphabetically sorted within each group)
- Use `from __future__ import annotations` for files with forward type hints
- Type-level imports use TYPE_CHECKING guard for runtime efficiency

Example:
```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from nanobot.config.schema import SomeConfig

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager
```

### Type Hints
- Use type hints for all function signatures and class attributes
- Prefer `str | None` over `Optional[str]` (Python 3.11+ style)
- Use `dict[str, Any]`, `list[str]` instead of `Dict`, `List`
- Dataclasses use `@dataclass(frozen=True)` for immutability

### Naming Conventions
- Classes: PascalCase (`AgentLoop`, `ToolRegistry`)
- Functions/variables: snake_case (`process_message`, `allowed_dir`)
- Constants: UPPER_SNAKE_CASE (`BOOTSTRAP_FILES`, `MAX_ITERATIONS`)
- Private attributes: leading underscore (`_running`, `_register_default_tools`)
- Async functions: explicitly named with async behavior (`run`, `process_message`)

### Formatting
- Max line length: 100 characters (configured in pyproject.toml)
- Use 4 spaces for indentation
- No trailing whitespace
- One blank line between top-level definitions (two between classes)
- Use f-strings for string formatting: `f"Processing: {value}"`

### Error Handling
- Log errors with loguru: `logger.error(f"Error: {e}")`
- Use try/except with specific exceptions when possible
- Return error messages as strings for tools (agents read them)
- Raise ValueError for invalid parameters, RuntimeError for operational issues

### Logging
- Use loguru logger: `from loguru import logger`
- Levels: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`
- Include context in messages: `logger.info(f"Processing {msg.session_key}: {preview}")`

### Classes & Data Structures
- Use Pydantic `BaseModel` for configuration and data validation
- Use `Field(default_factory=list)` for mutable defaults
- Use `@dataclass(frozen=True)` for simple immutable structures
- Include docstrings for classes and public methods

Example:
```python
from pydantic import BaseModel, Field

class SomeConfig(BaseModel):
    """Configuration for something."""
    enabled: bool = False
    items: list[str] = Field(default_factory=list)
    timeout: int = 30
```

### Async Patterns
- Always use `async def` for I/O operations
- Use `await` for async calls (no asyncio.run() inside async functions)
- Use `@pytest.mark.asyncio` for async tests
- Handle cancellation gracefully

### Docstrings
- Use triple-quoted strings at module/class/function level
- Keep descriptions concise but informative
- Args/Returns sections for complex functions

### Tool Development
- Inherit from `Tool` base class in `nanobot/agent/tools/base.py`
- Override `name`, `description`, `parameters`, and `execute`
- Execute returns string result (agent reads this)
- Use `validate_params()` for parameter validation

Example:
```python
class CustomTool(Tool):
    @property
    def name(self) -> str:
        return "custom_tool"

    @property
    def description(self) -> str:
        return "Does something custom"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        return "Result"
```

### Channel Development
- Inherit from `BaseChannel` in `nanobot/channels/base.py`
- Use `MessageBus.publish_inbound()` for incoming messages
- Handle `OutboundMessage` from the bus to send replies
- Use loguru for connection status logging
