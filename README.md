# Pi AI Gateway

Self-hosted AI gateway with smart routing between cloud (OpenRouter) and local (llama.cpp) models.

## Features

- **Smart Routing**: Automatically routes requests to cloud or local models based on intent classification and system resources
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Tool Calling Support**: Full support for OpenAI function/tool calling format
- **Dual Provider Support**:
  - Cloud: OpenRouter API (multiple model providers)
  - Local: llama.cpp via llama-server (on-device inference)
- **System Monitoring**: RAM and model availability tracking
- **Fallback Logic**: Automatic failover between cloud and local providers

## API Endpoints

### Chat Completions

```bash
POST /v1/chat/completions
POST /chat
```

**OpenAI-compatible request format:**

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Boston?"}
  ],
  "model": "anthropic/claude-3.5-sonnet",
  "temperature": 0.7,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "route": "auto"
}
```

**Gateway-specific parameters:**
- `route` (optional): `"auto"`, `"cloud"`, or `"local"` to override routing decision

**Response format includes tool calls:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "anthropic/claude-3.5-sonnet",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Boston\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 20,
    "total_tokens": 70
  },
  "x_route": "cloud",
  "x_provider": "openrouter",
  "x_intent": "quick_qa"
}
```

### Models

```bash
GET /v1/models
```

Returns available cloud and local models.

### Health Check

```bash
GET /health
GET /v1/health
GET /status
```

## Installation

See `deploy/install.sh` for deployment on Raspberry Pi.

## Development

```bash
# Create virtual environment (requires Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run locally
uvicorn ai_gateway.main:app --reload --port 8080
```

## Configuration

Create `.env` file (or use `.env.tpl` with 1Password CLI):

```bash
OPENROUTER_API_KEY=sk-or-v1-...
GATEWAY_API_KEY=optional-bearer-token
MODEL_BRIDGE_URL=http://localhost:9099
DEFAULT_CLOUD_MODEL=anthropic/claude-3.5-sonnet
```

## Routing Logic

The gateway classifies user intent and routes accordingly:

| Intent | Preferred Route | Keywords |
|--------|----------------|----------|
| Coding | Cloud | code, function, debug, python, etc. |
| Analysis | Cloud | analyze, explain, compare, review |
| Creative | Cloud | write, essay, story, article |
| Quick Q&A | Local | what is, who is, summary, tldr |
| Translation | Local | translate, in spanish, to english |

Routing can be overridden with the `route` parameter in the request.

## Tool Calling

The gateway fully supports OpenAI's tool/function calling format:

1. **Request with tools**: Include `tools` array with function definitions
2. **Tool choice**: Optional `tool_choice` parameter (`"auto"`, `"none"`, or specific function)
3. **Multi-turn conversations**: Support for assistant messages with `tool_calls` and tool response messages with `tool_call_id`

Both cloud (OpenRouter) and local (llama-server) providers pass through tool definitions and handle tool calls in responses.

## License

MIT
