# qvac-sdk

Python SDK for [Tether QVAC](https://github.com/tetherto/qvac) — local-first AI inference on any device.

QVAC runs AI models entirely on-device with no cloud dependency. This SDK provides a Pythonic interface to QVAC's inference capabilities: LLM completions, image generation, speech-to-text, translation, TTS, OCR, and embeddings.

## Installation

```bash
pip install qvac-sdk
```

With OpenAI-compatible client support:

```bash
pip install qvac-sdk[openai]
```

## Quick start

### Direct API

```python
from qvac_sdk import QVACClient

client = QVACClient()  # connects to local QVAC instance

# LLM completion
response = client.complete("Explain quantum computing in one paragraph")
print(response.text)

# Streaming
for chunk in client.complete_stream("Write a haiku about edge AI"):
    print(chunk.text, end="")
```

### Async

```python
from qvac_sdk import AsyncQVACClient

async def main():
    client = AsyncQVACClient()
    response = await client.complete("Hello from async Python")
    print(response.text)
```

### OpenAI-compatible mode

QVAC exposes an OpenAI-compatible API via `qvac serve openai`. This SDK wraps it natively:

```python
from qvac_sdk.openai_compat import QVACOpenAI

client = QVACOpenAI()  # points to local QVAC OpenAI endpoint

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

### Speech-to-text

```python
transcript = client.transcribe("meeting.wav")
print(transcript.text)
```

### Image generation

```python
image = client.generate_image("A cat riding a bicycle", width=512, height=512)
image.save("cat.png")
```

### Embeddings

```python
vectors = client.embed(["sentence one", "sentence two"])
print(vectors.shape)  # (2, 384)
```

## Configuration

```python
client = QVACClient(
    base_url="http://localhost:8080",  # QVAC instance URL
    timeout=30.0,                       # request timeout in seconds
    model="default",                    # model name override
)
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QVAC_BASE_URL` | `http://localhost:8080` | QVAC API endpoint |
| `QVAC_TIMEOUT` | `30.0` | Request timeout (seconds) |
| `QVAC_MODEL` | `default` | Default model name |

## Supported capabilities

| Capability | Method | Status |
|------------|--------|--------|
| LLM completion | `complete()` / `complete_stream()` | Planned |
| Chat (multi-turn) | `chat()` / `chat_stream()` | Planned |
| Speech-to-text | `transcribe()` | Planned |
| Translation | `translate()` | Planned |
| Text-to-speech | `speak()` | Planned |
| Image generation | `generate_image()` | Planned |
| OCR | `ocr()` | Planned |
| Embeddings | `embed()` | Planned |
| LoRA fine-tuning | `finetune()` | Planned |
| OpenAI compat | `qvac_sdk.openai_compat` | Planned |

## Development

```bash
git clone https://github.com/kinance/qvac-sdk
cd qvac-sdk
pip install -e ".[dev]"
ruff check .
mypy src/
pytest
```

## License

Apache 2.0 — same as QVAC.
