# co_steam_v1

Purpose
- Multi-modal pipeline workspace (ASR/SLM/T2I/State/Bridge).
- Current focus: realtime ASR on Windows host (CPU).

Locations
- Project root is now assumed relative to this repository (no hardcoded drive letters).

## Architecture (Microservices + WebSocket)

```
┌─────────────────────────────────────────────────────────────┐
│                     Bridge :5555                            │
│         (中心 hub，聚合所有消息，对外广播)                    │
│         TouchDesigner / Web 连接此端口                       │
└─────────────────────────────────────────────────────────────┘
        ▲              ▲              ▲              ▲
        │ ws           │ ws           │ ws           │ ws
   ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
   │   ASR   │───▶│   SLM   │───▶│  State  │───▶│   T2I   │
   │ :5551   │    │ :5552   │    │ :5553   │    │ :5554   │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

Ports:
- ASR: ws://localhost:5551
- SLM: ws://localhost:5552
- State: ws://localhost:5553 (todo)
- T2I: ws://localhost:5554 (todo)
- Bridge: ws://localhost:5555 (外部客户端连接)

Data flow:
- ASR -> SLM: transcribed text + context window
- SLM routes to:
  - State Machine -> T2I (image generation intent)
  - Agent keywords (conversation, extract topics/questions)

## Directory layout

- asr/    faster-whisper realtime ASR
- slm/    Ollama / Mistral
- t2i/    ComfyUI / Stream Diffusion
- state/  intent state machine
- bridge/ WebSocket / TD communication
- config/ portable configs
- logs/   experiment logs
- data/   audio / intermediate results
- models/ offline models

## 大文件（未纳入 Git，需手动下载）

以下资源体积较大，已写入 `.gitignore`，请从 Google Drive（或你提供的共享盘）手动下载并放到对应路径；如需我填入具体链接请告诉我。
- `ComfyUI_cu126/` (~15GB)：便携版 ComfyUI 及内置 Python/依赖，保持整个目录放在仓库根目录。
- `models/` (~5GB)：faster-whisper 模型权重，按 README 中列的子目录放置。
- `ollama-models.tar.gz` (~3.3GB)：导出的 Ollama 模型包，解压到 `%USERPROFILE%/.ollama/models`（或对应用户目录）。
- `OllamaSetup.exe` (~1.2GB)：Ollama 安装包，可从官网或共享盘下载后放在仓库根目录。
- 运行产物：`data/`（生成图片）、`snapshots/`、`logs/`、`state/` 等仅运行时需要，不纳入版本控制。

## ASR Models (AB testing)

faster-whisper variants:
| Model | Params | Status |
|-------|--------|--------|
| faster-whisper-tiny | 75M | ready |
| faster-whisper-base | 142M | ready |
| faster-whisper-small | 466M | ready (default) |
| faster-whisper-medium | 1.5B | ready |
| faster-whisper-large-v3 | 3B | ready |

Paths (relative to repo):
- models/faster-whisper-tiny
- models/faster-whisper-base
- models/faster-whisper-small
- models/faster-whisper-medium
- models/faster-whisper-large-v3

## SLM Models (AB testing)

Ollama Mistral variants:
| Model | Size | Params | Status |
|-------|------|--------|--------|
| ministral-3:3b-instruct-2512-q4_K_M | 3.0GB | 3B | deployed |
| ministral-3:8b-instruct-2512-q4_K_M | 6.0GB | 8B | todo |
| mistral-small3.1:24b | 15GB | 24B | todo |

Model comparison:
- Ministral 3B/8B: edge-optimized, multimodal (text+image), Apache 2.0
- Mistral Small 3.1: outperforms GPT-4o Mini, 150 tok/s, needs RTX 4090 or 32GB Mac

Install (local then scp to remote):
```
ollama pull ministral-3:3b-instruct-2512-q4_K_M
tar -czvf /tmp/ollama-models.tar.gz -C ~/.ollama models
scp /tmp/ollama-models.tar.gz 192.168.31.153:"D:\\co_steam_v1\\"
# remote: tar -xzf D:/co_steam_v1/ollama-models.tar.gz -C $env:USERPROFILE/.ollama/
```

SLM commands (Windows):
```
ollama list
ollama run ministral-3:3b-instruct-2512-q4_K_M "your prompt"
```

References:
- https://mistral.ai/news/mistral-3
- https://mistral.ai/news/mistral-small-3-1

## SLM Module

Scripts:
- slm/inference.py - intent classification and keyword extraction

Functions:
- `classify_intent(text)` - classify as IMAGE or CONVERSATION
- `extract_keywords(text)` - extract topics, questions, sentiment
- `route(text)` - main routing function

Routing:
- IMAGE intent → state_machine action (for T2I)
- CONVERSATION intent → agent_keywords action (topics + questions)

Test:
```
D:\Miniconda3\condabin\conda.bat run -n asr python D:\co_steam_v1\slm\inference.py "your text"
```

## Microservices (WebSocket)

Each service runs independently and communicates via WebSocket.

### Quick Start (Windows)
```batch
D:\co_steam_v1\run_services.bat
```
This starts ASR, SLM, and Bridge services in separate windows.

### Individual Services

ASR Service (captures audio, transcribes):
```
D:\Miniconda3\condabin\conda.bat run -n asr python D:\co_steam_v1\asr\service.py --device "Yeti X" --language en --context-sec 60
```

SLM Service (intent classification, keyword extraction):
```
D:\Miniconda3\condabin\conda.bat run -n asr python D:\co_steam_v1\slm\service.py
```

Bridge Service (hub for external clients):
```
D:\Miniconda3\condabin\conda.bat run -n asr python D:\co_steam_v1\bridge\service.py
```

### Message Protocol (JSON)

ASR output:
```json
{"type": "asr_text", "source": "asr", "data": {"text": "...", "chunk_id": 1, "context": [...]}}
```

SLM output (image intent):
```json
{"type": "intent", "source": "slm", "data": {"intent": "image", "confidence": "high", "prompt": "..."}}
```

SLM output (conversation):
```json
{"type": "keywords", "source": "slm", "data": {"topics": [...], "questions": [...], "sentiment": "..."}}
```

### TouchDesigner Integration

Connect WebSocket DAT to: `ws://192.168.31.153:5555`

## Legacy Pipeline (deprecated)

Old integrated pipeline (single process):
- asr/pipeline.py - realtime ASR with SLM routing

## ASR environment (Windows)

- Conda env: asr (Python 3.10)
- Packages: faster-whisper, sounddevice, numpy
- Scripts:
  - asr/realtime_asr.py
  - asr/list_input_devices.py
- Default model path: models/faster-whisper-small (relative to repo)
- Default mic: Yeti X

## ASR commands (Windows)

List input devices:
```
conda run -n asr python asr\list_input_devices.py
```

Realtime ASR:
```
conda run -n asr python asr\realtime_asr.py --device "Yeti X" --language en
```

## T2I References

- Reference images stored at `t2i/references/`:
  - `rural-makerspace-1.png`
  - `rural-makerspace-2.png`
