# PodMind

Download, transcribe, and summarize Xiaoyuzhou FM (小宇宙) podcasts.

## Quick Start

```bash
pip install -e .
cp .env.example .env          # add your DEEPSEEK_API_KEY

podmind full "https://www.xiaoyuzhoufm.com/episode/69f441cd5390b7cc928acdcc" --language zh
```

First run downloads the ASR model (~0.7–1.8 GB depending on backend). Subsequent runs hit the cache.

## ASR Backends

| Backend | Model | Size | Default | Speed (1h audio) | RTF | Output (1h) | Quality |
|---|---|---|---|---|---|---|---|
| `mlx-qwen-asr` | `mlx-community/Qwen3-ASR-0.6B-4bit` | 680 MB | ✅ | ~3 min | 0.051 | ~12K chars | Good accuracy and punctuation, output slightly condensed |
| `mlx-whisper` | `mlx-community/whisper-turbo` | 1.5 GB | | ~2.5 min | 0.043 | ~19K chars | Fast, but may drop punctuation and hallucinate on Chinese |
| `qwen-asr` | `Qwen/Qwen3-ASR-0.6B` | 1.8 GB | | ~6 min | 0.105 | ~21K chars | Highest accuracy, most detailed output |

All require Apple Silicon. `qwen-asr` needs `torch` and `ffmpeg`. `mlx-qwen-asr` needs `mlx-audio`.

## Setup

- Python >= 3.12, Apple Silicon Mac, `ffprobe` (`brew install ffmpeg`)
- [DeepSeek API key](https://platform.deepseek.com) for summarization

```bash
pip install -e .              # includes the default mlx-qwen-asr backend
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **Required** for summarization |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API endpoint |
| `PODMIND_DATA_DIR` | `./data` | Data directory |

## Usage

### Full pipeline

```bash
podmind full "https://www.xiaoyuzhoufm.com/episode/<id>" --language zh
```

### Step by step

```bash
podmind fetch "https://www.xiaoyuzhoufm.com/episode/<id>"
podmind transcribe <id> --language zh --profile
podmind summarize <id>
```

Use `--asr-backend qwen-asr` for highest quality. Replace `<id>` with the 24-char hex episode ID from the URL.

### Batch

```bash
podmind batch-transcribe <id1> <id2> --language zh --profile
```

### Benchmark

```bash
podmind bench <id> --language zh --profile
podmind bench <id> --backends qwen-asr --clip-seconds 120
```

### Preflight check

```bash
podmind doctor
podmind doctor --asr-backend qwen-asr
```

## CLI Reference

| Flag | Commands | Description |
|---|---|---|
| `--language` | full, transcribe, batch, bench | `zh`, `en`, `ja`, `ko`, or full name (`Chinese`) |
| `--asr-backend` | all except fetch | `mlx-qwen-asr` (default), `mlx-whisper`, or `qwen-asr` |
| `--asr-model` | full, transcribe, batch | Override default model ID |
| `--model` | full, summarize | LLM model (default: `deepseek-v4-pro`) |
| `--profile` | full, transcribe, batch, bench | Print timing breakdown (RTF, chunk times) |
| `--force` | all | Ignore cache and re-run |
| `--quiet` | transcribe, batch | Suppress transcript output |
| `--debug` | all | Print full traceback on errors |

### Qwen-specific flags

| Flag | Default | Description |
|---|---|---|
| `--chunk-seconds` | 30 | Seconds per chunk (30–1800) |
| `--batch-size` | 1 | Chunks per batch (1–8) |
| `--qwen-max-new-tokens` | 0 | Max tokens per chunk (0=auto-scale) |
| `--qwen-dtype` | float16 | `float16` (faster) or `bfloat16` |

Tuning notes: reducing `--chunk-seconds` from 600 to 30 improved RTF from ~0.66 to ~0.11. Increasing `--batch-size` beyond 1 showed negligible gains on MPS — inference is the bottleneck, not per-call overhead.

## Data Directory

```
data/
├── audio/         # Downloaded .m4a files
├── episodes/      # Scraped metadata (JSON)
├── transcripts/   # Transcription output + cache metadata
└── outputs/       # LLM summaries
```

Each backend writes to separate paths, so results from all backends coexist:

```
data/transcripts/<id>.txt                     # qwen-asr
data/transcripts/<id>.mlx-whisper.txt         # mlx-whisper
data/transcripts/<id>.mlx-qwen-asr.txt        # mlx-qwen-asr
data/outputs/<id>_summary.md                   # qwen-asr
data/outputs/<id>_summary.mlx-whisper.md       # mlx-whisper
data/outputs/<id>_summary.mlx-qwen-asr.md      # mlx-qwen-asr
```

## Cache

Transcripts and summaries are cached with content-addressed metadata. Cache is invalidated when audio, language, model, backend, or chunk parameters change. Use `--force` to bypass.

## Troubleshooting

**ffprobe missing:** `brew install ffmpeg`

**Qwen "MPS is required":** Install `torch` and ensure `PODMIND_ASR_DEVICE=mps`, or use `--asr-backend mlx-whisper`.

**Unexpected errors:** Re-run with `--debug` for a full traceback. Use `--force` to re-download corrupted audio or bypass stale cache.
