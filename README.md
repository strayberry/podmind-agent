# PodMind

Download, transcribe, and summarize Xiaoyuzhou FM (小宇宙) podcasts.

## Quick Start

```bash
pip install -e .
cp .env.example .env          # add your DEEPSEEK_API_KEY

podmind full "https://www.xiaoyuzhoufm.com/episode/69f441cd5390b7cc928acdcc" --language zh
```

First run downloads the ASR model (~1.6 GB) once. Subsequent runs hit the cache.

## ASR Backends

| Backend | Default | Speed (1h audio) | RTF | Output (1h) | Quality |
|---|---|---|---|---|---|
| `mlx-whisper` | ✅ | ~2.5 min | 0.043 | ~19K chars | Fast, but may drop punctuation and hallucinate on Chinese |
| `mlx-qwen` | | ~3 min | 0.051 | ~12K chars | Good accuracy and punctuation, output slightly condensed |
| `qwen` | | ~6 min | 0.105 | ~21K chars | Highest accuracy, most detailed output |

All require Apple Silicon. `qwen` needs `torch` and `ffmpeg`. `mlx-qwen` needs `mlx-audio`.

## Setup

- Python >= 3.12, Apple Silicon Mac, `ffprobe` (`brew install ffmpeg`)
- [DeepSeek API key](https://platform.deepseek.com) for summarization

```bash
pip install -e .              # mlx-whisper + qwen
pip install -e ".[mlx-qwen]"  # + mlx-qwen backend
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

Use `--asr-backend qwen` for higher quality. Replace `<id>` with the 24-char hex episode ID from the URL.

### Batch

```bash
podmind batch-transcribe <id1> <id2> --language zh --profile
```

### Benchmark

```bash
podmind bench <id> --language zh --profile
podmind bench <id> --backends qwen --clip-seconds 120
```

### Preflight check

```bash
podmind doctor
podmind doctor --asr-backend qwen
```

## CLI Reference

| Flag | Commands | Description |
|---|---|---|
| `--language` | full, transcribe, batch, bench | `zh`, `en`, `ja`, `ko`, or full name (`Chinese`) |
| `--asr-backend` | all except fetch | `mlx-whisper` (default) or `qwen` |
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

Each backend writes to separate paths, so results from both backends coexist:

```
data/transcripts/<id>.txt                  # qwen
data/transcripts/<id>.mlx-whisper.txt      # mlx-whisper
data/outputs/<id>_summary.md                # qwen
data/outputs/<id>_summary.mlx-whisper.md    # mlx-whisper
```

## Cache

Transcripts and summaries are cached with content-addressed metadata. Cache is invalidated when audio, language, model, backend, or chunk parameters change. Use `--force` to bypass.

## Troubleshooting

**ffprobe missing:** `brew install ffmpeg`

**Qwen "MPS is required":** Install `torch` and ensure `PODMIND_ASR_DEVICE=mps`, or use `--asr-backend mlx-whisper`.

**Unexpected errors:** Re-run with `--debug` for a full traceback. Use `--force` to re-download corrupted audio or bypass stale cache.
