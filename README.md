# PodMind

Download, transcribe, and summarize Xiaoyuzhou FM (小宇宙) podcasts.

## Quick Start

```bash
# 1. Install
pip install -e .
cp .env.example .env          # add your DEEPSEEK_API_KEY

# 2. Run
podmind full "https://www.xiaoyuzhoufm.com/episode/69f441cd5390b7cc928acdcc" --language zh
```

First run downloads the ASR model (~800 MB) and transcribes the audio — a 2-hour episode takes about 20 minutes on Apple Silicon. Subsequent runs hit the cache and skip straight to the summary.

## Workflow

```
Podcast URL → Scrape metadata → Download .m4a → ASR Transcription → DeepSeek Summary
```

Two **ASR backends** are included:

| Backend | Default | Speed (137 min ep) | Best for |
|---|---|---|---|
| `mlx-whisper` | ✅ | ~22 min (RTF 0.16) | Everyday use, fast turnaround |
| `qwen` | | Slower (RTF ~0.25) | Maximum transcription quality |

Both require Apple Silicon. Only `qwen` needs `torch` and `ffmpeg`.

## Setup

### Prerequisites

- Python >= 3.12
- Apple Silicon Mac
- `ffprobe` (included with `ffmpeg`)
- [DeepSeek API key](https://platform.deepseek.com) (for summarization)

```bash
# Install
pip install -e .

# Development tools (linter, type checker, tests)
pip install -e ".[dev]"

# Configure
cp .env.example .env
```

### Configuration

All settings via environment variables (edit `.env`):

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | **Required** for summarization |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API endpoint |
| `PODMIND_DATA_DIR` | `./data` | Where audio, transcripts, summaries live |
| `PODMIND_ASR_DEVICE` | `mps` | Device for Qwen backend |

## Usage

### Full pipeline (fetch → transcribe → summarize)

```bash
podmind full "https://www.xiaoyuzhoufm.com/episode/<id>" --language zh
```

### Step by step

```bash
# 1. Download audio
podmind fetch "https://www.xiaoyuzhoufm.com/episode/69f441cd5390b7cc928acdcc"

# 2. Transcribe with default backend (mlx-whisper)
podmind transcribe 69f441cd5390b7cc928acdcc --language zh --profile

# Or use Qwen for higher quality
podmind transcribe 69f441cd5390b7cc928acdcc --asr-backend qwen --language zh --profile

# 3. Generate summary
podmind summarize 69f441cd5390b7cc928acdcc --asr-backend mlx-whisper
```

> The episode ID is the 24-character hex string in the URL:
> `https://www.xiaoyuzhoufm.com/episode/`**`69f441cd5390b7cc928acdcc`**

### Batch transcribe

Process multiple episodes in one go:

```bash
podmind batch-transcribe <id1> <id2> <id3> --language zh --profile
```

Qwen loads the model once explicitly; mlx-whisper reuses an internal process-level cache.

### Benchmark

Compare speed and output across backends on a 5-minute clip:

```bash
podmind bench 69f441cd5390b7cc928acdcc --language zh --profile
podmind bench 69f441cd5390b7cc928acdcc --backends qwen --clip-seconds 120
```

Extracts a clip from the episode audio and runs each selected backend on it, printing RTF, character count, and a preview.

### Preflight check

```bash
podmind doctor                    # Verify mlx-whisper setup
podmind doctor --asr-backend qwen # Verify Qwen setup
```

Reports which dependencies are present, separating required from optional. Exits non-zero if anything required is missing.

## CLI reference

| Flag | Commands | Description |
|---|---|---|
| `--language` | full, transcribe, batch, bench | `zh`, `en`, `ja`, `ko`, or full name (`Chinese`) |
| `--asr-backend` | full, transcribe, summarize, batch, bench, doctor | `mlx-whisper` (default) or `qwen` |
| `--asr-model` | full, transcribe, batch | Override default model ID |
| `--model` | full, summarize | LLM model (default: `deepseek-v4-pro`) |
| `--profile` | full, transcribe, batch, bench | Print timing breakdown (RTF, chunk times) |
| `--force` | all | Ignore cache and re-run |
| `--quiet` | transcribe, batch | Suppress transcript text output |
| `--debug` | all | Print full traceback on errors |
| `--chunk-seconds` | full, transcribe, batch, bench | Qwen only: seconds per chunk (30–1800, default 600) |
| `--batch-size` | full, transcribe, batch, bench | Qwen only: chunks per batch (1, 2, or 4) |

## Data directory

```
data/
├── audio/         # Downloaded .m4a files
├── episodes/      # Scraped metadata (JSON)
├── transcripts/   # Transcription output + cache metadata
└── outputs/       # LLM summaries
```

Each backend writes to its own transcript and summary paths, so you can keep results from both side by side:

```
data/transcripts/69f441cd5390b7cc928acdcc.txt                  # qwen
data/transcripts/69f441cd5390b7cc928acdcc.mlx-whisper.txt      # mlx-whisper
data/outputs/69f441cd5390b7cc928acdcc_summary.md                # qwen
data/outputs/69f441cd5390b7cc928acdcc_summary.mlx-whisper.md    # mlx-whisper
```

Override the data directory:

```bash
export PODMIND_DATA_DIR="/path/to/podmind-data"
```

## Cache

Transcripts and summaries are cached with content-addressed metadata. Cache is invalidated automatically when any input changes: audio file, language, model, backend, chunk parameters, or prompt text.

Use `--force` to bypass. Delete files under `data/transcripts/` or `data/outputs/` to clear selectively.

## Development

```bash
pip install -e ".[dev]"

# Lint
python -m ruff check .

# Type check
python -m mypy src

# Tests
python -m pytest tests/ -v
```

## Troubleshooting

**`podmind doctor` reports ffprobe missing**
Install ffmpeg: `brew install ffmpeg`

**`ModuleNotFoundError: No module named 'mlx_whisper'`**
Re-run `pip install -e .` — mlx-whisper was added to base dependencies.

**Qwen backend: "MPS is required"**
Qwen only runs on Apple Silicon with PyTorch MPS backend. Use `--asr-backend mlx-whisper` instead, or ensure `torch` is installed and `PODMIND_ASR_DEVICE=mps`.

**Unexpected error during transcription**
Re-run with `--debug` for a full traceback. Common causes: corrupted audio download (use `--force` to re-download), or insufficient disk space for model weights.

**Summary repeats or is low quality**
The DeepSeek prompt is tuned for Chinese-language tech podcasts. For other content, adjust the prompt in `src/podmind/prompts/summary.txt`.
