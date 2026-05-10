"""PodMind — Download, transcribe, and summarize Xiaoyuzhou FM podcasts."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    ASR_DEVICE,
    DATA_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    PodmindError,
    ensure_dirs,
    validate_episode_id,
    validate_language,
)
from .downloader import audio_path, download_audio
from .scraper import fetch_episode, load_episode_info
from .summarizer import summarize, summary_path
from .transcriber import (
    _DEFAULT_BACKEND,
    _DEFAULT_MODELS,
    ASR_BACKENDS,
    ASRSession,
    TranscribeProfile,
    _get_duration,
    get_backend_spec,
    transcribe,
    transcript_path,
)


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


def positive_int(value: str) -> int:
    """Validate that value is a positive integer in range [30, 1800]."""
    try:
        ival = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from None
    if ival < 30 or ival > 1800:
        raise argparse.ArgumentTypeError(
            f"chunk-seconds must be between 30 and 1800, got {ival}"
        )
    return ival


def _batch_size_int(value: str) -> int:
    ival = int(value)
    if ival < 1 or ival > 8:
        raise argparse.ArgumentTypeError(
            f"batch-size must be between 1 and 8, got {ival}"
        )
    return ival


def _check_backend_issues(backend: str) -> list[str]:
    """Return backend-specific preflight issues (no ffmpeg/ffprobe)."""
    issues: list[str] = []
    spec = get_backend_spec(backend)

    from importlib.util import find_spec
    if find_spec(spec.dependency_module) is None:
        issues.append(f"{spec.dependency_module} not installed — {backend} backend unavailable")

    if spec.requires_mps:
        if ASR_DEVICE != "mps":
            issues.append(
                f"Unsupported ASR_DEVICE={ASR_DEVICE!r}. "
                "Qwen backend only supports Apple Silicon (MPS)."
            )
        try:
            import torch
            if not torch.backends.mps.is_available():
                issues.append("MPS is required — Qwen backend only runs on Apple Silicon")
        except ImportError:
            issues.append("torch not installed — Qwen backend unavailable")
    return issues


def _collect_preflight_issues(backend: str, *, need_summary: bool = False) -> list[str]:
    """System-level preflight only — no backend model checks.

    Backend model dependencies (torch, MPS, mlx-whisper) are checked lazily
    on cache miss by _instantiate_backend().  This keeps cache-hit paths fast
    and side-effect-free.
    """
    issues: list[str] = []
    if not _check_binary("ffprobe"):
        issues.append("ffprobe not found — required for audio duration detection")
    if need_summary and not DEEPSEEK_API_KEY:
        issues.append("DEEPSEEK_API_KEY not set — summarization will fail")
    if get_backend_spec(backend).requires_ffmpeg and not _check_binary("ffmpeg"):
        issues.append("ffmpeg not found — required for audio splitting")
    return issues


def _exit_on_preflight_issues(issues: list[str]) -> None:
    if issues:
        for i in issues:
            print(f"ERROR: {i}", file=sys.stderr)
        sys.exit("Preflight failed. Fix the issues above and retry.")


def _add_asr_args(parser: argparse.ArgumentParser, *, qwen: bool = True) -> None:
    """Add --asr-backend and --asr-model to a subparser."""
    parser.add_argument("--asr-backend", choices=ASR_BACKENDS,
                        default=_DEFAULT_BACKEND,
                        help=f"ASR backend (default: {_DEFAULT_BACKEND})")
    parser.add_argument("--asr-model", default=None,
                        help="ASR model ID (default: per-backend default)")
    if qwen:
        _add_qwen_args(parser)


def _add_qwen_args(parser: argparse.ArgumentParser) -> None:
    """Add Qwen-specific CLI flags to a subparser."""
    parser.add_argument("--chunk-seconds", type=positive_int, default=30,
                        help="Seconds per audio chunk for qwen-asr (30-1800, default 30)")
    parser.add_argument("--batch-size", type=_batch_size_int, default=1,
                        help="Chunks per qwen-asr batch (1-8, default 1)")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=0,
                        help="Max tokens for Qwen generation (0=auto-scale)")
    parser.add_argument("--qwen-dtype", choices=["bfloat16", "float16"], default="float16",
                        help="Torch dtype for Qwen backend (default: float16)")



def cmd_full(args: argparse.Namespace) -> None:
    """Run the full pipeline: fetch → download → transcribe → summarize."""
    print(f"=== Fetching episode: {args.url} ===")
    info = fetch_episode(args.url)
    print(f"  Title:    {info.title}")
    print(f"  Podcast:  {info.podcast_name}")
    print(f"  Duration: {info.duration_sec:.0f}s")
    print(f"  Audio:    {info.audio_url}")

    print("\n=== Downloading audio ===")
    ap = download_audio(info.episode_id, info.audio_url, force=args.force)

    print("\n=== Transcribing ===")
    text = transcribe(
        info.episode_id,
        str(ap),
        language=args.language,
        force=args.force,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        profile=args.profile,
        backend=args.asr_backend,
        backend_model=args.asr_model,
        max_new_tokens=getattr(args, "qwen_max_new_tokens", 0),
        dtype=getattr(args, "qwen_dtype", ""),
    )

    print("\n=== Summarizing ===")
    summarize(
        info.episode_id,
        text,
        title=info.title,
        model=args.model,
        force=args.force,
        backend=args.asr_backend,
    )

    print("\n=== Done ===")
    print(f"Summary: {summary_path(info.episode_id, backend=args.asr_backend)}")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Scrape metadata and download audio."""
    info = fetch_episode(args.url)
    print(f"Title:    {info.title}")
    print(f"Podcast:  {info.podcast_name}")
    print(f"Duration: {info.duration_sec:.0f}s")
    print(f"Episode:  {info.episode_id}")
    ensure_dirs()
    ap = download_audio(info.episode_id, info.audio_url, force=args.force)
    print(f"Audio:    {ap}")


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe a previously downloaded audio file."""
    eid = validate_episode_id(args.episode_id)
    ap = audio_path(eid)

    if not ap.exists():
        sys.exit(f"Audio not found: {ap}\nRun 'podmind fetch' first.")
    text = transcribe(
        eid,
        str(ap),
        language=args.language,
        force=args.force,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        profile=args.profile,
        backend=args.asr_backend,
        backend_model=args.asr_model,
        max_new_tokens=getattr(args, "qwen_max_new_tokens", 0),
        dtype=getattr(args, "qwen_dtype", ""),
    )
    if not args.quiet:
        print(text)


def cmd_summarize(args: argparse.Namespace) -> None:
    """Summarize a previously created transcript."""
    eid = validate_episode_id(args.episode_id)
    tp = transcript_path(eid, backend=args.asr_backend)
    if not tp.exists():
        sys.exit(f"Transcript not found: {tp}\nRun 'podmind transcribe' first.")
    text = tp.read_text(encoding="utf-8")

    # Try to reuse persisted episode metadata for the title
    title = eid
    info = load_episode_info(eid)
    if info:
        title = info.title

    result = summarize(
        eid,
        text,
        title=title,
        model=args.model,
        force=args.force,
        backend=args.asr_backend,
    )
    print(result)


def cmd_batch_transcribe(args: argparse.Namespace) -> None:
    """Transcribe multiple episodes with a single model load."""
    backend_name: str = args.asr_backend
    model_id: str = args.asr_model or _DEFAULT_MODELS[backend_name]

    # Validate IDs and check audio
    episode_ids: list[str] = []
    for raw in args.episode_ids:
        try:
            eid = validate_episode_id(raw)
        except PodmindError as e:
            sys.exit(str(e))
        ap = audio_path(eid)
        if not ap.exists():
            print(f"WARNING: audio not found for {eid}, skipping")
            continue
        episode_ids.append(eid)

    if not episode_ids:
        sys.exit("No episodes to process.")

    # Create session (no model load yet — lazy)
    session = ASRSession(
        backend_name, model_id, args.language,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
        max_new_tokens=getattr(args, "qwen_max_new_tokens", 0),
        dtype=getattr(args, "qwen_dtype", ""),
    )

    # Process each episode
    profiles: list[tuple[str, TranscribeProfile]] = []
    failed: list[str] = []
    cache_hits = 0

    for eid in episode_ids:
        print(f"\n=== Transcribing {eid} ===")
        try:
            result = session.transcribe_episode(
                eid, str(audio_path(eid)),
                force=args.force, profile=args.profile,
            )
            if result.from_cache:
                cache_hits += 1
            elif result.profile:
                profiles.append((eid, result.profile))
            if not args.quiet:
                print(result.text)
        except Exception as e:
            print(f"ERROR: Failed to transcribe {eid}: {e}")
            failed.append(eid)

    # Summary
    succeeded = len(episode_ids) - len(failed)
    total = len(episode_ids)
    print(f"\n=== Batch complete: {succeeded}/{total} succeeded "
          f"({cache_hits} from cache) ===")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    if args.profile and profiles:
        total_transcribe = sum(p.total_transcribe_seconds for _, p in profiles)
        total_audio = sum(p.total_audio_duration for _, p in profiles)
        print("\n--- Aggregate Profile ---")
        print(f"  Episodes:          {len(profiles)}")
        print(f"  Total transcribe:  {total_transcribe:.1f}s")
        print(f"  Total audio:       {total_audio:.0f}s")
        if total_audio > 0:
            print(f"  Aggregate RTF:     {total_transcribe / total_audio:.3f}")
        for eid, p in profiles:
            print(f"\n  [{eid}]")
            print(f"    RTF={p.rtf:.3f}, chunks={p.chunk_count}, "
                  f"transcribe={p.total_transcribe_seconds:.1f}s")

    if failed:
        sys.exit(1)


def cmd_bench(args: argparse.Namespace) -> None:
    """Benchmark ASR backends on a short clip from an episode."""
    issues: list[str] = []
    if not _check_binary("ffprobe"):
        issues.append("ffprobe not found — required for audio duration detection")
    if not _check_binary("ffmpeg"):
        issues.append("ffmpeg not found — required for clip extraction")
    for b in args.backends:
        issues.extend(_check_backend_issues(b))
    _exit_on_preflight_issues(issues)

    eid = validate_episode_id(args.episode_id)
    ap = audio_path(eid)
    if not ap.exists():
        sys.exit(f"Audio not found: {ap}\nRun 'podmind fetch' first.")

    # Extract clip
    import tempfile
    clip_path = Path(tempfile.gettempdir()) / f"podmind_bench_{eid}_{args.clip_seconds}s.wav"
    if not clip_path.exists() or args.force:
        print(f"Extracting {args.clip_seconds}s clip to {clip_path}...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(ap), "-t", str(args.clip_seconds),
                "-ar", "16000", "-ac", "1", str(clip_path),
            ],
            check=True,
        )
    else:
        print(f"Using existing clip: {clip_path}")

    duration = _get_duration(str(clip_path))
    print(f"Clip duration: {duration:.1f}s\n")

    results: dict[str, tuple[float, str]] = {}
    for backend_name in args.backends:
        print(f"--- {backend_name} ---")
        t0 = time.perf_counter()
        try:
            if args.asr_model and len(args.backends) != 1:
                sys.exit("--asr-model can only be used when benchmarking one backend")
            model_id = args.asr_model or _DEFAULT_MODELS[backend_name]
            session = ASRSession(
                backend_name, model_id, args.language,
                chunk_seconds=args.chunk_seconds,
                batch_size=args.batch_size,
                max_new_tokens=getattr(args, "qwen_max_new_tokens", 0),
                dtype=getattr(args, "qwen_dtype", ""),
            )
            result = session.transcribe_raw(str(clip_path), profile=args.profile)
            text = result.text
        except Exception as exc:
            print(f"ERROR: {backend_name} failed: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        rtf = elapsed / duration if duration > 0 else 0
        results[backend_name] = (elapsed, text)
        print(f"  Time:  {elapsed:.1f}s")
        print(f"  RTF:   {rtf:.3f}")
        print(f"  Chars: {len(text)}")
        print(f"  First 200 chars: {text[:200]}")
        print()

    if len(results) >= 2:
        backends_list = list(results.keys())
        t1, _ = results[backends_list[0]]
        t2, _ = results[backends_list[1]]
        ratio = t1 / t2 if t2 > 0 else float("inf")
        print(f"Speed ratio: {backends_list[0]} / {backends_list[1]} = {ratio:.2f}x")

    # Cleanup hint
    print(f"\nClip kept at: {clip_path}")


def main() -> None:
    # Prevent output buffering when stdout is redirected to a pipe or file.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(
        description="PodMind — download, transcribe, and summarize Xiaoyuzhou FM podcasts"
    )
    parser.add_argument("--debug", action="store_true",
                        help="Print traceback on unexpected errors")
    sub = parser.add_subparsers(dest="command", required=True)

    # podmind full
    p_full = sub.add_parser("full", help="Run full pipeline")
    p_full.add_argument("url", help="Xiaoyuzhou FM episode URL")
    p_full.add_argument("--language", type=validate_language, default=None,
                        help="ASR language (zh, en, ja, ko, yue, or full name like Chinese)")
    p_full.add_argument("--model", default="deepseek-v4-pro", help="LLM model for summarization")
    p_full.add_argument("--force", action="store_true", help="Skip cache")
    _add_asr_args(p_full)
    p_full.add_argument("--profile", action="store_true",
                        help="Print timing profile after transcription")
    p_full.set_defaults(func=cmd_full)

    # podmind fetch
    p_fetch = sub.add_parser("fetch", help="Scrape metadata and download audio")
    p_fetch.add_argument("url", help="Xiaoyuzhou FM episode URL")
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if cached")
    p_fetch.set_defaults(func=cmd_fetch)

    # podmind transcribe
    p_trans = sub.add_parser("transcribe", help="Transcribe downloaded audio")
    p_trans.add_argument("episode_id", help="Episode ID (from URL)")
    p_trans.add_argument("--language", type=validate_language, default=None,
                        help="ASR language (zh, en, ja, ko, yue, or full name like Chinese)")
    p_trans.add_argument("--force", action="store_true", help="Re-transcribe even if cached")
    p_trans.add_argument("--quiet", action="store_true", help="Suppress transcript output")
    _add_asr_args(p_trans)
    p_trans.add_argument("--profile", action="store_true",
                         help="Print timing profile after transcription")
    p_trans.set_defaults(func=cmd_transcribe)

    # podmind summarize
    p_summ = sub.add_parser("summarize", help="Summarize a transcript")
    p_summ.add_argument("episode_id", help="Episode ID (from URL)")
    p_summ.add_argument("--model", default="deepseek-v4-pro", help="LLM model for summarization")
    p_summ.add_argument("--force", action="store_true", help="Re-summarize even if cached")
    p_summ.add_argument("--asr-backend", choices=ASR_BACKENDS,
                        default=_DEFAULT_BACKEND,
                        help=f"ASR backend used for transcription (default: {_DEFAULT_BACKEND})")
    p_summ.set_defaults(func=cmd_summarize)

    # podmind batch-transcribe
    p_batch = sub.add_parser("batch-transcribe",
                             help="Transcribe multiple episodes in one process")
    p_batch.add_argument("episode_ids", nargs="+", help="Episode IDs to transcribe")
    p_batch.add_argument("--language", type=validate_language, default=None,
                        help="ASR language (zh, en, ja, ko, yue, or full name like Chinese)")
    _add_asr_args(p_batch)
    p_batch.add_argument("--profile", action="store_true",
                         help="Print timing profile")
    p_batch.add_argument("--force", action="store_true",
                         help="Re-transcribe even if cached")
    p_batch.add_argument("--quiet", action="store_true",
                         help="Suppress transcript output")
    p_batch.set_defaults(func=cmd_batch_transcribe)

    # podmind doctor
    p_doctor = sub.add_parser("doctor", help="Run preflight checks")
    p_doctor.add_argument("--asr-backend", choices=ASR_BACKENDS,
                          default=_DEFAULT_BACKEND,
                          help=f"Backend to check dependencies for (default: {_DEFAULT_BACKEND})")
    p_doctor.set_defaults(func=cmd_doctor)

    # podmind bench
    p_bench = sub.add_parser("bench", help="Benchmark ASR backends on a clip")
    p_bench.add_argument("episode_id", help="Episode ID")
    p_bench.add_argument("--clip-seconds", type=int, default=300,
                         help="Seconds to clip for benchmark (default: 300)")
    p_bench.add_argument("--language", type=validate_language, default=None,
                         help="ASR language")
    p_bench.add_argument(
        "--backends", nargs="+",
        default=list(ASR_BACKENDS),
        choices=ASR_BACKENDS,
        help="Backends to compare (default: all three)",
    )
    p_bench.add_argument("--asr-model", default=None,
                         help="ASR model ID override; only valid with one --backends value")
    _add_qwen_args(p_bench)
    p_bench.add_argument("--profile", action="store_true",
                         help="Print timing profile per backend")
    p_bench.add_argument("--force", action="store_true",
                         help="Re-extract clip even if cached")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    try:
        args.func(args)
    except PodmindError as e:
        sys.exit(str(e))
    except subprocess.CalledProcessError as e:
        sys.exit(f"Command failed: {e}")
    except KeyboardInterrupt:
        sys.exit("Interrupted")
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
            print("Re-run with --debug for a full traceback.", file=sys.stderr)
        sys.exit(1)


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run preflight checks and report status."""
    from contextlib import suppress

    print(f"PodMind preflight checks (backend: {args.asr_backend})\n")
    all_ok = True
    backend = args.asr_backend

    # --- ASR required ---
    asr_required: list[tuple[str, bool]] = [
        ("ffprobe", _check_binary("ffprobe")),
    ]
    spec = get_backend_spec(backend)
    from importlib.util import find_spec
    asr_required.append((spec.dependency_module, find_spec(spec.dependency_module) is not None))

    if spec.requires_ffmpeg:
        asr_required.append(("ffmpeg", _check_binary("ffmpeg")))

    if spec.requires_mps:
        asr_required.append(("ASR_DEVICE (must be mps)", ASR_DEVICE == "mps"))
        try:
            import torch
            device_ok = torch.backends.mps.is_available()
            asr_required.append(("torch", True))
            asr_required.append(("torch device (mps)", device_ok))
        except ImportError:
            asr_required.append(("torch", False))

    print("  ASR required:")
    for name, ok in asr_required:
        status = "OK" if ok else "MISSING"
        print(f"    [{status:7s}] {name}")
        if not ok:
            all_ok = False

    # --- Summary (only needed for podmind full / summarize) ---
    print()
    print("  Summary:")
    summary_ok = bool(DEEPSEEK_API_KEY)
    print(f"    [{'OK' if summary_ok else 'not set'}] DEEPSEEK_API_KEY")

    # --- Optional ---
    print()
    print("  Optional:")
    ffmpeg_ok = _check_binary("ffmpeg")
    print(f"    [{'OK ' if ffmpeg_ok else '   '}    ] ffmpeg (clip extraction / bench)")

    # --- Info ---
    print()
    print(f"  ASR model:    {_DEFAULT_MODELS.get(backend, '')}")
    if backend == "qwen-asr":
        with suppress(ImportError):
            import torch
            print(f"  PyTorch:      {torch.__version__}")
    elif backend == "mlx-whisper":
        from importlib.metadata import version
        from importlib.util import find_spec
        if find_spec("mlx_whisper") is not None:
            with suppress(Exception):
                print(f"  mlx-whisper:  {version('mlx-whisper')}")
    elif backend == "mlx-qwen-asr":
        from importlib.metadata import version
        from importlib.util import find_spec
        if find_spec("mlx_audio") is not None:
            with suppress(Exception):
                print(f"  mlx-audio:    {version('mlx-audio')}")
    print(f"  Data dir:     {DATA_DIR}")
    print(f"  DeepSeek URL: {DEEPSEEK_BASE_URL}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
