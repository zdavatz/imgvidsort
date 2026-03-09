#!/usr/bin/env python3
"""
imgvidsort - Sort and rename images/videos on a USB stick using a vision LLM via Ollama.

Videos: extracts 3 frames, sends them to the model, renames based on content description.
Images: sends the image to the model, renames based on content description.

Files are sorted into subdirectories by date (YYYY-MM-DD) and renamed with a
descriptive name from the vision model while preserving the original timestamp prefix.

Usage:
    python3 imgvidsort.py [--source /Volumes/WerniVid] [--output ./sorted] [--dry-run]
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:2b"

SUPPORTED_MODELS = [
    "qwen3-vl:8b",     # best quality — newest generation vision model
    "qwen2.5-vl:7b",   # strong vision model
    "gemma3:4b",        # good quality/speed tradeoff
    "qwen3-vl:2b",     # fastest — lightweight, good for large batches
]

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_DEFAULT_MODEL = "grok-4-1-fast-reasoning"


def _llmfit_recommend():
    """Use llmfit to find the best vision model for this hardware, return Ollama model name."""
    if not shutil.which("llmfit"):
        return None
    try:
        result = subprocess.run(
            ["llmfit", "recommend", "--capability", "vision", "--limit", "1", "--json"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        models = data.get("models", [])
        if not models:
            return None
        name = models[0]["name"].lower()
        params_b = models[0].get("params_b", 0)
        # Map llmfit model name to Ollama model name
        # e.g. "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ" -> "qwen3-vl:32b"
        #      "meta-llama/Llama-3.2-11B-Vision-Instruct" -> "llama3.2-vision:11b"
        #      "google/gemma-3-12b-it" -> "gemma3:12b"
        if "qwen3-vl" in name or "qwen3_vl" in name:
            if params_b > 20:
                return "qwen3-vl:32b"
            elif params_b > 5:
                return "qwen3-vl:8b"
            else:
                return "qwen3-vl:2b"
        elif "qwen2.5-vl" in name or "qwen2.5_vl" in name or "qwen2.5vl" in name:
            return "qwen2.5-vl:7b"
        elif "gemma-3" in name or "gemma3" in name:
            if params_b > 8:
                return "gemma3:12b"
            else:
                return "gemma3:4b"
        elif "llama-3.2" in name and "vision" in name:
            return "llama3.2-vision:11b"
        elif "phi-4" in name:
            return "phi4-mini"
        else:
            print(f"  llmfit recommends '{models[0]['name']}' but no Ollama mapping found")
            return None
    except Exception as e:
        print(f"  WARNING: llmfit failed: {e}")
        return None


def _get_system_ram_gb():
    """Return total system RAM in GB."""
    try:
        if sys.platform == "darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(result.stdout.strip()) / (1024**3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return None


def _get_model_size_gb(model):
    """Query Ollama for the model's size on disk in GB."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
            for m in tags.get("models", []):
                if model in m.get("name", ""):
                    return m.get("size", 0) / (1024**3)
    except Exception:
        pass
    return None


def _choose_num_ctx(model):
    """Choose an optimal num_ctx for vision classification based on system resources.

    Vision images use ~1000-3000 prompt tokens. We need enough headroom for
    the image tokens plus a short response, but not so much that we waste RAM.
    Returns the chosen num_ctx and a short explanation string.
    """
    ram_gb = _get_system_ram_gb()
    model_gb = _get_model_size_gb(model)

    # For vision classification, 4096 is the minimum (images can use ~2000 tokens)
    # 8192 gives comfortable headroom for larger/multiple images
    # More than 8192 is wasteful for short filename descriptions
    if ram_gb and model_gb:
        available_gb = ram_gb - model_gb - 2  # reserve 2 GB for OS
        if available_gb < 2:
            num_ctx = 4096  # tight on RAM, use minimum viable
        elif available_gb < 6:
            num_ctx = 4096  # modest RAM
        else:
            num_ctx = 8192  # plenty of RAM
        reason = f"RAM={ram_gb:.0f}GB, model={model_gb:.1f}GB"
    elif ram_gb:
        num_ctx = 4096 if ram_gb < 16 else 8192
        reason = f"RAM={ram_gb:.0f}GB"
    else:
        num_ctx = 4096
        reason = "default"

    return num_ctx, reason


def _format_size(size_bytes):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _prepare_image(path):
    """Convert HEIC to JPEG and resize large images for faster processing.
    Returns path to a ready-to-send image (may be a temp file)."""
    ext = os.path.splitext(path)[1].lower()
    needs_convert = ext == ".heic"
    # Check if image is large (> 1MB) and should be resized
    needs_resize = os.path.getsize(path) > 1_000_000

    if not needs_convert and not needs_resize:
        return path, None

    tmpdir = tempfile.mkdtemp(prefix="imgvidsort_prep_")
    out_path = os.path.join(tmpdir, "image.jpg")

    try:
        if sys.platform == "darwin":
            # Use macOS sips for conversion and resize
            if needs_convert:
                subprocess.run(
                    ["sips", "-s", "format", "jpeg", path, "--out", out_path],
                    capture_output=True, timeout=30
                )
            else:
                shutil.copy2(path, out_path)
            if needs_resize and os.path.exists(out_path):
                subprocess.run(
                    ["sips", "--resampleWidth", "1024", out_path],
                    capture_output=True, timeout=30
                )
        else:
            # On Linux, try ffmpeg for conversion/resize
            cmd = ["ffmpeg", "-y", "-i", path]
            if needs_resize:
                cmd += ["-vf", "scale=1024:-1"]
            cmd += ["-q:v", "2", out_path]
            subprocess.run(cmd, capture_output=True, timeout=30)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path, tmpdir
    except Exception as e:
        print(f"  WARNING: Image preprocessing failed: {e}")

    # Fallback: return original (will fail for HEIC but at least we tried)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return path, None


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frames(video_path, num_frames=3):
    """Extract evenly spaced frames from a video using ffmpeg."""
    tmpdir = tempfile.mkdtemp(prefix="imgvidsort_")
    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True
    )
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, json.JSONDecodeError, ValueError):
        duration = 10.0

    frame_paths = []
    for i in range(num_frames):
        timestamp = duration * (i + 1) / (num_frames + 1)
        out_path = os.path.join(tmpdir, f"frame_{i}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", out_path],
            capture_output=True
        )
        if os.path.exists(out_path):
            frame_paths.append(out_path)
    return frame_paths, tmpdir


VISION_PROMPT = (
    "Describe what you see in this image in 3-6 words suitable for a filename. "
    "Be specific about subjects, actions, and location. "
    "Use lowercase words separated by underscores. "
    "If the image contains Japanese or Chinese text, you may use those characters. "
    "Do NOT include file extensions. Examples: dog_playing_in_park, "
    "sunset_over_mountain_lake, child_riding_bicycle, "
    "東京_桜_公園, 猫_窓辺_昼寝. "
    "Reply with ONLY the filename, nothing else."
)


def _is_cjk(char):
    """Check if a character is CJK (Chinese, Japanese, Korean)."""
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF or      # CJK Unified Ideographs (Chinese/Kanji)
        0x3400 <= cp <= 0x4DBF or      # CJK Unified Ideographs Extension A
        0x3040 <= cp <= 0x309F or      # Hiragana
        0x30A0 <= cp <= 0x30FF or      # Katakana
        0xFF66 <= cp <= 0xFF9F or      # Half-width Katakana
        0xAC00 <= cp <= 0xD7AF         # Korean Hangul
    )


def clean_description(description):
    """Clean up model response into a valid filename component.
    Preserves CJK characters (Chinese, Japanese, Korean) directly in filenames.
    Transliterates Latin accented characters to ASCII equivalents.
    """
    # Transliterate common Latin accented characters
    _translit = {
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
        "à": "a", "á": "a", "â": "a", "ã": "a",
        "è": "e", "é": "e", "ê": "e", "ë": "e",
        "ì": "i", "í": "i", "î": "i", "ï": "i",
        "ò": "o", "ó": "o", "ô": "o", "õ": "o",
        "ù": "u", "ú": "u", "û": "u",
        "ñ": "n", "ç": "c",
    }
    description = description.lower()
    for char, repl in _translit.items():
        description = description.replace(char, repl)
    # Keep: ASCII alphanumerics, underscores, and CJK characters
    cleaned = []
    for ch in description:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            cleaned.append(ch)
        elif _is_cjk(ch):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    description = "".join(cleaned)
    description = re.sub(r"_+", "_", description).strip("_")
    if len(description) > 80:
        # For CJK, don't split mid-character — just truncate
        description = description[:80].rsplit("_", 1)[0] if "_" in description[:80] else description[:80]
    return description if description else "unknown"


def describe_with_ollama(image_paths, model, num_ctx=4096):
    """Send images to a vision model via Ollama and get a short description."""
    prepared = []
    tmpdirs = []
    for p in image_paths:
        prep_path, tmpdir = _prepare_image(p)
        prepared.append(prep_path)
        if tmpdir:
            tmpdirs.append(tmpdir)

    try:
        images_b64 = [encode_image_base64(p) for p in prepared]

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "think": False,
            "options": {"num_ctx": num_ctx},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw = result["message"]["content"].strip()
                cleaned = clean_description(raw)
                if cleaned == "unknown":
                    print(f"  DEBUG: Model raw response: {raw!r}")
                return cleaned
        except Exception as e:
            print(f"  WARNING: Ollama request failed: {e}")
            return "unknown"
    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)


def _upload_temp(filepath):
    """Upload a file to tmpfiles.org and return a direct download URL."""
    boundary = "----imgvidsort_boundary"
    filename = os.path.basename(filepath)

    with open(filepath, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "https://tmpfiles.org/api/v1/upload",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "imgvidsort/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())
        # Convert page URL to direct download URL
        # e.g. http://tmpfiles.org/12345/file.jpg -> https://tmpfiles.org/dl/12345/file.jpg
        page_url = result["data"]["url"]
        return page_url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1).replace("http://", "https://")


def describe_with_grok(image_paths, model):
    """Send images to Grok vision API via temporary URL uploads."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        print("  ERROR: XAI_API_KEY not set (this should not happen)")
        return "unknown"

    content = [{"type": "text", "text": VISION_PROMPT}]
    uploaded_urls = []
    for p in image_paths:
        try:
            url = _upload_temp(p)
            uploaded_urls.append(url)
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        except Exception as e:
            print(f"  WARNING: Failed to upload {p}: {e}")
            return "unknown"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "store": False,
        "max_tokens": 40,
        "temperature": 0.2,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROK_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "imgvidsort/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"].strip()
            return clean_description(text)
    except Exception as e:
        print(f"  WARNING: Grok API request failed: {e}")
        return "unknown"


def extract_date_from_filename(filename):
    """Extract date string (YYYY-MM-DD) from filename like 20250304_115207.mp4."""
    match = re.match(r"(\d{4})(\d{2})(\d{2})_", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "unsorted"


def sanitize_filename(name, ext, existing_names):
    """Ensure filename is unique by appending a counter if needed."""
    candidate = f"{name}{ext}"
    counter = 1
    while candidate in existing_names:
        candidate = f"{name}_{counter}{ext}"
        counter += 1
    existing_names.add(candidate)
    return candidate


def collect_media_files(source_dir):
    """Recursively collect all image and video files."""
    media_files = []
    for root, _dirs, files in os.walk(source_dir):
        # Skip hidden directories
        if any(part.startswith(".") for part in root.split(os.sep) if part):
            parent = os.path.relpath(root, source_dir)
            if parent.startswith("."):
                continue
        for f in files:
            if f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                media_files.append(os.path.join(root, f))
    # Sort by modification time, newest first
    media_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return media_files


def process_file(filepath, output_dir, existing_names, model, describe_fn, dry_run=False, inplace=False, file_index=0, file_total=0):
    """Process a single image or video file."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    date_str = extract_date_from_filename(filename)
    date_dir = os.path.join(output_dir, date_str)

    file_size = os.path.getsize(filepath)
    counter = f"[{file_index}/{file_total}]" if file_total else ""
    if file_size == 0:
        print(f"\n{counter} [SKIP] {filename} (0 bytes)")
        return None

    is_video = ext in VIDEO_EXTS
    file_type = "VIDEO" if is_video else "IMAGE"
    print(f"\n{counter} [{file_type}] {filename}")

    # Get description from vision model
    tmpdir = None
    t0 = time.time()
    try:
        if is_video:
            print(f"  Extracting 3 frames...")
            frame_paths, tmpdir = extract_frames(filepath)
            if not frame_paths:
                print(f"  WARNING: Could not extract frames, skipping rename")
                description = "unknown_video"
            else:
                print(f"  Analyzing {len(frame_paths)} frames with {model}...")
                description = describe_fn(frame_paths, model)
        else:
            print(f"  Analyzing with {model}...")
            description = describe_fn([filepath], model)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    elapsed = time.time() - t0

    # Build new filename: date_prefix + description + ext
    timestamp_match = re.match(r"(\d{8}_\d{6})", filename)
    prefix = timestamp_match.group(1) if timestamp_match else ""
    if prefix:
        new_name = f"{prefix}_{description}"
    else:
        new_name = description

    new_filename = sanitize_filename(new_name, ext, existing_names)

    if inplace:
        dest_path = os.path.join(os.path.dirname(filepath), new_filename)
        print(f"  -> {new_filename} ({elapsed:.1f}s)")

        if not dry_run:
            os.rename(filepath, dest_path)
    else:
        dest_path = os.path.join(date_dir, new_filename)
        print(f"  -> {date_str}/{new_filename} ({elapsed:.1f}s)")

        if not dry_run:
            os.makedirs(date_dir, exist_ok=True)
            shutil.copy2(filepath, dest_path)
            # Verify the copy has the same size as the source
            src_size = os.path.getsize(filepath)
            dst_size = os.path.getsize(dest_path)
            if dst_size != src_size:
                os.remove(dest_path)
                print(f"  ERROR: Copy verification failed ({dst_size} != {src_size} bytes), removed bad copy")
                return None

    return dest_path


def main():
    models_help = (
        "Supported Ollama models:\n" + "\n".join(f"  {m}" for m in SUPPORTED_MODELS) +
        "\n\nGrok models:\n  grok-2-vision-latest (default for --api grok)"
    )
    parser = argparse.ArgumentParser(
        description="Sort and rename images/videos using a vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=models_help,
    )
    parser.add_argument("--source", default="/Volumes/WerniVid",
                        help="Source directory (default: /Volumes/WerniVid)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <source>/sorted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without copying files")
    parser.add_argument("--api", default="ollama", choices=["ollama", "grok"],
                        help="API backend to use (default: ollama)")
    parser.add_argument("--model", default=None,
                        help=f"Vision model (default: {DEFAULT_MODEL} for ollama, {GROK_DEFAULT_MODEL} for grok)")
    parser.add_argument("--from-date", default=None,
                        help="Only process files from this date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--to-date", default=None,
                        help="Only process files up to this date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--prefix", default=None,
                        help="Only process files whose name starts with this string (e.g. PHOTO)")
    parser.add_argument("--inplace", action="store_true",
                        help="Rename files in place instead of copying to output directory")
    args = parser.parse_args()

    api = args.api
    if args.model:
        model = args.model
    elif api == "grok":
        model = GROK_DEFAULT_MODEL
    else:
        # Try llmfit to find the best vision model for this hardware
        recommended = _llmfit_recommend()
        if recommended:
            print(f"llmfit recommends: {recommended}")
            model = recommended
        else:
            model = DEFAULT_MODEL

    source = args.source
    output_dir = args.output or os.path.join(source, "sorted")

    if not os.path.isdir(source):
        print(f"ERROR: Source directory not found: {source}")
        sys.exit(1)

    if api == "grok":
        describe_fn = describe_with_grok
        if not os.environ.get("XAI_API_KEY"):
            print("XAI_API_KEY environment variable not set.")
            key = input("Enter your Grok API key (from https://console.x.ai): ").strip()
            if not key:
                print("ERROR: No API key provided")
                sys.exit(1)
            os.environ["XAI_API_KEY"] = key
            # Persist the key in ~/.bashrc
            bashrc = os.path.expanduser("~/.bashrc")
            export_line = f'export XAI_API_KEY="{key}"\n'
            with open(bashrc, "a") as f:
                f.write(export_line)
            print(f"API key saved to {bashrc}")
    else:
        # Choose optimal context window size for this system
        num_ctx, ctx_reason = _choose_num_ctx(model)
        describe_fn = lambda paths, m: describe_with_ollama(paths, m, num_ctx=num_ctx)
        # Verify Ollama is running and model is available
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
                model_names = [m["name"] for m in tags.get("models", [])]
                if not any(model in name for name in model_names):
                    print(f"Model '{model}' not found in Ollama. Pulling it now...")
                    result = subprocess.run(["ollama", "pull", model])
                    if result.returncode != 0:
                        print(f"ERROR: Failed to pull model '{model}'")
                        sys.exit(1)
                    print(f"Model '{model}' installed successfully.")
            # Check if model is already loaded with bad settings
            try:
                ps_req = urllib.request.Request("http://localhost:11434/api/ps")
                with urllib.request.urlopen(ps_req, timeout=5) as ps_resp:
                    ps_data = json.loads(ps_resp.read().decode("utf-8"))
                    for m in ps_data.get("models", []):
                        if model in m.get("name", ""):
                            size_gb = m.get("size", 0) / (1024**3)
                            ctx = m.get("details", {}).get("parameter_size", "")
                            if size_gb > 10:
                                print(f"  NOTE: Model using {size_gb:.1f} GB RAM — unloading to reload with optimal settings...")
                                unload_req = urllib.request.Request(
                                    OLLAMA_URL,
                                    data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                    headers={"Content-Type": "application/json"},
                                )
                                urllib.request.urlopen(unload_req, timeout=10)
                                print(f"  Model unloaded. Will reload with num_ctx={num_ctx}.")
            except Exception:
                pass  # Non-critical check
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama at localhost:11434: {e}")
            print("Make sure Ollama is running: ollama serve")
            sys.exit(1)

    print(f"Source:  {source}")
    if args.inplace:
        print(f"Mode:    inplace (rename files where they are)")
    else:
        print(f"Output:  {output_dir}")
    print(f"API:     {api}")
    print(f"Model:   {model}")
    if api == "ollama":
        print(f"Context: {num_ctx} tokens ({ctx_reason})")
    if args.prefix:
        print(f"Prefix:  {args.prefix}")
    if args.dry_run:
        print("DRY RUN - no files will be " + ("renamed" if args.inplace else "copied"))

    media_files = collect_media_files(source)
    # Exclude files already in the output directory
    media_files = [f for f in media_files if not f.startswith(os.path.join(source, "sorted"))]

    # Filter by date range if specified
    from_date = args.from_date.replace("-", "") if args.from_date else None
    to_date = args.to_date.replace("-", "") if args.to_date else None
    if from_date or to_date:
        def _in_date_range(filepath):
            match = re.match(r"(\d{8})_", os.path.basename(filepath))
            if not match:
                return False
            file_date = match.group(1)
            if from_date and file_date < from_date:
                return False
            if to_date and file_date > to_date:
                return False
            return True
        media_files = [f for f in media_files if _in_date_range(f)]

    # Filter by filename prefix if specified
    if args.prefix:
        media_files = [f for f in media_files if os.path.basename(f).startswith(args.prefix)]

    if args.limit:
        media_files = media_files[:args.limit]

    source_total = sum(os.path.getsize(f) for f in media_files)
    print(f"\nFound {len(media_files)} media files ({_format_size(source_total)})")
    images = [f for f in media_files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    videos = [f for f in media_files if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    print(f"  Images: {len(images)}")
    print(f"  Videos: {len(videos)}")

    if not media_files:
        print("No media files found.")
        return

    existing_names = set()
    processed = 0
    errors = 0

    total = len(media_files)
    for i, filepath in enumerate(media_files, 1):
        try:
            process_file(filepath, output_dir, existing_names, model, describe_fn, dry_run=args.dry_run, inplace=args.inplace, file_index=i, file_total=total)
            processed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  ERROR processing {filepath}: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"Done. Processed: {processed}, Errors: {errors}")
    print(f"Source size:  {_format_size(source_total)}")
    if args.inplace:
        print(f"Files renamed in place in: {source}")
    elif not args.dry_run and os.path.isdir(output_dir):
        output_total = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(output_dir)
            for f in files
        )
        print(f"Output size:  {_format_size(output_total)}")
        if output_total != source_total:
            print(f"WARNING: Size mismatch! Difference: {_format_size(abs(source_total - output_total))}")
        print(f"Sorted files are in: {output_dir}")


if __name__ == "__main__":
    main()
