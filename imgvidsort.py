#!/usr/bin/env python3
"""
imgvidsort - Sort and rename images/videos on a USB stick using Qwen2.5-VL via Ollama.

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
import urllib.request

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-vl:7b"


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


def describe_with_ollama(image_paths, model):
    """Send images to Qwen2.5-VL via Ollama and get a short description."""
    images_b64 = [encode_image_base64(p) for p in image_paths]

    prompt = (
        "Describe what you see in this image in 3-6 words suitable for a filename. "
        "Be specific about subjects, actions, and location. "
        "Use only lowercase English words separated by underscores. "
        "Do NOT include file extensions. Examples: dog_playing_in_park, "
        "sunset_over_mountain_lake, child_riding_bicycle. "
        "Reply with ONLY the filename, nothing else."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
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
            description = result["message"]["content"].strip()
            # Clean up: keep only alphanumeric and underscores
            description = re.sub(r"[^a-z0-9_]", "_", description.lower())
            description = re.sub(r"_+", "_", description).strip("_")
            # Truncate to reasonable length
            if len(description) > 80:
                description = description[:80].rsplit("_", 1)[0]
            return description if description else "unknown"
    except Exception as e:
        print(f"  WARNING: Ollama request failed: {e}")
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
    return sorted(media_files)


def process_file(filepath, output_dir, existing_names, model, dry_run=False):
    """Process a single image or video file."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    date_str = extract_date_from_filename(filename)
    date_dir = os.path.join(output_dir, date_str)

    is_video = ext in VIDEO_EXTS
    file_type = "VIDEO" if is_video else "IMAGE"
    print(f"\n[{file_type}] {filename}")

    # Get description from vision model
    tmpdir = None
    try:
        if is_video:
            print(f"  Extracting 3 frames...")
            frame_paths, tmpdir = extract_frames(filepath)
            if not frame_paths:
                print(f"  WARNING: Could not extract frames, skipping rename")
                description = "unknown_video"
            else:
                print(f"  Analyzing {len(frame_paths)} frames with {model}...")
                description = describe_with_ollama(frame_paths, model)
        else:
            print(f"  Analyzing with {model}...")
            description = describe_with_ollama([filepath], model)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Build new filename: date_prefix + description + ext
    timestamp_match = re.match(r"(\d{8}_\d{6})", filename)
    prefix = timestamp_match.group(1) if timestamp_match else ""
    if prefix:
        new_name = f"{prefix}_{description}"
    else:
        new_name = description

    new_filename = sanitize_filename(new_name, ext, existing_names)
    dest_path = os.path.join(date_dir, new_filename)

    print(f"  -> {date_str}/{new_filename}")

    if not dry_run:
        os.makedirs(date_dir, exist_ok=True)
        shutil.copy2(filepath, dest_path)

    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Sort and rename images/videos using Qwen2.5-VL")
    parser.add_argument("--source", default="/Volumes/WerniVid",
                        help="Source directory (default: /Volumes/WerniVid)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <source>/sorted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without copying files")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    model = args.model
    source = args.source
    output_dir = args.output or os.path.join(source, "sorted")

    if not os.path.isdir(source):
        print(f"ERROR: Source directory not found: {source}")
        sys.exit(1)

    # Verify Ollama is running and model is available
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
            model_names = [m["name"] for m in tags.get("models", [])]
            if not any(model in name for name in model_names):
                print(f"WARNING: Model '{model}' not found in Ollama. Available: {model_names}")
                print("Continuing anyway in case it's still pulling...")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama at localhost:11434: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print(f"Source:  {source}")
    print(f"Output:  {output_dir}")
    print(f"Model:   {model}")
    if args.dry_run:
        print("DRY RUN - no files will be copied")

    media_files = collect_media_files(source)
    # Exclude files already in the output directory
    media_files = [f for f in media_files if not f.startswith(os.path.join(source, "sorted"))]

    print(f"\nFound {len(media_files)} media files")
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

    for filepath in media_files:
        try:
            process_file(filepath, output_dir, existing_names, model, dry_run=args.dry_run)
            processed += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  ERROR processing {filepath}: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"Done. Processed: {processed}, Errors: {errors}")
    if not args.dry_run:
        print(f"Sorted files are in: {output_dir}")


if __name__ == "__main__":
    main()
