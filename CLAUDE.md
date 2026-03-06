# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**imgvidsort** — Sorts and renames images/videos from a USB stick using a local vision LLM via Ollama. Licensed under GPL-3.0.

## How It Works

1. Scans source directory recursively for images (.jpg, .jpeg, .png, .heic, .webp) and videos (.mp4, .mov, .avi, .mkv)
2. For videos: extracts 3 evenly-spaced frames via ffmpeg, sends them to the vision model
3. For images: sends the image directly to the vision model
4. Model returns a short description used as the new filename
5. Files are copied (not moved) into `<output>/YYYY-MM-DD/` directories, preserving original timestamps in the filename

## Commands

```bash
# Basic usage (reads from /Volumes/WerniVid, writes to /Volumes/WerniVid/sorted)
python3 imgvidsort.py

# Preview without copying
python3 imgvidsort.py --dry-run

# Custom source/output
python3 imgvidsort.py --source /path/to/media --output /path/to/sorted

# Use a specific vision model
python3 imgvidsort.py --model qwen3-vl:8b
python3 imgvidsort.py --model qwen2.5-vl:7b
python3 imgvidsort.py --model gemma3:4b
python3 imgvidsort.py --model qwen3-vl:2b
```

## Supported Vision Models

- `qwen3-vl:8b` (default) — best quality, newest generation
- `qwen2.5-vl:7b` — strong vision model
- `gemma3:4b` — good quality/speed tradeoff
- `qwen3-vl:2b` — fastest, lightweight, good for large batches

## Dependencies

- Python 3 (no pip packages needed — uses only stdlib)
- ffmpeg/ffprobe (for video frame extraction)
- Ollama running locally with one of the supported vision models
