# imgvidsort

Sort and rename images and videos using a vision LLM via Ollama or the Grok API.

## What it does

- Scans a USB stick (or any directory) for images and videos
- Sends images to a vision model for content recognition and renames them based on what the model sees
- Extracts 3 frames from each video, analyzes them, and renames the video accordingly
- Sorts everything into date-based subdirectories (`YYYY-MM-DD/`)
- Copies files — originals are never modified or deleted
- Skips 0-byte files (corrupt or incomplete transfers) to avoid hallucinated filenames
- Verifies file integrity after copying (removes bad copies if size mismatch)
- Reports source and output directory sizes, warns on size mismatch
- Automatically pulls the Ollama model if not installed
- Uses [llmfit](https://github.com/jeroenherczeg/llmfit) to auto-select the best vision model for your hardware (if installed)
- Supports the [Grok](https://x.ai/) vision API as an alternative backend (`--api grok`)
- Filter files by date range with `--from-date` and `--to-date`
- Filter files by filename prefix with `--prefix` (e.g. `--prefix PHOTO`)
- Limit processing to N files with `--limit`
- Rename files in place with `--inplace` (no copying, no date subdirectories)
- Logs per-file classification time
- Automatically converts HEIC images to JPEG and resizes large images (>1MB) before sending to the model for faster processing
- Disables thinking/reasoning mode in Ollama models (e.g. qwen3-vl) for fast, reliable image recognition
- Auto-detects optimal context window size (num_ctx) based on system RAM and model size
- Detects and unloads models running with excessive memory usage before reloading with optimal settings
- Transliterates non-ASCII characters (umlauts, accents) in model responses for safe filenames

## Requirements

- Python 3 (no external packages needed)
- [ffmpeg](https://ffmpeg.org/) (for video frame extraction)
- [Ollama](https://ollama.com/) running locally with a vision model, **or**
- A [Grok API key](https://console.x.ai) (set `XAI_API_KEY` env var)

```bash
# Install one of the supported vision models
ollama pull qwen3-vl:8b         # best quality
ollama pull qwen2.5-vl:7b       # strong vision model
ollama pull gemma3:4b            # good quality/speed tradeoff
ollama pull qwen3-vl:2b         # default — lightweight, recommended for Apple M5 Macs
```

## Usage

```bash
# Preview what would happen (no files copied)
python3 imgvidsort.py --dry-run

# Sort from default USB stick (/Volumes/WerniVid) into /Volumes/WerniVid/sorted/
python3 imgvidsort.py

# Custom source and output
python3 imgvidsort.py --source /path/to/media --output /path/to/sorted

# Use a specific vision model
python3 imgvidsort.py --model qwen3-vl:8b     # best quality
python3 imgvidsort.py --model qwen2.5-vl:7b   # strong alternative
python3 imgvidsort.py --model gemma3:4b        # balanced
python3 imgvidsort.py --model qwen3-vl:2b     # fastest, recommended for M5 Macs (default)

# Use Grok API instead of Ollama
python3 imgvidsort.py --api grok

# Filter by date range
python3 imgvidsort.py --from-date 2025-03-01 --to-date 2025-03-15

# Limit number of files to process
python3 imgvidsort.py --limit 10

# Only process files starting with "PHOTO"
python3 imgvidsort.py --source /path/to/media --prefix PHOTO --limit 10

# Rename files in place (no copying, stays in source directory)
python3 imgvidsort.py --source /path/to/media --prefix PHOTO --inplace
```

## Output structure

```
sorted/
  2024-05-23/
    20240523_170328_dog_playing_in_park.jpg
    20240523_171402_sunset_over_lake.jpg
  2025-03-04/
    20250304_115207_child_riding_bicycle.mp4
```

## License

GPL-3.0
