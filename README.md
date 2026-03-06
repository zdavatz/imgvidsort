# imgvidsort

Sort and rename images and videos using a local vision LLM via Ollama.

## What it does

- Scans a USB stick (or any directory) for images and videos
- Sends images to a vision model for content recognition and renames them based on what the model sees
- Extracts 3 frames from each video, analyzes them, and renames the video accordingly
- Sorts everything into date-based subdirectories (`YYYY-MM-DD/`)
- Copies files — originals are never modified or deleted

## Requirements

- Python 3 (no external packages needed)
- [ffmpeg](https://ffmpeg.org/) (for video frame extraction)
- [Ollama](https://ollama.com/) running locally with a vision model

```bash
# Install one of the supported vision models
ollama pull qwen2.5-vl:7b      # default, strong vision model
ollama pull qwen3-vl:2b         # lightweight & fast, good for large batches
ollama pull gemma3:4b            # good quality/speed tradeoff
```

## Usage

```bash
# Preview what would happen (no files copied)
python3 imgvidsort.py --dry-run

# Sort from default USB stick (/Volumes/WerniVid) into /Volumes/WerniVid/sorted/
python3 imgvidsort.py

# Custom source and output
python3 imgvidsort.py --source /path/to/media --output /path/to/sorted

# Use a different vision model
python3 imgvidsort.py --model qwen3-vl:2b
python3 imgvidsort.py --model gemma3:4b
python3 imgvidsort.py --model llama3.2-vision:latest
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
