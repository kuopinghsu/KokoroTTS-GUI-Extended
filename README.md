# Kokoro TTS Local Web UI (Extended)

A local Gradio interface for the **Kokoro** open-weight Text-to-Speech model. Generate high-quality speech with parallel chunk processing, text cleaning, batch file conversion, and WAV/MP3 export.

<img width="1138" height="1283" alt="image" src="https://github.com/user-attachments/assets/3d4b74d2-33d8-4cef-ad4e-bc507d490270" />

## Features

* **High-quality TTS** — US & UK Kokoro voices
* **Single Text tab** — type or drag-and-drop a `.txt` / `.md` file
* **Batch Files tab** — build a file list (upload, paste paths, or a whole folder) and convert many files at once
* **WAV / MP3 export** — choose output format (MP3 needs `ffmpeg`)
* **Batch preview** — play completed files with Prev/Next after a batch run
* **Parallel processing** — long text is split into chunks and processed on multiple threads
* **Hardware acceleration** — uses NVIDIA CUDA when available, otherwise CPU
* **Text preprocessing** — lowercase, whitespace normalize, reference-number removal, initials formatting
* **Tokenization preview** — inspect phonemes before synthesis
* **Sample library** — Random Quote, Gatsby, Frankenstein shortcuts

## Prerequisites

1. **Python 3.10+** (recommended; matches Kokoro’s supported range)
2. **eSpeak-ng** — required for phonemization
   * **Windows:**
     1. Install from [eSpeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases).
     2. In an **Administrator** PowerShell:
        ```powershell
        $env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
        $env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
        setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
        setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
        ```
   * **Linux:** `sudo apt-get install espeak-ng`
   * **Mac:** `brew install espeak`
3. **ffmpeg** (optional, for MP3) — must include `libmp3lame`
   * **Linux:** `sudo apt-get install ffmpeg`
   * **Mac:** `brew install ffmpeg`
   * **Windows:** install from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to `PATH`

## Installation

1. **Clone or open this project**, then create a virtual environment in the project folder:

     ```bash
     uv venv --python 3.12.2 .venv
     source .venv/bin/activate
     ```

2. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

   For a specific CUDA build of PyTorch, install Torch from [pytorch.org](https://pytorch.org/) first, then run the command above.

   Model weights are downloaded automatically from Hugging Face on first run (`hexgrad/Kokoro-82M`). Setting `HF_TOKEN` is optional but helps with Hub rate limits.

## Usage

1. Activate the environment and start the app:

   ```bash
   source .venv/bin/activate
   python app.py
   ```

2. The UI opens in your browser. Shared controls (voice, speed, cleaning, parallel chunks, output format) apply to both tabs.

### Single Text

* Enter text, or drag a `.txt` / `.md` file onto the drop zone
* Click **Generate** — preview plays in the UI; a downloadable WAV/MP3 appears below
* Use **Tokenize** to inspect phonemes

### Batch Files

1. Add files via upload, or paste paths (one per line):
   * a single file: `/path/to/chapter1.txt`
   * a **folder**: `/path/to/text` (adds all `.txt` / `.md` inside)
   * a glob: `/path/to/*.txt`
2. Click **Add Paths to List** (or use Clear / Remove by index)
3. Set the **Output folder** and **Output Format** (`wav` or `mp3`)
4. Click **Batch Convert**
5. When finished, use **Play completed file** / Prev / Next to preview results

## Configuration

| Control | Description |
|---|---|
| **Voice** | US / UK Kokoro voices |
| **Speed** | 0.5× – 2.0× |
| **Output Format** | `wav` or `mp3` |
| **Text Cleaning** | Lowercase, whitespace, references, initials |
| **Parallel chunks** | 1–10 concurrent chunk workers (lower if you hit OOM) |
| **GRADIO_SERVER_PORT** | Optional env var to force a port; if that port cannot bind, the app picks a free one |

On some WSL/Windows setups, ports in the 7000–9000 range are reserved. The app prefers `7860`, then falls back (e.g. around `17860`).

## Troubleshooting

* **NLTK `punkt_tab` / `punkt`:** The app downloads these on startup. If that fails:
  ```bash
  python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
  ```
* **eSpeak / phonemizer errors:** Install `espeak-ng` and ensure it is on `PATH` (see Prerequisites). The app includes a small compatibility patch for newer phonemizer builds.
* **MP3 export fails:** Install `ffmpeg` with `libmp3lame`, or switch **Output Format** to `wav`.
* **Gradio `InvalidPathError` when returning batch files:** The app allows serving files under your home directory. Restart after updating so `allowed_paths` is applied. Outputs are still written to your chosen folder even if the download widget fails.
* **Port already in use / empty port range:** Unset a bad `GRADIO_SERVER_PORT`, or set one that can bind, e.g. `GRADIO_SERVER_PORT=17860 python app.py`.

## Project structure

```text
├── app.py                 # Gradio UI (single + batch)
├── requirements.txt       # Python dependencies
├── kokoro/                # Kokoro package (local / installed)
├── en.txt                 # Optional random-quote source
├── gatsby5k.md            # Optional sample text
├── frankenstein5k.md      # Optional sample text
├── batch_output/          # Default batch output folder (created on use)
└── .venv/                 # Virtual environment (created during install)
```

## License

This project uses the Kokoro TTS model. See the original model’s license for usage terms.

### Acknowledgements

- [@yl4579](https://huggingface.co/yl4579) for architecting StyleTTS 2
- [@Pendrokar](https://huggingface.co/Pendrokar) for adding Kokoro to the TTS Spaces Arena
- Thanks to everyone who contributed synthetic training data and compute
- Discord: https://discord.gg/QuGxSWBfQy
- “Kokoro” (心) means “heart” / “spirit” in Japanese; also a [Terminator franchise character](https://terminator.fandom.com/wiki/Kokoro) alongside [Misaki](https://github.com/hexgrad/misaki?tab=readme-ov-file#acknowledgements)

<img src="https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg" width="400" alt="kokoro" />
