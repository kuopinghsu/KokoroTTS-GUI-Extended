# Kokoro TTS Local Web UI (Extended)

A local Gradio interface for Kokoro (StyleTTS2 + iSTFTNet, 82M params, 24 kHz). Switch checkpoints in the UI:

* **[v1.0](https://huggingface.co/hexgrad/Kokoro-82M)** — more US/UK English and Japanese voices, plus 8 older Chinese names
* **[v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)** — 100 Mandarin speakers, plus English Maple / Sol / Vale

See the [RapidSpeech Kokoro guide](https://github.com/RapidAI/RapidSpeech.cpp/blob/main/docs/kokoro.md) for architecture and G2P notes.

<img width="1138" height="1283" alt="image" src="https://github.com/user-attachments/assets/3d4b74d2-33d8-4cef-ad4e-bc507d490270" />

## Features

* **Two checkpoints** — switch in the UI:
  * **v1.0 舊版** (`hexgrad/Kokoro-82M`): more US/UK English + Japanese + 8 older Chinese names
  * **v1.1-zh** (`hexgrad/Kokoro-82M-v1.1-zh`): 55 female + 45 male Mandarin voices, plus Maple / Sol / Vale
* **Chinese G2P (v1.1-zh)** — `misaki[zh]` `ZHG2P(version='1.1')`
* **Single Text tab** — type or drag-and-drop a `.txt` / `.md` file
* **Batch Files tab** — build a file list (upload, paste paths, or a whole folder) and convert many files at once
* **WAV / MP3 export** — choose output format (MP3 needs `ffmpeg`)
* **Batch preview** — play completed files with Prev/Next after a batch run
* **Parallel processing** — long text is split into chunks and processed on multiple threads
* **Hardware acceleration** — uses NVIDIA CUDA when available, otherwise CPU
* **Text preprocessing** — lowercase, whitespace normalize, reference-number removal, initials formatting
* **Tokenization preview** — inspect phonemes before synthesis
* **Sample library** — 中文示例, Random Quote, Gatsby, Frankenstein

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

   Model weights are downloaded from Hugging Face on first use of each checkpoint (`hexgrad/Kokoro-82M` for v1.0, `hexgrad/Kokoro-82M-v1.1-zh` for Chinese). Setting `HF_TOKEN` is optional but helps with Hub rate limits.

## Usage

1. Activate the environment and start the app:

   ```bash
   source .venv/bin/activate
   uv run python app.py
   ```

2. The UI opens in your browser. Shared controls (model, voice, speed, cleaning, parallel chunks, output format) apply to both tabs.

3. **Choose a checkpoint** with the **Model** dropdown (left panel):
   * **v1.0 舊版** — Heart, Bella, Emma, Japanese Alpha / Kumo, older Xiaoxiao / Yunxi, …
   * **v1.1-zh** — 55 female (`zf_*`) + 45 male (`zm_*`) Mandarin voices, plus Maple / Sol / Vale
   Switching reloads weights (first time downloads from Hugging Face). Then pick **Language** and **Voice**.

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
| **Model** | `v1.0 舊版` for English/Japanese, or `v1.1-zh` for Chinese |
| **Voice** | Depends on the selected model |
| **Speed** | Duration scale (0.5×–2.0×). Values outside 0.7–1.3 may distort prosody |
| **Output Format** | `wav` or `mp3` |
| **Text Cleaning** | Lowercase, whitespace, references, initials |
| **Parallel chunks** | 1–10 concurrent chunk workers (lower if you hit OOM) |
| **GRADIO_SERVER_PORT** | Optional env var to force a port; if that port cannot bind, the app picks a free one |

On some WSL/Windows setups, ports in the 7000–9000 range are reserved. The app prefers `7860`, then falls back (e.g. around `17860`).

## Troubleshooting

* **NLTK `punkt_tab` / `punkt`:** The app downloads these on startup. If that fails:
  ```bash
  uv run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
  ```
* **Chinese voices unavailable:** Select **v1.1-zh** in **Model**, then install G2P extras and restart:
  ```bash
  uv pip install "misaki[zh]"
  ```
  v1.0 has only 8 older Chinese names (Xiaobei, Xiaoxiao, Yunxi, …). The 100 numbered Mandarin voices (`zf_001` / `zm_009`, …) are in **v1.1-zh** only.
* **Japanese voices unavailable:** Select **v1.0 舊版** (Japanese is not in v1.1-zh), then:
  ```bash
  uv pip install "misaki[ja]"
  uv run python -m unidic download
  ```
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
