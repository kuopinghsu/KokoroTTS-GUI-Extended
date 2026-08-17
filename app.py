import gradio as gr
import os
import random
import shutil
import subprocess
import sys
import tempfile
import torch
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# --- MONKEY PATCH: Fix for 'EspeakWrapper' has no attribute 'set_data_path' ---
# This block must run BEFORE importing kokoro or misaki.
# It patches a compatibility issue between misaki and newer phonemizer versions.
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, 'set_data_path'):
        print("DEBUG: Patching EspeakWrapper.set_data_path for compatibility...")
        def set_data_path(path):
            # This environment variable is often used by espeak-ng as a fallback
            os.environ["PHONEMIZER_ESPEAK_DATA"] = path
        EspeakWrapper.set_data_path = set_data_path
except ImportError:
    pass # If phonemizer isn't installed, the main import block will catch it.

# --- NEW: Imports and setup for text chunking ---
import re
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    raise ImportError("NLTK not found. Please install it using: pip install nltk")

# Download NLTK sentence tokenizer data if not already present.
# Newer NLTK versions use 'punkt_tab'; older ones use 'punkt'.
for resource, path in (
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("punkt", "tokenizers/punkt"),
):
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"DEBUG: NLTK '{resource}' tokenizer not found. Downloading...")
        nltk.download(resource)
        print(f"DEBUG: Download of '{resource}' complete.")

# --- NEW: Progress Logging ---
def log_progress(message, level="INFO"):
    """Simple colored and timestamped logger."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "DEBUG": "\033[92m",   # Green
        "WARN": "\033[93m",    # Yellow
        "ERROR": "\033[91m",   # Red
        "ENDC": "\033[0m",     # End color
    }
    color = colors.get(level, colors["INFO"])
    timestamp = time.strftime('%H:%M:%S')
    print(f"{color}[{timestamp} - {level}] {message}{colors['ENDC']}")

# --- LOCAL SETUP: Import Kokoro modules ---
# Ensure the 'kokoro' folder/module is in the same directory or installed
try:
    from kokoro import KModel, KPipeline
    import kokoro
    import misaki
except ImportError as e:
    raise ImportError(f"Could not import kokoro or misaki. Ensure you are in the correct directory with the library files. Error: {e}")

# --- DEVICE DETECTION ---
# Check for NVIDIA GPU (CUDA) on Windows
CUDA_AVAILABLE = torch.cuda.is_available()
device = 'cuda' if CUDA_AVAILABLE else 'cpu'

print(f"DEBUG: Kokoro Version: {kokoro.__version__}")
print(f"DEBUG: Misaki Version: {misaki.__version__}")
print(f"DEBUG: Running on {device.upper()}")

# --- MODEL PRESETS ---
# v1.0: more English + Japanese (+ older named Chinese voices)
# v1.1-zh: 100 Mandarin speakers + Maple/Sol/Vale
# https://huggingface.co/hexgrad/Kokoro-82M
# https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
# https://github.com/RapidAI/RapidSpeech.cpp/blob/main/docs/kokoro.md
ZH_G2P_VERSION = '1.1'
LANG_EXTRAS = {'j': 'ja', 'z': 'zh'}

CHOICES_V10 = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
    '🇯🇵 🚺 Alpha': 'jf_alpha',
    '🇯🇵 🚺 Gongitsune': 'jf_gongitsune',
    '🇯🇵 🚺 Nezumi': 'jf_nezumi',
    '🇯🇵 🚺 Tebukuro': 'jf_tebukuro',
    '🇯🇵 🚹 Kumo': 'jm_kumo',
    '🇨🇳 🚺 Xiaobei': 'zf_xiaobei',
    '🇨🇳 🚺 Xiaoni': 'zf_xiaoni',
    '🇨🇳 🚺 Xiaoxiao': 'zf_xiaoxiao',
    '🇨🇳 🚺 Xiaoyi': 'zf_xiaoyi',
    '🇨🇳 🚹 Yunjian': 'zm_yunjian',
    '🇨🇳 🚹 Yunxi': 'zm_yunxi',
    '🇨🇳 🚹 Yunxia': 'zm_yunxia',
    '🇨🇳 🚹 Yunyang': 'zm_yunyang',
}

CHOICES_V11 = {
    '🇺🇸 🚺 Maple': 'af_maple',
    '🇺🇸 🚺 Sol': 'af_sol',
    '🇬🇧 🚺 Vale': 'bf_vale',
}
for _id in (
    '001', '002', '003', '004', '005', '006', '007', '008', '017', '018', '019',
    '021', '022', '023', '024', '026', '027', '028', '032', '036', '038', '039',
    '040', '042', '043', '044', '046', '047', '048', '049', '051', '059', '060',
    '067', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079',
    '083', '084', '085', '086', '087', '088', '090', '092', '093', '094', '099',
):
    CHOICES_V11[f'🇨🇳 🚺 {_id}'] = f'zf_{_id}'
for _id in (
    '009', '010', '011', '012', '013', '014', '015', '016', '020', '025', '029',
    '030', '031', '033', '034', '035', '037', '041', '045', '050', '052', '053',
    '054', '055', '056', '057', '058', '061', '062', '063', '064', '065', '066',
    '068', '069', '080', '081', '082', '089', '091', '095', '096', '097', '098',
    '100',
):
    CHOICES_V11[f'🇨🇳 🚹 {_id}'] = f'zm_{_id}'


def _voice_groups(choices):
    groups = {
        "All": list(choices.items()),
        "🇺🇸 English (US)": [(k, v) for k, v in choices.items() if v.startswith('a')],
        "🇬🇧 English (UK)": [(k, v) for k, v in choices.items() if v.startswith('b')],
    }
    ja = [(k, v) for k, v in choices.items() if v.startswith('j')]
    zh = [(k, v) for k, v in choices.items() if v.startswith('z')]
    if ja:
        groups["🇯🇵 Japanese"] = ja
    if zh:
        groups["🇨🇳 Chinese"] = zh
    return groups


MODEL_PRESETS = {
    "v1.0": {
        "label": "v1.0 舊版 · 英文／日文／舊中文",
        "repo_id": "hexgrad/Kokoro-82M",
        "choices": CHOICES_V10,
        "groups": _voice_groups(CHOICES_V10),
        "default_lang": "🇺🇸 English (US)",
        "default_voice": "af_heart",
        "init_langs": "abjz",
        "subtitle": "v1.0 · 英文／日文較完整 · 24 kHz",
        "voice_info": "舊版 Kokoro-82M：美／英多聲線、5 個日文、8 個舊中文名。切到 v1.1-zh 才有 100 條中文聲線。",
    },
    "v1.1-zh": {
        "label": "v1.1-zh · 中文 100 聲線",
        "repo_id": "hexgrad/Kokoro-82M-v1.1-zh",
        "choices": CHOICES_V11,
        "groups": _voice_groups(CHOICES_V11),
        "default_lang": "🇨🇳 Chinese",
        "default_voice": "zf_001",
        "init_langs": "abz",
        "subtitle": "v1.1-zh · 中文為主 · 24 kHz",
        "voice_info": "v1.1-zh：55 女 + 45 男中文聲線，外加 Maple／Sol／Vale。沒有日文與舊版 Heart／Xiaoxiao 等。",
    },
}

DEFAULT_PRESET = "v1.0"
MODEL_DROPDOWN = [(spec["label"], key) for key, spec in MODEL_PRESETS.items()]

REPO_ID = MODEL_PRESETS[DEFAULT_PRESET]["repo_id"]
CURRENT_PRESET = None
model_instance = None
models = {}
pipelines = {}


def _ensure_unidic():
    """UniDic's pip package does not include MeCab dictionary files; download them once."""
    try:
        import unidic
    except ImportError:
        return
    mecabrc = os.path.join(unidic.DICDIR, 'mecabrc')
    if os.path.isfile(mecabrc):
        return
    print("DEBUG: UniDic dictionary missing. Downloading (this can take several minutes)...")
    import unidic.download
    unidic.download.download_version()
    if not os.path.isfile(mecabrc):
        raise RuntimeError(
            "UniDic dictionary download failed. Run: "
            f"{sys.executable} -m unidic download"
        )
    print("DEBUG: UniDic dictionary download complete.")


def get_pipeline(lang_code):
    """Return (and lazily create) a quiet KPipeline for the voice language."""
    if lang_code in pipelines:
        return pipelines[lang_code]
    extra = LANG_EXTRAS.get(lang_code)
    if lang_code == 'j':
        _ensure_unidic()
    try:
        pipe = KPipeline(lang_code=lang_code, model=False, repo_id=REPO_ID)
        if lang_code == 'z' and not REPO_ID.endswith('/Kokoro-82M'):
            print(f"DEBUG: Chinese G2P is misaki[zh] ZHG2P(version='{ZH_G2P_VERSION}')")
    except ImportError as exc:
        if extra:
            raise ImportError(
                f"Language '{lang_code}' requires: uv pip install 'misaki[{extra}]'"
            ) from exc
        raise
    if lang_code == 'a' and hasattr(pipe, 'g2p') and hasattr(pipe.g2p, 'lexicon'):
        pipe.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
    elif lang_code == 'b' and hasattr(pipe, 'g2p') and hasattr(pipe.g2p, 'lexicon'):
        pipe.g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    pipelines[lang_code] = pipe
    return pipe


def load_checkpoint(preset_id):
    """Load a Kokoro checkpoint and rebuild language pipelines."""
    global REPO_ID, CURRENT_PRESET, model_instance, models, pipelines
    spec = MODEL_PRESETS[preset_id]
    if CURRENT_PRESET == preset_id and model_instance is not None:
        return spec
    print(f"DEBUG: Loading {preset_id} from {spec['repo_id']}...")
    pipelines.clear()
    if model_instance is not None:
        del model_instance
        models.clear()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        model_instance = None
    REPO_ID = spec["repo_id"]
    model_instance = KModel(repo_id=REPO_ID).to(device).eval()
    models = {True: model_instance, False: model_instance}
    CURRENT_PRESET = preset_id
    for lang in spec["init_langs"]:
        try:
            get_pipeline(lang)
            print(f"DEBUG: Pipeline ready for lang_code='{lang}'")
        except Exception as exc:
            extra = LANG_EXTRAS.get(lang, lang)
            print(f"WARN: lang_code='{lang}' unavailable. uv pip install 'misaki[{extra}]' ({exc})")
    print(f"DEBUG: Active model: {preset_id} ({REPO_ID})")
    return spec


print("Loading default model... this may take a moment.")
load_checkpoint(DEFAULT_PRESET)


def forward_device(ps, ref_s, speed):
    """Simplified forward pass that uses the globally loaded model."""
    return models[CUDA_AVAILABLE](ps, ref_s, speed)

# --- TEXT CHUNKING HELPERS (Adapted from Chatter.py) ---

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text.strip())

def remove_inline_reference_numbers(text):
    # Removes bracketed or superscript-style reference numbers (e.g., [1], .”3)
    return re.sub(r'([.!?,\"\'”’)\]])(\d+)(?=\s|$)', r'\1', text)

def replace_letter_period_sequences(text: str) -> str:
    # Converts "J.R.R." to "J R R"
    def replacer(match):
        cleaned = match.group(0).rstrip('.')
        letters = cleaned.split('.')
        return ' '.join(letters)
    return re.sub(r'\b(?:[A-Za-z]\.){2,}', replacer, text)

def split_long_sentence(sentence, max_len=300, seps=None):
    """
    Recursively split a sentence into chunks of <= max_len using a sequence of separators.
    Tries each separator in order, splitting further as needed.
    """
    if seps is None:
        seps = [';', ':', '-', ',', ' ']

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        # Fallback: force split every max_len chars
        return [sentence[i:i + max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)

    if len(parts) == 1:
        # Separator not found, try next separator
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    # Recursively process each part, joining the separator back
    chunks = []
    current = parts[0].strip()
    for part in parts[1:]:
        candidate = (current + sep + part).strip()
        if len(candidate) > max_len:
            chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
            current = part.strip()
        else:
            current = candidate
    
    # Add the last processed part
    if current:
        chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
        
    return [c for c in chunks if c]

def group_sentences(sentences, max_chars=280, joiner=" "):
    """
    Groups sentences into chunks of a specified maximum character length.
    """
    chunks = []
    current_chunk = []
    current_length = 0
    sep_len = len(joiner)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_len = len(sentence)

        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(joiner.join(current_chunk))
            
            # Split the oversized sentence and add its parts as separate chunks
            chunks.extend(split_long_sentence(sentence, max_chars))
            
            current_chunk = []
            current_length = 0
        elif current_length + sentence_len + (sep_len if current_chunk else 0) <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_len + (sep_len if current_chunk else 0)
        else:
            chunks.append(joiner.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len

    if current_chunk:
        chunks.append(joiner.join(current_chunk))

    return chunks


def split_sentences(text, lang_code):
    """Split text into sentences, using CJK punctuation for Japanese/Chinese."""
    if lang_code in ('j', 'z'):
        parts = re.split(r'(?<=[。！？!?．…\n])', text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences or [text.strip()]
    return sent_tokenize(text)


def prepare_text(text, voice, clean_lowercase, clean_whitespace, clean_references, clean_initials):
    """Apply cleaning and chunking appropriate for the selected voice language."""
    lang_code = voice[0]
    english = lang_code in ('a', 'b')
    if clean_lowercase and english:
        text = text.lower()
    if clean_whitespace:
        text = normalize_whitespace(text)
    if clean_references:
        text = remove_inline_reference_numbers(text)
    if clean_initials and english:
        text = replace_letter_period_sequences(text)

    sentences = split_sentences(text, lang_code)
    max_chars = 80 if lang_code in ('j', 'z') else 280
    joiner = '' if lang_code in ('j', 'z') else ' '
    return group_sentences(sentences, max_chars=max_chars, joiner=joiner)


def process_chunk(chunk_text, index, voice, speed):
    """Processes a single chunk of text to generate audio."""
    try:
        pipeline = get_pipeline(voice[0])
        pack = pipeline.load_voice(voice)

        chunk_text = chunk_text.strip()
        if not chunk_text:
            return index, None, None

        processed_chunk = next(pipeline(chunk_text, voice, speed), None)
        if processed_chunk is None:
            log_progress(f"Pipeline returned no output for chunk {index+1}.", "WARN")
            return index, None, None

        _, ps, _ = processed_chunk

        ref_s_index = len(ps) - 1
        if ref_s_index >= len(pack):
            log_progress(f"ref_s index {ref_s_index} out of bounds for pack length {len(pack)} in chunk {index+1}. Using last element.", "WARN")
            ref_s_index = len(pack) - 1
        
        ref_s = pack[ref_s_index]

        audio = forward_device(ps, ref_s, speed)
        return index, audio, str(ps)
    except Exception as e:
        log_progress(f"Audio generation failed for chunk {index+1}: {e}", "ERROR")
        return index, None, None

# --- GENERATION FUNCTIONS ---

def generate_first(text, voice, speed, use_gpu, clean_lowercase, clean_whitespace, clean_references, clean_initials, parallel_chunks):
    """
    Generates audio for the given text. For long text, it splits it into chunks,
    generates audio for each in parallel, and concatenates them.
    """
    log_progress("Generation started.", "INFO")
    text = text.strip()
    if not text:
        log_progress("Input text is empty. Aborting.", "WARN")
        return None, ''

    # --- Apply text cleaning ---
    log_progress("Applying text cleaning options...", "DEBUG")
    log_progress("Splitting text into sentences and grouping into chunks...")
    chunks = prepare_text(
        text, voice, clean_lowercase, clean_whitespace, clean_references, clean_initials
    )
    log_progress(f"Text divided into {len(chunks)} chunks for parallel processing.", "DEBUG")

    results = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=int(parallel_chunks)) as executor:
        log_progress(f"Submitting {len(chunks)} chunks to thread pool ({int(parallel_chunks)} workers)...", "DEBUG")
        futures = [executor.submit(process_chunk, chunk, i, voice, speed) for i, chunk in enumerate(chunks)]
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            index, audio, ps = future.result()
            if audio is not None:
                results[index] = (audio, ps)
            log_progress(f"Completed chunk processing: {completed_count}/{len(chunks)}.", "INFO")

    log_progress("Collating results...", "DEBUG")
    all_audio = [res[0] for res in results if res]
    all_ps = [res[1] for res in results if res]

    if not all_audio:
        log_progress("No audio was generated for any chunk.", "ERROR")
        return None, ''

    log_progress("Concatenating audio chunks...", "INFO")
    silence = torch.zeros(int(24000 * 0.25))  # Silence on CPU
    final_audio_list = []
    for i, audio_chunk in enumerate(all_audio):
        final_audio_list.append(audio_chunk.cpu())
        if i < len(all_audio) - 1:
            final_audio_list.append(silence)

    final_audio = torch.cat(final_audio_list)
    final_ps = "\n---\n".join(all_ps)
    
    log_progress("Generation finished successfully.", "INFO")
    return (24000, final_audio.numpy()), final_ps

def tokenize_first(text, voice, clean_lowercase, clean_whitespace, clean_references, clean_initials):
    text = text.strip()
    if not text:
        return ''
    
    log_progress("Tokenizing first chunk for display...", "DEBUG")
    chunks = prepare_text(
        text, voice, clean_lowercase, clean_whitespace, clean_references, clean_initials
    )
    first_chunk = chunks[0] if chunks else ''

    pipeline = get_pipeline(voice[0])
    for _, ps, _ in pipeline(first_chunk, voice):
        return ps
    return ''

# --- FILE HANDLING (Robustness for Local Run) ---
# We use try/except blocks so the app runs even if you don't have the text files.

def load_text_file(filename, default_text):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as r:
                return r.read().strip()
        except Exception:
            return default_text
    return default_text

def get_random_quote():
    if os.path.exists('en.txt'):
        with open('en.txt', 'r', encoding='utf-8') as r:
            lines = [line.strip() for line in r if line.strip()]
            if lines:
                return random.choice(lines)
    return "Kokoro is an open-weight TTS model with 82 million parameters."

def get_gatsby():
    return load_text_file('gatsby5k.md', "The Great Gatsby text file was not found.")

def get_frankenstein():
    return load_text_file('frankenstein5k.md', "The Frankenstein text file was not found.")


def get_zh_sample():
    return "你好，世界，今天天气真好。"


def load_dropped_text_file(file_path):
    """Load text from a dragged/dropped file into the Single Text input."""
    if not file_path:
        return gr.update()
    path = file_path if isinstance(file_path, str) else str(file_path)
    path = path.strip().strip('"').strip("'")
    if not path or not os.path.isfile(path):
        return gr.update()
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            content = handle.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            content = handle.read()
    except OSError as exc:
        log_progress(f"Failed to load dropped file {path}: {exc}", "ERROR")
        return gr.update()
    log_progress(f"Loaded text from {path} ({len(content)} chars)", "INFO")
    return content

# --- BATCH FILE LIST / CONVERSION ---

TEXT_FILE_SUFFIXES = {'.txt', '.md', '.text', '.markdown'}


def _to_wsl_path(path: str) -> str:
    """Convert a Windows path to a WSL path when running under WSL."""
    # C:\foo\bar or C:/foo/bar -> /mnt/c/foo/bar
    if len(path) >= 3 and path[1] == ':' and path[0].isalpha() and path[2] in '\\/':
        drive = path[0].lower()
        rest = path[3:].replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    # \\wsl$\Distro\home\... already accessible differently; leave as-is
    return path.replace('\\', '/')


def _expand_path_entry(entry: str):
    """Expand one pasted entry into concrete text-file paths.

    Supports:
    - a single file path
    - a directory (all text files inside)
    - a glob pattern (e.g. /path/*.txt)
    """
    import glob as _glob

    entry = _to_wsl_path(entry.strip().strip('"').strip("'"))
    if not entry or entry.startswith('#'):
        return [], None

    # Glob patterns
    if any(ch in entry for ch in '*?['):
        matches = sorted(
            os.path.abspath(p)
            for p in _glob.glob(entry)
            if os.path.isfile(p)
            and Path(p).suffix.lower() in TEXT_FILE_SUFFIXES
        )
        if not matches:
            return [], f"no text files matched: {entry}"
        return matches, None

    abs_entry = os.path.abspath(entry)

    if os.path.isdir(abs_entry):
        files = []
        for child in sorted(Path(abs_entry).iterdir()):
            if child.is_file() and child.suffix.lower() in TEXT_FILE_SUFFIXES:
                files.append(str(child.resolve()))
        if not files:
            for child in sorted(Path(abs_entry).rglob('*')):
                if child.is_file() and child.suffix.lower() in TEXT_FILE_SUFFIXES:
                    files.append(str(child.resolve()))
        if not files:
            return [], f"no text files in folder: {abs_entry}"
        return files, None

    if os.path.isfile(abs_entry):
        suffix = Path(abs_entry).suffix.lower()
        name = Path(abs_entry).name.lower()
        # Manifest files: one path per line
        if (
            suffix in {'.list', '.files'}
            or 'filelist' in name
            or name.endswith('_batch.txt')
            or name == 'batch.txt'
        ):
            expanded = []
            try:
                with open(abs_entry, 'r', encoding='utf-8') as handle:
                    for sub in handle:
                        sub = sub.strip().strip('"').strip("'")
                        if not sub or sub.startswith('#'):
                            continue
                        sub_files, err = _expand_path_entry(sub)
                        if sub_files:
                            expanded.extend(sub_files)
            except OSError as exc:
                return [], f"cannot read manifest {abs_entry}: {exc}"
            return expanded, None if expanded else f"empty manifest: {abs_entry}"

        if suffix and suffix not in TEXT_FILE_SUFFIXES:
            return [], f"unsupported type: {abs_entry}"
        return [abs_entry], None

    return [], f"not found: {entry}"


def _normalize_path_list(paths):
    """Flatten Gradio file values into a clean list of paths."""
    if not paths:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    cleaned = []
    for path in paths:
        if not path:
            continue
        path = _to_wsl_path(str(path).strip().strip('"').strip("'"))
        if path:
            cleaned.append(os.path.abspath(path))
    return cleaned


def format_file_list(file_list):
    if not file_list:
        return "(no files added)"
    return "\n".join(f"{i+1}. {path}" for i, path in enumerate(file_list))


def add_files_to_list(current_list, uploaded_files):
    """Append uploaded files to the batch list (skip duplicates)."""
    current_list = list(current_list or [])
    added = 0
    skipped = 0
    notes = []
    for path in _normalize_path_list(uploaded_files):
        suffix = Path(path).suffix.lower()
        if suffix and suffix not in TEXT_FILE_SUFFIXES:
            skipped += 1
            notes.append(f"unsupported type: {path}")
            continue
        if not os.path.isfile(path):
            skipped += 1
            notes.append(f"not found: {path}")
            continue
        if path in current_list:
            skipped += 1
            notes.append(f"duplicate: {path}")
            continue
        current_list.append(path)
        added += 1
    status = f"Added {added} file(s). Total: {len(current_list)}."
    if skipped:
        status += f" Skipped {skipped}."
        if notes:
            status += "\n" + "\n".join(f"  - {n}" for n in notes[:20])
            if len(notes) > 20:
                status += f"\n  ... and {len(notes) - 20} more"
    return current_list, format_file_list(current_list), status


def add_paths_to_list(current_list, paths_text):
    """Add paths pasted one-per-line (files, folders, or globs)."""
    current_list = list(current_list or [])
    if paths_text is None:
        paths_text = ""
    # Gradio may pass non-str in edge cases
    paths_text = str(paths_text)
    if not paths_text.strip():
        return current_list, format_file_list(current_list), "No paths provided. Paste a file path, folder, or glob (one per line)."

    collected = []
    errors = []
    for line in paths_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        files, err = _expand_path_entry(line)
        if err:
            errors.append(err)
        if files:
            collected.extend(files)

    if not collected and errors:
        return (
            current_list,
            format_file_list(current_list),
            "Nothing added:\n" + "\n".join(f"  - {e}" for e in errors),
        )

    new_list, view, status = add_files_to_list(current_list, collected)
    if errors:
        status += "\nPath notes:\n" + "\n".join(f"  - {e}" for e in errors)
    return new_list, view, status


def clear_file_list():
    return [], format_file_list([]), "File list cleared."


def remove_selected_files(current_list, selected_indices_text):
    """Remove 1-based indices from the list, e.g. '1,3,5' or '2-4'."""
    current_list = list(current_list or [])
    if not current_list:
        return current_list, format_file_list(current_list), "List is empty."
    if not selected_indices_text or not selected_indices_text.strip():
        return current_list, format_file_list(current_list), "Enter indices to remove (e.g. 1,3 or 2-4)."

    to_remove = set()
    for token in selected_indices_text.replace(' ', '').split(','):
        if not token:
            continue
        if '-' in token:
            try:
                start_s, end_s = token.split('-', 1)
                start, end = int(start_s), int(end_s)
                to_remove.update(range(start, end + 1))
            except ValueError:
                return current_list, format_file_list(current_list), f"Invalid range: {token}"
        else:
            try:
                to_remove.add(int(token))
            except ValueError:
                return current_list, format_file_list(current_list), f"Invalid index: {token}"

    new_list = [path for i, path in enumerate(current_list, start=1) if i not in to_remove]
    removed = len(current_list) - len(new_list)
    return new_list, format_file_list(new_list), f"Removed {removed} file(s)."


EXPORT_SAMPLE_RATE = 24000
EXPORT_BITRATE = "32k"
COMPRESSED_FORMATS = {"mp3", "aac"}


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _to_mono(audio_data):
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=-1)
    return np.squeeze(audio_data)


def save_audio_file(path_no_ext, sample_rate, audio_data, audio_format="mp3"):
    """Write 24 kHz mono audio as MP3, AAC, or WAV.

    Compressed formats use 32 kbps. WAV is 16-bit PCM at 24 kHz mono.
    """
    audio_format = (audio_format or "mp3").lower().lstrip(".")
    if audio_format not in {"mp3", "aac", "wav"}:
        raise ValueError(f"Unsupported audio format: {audio_format}")

    audio_data = _to_mono(audio_data)
    out_path = f"{path_no_ext}.{audio_format}"

    if audio_format == "wav" and not _ffmpeg_available():
        sf.write(out_path, audio_data, EXPORT_SAMPLE_RATE if sample_rate == EXPORT_SAMPLE_RATE else sample_rate)
        return out_path

    if audio_format in COMPRESSED_FORMATS and not _ffmpeg_available():
        raise RuntimeError(
            f"{audio_format.upper()} export requires ffmpeg on PATH. "
            "Install ffmpeg or choose WAV."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        sf.write(tmp_wav, audio_data, int(sample_rate))
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", tmp_wav,
            "-ar", str(EXPORT_SAMPLE_RATE),
            "-ac", "1",
        ]
        if audio_format == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-b:a", EXPORT_BITRATE])
        elif audio_format == "aac":
            cmd.extend(["-codec:a", "aac", "-b:a", EXPORT_BITRATE, "-f", "adts"])
        else:
            cmd.extend(["-codec:a", "pcm_s16le"])
        cmd.append(out_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
            raise RuntimeError(f"ffmpeg {audio_format.upper()} encode failed: {detail}")
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass
    return out_path


def generate_with_export(
    text,
    voice,
    speed,
    use_gpu,
    clean_lowercase,
    clean_whitespace,
    clean_references,
    clean_initials,
    parallel_chunks,
    audio_format,
):
    """Generate audio for playback and also export a downloadable audio file."""
    audio, ps = generate_first(
        text,
        voice,
        speed,
        use_gpu,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
        parallel_chunks,
    )
    if audio is None:
        return None, '', None

    sample_rate, audio_data = audio
    export_dir = tempfile.mkdtemp(prefix="kokoro_export_")
    try:
        export_path = save_audio_file(
            os.path.join(export_dir, "kokoro_output"),
            sample_rate,
            audio_data,
            audio_format,
        )
    except Exception as exc:
        log_progress(f"Export failed: {exc}", "ERROR")
        return audio, ps, None
    return audio, ps, export_path


def batch_convert(
    file_list,
    voice,
    speed,
    use_gpu,
    clean_lowercase,
    clean_whitespace,
    clean_references,
    clean_initials,
    parallel_chunks,
    output_dir,
    audio_format,
    progress=gr.Progress(track_tqdm=False),
):
    """Convert every text file in the list using current voice and export settings."""
    empty_player = gr.update(choices=[], value=None)
    file_list = list(file_list or [])
    if not file_list:
        return "No files in the list. Add files first.", None, empty_player, None, []

    audio_format = (audio_format or "mp3").lower().lstrip(".")
    if audio_format in COMPRESSED_FORMATS and not _ffmpeg_available():
        return (
            f"{audio_format.upper()} export requires ffmpeg on PATH. "
            "Install ffmpeg or choose WAV.",
            None,
            empty_player,
            None,
            [],
        )

    output_dir = (output_dir or "").strip() or "batch_output"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_progress(
        f"Batch conversion started: {len(file_list)} file(s) → {output_dir} ({audio_format})",
        "INFO",
    )
    written = []
    errors = []

    for i, path in enumerate(file_list):
        name = os.path.basename(path)
        progress((i) / len(file_list), desc=f"Converting {name}")
        log_progress(f"Batch [{i+1}/{len(file_list)}]: {path}", "INFO")

        if not os.path.isfile(path):
            errors.append(f"{name}: file not found")
            continue

        try:
            with open(path, 'r', encoding='utf-8') as handle:
                text = handle.read()
        except Exception as exc:
            errors.append(f"{name}: read failed ({exc})")
            continue

        if not text.strip():
            errors.append(f"{name}: empty file")
            continue

        try:
            audio, _ps = generate_first(
                text,
                voice,
                speed,
                use_gpu,
                clean_lowercase,
                clean_whitespace,
                clean_references,
                clean_initials,
                parallel_chunks,
            )
        except Exception as exc:
            errors.append(f"{name}: generation failed ({exc})")
            log_progress(f"Batch failed for {name}: {exc}", "ERROR")
            continue

        if audio is None:
            errors.append(f"{name}: no audio generated")
            continue

        sample_rate, audio_data = audio
        stem = Path(path).stem
        path_no_ext = os.path.join(output_dir, stem)
        # Avoid overwriting when multiple inputs share a stem
        if os.path.exists(f"{path_no_ext}.{audio_format}"):
            path_no_ext = os.path.join(output_dir, f"{stem}_{i+1}")
        try:
            out_path = save_audio_file(path_no_ext, sample_rate, audio_data, audio_format)
            written.append(out_path)
            log_progress(f"Wrote {out_path}", "INFO")
        except Exception as exc:
            errors.append(f"{name}: write failed ({exc})")

    progress(1.0, desc="Done")
    status_lines = [
        f"Batch complete: {len(written)}/{len(file_list)} succeeded.",
        f"Format: {audio_format.upper()}",
        f"Output folder: {output_dir}",
    ]
    if written:
        status_lines.append("Files:")
        status_lines.extend(f"  ✓ {p}" for p in written)
    if errors:
        status_lines.append("Errors:")
        status_lines.extend(f"  ✗ {e}" for e in errors)

    status = "\n".join(status_lines)
    log_progress(status, "INFO" if not errors else "WARN")

    if not written:
        return status, None, empty_player, None, []

    choices = [(os.path.basename(p), p) for p in written]
    first = written[0]
    player_update = gr.update(choices=choices, value=first)
    return status, written, player_update, first, written


def play_batch_audio(selected_path):
    """Load a completed batch output into the audio player."""
    if not selected_path:
        return None
    path = str(selected_path)
    if not os.path.isfile(path):
        log_progress(f"Batch play: file not found: {path}", "WARN")
        return None
    return path


def step_batch_audio(selected_path, written_files, direction):
    """Move to previous/next completed batch file."""
    written_files = list(written_files or [])
    if not written_files:
        return gr.update(), None
    try:
        idx = written_files.index(selected_path) if selected_path in written_files else 0
    except ValueError:
        idx = 0
    idx = (idx + int(direction)) % len(written_files)
    path = written_files[idx]
    return gr.update(value=path), path

# --- UI CONFIGURATION ---

def voices_for_language(lang):
    spec = MODEL_PRESETS[CURRENT_PRESET or DEFAULT_PRESET]
    groups = spec["groups"]
    items = groups.get(lang) or list(spec["choices"].items())
    return gr.update(choices=items, value=items[0][1])


def switch_model(preset_id):
    """Swap checkpoint and refresh language/voice dropdowns."""
    spec = load_checkpoint(preset_id)
    groups = spec["groups"]
    header = f"# Kokoro TTS\n{spec['subtitle']} · **{HARDWARE_BADGE}**"
    return (
        header,
        gr.update(choices=list(groups.keys()), value=spec["default_lang"]),
        gr.update(
            choices=groups[spec["default_lang"]],
            value=spec["default_voice"],
            info=spec["voice_info"],
        ),
    )


TOKEN_NOTE = '''💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`
⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`
⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)'''

# --- THEME / CSS / JS ---

KOKORO_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("DM Sans"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca 0%, #6d28d9 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    shadow_drop="0 8px 24px rgba(15, 23, 42, 0.08)",
    shadow_drop_lg="0 16px 40px rgba(15, 23, 42, 0.12)",
)

APP_CSS = """
:root {
  --kokoro-radius: 16px;
  --kokoro-gap: 14px;
}
.gradio-container {
  max-width: 1280px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 18px 18px 32px !important;
  font-feature-settings: "ss01", "kern";
}
.app-header {
  align-items: center !important;
  margin-bottom: 8px;
}
.brand-title h1, .brand-title h2 {
  margin: 0 !important;
  letter-spacing: -0.03em;
  font-weight: 700 !important;
}
.brand-title p {
  margin: 4px 0 0 !important;
  opacity: 0.72;
  font-size: 0.95rem;
}
.theme-toggle {
  justify-content: flex-end !important;
  align-items: center !important;
}
.theme-toggle button {
  min-width: 88px;
  border-radius: 999px !important;
}
.settings-panel, .main-panel, .batch-card {
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: var(--kokoro-radius) !important;
  padding: 14px 14px 8px !important;
  box-shadow: var(--shadow-drop);
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 0.85rem;
  font-weight: 600;
  border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill);
}
.hw-ok { color: #059669; }
.hw-cpu { color: #d97706; }
.gradio-container .tabs {
  margin-top: 4px;
}
button.primary, .generate-row button {
  border-radius: 12px !important;
}
#theme-light, #theme-dark {
  border-radius: 999px !important;
}
.dark #theme-dark,
#theme-light {
  font-weight: 700 !important;
}
.dark #theme-light {
  font-weight: 500 !important;
}
.dark .settings-panel,
.dark .main-panel,
.dark .batch-card {
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
}
footer { display: none !important; }
"""

APP_JS = """
function initKokoroTheme() {
  const apply = (mode) => {
    try { localStorage.setItem('kokoro-theme', mode); } catch (e) {}
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== mode) {
      url.searchParams.set('__theme', mode);
      window.location.replace(url.href);
    }
  };
  try {
    const url = new URL(window.location);
    const current = url.searchParams.get('__theme');
    if (!current) {
      const saved = localStorage.getItem('kokoro-theme');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      apply(saved || (prefersDark ? 'dark' : 'light'));
    } else {
      localStorage.setItem('kokoro-theme', current);
    }
  } catch (e) {}
  return 'theme-ready';
}
"""

THEME_LIGHT_JS = """
() => {
  try { localStorage.setItem('kokoro-theme', 'light'); } catch (e) {}
  const url = new URL(window.location);
  url.searchParams.set('__theme', 'light');
  window.location.href = url.href;
}
"""

THEME_DARK_JS = """
() => {
  try { localStorage.setItem('kokoro-theme', 'dark'); } catch (e) {}
  const url = new URL(window.location);
  url.searchParams.set('__theme', 'dark');
  window.location.href = url.href;
}
"""

HARDWARE_BADGE = "CUDA GPU" if CUDA_AVAILABLE else "CPU"
_startup = MODEL_PRESETS[DEFAULT_PRESET]

# --- GRADIO INTERFACE ---

with gr.Blocks(title="Kokoro TTS") as app:
    with gr.Row(elem_classes="app-header"):
        with gr.Column(scale=5, elem_classes="brand-title"):
            header_md = gr.Markdown(
                f"# Kokoro TTS\n"
                f"{_startup['subtitle']} · **{HARDWARE_BADGE}**"
            )
        with gr.Column(scale=2, min_width=220, elem_classes="theme-toggle"):
            with gr.Row():
                light_btn = gr.Button("Light", elem_id="theme-light", variant="secondary")
                dark_btn = gr.Button("Dark", elem_id="theme-dark", variant="secondary")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=280, elem_classes="settings-panel"):
            gr.Markdown("### Voice")
            model_select = gr.Dropdown(
                MODEL_DROPDOWN,
                value=DEFAULT_PRESET,
                label="Model",
                info="舊版 v1.0 支援較多英文與日文；v1.1-zh 專精中文 100 聲線。切換會重新載入權重。",
            )
            voice_lang = gr.Dropdown(
                list(_startup["groups"].keys()),
                value=_startup["default_lang"],
                label="Language",
            )
            voice = gr.Dropdown(
                _startup["groups"][_startup["default_lang"]],
                value=_startup["default_voice"],
                label='Voice',
                filterable=True,
                info=_startup["voice_info"],
            )
            speed = gr.Slider(
                minimum=0.5, maximum=2, value=1, step=0.1, label='Speed',
                info='Duration scale. Values outside 0.7–1.3 may distort prosody.',
            )
            audio_format = gr.Dropdown(
                choices=['mp3', 'aac', 'wav'],
                value='mp3',
                label='Output Format',
                info='24 kHz / mono / 32 kbps (WAV is 16-bit PCM). MP3 and AAC need ffmpeg.',
            )
            use_gpu = gr.Dropdown(
                [('GPU (Detected)' if CUDA_AVAILABLE else 'GPU (Not Found)', True), ('CPU', False)],
                value=CUDA_AVAILABLE,
                label='Hardware',
                interactive=False,
            )
            with gr.Accordion("Text cleaning", open=False):
                clean_lowercase = gr.Checkbox(label="Convert to lowercase", value=True)
                clean_whitespace = gr.Checkbox(label="Normalize whitespace", value=True)
                clean_references = gr.Checkbox(label="Remove reference numbers", value=True)
                clean_initials = gr.Checkbox(label="Format initials (J.R.R.)", value=True)
            with gr.Accordion("Performance", open=False):
                parallel_chunks = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Parallel chunks",
                    info="Lower this if you run out of memory",
                )

        with gr.Column(scale=3, elem_classes="main-panel"):
            with gr.Tabs():
                with gr.Tab("Single"):
                    single_file = gr.File(
                        label='Drop a .txt or .md file',
                        file_count='single',
                        file_types=['.txt', '.md', '.text', '.markdown'],
                        type='filepath',
                    )
                    text = gr.Textbox(
                        label='Input text',
                        lines=8,
                        value="Hello, this is Kokoro. Choose v1.0 for English and Japanese, or v1.1-zh for Chinese voices.",
                    )
                    with gr.Row():
                        zh_sample_btn = gr.Button('中文示例', variant='secondary')
                        random_btn = gr.Button('Random quote', variant='secondary')
                        gatsby_btn = gr.Button('Gatsby', variant='secondary')
                        frankenstein_btn = gr.Button('Frankenstein', variant='secondary')
                    with gr.Row(elem_classes="generate-row"):
                        generate_btn = gr.Button('Generate', variant='primary')
                        stop_generate_btn = gr.Button('Stop', variant='stop')
                    out_audio = gr.Audio(label='Output', interactive=False)
                    out_file = gr.File(label='Download', interactive=False)
                    with gr.Accordion('Phonemes / tokens', open=False):
                        out_ps = gr.Textbox(interactive=False, show_label=False, lines=4)
                        tokenize_btn = gr.Button('Tokenize', variant='secondary')
                        gr.Markdown(TOKEN_NOTE)

                with gr.Tab("Batch"):
                    gr.Markdown(
                        "Add `.txt` / `.md` files by upload, folder path, or glob "
                        "(`~/text/*.txt`). Uses the voice settings on the left."
                    )
                    batch_files_state = gr.State([])
                    batch_results_state = gr.State([])
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes="batch-card"):
                            batch_upload = gr.File(
                                label='Add text files',
                                file_count='multiple',
                                file_types=['.txt', '.md', '.text', '.markdown'],
                                type='filepath',
                            )
                            batch_paths = gr.Textbox(
                                label='File or folder paths (one per line)',
                                lines=4,
                                placeholder='/home/kuoping/text\n/path/to/chapter1.txt\n/path/to/*.md',
                            )
                            with gr.Row():
                                add_paths_btn = gr.Button('Add paths', variant='secondary')
                                clear_list_btn = gr.Button('Clear', variant='secondary')
                            batch_file_list_view = gr.Textbox(
                                label='Queue',
                                lines=8,
                                value=format_file_list([]),
                                interactive=False,
                            )
                            with gr.Row():
                                remove_indices = gr.Textbox(
                                    label='Remove by index',
                                    placeholder='1,3 or 2-4',
                                    scale=3,
                                )
                                remove_btn = gr.Button('Remove', variant='secondary', scale=1)

                        with gr.Column(elem_classes="batch-card"):
                            batch_output_dir = gr.Textbox(
                                label='Output folder',
                                value=str(Path('batch_output').resolve()),
                            )
                            with gr.Row():
                                batch_convert_btn = gr.Button('Batch convert', variant='primary')
                                stop_batch_btn = gr.Button('Stop', variant='stop')
                            batch_status = gr.Textbox(label='Status', lines=8, interactive=False)
                            batch_outputs = gr.File(
                                label='Generated files',
                                file_count='multiple',
                                interactive=False,
                            )
                            with gr.Row():
                                batch_play_select = gr.Dropdown(
                                    label='Preview',
                                    choices=[],
                                    value=None,
                                    interactive=True,
                                )
                                batch_prev_btn = gr.Button('Prev', variant='secondary', scale=0)
                                batch_next_btn = gr.Button('Next', variant='secondary', scale=0)
                            batch_audio = gr.Audio(
                                label='Player',
                                interactive=False,
                                autoplay=True,
                            )

    # Event Handlers — Single Text
    model_select.change(
        fn=switch_model,
        inputs=[model_select],
        outputs=[header_md, voice_lang, voice],
    )
    voice_lang.change(fn=voices_for_language, inputs=[voice_lang], outputs=[voice])
    light_btn.click(fn=None, js=THEME_LIGHT_JS)
    dark_btn.click(fn=None, js=THEME_DARK_JS)
    single_file.change(fn=load_dropped_text_file, inputs=[single_file], outputs=[text])
    zh_sample_btn.click(fn=get_zh_sample, inputs=[], outputs=[text])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])

    generation_inputs = [
        text,
        voice,
        speed,
        use_gpu,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
        parallel_chunks,
        audio_format,
    ]

    tokenization_inputs = [
        text,
        voice,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
    ]

    generation_event = generate_btn.click(
        fn=generate_with_export,
        inputs=generation_inputs,
        outputs=[out_audio, out_ps, out_file],
    )
    tokenize_btn.click(fn=tokenize_first, inputs=tokenization_inputs, outputs=[out_ps])
    stop_generate_btn.click(fn=None, cancels=generation_event)

    # Event Handlers — Batch Files
    batch_upload.change(
        fn=add_files_to_list,
        inputs=[batch_files_state, batch_upload],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )
    add_paths_btn.click(
        fn=add_paths_to_list,
        inputs=[batch_files_state, batch_paths],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )
    clear_list_btn.click(
        fn=clear_file_list,
        inputs=[],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )
    remove_btn.click(
        fn=remove_selected_files,
        inputs=[batch_files_state, remove_indices],
        outputs=[batch_files_state, batch_file_list_view, batch_status],
    )

    batch_inputs = [
        batch_files_state,
        voice,
        speed,
        use_gpu,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
        parallel_chunks,
        batch_output_dir,
        audio_format,
    ]
    batch_event = batch_convert_btn.click(
        fn=batch_convert,
        inputs=batch_inputs,
        outputs=[batch_status, batch_outputs, batch_play_select, batch_audio, batch_results_state],
    )
    stop_batch_btn.click(fn=None, cancels=batch_event)
    batch_play_select.change(
        fn=play_batch_audio,
        inputs=[batch_play_select],
        outputs=[batch_audio],
    )
    batch_prev_btn.click(
        fn=lambda path, files: step_batch_audio(path, files, -1),
        inputs=[batch_play_select, batch_results_state],
        outputs=[batch_play_select, batch_audio],
    )
    batch_next_btn.click(
        fn=lambda path, files: step_batch_audio(path, files, 1),
        inputs=[batch_play_select, batch_results_state],
        outputs=[batch_play_select, batch_audio],
    )

def _find_free_port(preferred=7860, fallback_start=17860, tries=100):
    """Pick a bindable localhost port.

    On some WSL/Windows setups, Gradio's default range (e.g. 7860+) is
    reserved even when nothing is listening, so fall back to higher ports.
    """
    import socket

    candidates = [preferred, *range(fallback_start, fallback_start + tries)]
    for port in candidates:
        with socket.socket() as sock:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(
        f"Cannot find an empty port among {preferred} or "
        f"{fallback_start}-{fallback_start + tries - 1}."
    )


if __name__ == '__main__':
    # Launch in browser automatically. Prefer GRADIO_SERVER_PORT when it can bind;
    # otherwise pick a free port (WSL/Windows often reserves Gradio's 7860 range).
    import signal
    import socket

    env_port = os.getenv("GRADIO_SERVER_PORT")
    server_port = None
    if env_port:
        candidate = int(env_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", candidate))
            server_port = candidate
        except OSError:
            print(
                f"WARN: Port {candidate} from GRADIO_SERVER_PORT is unavailable; "
                "selecting a free port instead."
            )
    if server_port is None:
        server_port = _find_free_port()
    print(f"DEBUG: Launching Gradio on port {server_port}")
    # Allow returning batch outputs from folders outside the project dir
    # (Gradio only serves files from cwd/temp unless listed here).
    allowed_paths = [
        str(Path.cwd().resolve()),
        str(Path.home().resolve()),
        tempfile.gettempdir(),
    ]

    # First Ctrl-C warns; second Ctrl-C force-exits (Gradio/uvicorn often swallows SIGINT).
    _interrupt_count = {"n": 0}

    def _handle_interrupt(signum, frame):
        _interrupt_count["n"] += 1
        if _interrupt_count["n"] == 1:
            print("\nPress Ctrl-C again to close the app.", flush=True)
            return
        print("\nClosing.", flush=True)
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_interrupt)

    print("DEBUG: Press Ctrl-C twice to close the app.")
    app.queue().launch(
        inbrowser=True,
        server_port=server_port,
        allowed_paths=allowed_paths,
        theme=KOKORO_THEME,
        css=APP_CSS,
        js=APP_JS,
    )