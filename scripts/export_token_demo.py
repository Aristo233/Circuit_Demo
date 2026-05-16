#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import wave
import zlib
from binascii import crc32
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_transparency_tool.server.dataset_utils import extract_abc_segment, load_dataset_file


DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "demo_page" / "assets" / "token-demo"
DEFAULT_BROWSER = Path("/usr/bin/google-chrome")
DEFAULT_CONTRIBUTION_THRESHOLD = 0.038


@dataclass(frozen=True)
class DemoSample:
    id: str
    label: str
    backbone: str
    iteration: int
    dataset_file: str
    line_index: int


SAMPLES: tuple[DemoSample, ...] = (
    DemoSample(
        id="i2-iteration-10025",
        label="I-II-V-I / Iteration 10025",
        backbone="I-II-V-I",
        iteration=10025,
        dataset_file="dataset/I-II-V-I/sample_input_nottingham.txt",
        line_index=2,
    ),
    DemoSample(
        id="i4-iteration-30008",
        label="I-IV-V-I / Iteration 30008",
        backbone="I-IV-V-I",
        iteration=30008,
        dataset_file="dataset/I-IV-V-I/sample_input_nottingham.txt",
        line_index=10,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export static token-click screenshots from the Streamlit demo."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--browser-executable", type=Path, default=DEFAULT_BROWSER)
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu", "auto"],
        default="gpu",
        help="Preferred Streamlit device. The default forces the app device selector to gpu when CUDA is visible.",
    )
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--ld-preload-stub", default="/tmp/libittnotify_stub.so")
    parser.add_argument("--startup-timeout", type=float, default=420.0)
    parser.add_argument("--after-click-ms", type=int, default=1800)
    parser.add_argument("--viewport-width", type=int, default=2200)
    parser.add_argument("--viewport-height", type=int, default=2200)
    parser.add_argument(
        "--contribution-threshold",
        type=float,
        default=DEFAULT_CONTRIBUTION_THRESHOLD,
        help="Contribution graph threshold to force in Streamlit during export.",
    )
    parser.add_argument(
        "--sample",
        dest="sample_ids",
        action="append",
        choices=[sample.id for sample in SAMPLES],
        help="Export only this sample id. May be repeated.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Export only the first two tokens for each selected sample.",
    )
    parser.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=0,
        help="Optional hard limit for token screenshots per sample.",
    )
    parser.add_argument(
        "--keep-streamlit-logs",
        action="store_true",
        help="Print Streamlit stdout/stderr after each sample finishes.",
    )
    return parser.parse_args()


def selected_samples(sample_ids: Optional[List[str]]) -> List[DemoSample]:
    if not sample_ids:
        return list(SAMPLES)
    wanted = set(sample_ids)
    return [sample for sample in SAMPLES if sample.id in wanted]


def resolve_project_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def read_sample_sentence(sample: DemoSample) -> str:
    rows = load_dataset_file(str(resolve_project_path(sample.dataset_file)))
    if sample.line_index < 0 or sample.line_index >= len(rows):
        raise IndexError(
            f"{sample.id}: line_index={sample.line_index} is outside "
            f"{sample.dataset_file} with {len(rows)} rows"
        )
    sentence = rows[sample.line_index]
    marker = f"Iteration {sample.iteration}"
    if marker not in sentence:
        raise ValueError(f"{sample.id}: expected '{marker}' in selected dataset row")
    return sentence


KEY_SIGNATURES: Dict[str, Dict[str, int]] = {
    "C": {},
    "G": {"F": 1},
    "D": {"F": 1, "C": 1},
    "A": {"F": 1, "C": 1, "G": 1},
    "E": {"F": 1, "C": 1, "G": 1, "D": 1},
    "B": {"F": 1, "C": 1, "G": 1, "D": 1, "A": 1},
    "F#": {"F": 1, "C": 1, "G": 1, "D": 1, "A": 1, "E": 1},
    "C#": {"F": 1, "C": 1, "G": 1, "D": 1, "A": 1, "E": 1, "B": 1},
    "F": {"B": -1},
    "Bb": {"B": -1, "E": -1},
    "Eb": {"B": -1, "E": -1, "A": -1},
    "Ab": {"B": -1, "E": -1, "A": -1, "D": -1},
    "Db": {"B": -1, "E": -1, "A": -1, "D": -1, "G": -1},
    "Gb": {"B": -1, "E": -1, "A": -1, "D": -1, "G": -1, "C": -1},
    "Cb": {"B": -1, "E": -1, "A": -1, "D": -1, "G": -1, "C": -1, "F": -1},
}

NOTE_OFFSETS = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def abc_header_value(abc_text: str, prefix: str) -> Optional[str]:
    for line in abc_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return None


def parse_fraction(value: Optional[str], fallback: float) -> float:
    if not value:
        return fallback
    raw = value.strip().split()[0]
    with contextlib.suppress(ValueError, ZeroDivisionError):
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            return float(numerator or "1") / float(denominator)
        return float(raw)
    return fallback


def parse_default_note_seconds(abc_text: str) -> float:
    default_note = parse_fraction(abc_header_value(abc_text, "L:"), 1.0 / 8.0)
    beat_note = 1.0 / 4.0
    bpm = 120.0
    q_value = abc_header_value(abc_text, "Q:")
    if q_value:
        if "=" in q_value:
            left, right = q_value.split("=", 1)
            beat_note = parse_fraction(left, beat_note)
            with contextlib.suppress(ValueError):
                bpm = float(right.strip())
        else:
            with contextlib.suppress(ValueError):
                bpm = float(q_value.strip())
    return (60.0 / bpm) * (default_note / beat_note)


def parse_key_signature(abc_text: str) -> Dict[str, int]:
    key_value = abc_header_value(abc_text, "K:") or "C"
    key = key_value.split()[0].strip()
    key = key.replace("maj", "").replace("major", "")
    key = key[:1].upper() + key[1:]
    return KEY_SIGNATURES.get(key, {})


def parse_abc_duration(body: str, start: int) -> tuple[float, int]:
    i = start
    numerator = ""
    while i < len(body) and body[i].isdigit():
        numerator += body[i]
        i += 1

    duration = float(numerator) if numerator else 1.0
    if i < len(body) and body[i] == "/":
        slash_count = 0
        while i < len(body) and body[i] == "/":
            slash_count += 1
            i += 1
        denominator = ""
        while i < len(body) and body[i].isdigit():
            denominator += body[i]
            i += 1
        if denominator:
            duration /= float(denominator)
        else:
            duration /= 2.0 ** slash_count

    return duration, i


def abc_note_to_midi(note: str, accidental: int, explicit_accidental: bool, key_signature: Dict[str, int], octave_marks: str) -> int:
    name = note.upper()
    octave = 5 if note.islower() else 4
    midi = 12 * (octave + 1) + NOTE_OFFSETS[name]
    if explicit_accidental:
        midi += accidental
    else:
        midi += key_signature.get(name, 0)
    midi += 12 * octave_marks.count("'")
    midi -= 12 * octave_marks.count(",")
    return midi


def abc_to_note_events(abc_text: str) -> List[tuple[Optional[float], float]]:
    body_span = abc_text.split("K:", 1)
    body = body_span[1].split("\n", 1)[1] if len(body_span) == 2 and "\n" in body_span[1] else abc_text
    key_signature = parse_key_signature(abc_text)
    base_seconds = parse_default_note_seconds(abc_text)
    events: List[tuple[Optional[float], float]] = []

    i = 0
    while i < len(body):
        char = body[i]
        if char == '"':
            i += 1
            while i < len(body) and body[i] != '"':
                i += 1
            i += 1
            continue
        if char in "^_=" or char.upper() in NOTE_OFFSETS or char in "zZ":
            accidental = 0
            explicit_accidental = False
            while i < len(body) and body[i] in "^_=":
                explicit_accidental = True
                if body[i] == "^":
                    accidental += 1
                elif body[i] == "_":
                    accidental -= 1
                else:
                    accidental = 0
                i += 1
            if i >= len(body):
                break

            note = body[i]
            if note in "zZ":
                i += 1
                duration, i = parse_abc_duration(body, i)
                events.append((None, max(0.06, duration * base_seconds)))
                continue
            if note.upper() not in NOTE_OFFSETS:
                i += 1
                continue

            i += 1
            octave_marks = ""
            while i < len(body) and body[i] in "',":
                octave_marks += body[i]
                i += 1
            duration, i = parse_abc_duration(body, i)
            midi = abc_note_to_midi(note, accidental, explicit_accidental, key_signature, octave_marks)
            frequency = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            events.append((frequency, max(0.06, duration * base_seconds)))
            continue
        i += 1

    return events


def piano_harmonic_sample(frequency: float, t: float, duration: float) -> float:
    attack = min(1.0, t / 0.008)
    release = 1.0 if t <= duration else math.exp(-(t - duration) / 0.18)
    base_decay = max(0.32, min(1.15, 1.25 - (frequency - 220.0) / 1600.0))
    partials = (
        (1, 1.00),
        (2, 0.50),
        (3, 0.24),
        (4, 0.12),
        (5, 0.075),
        (6, 0.045),
        (8, 0.022),
    )
    sample = 0.0
    for harmonic, weight in partials:
        inharmonicity = 1.0 + 0.00045 * harmonic * harmonic
        partial_frequency = frequency * harmonic * inharmonicity
        partial_decay = math.exp(-t / max(0.06, base_decay / (harmonic ** 0.72)))
        sample += weight * partial_decay * math.sin(2.0 * math.pi * partial_frequency * t)

    hammer = math.exp(-t / 0.012) * math.sin(2.0 * math.pi * frequency * 12.0 * t)
    return attack * release * (sample + 0.055 * hammer)


def write_score_audio(output_path: Path, sentence: str) -> None:
    abc_text = extract_abc_segment(sentence)
    if not abc_text:
        raise RuntimeError("Cannot generate score audio: selected sentence has no ABC segment")

    events = abc_to_note_events(abc_text)
    if not events:
        raise RuntimeError("Cannot generate score audio: ABC segment has no note events")

    sample_rate = 44100
    release_tail_seconds = 0.42
    master_gain = 0.34
    output_path.parent.mkdir(parents=True, exist_ok=True)

    starts: List[float] = []
    cursor = 0.0
    for _frequency, duration in events:
        starts.append(cursor)
        cursor += duration

    total_seconds = cursor + release_tail_seconds
    total_frames = max(1, int(math.ceil(total_seconds * sample_rate)))
    mix = [0.0] * total_frames

    for (frequency, duration), start_seconds in zip(events, starts):
        if frequency is None:
            continue
        start_frame = int(start_seconds * sample_rate)
        note_frames = max(1, int(math.ceil((duration + release_tail_seconds) * sample_rate)))
        for frame in range(note_frames):
            target = start_frame + frame
            if target >= total_frames:
                break
            t = frame / sample_rate
            mix[target] += piano_harmonic_sample(frequency, t, duration)

    peak = max((abs(value) for value in mix), default=1.0)
    scale = min(master_gain / peak, master_gain) if peak > 0 else master_gain

    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for value in mix:
            sample = max(-1.0, min(1.0, value * scale))
            frames.extend(int(sample * 32767).to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(frames)


def write_temp_streamlit_config(tmpdir: Path, sample: DemoSample, sentence: str) -> Path:
    dataset_path = tmpdir / f"{sample.id}.txt"
    dataset_path.write_text(sentence.replace("\n", "\\n") + "\n", encoding="utf-8")
    config_path = tmpdir / f"{sample.id}.json"
    config = {
        "allow_loading_dataset_files": False,
        "max_user_string_length": 100,
        "preloaded_dataset_filename": str(dataset_path),
        "debug": False,
        "demo_mode": True,
        "models": {
            "m-a-p/ChatMusician": None,
        },
        "default_model": "m-a-p/ChatMusician",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def build_streamlit_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.device != "auto":
        env["LLMTT_DEFAULT_DEVICE"] = str(args.device)
        env["LLMTT_FORCE_DEVICE"] = str(args.device)
    env["LLMTT_HIDE_SCORE_AUDIO"] = "1"
    env["LLMTT_CONTRIBUTION_THRESHOLD"] = f"{args.contribution_threshold:.3f}"
    stub = Path(args.ld_preload_stub)
    if stub.exists():
        current = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = f"{stub}:{current}" if current else str(stub)
    elif args.ld_preload_stub:
        print(f"[export] LD_PRELOAD stub not found: {stub}", file=sys.stderr)
    return env


def wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {host}:{port}") from last_error


def verify_requested_device(args: argparse.Namespace) -> None:
    if args.device != "gpu":
        return
    cmd = [
        str(args.python),
        "-c",
        (
            "import os, torch; "
            "print('CUDA_VISIBLE_DEVICES=' + str(os.environ.get('CUDA_VISIBLE_DEVICES'))); "
            "print('cuda_available=' + str(torch.cuda.is_available())); "
            "print('cuda_device_count=' + str(torch.cuda.device_count())); "
            "raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"
        ),
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=build_streamlit_env(args),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise SystemExit(
            "GPU export was requested, but the llmtt subprocess cannot see CUDA.\n"
            f"{result.stdout.strip()}\n"
            "Try running with a different GPU, for example `--cuda-visible-devices 0`, "
            "or use `--device cpu` if GPU is unavailable."
        )
    print("[export] GPU check passed", flush=True)
    print(result.stdout.strip(), flush=True)


@contextlib.contextmanager
def streamlit_server(args: argparse.Namespace, config_path: Path) -> Iterable[subprocess.Popen[str]]:
    cmd = [
        str(args.python),
        "-m",
        "streamlit",
        "run",
        "llm_transparency_tool/server/app.py",
        f"--server.port={args.port}",
        f"--server.address={args.host}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
        "--",
        str(config_path),
    ]
    env = build_streamlit_env(args)
    print(
        "[export] starting Streamlit with "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r} "
        f"LLMTT_FORCE_DEVICE={env.get('LLMTT_FORCE_DEVICE')!r} "
        f"LLMTT_CONTRIBUTION_THRESHOLD={env.get('LLMTT_CONTRIBUTION_THRESHOLD')!r} "
        f"LD_PRELOAD={env.get('LD_PRELOAD', '')!r}",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=None if args.keep_streamlit_logs else subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_port(args.host, args.port, args.startup_timeout)
        yield proc
    finally:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate(timeout=10)
        if args.keep_streamlit_logs and output:
            print(output)


def import_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for screenshot export.\n"
            "Install it in the llmtt environment with:\n"
            "  /home/zixiao.wang/.conda/envs/llmtt/bin/python -m pip install -r demo_page/scripts/requirements-token-demo-export.txt\n"
            "The script uses /usr/bin/google-chrome by default, so a Playwright browser download is not required."
        ) from exc
    return sync_playwright


def find_graph_frame(page: Any, timeout: float = 240.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in page.frames:
            try:
                if frame.locator("svg polygon").count() > 0:
                    return frame
            except Exception:
                continue
        page.wait_for_timeout(500)
    raise TimeoutError("Could not find contribution graph iframe with token selectors")


def find_abc_score_frame(page: Any) -> Optional[Any]:
    for frame in page.frames:
        try:
            if frame.locator("#abc-score-shell, #abc-paper").count() > 0:
                return frame
        except Exception:
            continue
    return None


def abc_score_frame_and_locator(page: Any) -> Optional[tuple[Any, Any]]:
    abc_frame = find_abc_score_frame(page)
    if abc_frame is None:
        return None
    for selector in ("#abc-paper svg", "#abc-score-shell"):
        with contextlib.suppress(Exception):
            locator = abc_frame.locator(selector).first
            box = locator.bounding_box()
            if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
                return abc_frame, locator
    return None


def png_chunks(data: bytes) -> Iterable[tuple[bytes, bytes]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a PNG file")
    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        yield kind, payload
        offset += length + 12


def read_png_rgb(path: Path) -> tuple[int, int, List[bytearray]]:
    data = path.read_bytes()
    width = height = bit_depth = color_type = None
    idat_parts: List[bytes] = []
    for kind, payload in png_chunks(data):
        if kind == b"IHDR":
            width = int.from_bytes(payload[0:4], "big")
            height = int.from_bytes(payload[4:8], "big")
            bit_depth = payload[8]
            color_type = payload[9]
            if payload[12] != 0:
                raise ValueError("Interlaced PNG files are not supported")
        elif kind == b"IDAT":
            idat_parts.append(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in (2, 6):
        raise ValueError("Only 8-bit RGB/RGBA PNG files are supported")

    channels = 4 if color_type == 6 else 3
    stride = width * channels
    raw = zlib.decompress(b"".join(idat_parts))
    rows: List[bytearray] = []
    offset = 0
    prev = bytearray(stride)
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride
        for i, value in enumerate(row):
            left = row[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 1:
                row[i] = (value + left) & 0xFF
            elif filter_type == 2:
                row[i] = (value + up) & 0xFF
            elif filter_type == 3:
                row[i] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                predictor = left + up - up_left
                pa = abs(predictor - left)
                pb = abs(predictor - up)
                pc = abs(predictor - up_left)
                row[i] = (value + (left if pa <= pb and pa <= pc else up if pb <= pc else up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"Unsupported PNG filter type {filter_type}")
        prev = row
        if channels == 4:
            rows.append(bytearray(channel for x in range(width) for channel in row[x * 4 : x * 4 + 3]))
        else:
            rows.append(row)
    return width, height, rows


def write_png_rgb(path: Path, width: int, height: int, rows: Sequence[bytes]) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = crc32(kind)
        checksum = crc32(payload, checksum) & 0xFFFFFFFF
        return len(payload).to_bytes(4, "big") + kind + payload + checksum.to_bytes(4, "big")

    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def crop_score_png(path: Path, padding: int = 10, threshold: int = 245) -> None:
    width, height, rows = read_png_rgb(path)

    def is_foreground(x: int, y: int) -> bool:
        offset = x * 3
        return any(channel < threshold for channel in rows[y][offset : offset + 3])

    score_rows = range(max(0, int(height * 0.22)), height)
    score_xs = [x for y in score_rows for x in range(width) if is_foreground(x, y)]
    if not score_xs:
        return
    content_right = min(width - 1, max(score_xs) + padding)
    points = [
        (x, y)
        for y in range(height)
        for x in range(content_right + 1)
        if is_foreground(x, y)
    ]
    if not points:
        return

    left = max(0, min(x for x, _ in points) - padding)
    right = content_right
    top = max(0, min(y for _, y in points) - padding)
    bottom = min(height - 1, max(y for _, y in points) + padding)
    if left == 0 and right == width - 1 and top == 0 and bottom == height - 1:
        return

    cropped_rows = [row[left * 3 : (right + 1) * 3] for row in rows[top : bottom + 1]]
    write_png_rgb(path, right - left + 1, bottom - top + 1, cropped_rows)


def graph_svg_box(frame: Any) -> Optional[Dict[str, float]]:
    for selector in ("svg",):
        with contextlib.suppress(Exception):
            box = frame.locator(selector).first.bounding_box()
            if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
                return box
    with contextlib.suppress(Exception):
        return frame.frame_element().bounding_box()
    return None


def style_score_for_export(page: Any, graph_width: float) -> None:
    abc_frame = find_abc_score_frame(page)
    if abc_frame is None:
        return
    width = max(320, int(round(graph_width)))
    with contextlib.suppress(Exception):
        abc_frame.evaluate(
            """
            (width) => {
              if (typeof window.__LLMTT_TOKEN_DEMO_SET_SCORE_WIDTH__ === "function") {
                window.__LLMTT_TOKEN_DEMO_SET_SCORE_WIDTH__(width);
              }
              const shell = document.getElementById("abc-score-shell");
              const paper = document.getElementById("abc-paper");
              const svg = document.querySelector("#abc-paper svg");
              const toolbar = document.getElementById("abc-toolbar");
              const audio = document.getElementById("abc-audio");

              document.documentElement.style.width = `${width}px`;
              document.body.style.width = `${width}px`;
              if (shell) {
                shell.style.width = `${width}px`;
                shell.style.maxWidth = `${width}px`;
                shell.style.overflow = "visible";
              }
              if (paper) {
                paper.style.width = `${width}px`;
                paper.style.maxWidth = `${width}px`;
              }
              if (svg) {
                svg.setAttribute("width", String(width));
                svg.style.width = `${width}px`;
                svg.style.maxWidth = `${width}px`;
                svg.style.height = "auto";
                svg.style.overflow = "visible";
              }
              for (const node of [toolbar, audio, ...document.querySelectorAll(".abcjs-inline-audio")]) {
                if (node) {
                  node.style.display = "none";
                  node.style.visibility = "hidden";
                }
              }
            }
            """,
            width,
        )


def write_page_diagnostics(page: Any, output_root: Path, sample_id: str, reason: str) -> None:
    debug_dir = output_root / "_debug" / sample_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)[:80]
    screenshot_path = debug_dir / f"{safe_reason}.png"
    text_path = debug_dir / f"{safe_reason}.txt"
    html_path = debug_dir / f"{safe_reason}.html"

    with contextlib.suppress(Exception):
        page.screenshot(path=str(screenshot_path), full_page=True)
    with contextlib.suppress(Exception):
        visible_text = page.locator("body").inner_text(timeout=2000)
        text_path.write_text(visible_text, encoding="utf-8")
    with contextlib.suppress(Exception):
        html_path.write_text(page.content(), encoding="utf-8")

    print(f"[export] wrote debug diagnostics to {debug_dir}", flush=True)


def extract_token_labels(frame: Any) -> List[str]:
    return frame.evaluate(
        """
        () => {
          const polygons = Array.from(document.querySelectorAll("svg polygon"));
          const labels = Array.from(document.querySelectorAll("svg text"))
            .slice(0, polygons.length)
            .map((node) => node.textContent || "");
          return polygons.map((_, index) => labels[index] || "");
        }
        """
    )


def chord_element_role(token_text: str) -> str:
    stripped = token_text.strip()
    if stripped == '"':
        return "quote"
    if stripped and all(char.isdigit() for char in stripped):
        return "chord_number"
    return "chord_text"


def append_chord_element_entry(
    entries: List[Dict[str, Any]],
    *,
    index: int,
    display: str,
    chord_label: str,
    chord_ordinal: int,
    role: str,
    chord_span: List[int],
) -> None:
    entries.append(
        {
            "index": index,
            "display": display,
            "kind": "chord_element",
            "chord_ordinal": chord_ordinal,
            "chord_label": chord_label,
            "chord_element": role,
            "token_span": [index, index],
            "chord_span": chord_span,
        }
    )


def chord_token_entries(token_labels: Sequence[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    i = 0
    chord_ordinal = 0
    while i < len(token_labels):
        if token_labels[i] != '"':
            i += 1
            continue

        opening_quote_index = i
        i += 1
        part_indices: List[int] = []
        parts: List[str] = []
        while i < len(token_labels) and token_labels[i] != '"':
            piece = token_labels[i]
            if piece:
                part_indices.append(i)
                parts.append(piece)
            i += 1

        if i >= len(token_labels):
            break

        closing_quote_index = i
        label = "".join(parts).strip()
        if label and part_indices:
            chord_ordinal += 1
            chord_span = [opening_quote_index, closing_quote_index]
            append_chord_element_entry(
                entries,
                index=opening_quote_index,
                display=token_labels[opening_quote_index],
                chord_label=label,
                chord_ordinal=chord_ordinal,
                role="opening_quote",
                chord_span=chord_span,
            )
            for part_index in part_indices:
                append_chord_element_entry(
                    entries,
                    index=part_index,
                    display=token_labels[part_index],
                    chord_label=label,
                    chord_ordinal=chord_ordinal,
                    role=chord_element_role(token_labels[part_index]),
                    chord_span=chord_span,
                )
            append_chord_element_entry(
                entries,
                index=closing_quote_index,
                display=token_labels[closing_quote_index],
                chord_label=label,
                chord_ordinal=chord_ordinal,
                role="closing_quote",
                chord_span=chord_span,
            )
        i += 1
    return entries


def click_token(frame: Any, index: int) -> None:
    frame.evaluate(
        """
        (index) => {
          const polygons = Array.from(document.querySelectorAll("svg polygon"));
          const polygon = polygons[index];
          if (!polygon) {
            throw new Error(`Missing token polygon ${index}`);
          }
          polygon.dispatchEvent(new MouseEvent("click", {
            bubbles: true,
            cancelable: true,
            view: window
          }));
        }
        """,
        index,
    )


def screenshot_token_assets(page: Any, frame: Any, graph_path: Path, score_path: Path) -> None:
    graph_box = graph_svg_box(frame)
    if not graph_box:
        raise RuntimeError("Contribution graph SVG has no visible bounding box")

    style_score_for_export(page, graph_box["width"])
    page.wait_for_timeout(150)

    graph_locator = frame.locator("svg").first
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_locator.scroll_into_view_if_needed()
    graph_locator.screenshot(path=str(graph_path), omit_background=False)

    score_result = abc_score_frame_and_locator(page)
    if score_result is None:
        raise RuntimeError("ABC score SVG has no visible bounding box")
    _, score_locator = score_result
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_locator.scroll_into_view_if_needed()
    score_locator.screenshot(path=str(score_path), omit_background=False)
    crop_score_png(score_path)


def clear_existing_token_pngs(*directories: Path) -> None:
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("token_*.png"):
            path.unlink()


def prepare_page(page: Any, args: argparse.Namespace, url: str, sample_id: str) -> Any:
    page.set_viewport_size({"width": args.viewport_width, "height": args.viewport_height})
    page.goto(url, wait_until="domcontentloaded", timeout=int(args.startup_timeout * 1000))
    page.add_style_tag(
        content="""
          [data-testid="stSidebar"], [data-testid="stHeader"], header, footer {
            display: none !important;
          }
          .block-container {
            padding-top: 0.75rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: none !important;
          }
        """
    )
    try:
        frame = find_graph_frame(page, timeout=args.startup_timeout)
    except TimeoutError:
        write_page_diagnostics(page, args.output_root, sample_id, "graph_frame_timeout")
        raise
    frame.frame_element().scroll_into_view_if_needed()
    page.wait_for_timeout(args.after_click_ms)
    return frame


def export_sample(sync_playwright: Any, args: argparse.Namespace, sample: DemoSample, sentence: str, config_path: Path) -> Dict[str, Any]:
    sample_dir = args.output_root / sample.id
    graph_dir = sample_dir / "graph"
    score_dir = sample_dir / "score"
    audio_dir = sample_dir / "audio"
    graph_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_token_pngs(sample_dir, graph_dir, score_dir)
    score_audio_path = audio_dir / "score.wav"
    write_score_audio(score_audio_path, sentence)
    url = f"http://{args.host}:{args.port}"
    token_limit = 2 if args.smoke else args.max_tokens_per_sample
    tokens: List[Dict[str, Any]] = []

    with streamlit_server(args, config_path):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(args.browser_executable) if args.browser_executable.exists() else None,
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                page = browser.new_page()
                frame = prepare_page(page, args, url, sample.id)
                token_labels = extract_token_labels(frame)
                token_entries = chord_token_entries(token_labels)
                if not token_entries:
                    raise RuntimeError(f"{sample.id}: no chord element entries found")
                if token_limit and token_limit > 0:
                    token_entries = token_entries[:token_limit]

                for export_index, token_entry in enumerate(token_entries):
                    token_index = int(token_entry["index"])
                    label = str(token_entry["display"])
                    chord_label = str(token_entry.get("chord_label", ""))
                    role = str(token_entry.get("chord_element", ""))
                    print(
                        f"[export] {sample.id}: chord element {export_index + 1}/{len(token_entries)} "
                        f"token {token_index} {label!r} chord={chord_label!r} role={role}",
                        flush=True,
                    )
                    frame = find_graph_frame(page)
                    click_token(frame, token_index)
                    page.wait_for_timeout(args.after_click_ms)
                    frame = find_graph_frame(page)
                    frame.frame_element().scroll_into_view_if_needed()
                    image_name = f"token_{token_index:03d}.png"
                    graph_path = graph_dir / image_name
                    score_path = score_dir / image_name
                    screenshot_token_assets(page, frame, graph_path, score_path)
                    graph_image = f"assets/token-demo/{sample.id}/graph/{image_name}"
                    score_image = f"assets/token-demo/{sample.id}/score/{image_name}"
                    tokens.append(
                        {
                            **token_entry,
                            "graph_image": graph_image,
                            "score_image": score_image,
                            "image": graph_image,
                        }
                    )
            finally:
                browser.close()

    return {
        "id": sample.id,
        "label": sample.label,
        "backbone": sample.backbone,
        "iteration": sample.iteration,
        "dataset_file": sample.dataset_file,
        "line_index": sample.line_index,
        "sentence": sentence,
        "score_audio": f"assets/token-demo/{sample.id}/audio/score.wav",
        "tokens": tokens,
    }


def write_manifest(output_root: Path, manifest: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "manifest.json"
    js_path = output_root / "manifest.js"
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    json_path.write_text(payload + "\n", encoding="utf-8")
    js_path.write_text(f"window.TOKEN_DEMO_MANIFEST = {payload};\n", encoding="utf-8")
    print(f"[export] wrote {json_path}")
    print(f"[export] wrote {js_path}")


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    samples = selected_samples(args.sample_ids)
    verify_requested_device(args)
    sync_playwright = import_playwright()
    exported_samples: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="token-demo-export-") as tmp:
        tmpdir = Path(tmp)
        for sample in samples:
            sentence = read_sample_sentence(sample)
            config_path = write_temp_streamlit_config(tmpdir, sample, sentence)
            exported_samples.append(export_sample(sync_playwright, args, sample, sentence, config_path))

    manifest = {
        "version": 6,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export": {
            "status": "smoke" if args.smoke else "full",
            "viewport": {
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            "graph_defaults": {
                "contribution_threshold": args.contribution_threshold,
                "renormalize_after_threshold": True,
                "normalize_before_unembedding": True,
            },
        },
        "samples": exported_samples,
    }
    write_manifest(args.output_root, manifest)

    for sample in exported_samples:
        expected = len(sample["tokens"])
        graph_count = len(list((args.output_root / sample["id"] / "graph").glob("token_*.png")))
        score_count = len(list((args.output_root / sample["id"] / "score").glob("token_*.png")))
        audio_exists = (args.output_root / sample["id"] / "audio" / "score.wav").is_file()
        if expected != graph_count or expected != score_count:
            raise RuntimeError(
                f"{sample['id']}: manifest tokens={expected}, "
                f"graph images={graph_count}, score images={score_count}"
            )
        if not audio_exists:
            raise RuntimeError(f"{sample['id']}: missing score audio")
        print(
            f"[export] verified {sample['id']}: "
            f"{graph_count} graph screenshots, {score_count} score screenshots, 1 score audio"
        )


if __name__ == "__main__":
    main()
