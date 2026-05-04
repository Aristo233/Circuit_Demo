#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_transparency_tool.server.dataset_utils import load_dataset_file


DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "demo_page" / "assets" / "token-demo"
DEFAULT_BROWSER = Path("/usr/bin/google-chrome")


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
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--ld-preload-stub", default="/tmp/libittnotify_stub.so")
    parser.add_argument("--startup-timeout", type=float, default=420.0)
    parser.add_argument("--after-click-ms", type=int, default=1800)
    parser.add_argument("--viewport-width", type=int, default=2200)
    parser.add_argument("--viewport-height", type=int, default=2200)
    parser.add_argument("--capture-width", type=int, default=0)
    parser.add_argument("--capture-height", type=int, default=0)
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
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=build_streamlit_env(args),
        text=True,
        stdout=subprocess.PIPE,
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


def screenshot_graph_region(page: Any, frame: Any, args: argparse.Namespace, output_path: Path) -> None:
    from PIL import Image

    frame_element = frame.frame_element()
    frame_box = frame_element.bounding_box()
    if not frame_box:
        raise RuntimeError("Contribution graph iframe has no visible bounding box")

    main_box = None
    for selector in ('section[data-testid="stMain"]', 'div[data-testid="stAppViewContainer"]', "main"):
        locator = page.locator(selector).first
        try:
            box = locator.bounding_box()
        except Exception:
            box = None
        if box:
            main_box = box
            break

    viewport = page.viewport_size or {"width": args.viewport_width, "height": args.viewport_height}
    left = int(max(0, (main_box or frame_box)["x"]))
    top = int(max(0, frame_box["y"] - 8))
    width = int(args.capture_width or min(viewport["width"] - left, (main_box or {"width": viewport["width"]})["width"]))
    height = int(args.capture_height or (viewport["height"] - top))
    width = max(320, min(width, viewport["width"] - left))
    height = max(320, min(height, viewport["height"] - top))

    raw = page.screenshot(full_page=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        crop = image.crop((left, top, left + width, top + height))
        crop.save(output_path)


def prepare_page(page: Any, args: argparse.Namespace, url: str) -> Any:
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
    frame = find_graph_frame(page)
    frame.frame_element().scroll_into_view_if_needed()
    page.wait_for_timeout(args.after_click_ms)
    return frame


def export_sample(sync_playwright: Any, args: argparse.Namespace, sample: DemoSample, sentence: str, config_path: Path) -> Dict[str, Any]:
    sample_dir = args.output_root / sample.id
    sample_dir.mkdir(parents=True, exist_ok=True)
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
                frame = prepare_page(page, args, url)
                token_labels = extract_token_labels(frame)
                if token_limit and token_limit > 0:
                    token_labels = token_labels[:token_limit]

                for token_index, label in enumerate(token_labels):
                    print(f"[export] {sample.id}: token {token_index + 1}/{len(token_labels)} {label!r}")
                    frame = find_graph_frame(page)
                    click_token(frame, token_index)
                    page.wait_for_timeout(args.after_click_ms)
                    frame = find_graph_frame(page)
                    frame.frame_element().scroll_into_view_if_needed()
                    image_name = f"token_{token_index:03d}.png"
                    image_path = sample_dir / image_name
                    screenshot_graph_region(page, frame, args, image_path)
                    tokens.append(
                        {
                            "index": token_index,
                            "display": label,
                            "image": f"assets/token-demo/{sample.id}/{image_name}",
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
    sync_playwright = import_playwright()
    exported_samples: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="token-demo-export-") as tmp:
        tmpdir = Path(tmp)
        for sample in samples:
            sentence = read_sample_sentence(sample)
            config_path = write_temp_streamlit_config(tmpdir, sample, sentence)
            exported_samples.append(export_sample(sync_playwright, args, sample, sentence, config_path))

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export": {
            "status": "smoke" if args.smoke else "full",
            "viewport": {
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            "graph_defaults": {
                "contribution_threshold": 0.05,
                "renormalize_after_threshold": True,
                "normalize_before_unembedding": True,
            },
        },
        "samples": exported_samples,
    }
    write_manifest(args.output_root, manifest)

    for sample in exported_samples:
        expected = len(sample["tokens"])
        image_count = len(list((args.output_root / sample["id"]).glob("token_*.png")))
        if expected != image_count:
            raise RuntimeError(f"{sample['id']}: manifest tokens={expected}, images={image_count}")
        print(f"[export] verified {sample['id']}: {image_count} screenshots")


if __name__ == "__main__":
    main()
