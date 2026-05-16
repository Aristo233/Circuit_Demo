# Static Token-Click Demo Page

This folder contains the static GitHub Pages demo and the export helper used to generate token screenshots from the Streamlit app.

## Open The Static Page

From the repository root:

```bash
python -m http.server 8123 --directory demo_page
```

Then open:

```text
http://127.0.0.1:8123/
```

Opening `demo_page/index.html` directly also works for the placeholder manifest. Use a local HTTP server after generating screenshots so `manifest.json` is loaded consistently.

## Export Screenshots

Install the browser automation dependency in the `llmtt` environment:

```bash
/home/zixiao.wang/.conda/envs/llmtt/bin/python -m pip install -r demo_page/scripts/requirements-token-demo-export.txt
```

Smoke export, two chord elements per sample:

```bash
/home/zixiao.wang/.conda/envs/llmtt/bin/python demo_page/scripts/export_token_demo.py --smoke
```

Full export:

```bash
/home/zixiao.wang/.conda/envs/llmtt/bin/python demo_page/scripts/export_token_demo.py
```

By default, the exporter starts Streamlit with `CUDA_VISIBLE_DEVICES=1`, `LLMTT_DEFAULT_DEVICE=gpu`, and `LLMTT_FORCE_DEVICE=gpu`. It also checks CUDA visibility before starting Streamlit. To use a different GPU or CPU:

```bash
/home/zixiao.wang/.conda/envs/llmtt/bin/python demo_page/scripts/export_token_demo.py --cuda-visible-devices 0
/home/zixiao.wang/.conda/envs/llmtt/bin/python demo_page/scripts/export_token_demo.py --device cpu
```

The exporter writes manifests plus separate graph and score screenshots into:

```text
demo_page/assets/token-demo/
demo_page/assets/token-demo/<sample-id>/graph/token_<original-index>.png
demo_page/assets/token-demo/<sample-id>/score/token_<original-index>.png
demo_page/assets/token-demo/<sample-id>/audio/score.wav
```

It exports only chord-literal elements: the opening quote, chord text, chord number tokens such as `7`, and the closing quote. The default graph settings are contribution threshold `0.038`, renormalize enabled, normalize enabled, and the current graph scale. The generated `score.wav` files use the export helper's piano-like offline synthesizer.
