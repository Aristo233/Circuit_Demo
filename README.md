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

Smoke export, two tokens per sample:

```bash
/home/zixiao.wang/.conda/envs/llmtt/bin/python demo_page/scripts/export_token_demo.py --smoke
```

Full export:

```bash
/home/zixiao.wang/.conda/envs/llmtt/bin/python demo_page/scripts/export_token_demo.py
```

The exporter writes screenshots and manifests into:

```text
demo_page/assets/token-demo/
```

It uses the current Streamlit defaults: contribution threshold `0.05`, renormalize enabled, normalize enabled, and the current graph scale.
