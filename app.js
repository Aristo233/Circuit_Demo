(function () {
  const FALLBACK_MANIFEST = {
    version: 1,
    generated_at: null,
    samples: [],
  };

  const state = {
    manifest: FALLBACK_MANIFEST,
    sampleIndex: 0,
    tokenIndex: 0,
  };

  const els = {
    sampleTabs: document.getElementById("sample-tabs"),
    sampleMeta: document.getElementById("sample-meta"),
    tokenSelect: document.getElementById("token-select"),
    prevToken: document.getElementById("prev-token"),
    nextToken: document.getElementById("next-token"),
    tokenCount: document.getElementById("token-count"),
    tokenStrip: document.getElementById("token-strip"),
    viewerTitle: document.getElementById("viewer-title"),
    screenshot: document.getElementById("screenshot"),
    emptyState: document.getElementById("empty-state"),
    imageMissing: document.getElementById("image-missing"),
    openImage: document.getElementById("open-image"),
  };

  function normalizeManifest(raw) {
    if (!raw || !Array.isArray(raw.samples)) {
      return FALLBACK_MANIFEST;
    }
    return {
      ...raw,
      samples: raw.samples.map((sample) => ({
        ...sample,
        tokens: Array.isArray(sample.tokens) ? sample.tokens : [],
      })),
    };
  }

  async function loadManifest() {
    const globalManifest = normalizeManifest(window.TOKEN_DEMO_MANIFEST);
    if (window.location.protocol === "file:" && globalManifest.samples.length > 0) {
      return globalManifest;
    }

    try {
      const response = await fetch("assets/token-demo/manifest.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Manifest request failed: ${response.status}`);
      }
      return normalizeManifest(await response.json());
    } catch (error) {
      if (globalManifest.samples.length > 0) {
        return globalManifest;
      }
      return FALLBACK_MANIFEST;
    }
  }

  function selectedSample() {
    return state.manifest.samples[state.sampleIndex] || null;
  }

  function selectedToken() {
    const sample = selectedSample();
    return sample ? sample.tokens[state.tokenIndex] || null : null;
  }

  function tokenLabel(token) {
    if (!token) {
      return "";
    }
    const text = token.display || token.text || "";
    return text === "" ? "(empty)" : text;
  }

  function clearElement(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function setHidden(node, hidden) {
    node.classList.toggle("hidden", Boolean(hidden));
  }

  function renderSampleTabs() {
    clearElement(els.sampleTabs);
    const samples = state.manifest.samples;

    samples.forEach((sample, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.role = "tab";
      button.textContent = sample.label || sample.id || `Sample ${index + 1}`;
      button.setAttribute("aria-selected", String(index === state.sampleIndex));
      button.addEventListener("click", () => {
        state.sampleIndex = index;
        state.tokenIndex = 0;
        render();
      });
      els.sampleTabs.appendChild(button);
    });
  }

  function renderSampleMeta(sample) {
    clearElement(els.sampleMeta);
    if (!sample) {
      els.sampleMeta.textContent = "No sample manifest is available.";
      return;
    }

    const rows = [
      ["Backbone", sample.backbone],
      ["Iteration", sample.iteration],
      ["Dataset", sample.dataset_file],
      ["Line index", sample.line_index],
    ];

    rows.forEach(([label, value]) => {
      if (value === undefined || value === null || value === "") {
        return;
      }
      const row = document.createElement("div");
      const strong = document.createElement("strong");
      strong.textContent = `${label}: `;
      const code = document.createElement("code");
      code.textContent = String(value);
      row.append(strong, code);
      els.sampleMeta.appendChild(row);
    });
  }

  function renderTokenSelect(sample) {
    clearElement(els.tokenSelect);
    if (!sample || sample.tokens.length === 0) {
      const option = document.createElement("option");
      option.textContent = "No tokens exported";
      option.value = "";
      els.tokenSelect.appendChild(option);
      els.tokenSelect.disabled = true;
      return;
    }

    els.tokenSelect.disabled = false;
    sample.tokens.forEach((token, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${index}: ${tokenLabel(token)}`;
      option.selected = index === state.tokenIndex;
      els.tokenSelect.appendChild(option);
    });
  }

  function renderTokenStrip(sample) {
    clearElement(els.tokenStrip);
    if (!sample || sample.tokens.length === 0) {
      return;
    }

    sample.tokens.forEach((token, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `token-chip${index === state.tokenIndex ? " active" : ""}`;
      button.title = `${index}: ${tokenLabel(token)}`;
      button.textContent = tokenLabel(token);
      button.addEventListener("click", () => {
        state.tokenIndex = index;
        render();
      });
      els.tokenStrip.appendChild(button);
    });
  }

  function renderViewer(sample, token) {
    const hasToken = Boolean(sample && token);
    const label = hasToken ? tokenLabel(token) : "";
    els.viewerTitle.textContent = hasToken
      ? `${sample.label || sample.id} - token ${state.tokenIndex}: ${label}`
      : "No screenshot selected";

    setHidden(els.emptyState, hasToken);
    setHidden(els.imageMissing, true);
    setHidden(els.screenshot, true);
    setHidden(els.openImage, true);
    els.screenshot.removeAttribute("src");
    els.screenshot.alt = "";

    if (!hasToken || !token.image) {
      setHidden(els.emptyState, false);
      return;
    }

    els.screenshot.onload = () => {
      setHidden(els.screenshot, false);
      setHidden(els.emptyState, true);
      setHidden(els.imageMissing, true);
      setHidden(els.openImage, false);
    };
    els.screenshot.onerror = () => {
      setHidden(els.screenshot, true);
      setHidden(els.emptyState, true);
      setHidden(els.imageMissing, false);
      setHidden(els.openImage, true);
    };
    els.screenshot.alt = `${sample.label || sample.id}, token ${state.tokenIndex}: ${label}`;
    els.screenshot.src = token.image;
    els.openImage.href = token.image;
  }

  function renderStepper(sample) {
    const count = sample ? sample.tokens.length : 0;
    els.prevToken.disabled = count === 0 || state.tokenIndex <= 0;
    els.nextToken.disabled = count === 0 || state.tokenIndex >= count - 1;
    els.tokenCount.value = count > 0 ? `${state.tokenIndex + 1} / ${count}` : "0 / 0";
  }

  function clampState() {
    const samples = state.manifest.samples;
    state.sampleIndex = Math.min(Math.max(state.sampleIndex, 0), Math.max(samples.length - 1, 0));
    const sample = selectedSample();
    const tokenCount = sample ? sample.tokens.length : 0;
    state.tokenIndex = Math.min(Math.max(state.tokenIndex, 0), Math.max(tokenCount - 1, 0));
  }

  function render() {
    clampState();
    const sample = selectedSample();
    const token = selectedToken();
    renderSampleTabs();
    renderSampleMeta(sample);
    renderTokenSelect(sample);
    renderTokenStrip(sample);
    renderStepper(sample);
    renderViewer(sample, token);
  }

  els.tokenSelect.addEventListener("change", () => {
    const next = Number.parseInt(els.tokenSelect.value, 10);
    if (Number.isFinite(next)) {
      state.tokenIndex = next;
      render();
    }
  });

  els.prevToken.addEventListener("click", () => {
    state.tokenIndex -= 1;
    render();
  });

  els.nextToken.addEventListener("click", () => {
    state.tokenIndex += 1;
    render();
  });

  loadManifest().then((manifest) => {
    state.manifest = manifest;
    render();
  });
})();
