function apiBase() {
  // When served by FastAPI, APIs are on the same origin.
  // On GitHub Pages / plain static hosting, there is no API.
  return "";
}

async function getJSON(url) {
  const res = await fetch(apiBase() + url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

function renderHighlights(card) {
  const root = document.getElementById("highlights");
  root.innerHTML = "";
  for (const h of card.highlights || []) {
    const el = document.createElement("article");
    el.innerHTML =
      `<div class="value">${h.value}</div>` +
      `<div class="label">${h.label}</div>` +
      `<div class="detail">${h.detail || ""}</div>`;
    root.appendChild(el);
  }
  const methods = document.getElementById("methods");
  if (methods) {
    methods.innerHTML = "";
    for (const m of card.methods || []) {
      const li = document.createElement("li");
      li.textContent = m;
      methods.appendChild(li);
    }
  }
  document.getElementById("disclaimer").textContent = card.disclaimer || "";
  if (card.subtitle) document.getElementById("subtitle").textContent = card.subtitle;
}

async function loadPortfolio() {
  // 1) Embedded data (works on GitHub Pages / file open / static host)
  if (window.POLICYSHIFT_CARD) {
    renderHighlights(window.POLICYSHIFT_CARD);
  }
  // 2) Prefer live API card when available
  try {
    const card = await getJSON("/api/portfolio");
    renderHighlights(card);
  } catch (_) {
    // static mode — embedded card is enough
    if (!window.POLICYSHIFT_CARD) {
      document.getElementById("highlights").innerHTML =
        `<p class="support">Missing metrics. Run <code>python scripts/export_portfolio.py</code>.</p>`;
    }
  }
}

async function apiAvailable() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    return res.ok;
  } catch (_) {
    return false;
  }
}

async function loadExperiments() {
  const help = document.getElementById("playback-help");
  const panel = document.getElementById("playback-panel");
  const select = document.getElementById("experiment-select");
  const view = document.getElementById("experiment-view");

  const ok = await apiAvailable();
  if (!ok) {
    // Keep static instructions; hide empty controls
    if (panel) panel.classList.add("playback-hidden");
    return;
  }

  if (panel) panel.classList.remove("playback-hidden");
  if (help) {
    help.textContent = "Select a local experiment to inspect summaries and trajectories.";
  }

  try {
    const data = await getJSON("/api/experiments");
    select.innerHTML = "";
    if (!data.experiments.length) {
      view.textContent =
        "No local experiments in artifacts/experiments/. Run make evaluate-phase2 (etc.).";
      return;
    }
    for (const exp of data.experiments) {
      const opt = document.createElement("option");
      opt.value = exp.id;
      opt.textContent = exp.id;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => showExperiment(select.value));
    await showExperiment(select.value);
  } catch (err) {
    view.textContent = String(err);
  }
}

async function showExperiment(id) {
  const view = document.getElementById("experiment-view");
  const trajView = document.getElementById("traj-view");
  view.textContent = "Loading…";
  try {
    const exp = await getJSON(`/api/experiments/${encodeURIComponent(id)}`);
    const slim = { ...exp };
    delete slim.preference_explorer;
    view.textContent = JSON.stringify(slim, null, 2);
    const traj = await getJSON(
      `/api/experiments/${encodeURIComponent(id)}/trajectories?limit=3`
    );
    trajView.textContent = JSON.stringify(traj.trajectories, null, 2);
  } catch (err) {
    view.textContent = String(err);
    trajView.textContent = "—";
  }
}

loadPortfolio();
loadExperiments();
