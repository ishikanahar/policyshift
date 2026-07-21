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

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

async function apiAvailable() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    return res.ok;
  } catch (_) {
    return false;
  }
}

function embeddedExperiments() {
  return (window.POLICYSHIFT_EXPERIMENTS && window.POLICYSHIFT_EXPERIMENTS.experiments) || [];
}

async function loadPortfolio() {
  if (window.POLICYSHIFT_CARD) {
    renderHighlights(window.POLICYSHIFT_CARD);
  }
  try {
    const card = await getJSON("/api/portfolio");
    renderHighlights(card);
  } catch (_) {
    if (!window.POLICYSHIFT_CARD) {
      document.getElementById("highlights").innerHTML =
        `<p class="support">Missing metrics. Run <code>python scripts/export_portfolio.py</code>.</p>`;
    }
  }
}

function fillExperimentSelect(ids) {
  const select = document.getElementById("experiment-select");
  select.innerHTML = "";
  for (const id of ids) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    select.appendChild(opt);
  }
}

function showEmbeddedExperiment(id) {
  const view = document.getElementById("experiment-view");
  const trajView = document.getElementById("traj-view");
  const exp = embeddedExperiments().find((e) => e.id === id);
  if (!exp) {
    view.textContent = `No embedded data for ${id}. Restart with: python scripts/serve_playback.py`;
    trajView.textContent = "—";
    return;
  }
  const slim = { ...exp };
  const trajs = slim.sample_trajectories || [];
  delete slim.sample_trajectories;
  view.textContent = JSON.stringify(slim, null, 2);
  trajView.textContent = JSON.stringify(trajs, null, 2);
}

async function showLiveExperiment(id) {
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
    // Fall back to embedded snapshot if the API died mid-session
    const embedded = embeddedExperiments().find((e) => e.id === id);
    if (embedded) {
      showEmbeddedExperiment(id);
      view.textContent =
        `// Live API unavailable (${err}). Showing embedded snapshot.\n` +
        view.textContent;
      return;
    }
    view.textContent =
      `${err}\n\nServer looks down. In the project folder run:\n` +
      `  python scripts/serve_playback.py\n` +
      `then refresh this page.`;
    trajView.textContent = "—";
  }
}

async function loadExperiments() {
  const help = document.getElementById("playback-help");
  const panel = document.getElementById("playback-panel");
  const select = document.getElementById("experiment-select");
  const embedded = embeddedExperiments();
  const live = await apiAvailable();

  if (panel) panel.classList.remove("playback-hidden");

  if (live) {
    if (help) {
      help.textContent =
        "Live API connected. Select an experiment to inspect summaries and trajectories.";
    }
    try {
      const data = await getJSON("/api/experiments");
      const ids = (data.experiments || []).map((e) => e.id);
      if (!ids.length && embedded.length) {
        fillExperimentSelect(embedded.map((e) => e.id));
        select.onchange = () => showEmbeddedExperiment(select.value);
        showEmbeddedExperiment(select.value);
        return;
      }
      fillExperimentSelect(ids);
      select.onchange = () => showLiveExperiment(select.value);
      if (ids.length) await showLiveExperiment(ids[0]);
      return;
    } catch (err) {
      // continue to embedded
      console.warn(err);
    }
  }

  if (embedded.length) {
    if (help) {
      help.textContent =
        "Showing embedded experiment snapshots (no live API). For live playback run: python scripts/serve_playback.py";
    }
    fillExperimentSelect(embedded.map((e) => e.id));
    select.onchange = () => showEmbeddedExperiment(select.value);
    showEmbeddedExperiment(select.value);
    return;
  }

  if (panel) panel.classList.add("playback-hidden");
  if (help) {
    help.innerHTML =
      `No experiment data loaded. Run <code>python scripts/export_portfolio.py</code> then ` +
      `<code>python scripts/serve_playback.py</code> and open <code>http://127.0.0.1:8000</code>.`;
  }
}

loadPortfolio();
loadExperiments();
