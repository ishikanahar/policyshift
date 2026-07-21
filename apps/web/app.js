async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

function renderHighlights(card) {
  const root = document.getElementById("highlights");
  root.innerHTML = "";
  for (const h of card.highlights || []) {
    const el = document.createElement("article");
    el.innerHTML = `<div class="value">${h.value}</div><div class="label">${h.label}</div><div class="detail">${h.detail || ""}</div>`;
    root.appendChild(el);
  }
  document.getElementById("disclaimer").textContent = card.disclaimer || "";
  if (card.subtitle) document.getElementById("subtitle").textContent = card.subtitle;
}

async function loadPortfolio() {
  try {
    const card = await getJSON("/api/portfolio");
    renderHighlights(card);
  } catch (err) {
    document.getElementById("highlights").innerHTML =
      `<p class="support">Run <code>python scripts/export_portfolio.py</code> then refresh. (${err.message})</p>`;
  }
}

async function loadExperiments() {
  const select = document.getElementById("experiment-select");
  const view = document.getElementById("experiment-view");
  try {
    const data = await getJSON("/api/experiments");
    select.innerHTML = "";
    if (!data.experiments.length) {
      view.textContent = "No local experiments in artifacts/experiments/. Run make evaluate-phase2 (etc.).";
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
    const traj = await getJSON(`/api/experiments/${encodeURIComponent(id)}/trajectories?limit=3`);
    trajView.textContent = JSON.stringify(traj.trajectories, null, 2);
  } catch (err) {
    view.textContent = String(err);
    trajView.textContent = "—";
  }
}

loadPortfolio();
loadExperiments();
