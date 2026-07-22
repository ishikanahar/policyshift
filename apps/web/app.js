const SCENES = {
  ops: {
    kicker: "Ops / supply chain",
    title: "Materials receiving policy just got stricter",
    body: "Last month: release if the COA is present. This month: dual approval + quarantine if temperature logs are incomplete. An agent trained on last month’s SOP will “helpfully” release inventory that should be held — expensive and unsafe.",
    map: "Maps to: warehouse ops · quality · ERP / WMS copilots",
  },
  lab: {
    kicker: "Lab / safety",
    title: "Instrument access rules changed overnight",
    body: "Calibration windows, after-hours restrictions, and supervisor approvals shift between policy versions. A stale agent books the wrong instrument slot or skips a failed QC gate — the kind of miss safety teams lose sleep over.",
    map: "Maps to: EHS · lab ops · regulated R&D environments",
  },
  ai: {
    kicker: "AI / data governance",
    title: "Approved-model and data-handling rules update",
    body: "External APIs get blocked, retention rules tighten, high-impact actions need a human. Yesterday’s agent happily calls a banned vendor model. Today that is a compliance incident.",
    map: "Maps to: AI platform · security · legal / GRC copilots",
  },
};

const MODES = {
  naive: {
    source: "Stale PDF excerpt from last year’s training pack",
    action: "Approve release",
    outcome: "Policy miss · unsafe release",
  },
  aware: {
    source: "Policy version effective at the event timestamp",
    action: "Quarantine + request dual approval",
    outcome: "Aligned with current SOP · safe hold",
  },
};

function renderImpact(card) {
  const root = document.getElementById("impact-grid");
  if (!root) return;
  const base = (card && card.highlights) || [];
  const byLabel = Object.fromEntries(base.map((h) => [h.label, h]));
  const items = [
    {
      value: (byLabel["RAG task success"] || {}).value || "0.75",
      label: "Higher task success with version-aware RAG",
      detail: `up from ${(byLabel["RAG task success"] || {}).detail || "baseline 0.58"}`.replace(
        /^up from vs /,
        "up from "
      ),
    },
    {
      value: (byLabel["Stale@5 (date-filtered)"] || {}).value || "0.00",
      label: "Stale documents in top-5 retrieval",
      detail: "down from 0.45 with naive search",
    },
    {
      value: "80%",
      label: "Fewer teacher / labeling calls",
      detail: "budgeted selection matched label-all success in smoke",
    },
    {
      value: (byLabel["Domains × versions"] || {}).value || "3 × 3",
      label: "Domains × policy versions under test",
      detail: "120+ executable agent cases",
    },
  ];
  // Fix first detail if we have the vs baseline string
  if (byLabel["RAG task success"] && byLabel["RAG task success"].detail) {
    items[0].detail = byLabel["RAG task success"].detail.replace(/^vs /, "up from ");
  }
  root.innerHTML = "";
  for (const h of items) {
    const el = document.createElement("article");
    el.className = "impact-item";
    el.innerHTML =
      `<div class="value">${h.value}</div>` +
      `<div class="label">${h.label}</div>` +
      `<div class="detail">${h.detail || ""}</div>`;
    root.appendChild(el);
  }
  const disc = document.getElementById("disclaimer");
  if (disc) {
    disc.textContent =
      "Synthetic environment with measured smoke artifacts. Built to show evaluation + agent-systems craft for hiring — not a claim of production deployment or frontier-scale training.";
  }
}

function setScene(key) {
  const scene = SCENES[key];
  if (!scene) return;
  const board = document.getElementById("scenario-board");
  const copy = board.querySelector(".scene-copy");
  copy.style.animation = "none";
  // retrigger
  void copy.offsetWidth;
  copy.style.animation = "";
  document.getElementById("scene-kicker").textContent = scene.kicker;
  document.getElementById("scene-title").textContent = scene.title;
  document.getElementById("scene-body").textContent = scene.body;
  document.getElementById("scene-map").textContent = scene.map;
  document.querySelectorAll(".tab").forEach((btn) => {
    const on = btn.dataset.scene === key;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
}

function setMode(mode) {
  const stage = document.getElementById("decision-stage");
  const data = MODES[mode];
  if (!stage || !data) return;
  stage.classList.add("is-flipping");
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.mode === mode);
  });
  setTimeout(() => {
    stage.dataset.mode = mode;
    document.getElementById("dec-source").textContent = data.source;
    document.getElementById("dec-action").textContent = data.action;
    document.getElementById("dec-outcome").textContent = data.outcome;
    stage.classList.remove("is-flipping");
  }, 160);
}

function wireInteractions() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => setScene(btn.dataset.scene));
  });
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });
  // Auto-rotate scenes for dynamism
  const keys = Object.keys(SCENES);
  let i = 0;
  setInterval(() => {
    i = (i + 1) % keys.length;
    // don't fight a focused user on tabs
    if (document.activeElement && document.activeElement.classList.contains("tab")) return;
    setScene(keys[i]);
  }, 7000);
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
    view.textContent = `No embedded data for ${id}.`;
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
    const embedded = embeddedExperiments().find((e) => e.id === id);
    if (embedded) {
      showEmbeddedExperiment(id);
      return;
    }
    view.textContent = String(err);
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
    try {
      const data = await getJSON("/api/experiments");
      const ids = (data.experiments || []).map((e) => e.id);
      fillExperimentSelect(ids.length ? ids : embedded.map((e) => e.id));
      select.onchange = () =>
        ids.length ? showLiveExperiment(select.value) : showEmbeddedExperiment(select.value);
      if (select.value) {
        if (ids.length) await showLiveExperiment(select.value);
        else showEmbeddedExperiment(select.value);
      }
      return;
    } catch (_) {
      /* fall through */
    }
  }

  if (embedded.length) {
    if (help) help.textContent = "Embedded experiment snapshots from measured smoke runs.";
    fillExperimentSelect(embedded.map((e) => e.id));
    select.onchange = () => showEmbeddedExperiment(select.value);
    showEmbeddedExperiment(select.value);
    return;
  }
  if (panel) panel.classList.add("playback-hidden");
}

async function boot() {
  wireInteractions();
  setScene("ops");
  setMode("naive");
  const card = window.POLICYSHIFT_CARD;
  renderImpact(card);
  try {
    const live = await getJSON("/api/portfolio");
    renderImpact(live);
  } catch (_) {
    /* embedded card is enough */
  }
  await loadExperiments();
}

boot();
