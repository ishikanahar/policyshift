const SCENES = {
  ops: {
    kicker: "Ops / supply chain",
    title: "Receiving SOP got stricter this quarter",
    body: "Old rule: release if COA exists. New rule: quarantine + dual approval when temperature logs are incomplete. Stale agents still rubber-stamp releases.",
    map: "Relevant to: warehouse ops · quality · ERP / WMS copilots",
    naive: {
      steps: [
        ["Read case", "Inbound lot, missing continuous temp log"],
        ["Grab policy", "Uses last year’s PDF excerpt (v1.0)"],
        ["Decide", "Approve release into inventory"],
      ],
      result: "FAIL · unsafe release against current SOP",
      good: false,
    },
    aware: {
      steps: [
        ["Read case", "Inbound lot, missing continuous temp log"],
        ["Resolve version", "Selects policy effective at event time (v2.0)"],
        ["Decide", "Quarantine + request dual approval"],
      ],
      result: "PASS · held safely under current SOP",
      good: true,
    },
  },
  lab: {
    kicker: "Lab / safety",
    title: "Instrument access rules changed overnight",
    body: "Calibration windows and after-hours approvals shifted. A stale agent books an instrument outside the allowed window.",
    map: "Relevant to: EHS · lab ops · regulated R&D",
    naive: {
      steps: [
        ["Read case", "After-hours HPLC request"],
        ["Grab policy", "Old access matrix without new gate"],
        ["Decide", "Auto-approve reservation"],
      ],
      result: "FAIL · safety / access violation",
      good: false,
    },
    aware: {
      steps: [
        ["Read case", "After-hours HPLC request"],
        ["Resolve version", "Loads active lab policy + calibration check"],
        ["Decide", "Require supervisor approval"],
      ],
      result: "PASS · blocked until approval",
      good: true,
    },
  },
  ai: {
    kicker: "AI governance",
    title: "Approved-model list was updated",
    body: "External API vendors got restricted. Yesterday’s agent still calls a banned model endpoint.",
    map: "Relevant to: AI platform · security · GRC",
    naive: {
      steps: [
        ["Read case", "Summarize sensitive customer notes"],
        ["Grab policy", "Cached ‘any approved LLM’ guidance"],
        ["Decide", "Call external vendor API"],
      ],
      result: "FAIL · compliance incident",
      good: false,
    },
    aware: {
      steps: [
        ["Read case", "Summarize sensitive customer notes"],
        ["Resolve version", "Checks current approved-model + retention rules"],
        ["Decide", "Route to internal model + human review"],
      ],
      result: "PASS · policy-aligned path",
      good: true,
    },
  },
};

let sceneKey = "ops";
let mode = "naive";
let running = false;
let timers = [];
let userTouched = false;

function clearTimers() {
  timers.forEach(clearTimeout);
  timers = [];
}

function renderImpact(card) {
  const root = document.getElementById("impact-grid");
  if (!root) return;
  const base = (card && card.highlights) || [];
  const byLabel = Object.fromEntries(base.map((h) => [h.label, h]));
  const items = [
    {
      value: (byLabel["RAG task success"] || {}).value || "0.75",
      label: "Higher task success with version-aware RAG",
      detail: ((byLabel["RAG task success"] || {}).detail || "vs baseline 0.58").replace(/^vs /, "up from "),
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
  root.innerHTML = items
    .map(
      (h) =>
        `<article class="impact-item"><div class="value">${h.value}</div><div class="label">${h.label}</div><div class="detail">${h.detail}</div></article>`
    )
    .join("");
  const disc = document.getElementById("disclaimer");
  if (disc) {
    disc.textContent =
      "Synthetic environment with measured smoke artifacts — built for hiring portfolio demos, not as a production deployment claim.";
  }
}

function renderTabs() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = "";
  Object.keys(SCENES).forEach((key) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tab" + (key === sceneKey ? " is-active" : "");
    btn.dataset.scene = key;
    btn.textContent = SCENES[key].kicker;
    tabs.appendChild(btn);
  });
}

function renderCopy() {
  const s = SCENES[sceneKey];
  document.getElementById("scene-kicker").textContent = s.kicker;
  document.getElementById("scene-title").textContent = s.title;
  document.getElementById("scene-body").textContent = s.body;
  document.getElementById("scene-map").textContent = s.map;
  document.querySelectorAll(".mode-btn").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.mode === mode);
  });
}

function resetTrace() {
  clearTimers();
  running = false;
  const runBtn = document.getElementById("run-btn");
  if (runBtn) runBtn.disabled = false;
  const hint = document.getElementById("run-hint");
  if (hint) hint.textContent = "You’ll see each step animate";
  const path = SCENES[sceneKey][mode];
  const steps = document.getElementById("steps");
  steps.innerHTML = path.steps
    .map(
      ([a, b], i) =>
        `<div class="step" data-i="${i}"><div class="dot"></div><div><strong>${a}</strong><span>${b}</span></div></div>`
    )
    .join("");
  const result = document.getElementById("trace-result");
  result.className = "trace-result";
  result.textContent = "Waiting to run…";
}

function runDecision() {
  if (running) return;
  userTouched = true;
  running = true;
  const runBtn = document.getElementById("run-btn");
  runBtn.disabled = true;
  document.getElementById("run-hint").textContent = "Running…";
  const path = SCENES[sceneKey][mode];
  const nodes = [...document.querySelectorAll("#steps .step")];
  nodes.forEach((n) => n.classList.remove("on", "done", "fail"));
  const result = document.getElementById("trace-result");
  result.className = "trace-result";
  result.textContent = "Tracing decision…";

  path.steps.forEach((_, i) => {
    timers.push(
      setTimeout(() => {
        nodes.forEach((n, j) => {
          if (j <= i) n.classList.add("on");
        });
        nodes[i].classList.add(path.good ? "done" : i === path.steps.length - 1 ? "fail" : "done");
      }, 400 + i * 520)
    );
  });

  timers.push(
    setTimeout(() => {
      result.textContent = path.result;
      result.className = "trace-result " + (path.good ? "good" : "bad");
      running = false;
      runBtn.disabled = false;
      document.getElementById("run-hint").textContent = path.good
        ? "Version-aware path held the line."
        : "Naive path used a stale rule — common production bug.";
    }, 400 + path.steps.length * 520)
  );
}

function setScene(key) {
  if (!SCENES[key]) return;
  sceneKey = key;
  renderTabs();
  renderCopy();
  resetTrace();
}

function setMode(next) {
  mode = next;
  renderCopy();
  resetTrace();
}

function wireDemo() {
  const root = document.getElementById("demo-root");
  if (!root) return;

  // Event delegation — clicks always work
  root.addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab && tab.dataset.scene) {
      e.preventDefault();
      userTouched = true;
      setScene(tab.dataset.scene);
      return;
    }
    const modeBtn = e.target.closest(".mode-btn");
    if (modeBtn && modeBtn.dataset.mode) {
      e.preventDefault();
      userTouched = true;
      setMode(modeBtn.dataset.mode);
      return;
    }
    if (e.target.closest("#run-btn")) {
      e.preventDefault();
      runDecision();
    }
  });

  document.getElementById("copy-embed")?.addEventListener("click", async () => {
    const text = document.getElementById("embed-code")?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      const btn = document.getElementById("copy-embed");
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = "Copy embed code";
      }, 1500);
    } catch (_) {
      alert("Copy failed — select the code block manually.");
    }
  });

  // Gentle auto-rotate ONLY until user interacts
  const keys = Object.keys(SCENES);
  let i = 0;
  setInterval(() => {
    if (userTouched || running) return;
    i = (i + 1) % keys.length;
    setScene(keys[i]);
  }, 9000);
}

async function boot() {
  renderTabs();
  renderCopy();
  resetTrace();
  wireDemo();
  renderImpact(window.POLICYSHIFT_CARD);
  try {
    const res = await fetch("/api/portfolio", { cache: "no-store" });
    if (res.ok) renderImpact(await res.json());
  } catch (_) {
    /* static hosting is fine */
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    boot();
  });
} else {
  boot();
}
