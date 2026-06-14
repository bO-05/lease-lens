const bootstrap = JSON.parse(document.getElementById("lease-lens-bootstrap").textContent);

const els = {
  proofChips: document.getElementById("proofChips"),
  exampleSelect: document.getElementById("exampleSelect"),
  sourceBanner: document.getElementById("sourceBanner"),
  fileInput: document.getElementById("fileInput"),
  fileName: document.getElementById("fileName"),
  contractText: document.getElementById("contractText"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  clearBtn: document.getElementById("clearBtn"),
  runState: document.getElementById("runState"),
  emptyState: document.getElementById("emptyState"),
  results: document.getElementById("results"),
  scoreSeal: document.getElementById("scoreSeal"),
  scoreValue: document.getElementById("scoreValue"),
  verdictText: document.getElementById("verdictText"),
  summaryText: document.getElementById("summaryText"),
  findingCards: document.getElementById("findingCards"),
  coverageNote: document.getElementById("coverageNote"),
  highlightedDoc: document.getElementById("highlightedDoc"),
  emailBtn: document.getElementById("emailBtn"),
  emailOut: document.getElementById("emailOut"),
};

let client = null;
let lastFindings = [];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function unwrap(result) {
  if (result && Array.isArray(result.data)) return result.data[0];
  return result;
}

async function connectClient() {
  try {
    const mod = await import("https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js");
    client = await mod.Client.connect(window.location.origin);
    els.runState.textContent = bootstrap.mock_mode ? "Mock preview active" : "Ready on ZeroGPU";
  } catch (error) {
    client = null;
    els.runState.textContent = bootstrap.mock_mode ? "Mock REST fallback" : "Client loading failed";
  }
}

async function predict(apiName, payload, restPath, method = "POST") {
  if (client) {
    const result = await client.predict(apiName, payload);
    return unwrap(result);
  }
  if (method === "GET") {
    const params = new URLSearchParams(payload);
    const response = await fetch(`${restPath}?${params}`);
    return response.json();
  }
  const response = await fetch(restPath, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

function setBusy(isBusy, label = "Running redline scan") {
  els.analyzeBtn.disabled = isBusy;
  els.emailBtn.disabled = isBusy;
  els.runState.textContent = isBusy ? label : (bootstrap.mock_mode ? "Mock preview active" : "Ready on ZeroGPU");
}

function populateProof() {
  els.proofChips.innerHTML = "";
  for (const chip of bootstrap.proof_chips || []) {
    const el = document.createElement("span");
    el.textContent = chip;
    els.proofChips.appendChild(el);
  }
}

function populateExamples() {
  els.exampleSelect.innerHTML = "";
  for (const name of bootstrap.examples || []) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    els.exampleSelect.appendChild(option);
  }
  els.exampleSelect.value = bootstrap.default_example;
}

async function loadExample(name) {
  setBusy(true, "Loading docket");
  try {
    const payload = await predict("/get_example", { name }, "/api/get_example", "GET");
    els.contractText.value = payload.text || "";
    els.sourceBanner.innerHTML = payload.source_banner_html || '<div class="source-banner"><span class="source-dot">SMP</span>Synthetic teaching sample. Upload a .txt contract or choose a real SEC filing.</div>';
    els.fileName.textContent = payload.is_real ? "real public filing loaded" : "sample loaded";
  } finally {
    setBusy(false);
  }
}

function renderEmpty(message) {
  els.emptyState.classList.remove("hidden");
  els.results.classList.add("hidden");
  if (message) {
    els.emptyState.querySelector("p").textContent = message;
  }
}

function scoreClass(score, highCount, flagCount) {
  if (highCount > 0) return "high";
  if (flagCount > 0 || score > 0) return "med";
  return "clean";
}

function renderFindings(findings) {
  if (!findings.length) {
    els.findingCards.innerHTML = '<div class="finding-card"><h3>No risky clauses flagged <span class="risk-pill">clear</span></h3><p>The checked clause categories did not produce a grounded risky-clause flag.</p></div>';
    return;
  }
  const sorted = [...findings].sort((a, b) => (a.risk === "high" ? -1 : 1) - (b.risk === "high" ? -1 : 1));
  els.findingCards.innerHTML = sorted.map((finding) => `
    <article class="finding-card ${finding.risk === "high" ? "high" : ""}">
      <h3>${escapeHtml(finding.label)} <span class="risk-pill">${escapeHtml(finding.risk)}</span></h3>
      <pre class="quote">${escapeHtml(finding.text)}</pre>
      <p><b>Why it matters:</b> ${escapeHtml(finding.why)}</p>
      <p><b>Push back:</b> ${escapeHtml(finding.tip)}</p>
    </article>
  `).join("");
}

function renderResults(data) {
  if (!data || data.status === "empty") {
    renderEmpty(data?.message || "Paste or pick a contract first.");
    return;
  }
  if (data.status === "error") {
    renderEmpty(data.message || "Analysis failed. Try again shortly.");
    return;
  }

  lastFindings = data.findings || [];
  els.emptyState.classList.add("hidden");
  els.results.classList.remove("hidden");

  const sealClass = scoreClass(data.score || 0, data.high_count || 0, data.flag_count || 0);
  els.scoreSeal.className = `score-seal ${sealClass}`;
  els.scoreValue.textContent = data.score ?? 0;
  els.verdictText.textContent = data.verdict || "Review complete";
  els.summaryText.textContent = `${data.flag_count || 0} clauses flagged (${data.high_count || 0} high-risk) of ${data.checked_count || 0} checked.`;
  renderFindings(lastFindings);

  const skipped = data.skipped?.length ? `Skipped because keywords were absent: ${data.skipped.join(", ")}.` : "";
  const note = data.coverage_note ? `${data.coverage_note} ` : "";
  els.coverageNote.textContent = `${note}${skipped} ${data.disclaimer || ""}`.trim();
  els.highlightedDoc.innerHTML = data.highlighted_html || '<div class="contract-page">No source text available.</div>';
  els.emailOut.value = "";
}

async function analyze() {
  const text = els.contractText.value.trim();
  setBusy(true, "Running batched checks");
  try {
    const data = await predict("/analyze_contract", { text }, "/api/analyze_contract");
    renderResults(data);
  } catch (error) {
    renderEmpty(`Analysis failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function draftEmail() {
  if (!lastFindings.length) {
    els.emailOut.value = "Run an analysis first - then I can draft the email from the flagged clauses.";
    return;
  }
  setBusy(true, "Drafting pushback");
  try {
    const data = await predict("/draft_email", { state_json: JSON.stringify(lastFindings) }, "/api/draft_email");
    els.emailOut.value = data.email || "";
  } catch (error) {
    els.emailOut.value = `Draft failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  els.exampleSelect.addEventListener("change", () => loadExample(els.exampleSelect.value));
  els.analyzeBtn.addEventListener("click", analyze);
  els.emailBtn.addEventListener("click", draftEmail);
  els.clearBtn.addEventListener("click", () => {
    els.contractText.value = "";
    els.sourceBanner.innerHTML = "";
    els.fileName.textContent = "or edit the docket text below";
    lastFindings = [];
    renderEmpty("Paste or upload a contract, then run the scan.");
  });
  els.fileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    els.contractText.value = await file.text();
    els.sourceBanner.innerHTML = "";
    els.fileName.textContent = file.name;
    lastFindings = [];
    renderEmpty("Uploaded text is ready for analysis.");
  });
}

async function init() {
  populateProof();
  populateExamples();
  bindEvents();
  await connectClient();
  await loadExample(bootstrap.default_example);
}

init();
