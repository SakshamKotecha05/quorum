"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icons = {
  search: '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 10 9 5 9-5M3 18l9 5 9-5"/>',
  chart: '<path d="M5 20v-7m7 7V7m7 13V3"/>',
  refresh: '<path d="M3 10a9 9 0 1 1 1 9M3 4v6h6"/>',
  document: '<path d="M5 2h9l5 5v15H5V2Zm9 0v6h5M8 13h8M8 17h6"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6h6"/>',
  screen: '<path d="M2 3h20v14H2zM8 21h8m-4-4v4"/>',
  quote: '<path d="M9 5C3 6 3 10 3 14h6v5H3v-5m18-9c-6 1-6 5-6 9h6v5h-6v-5"/>',
  shield: '<path d="m12 2 9 4v6c0 6-9 10-9 10S3 18 3 12V6l9-4Z"/>',
  pen: '<path d="m4 16 12-12 4 4L8 20l-5 1 1-5Zm9-9 4 4"/>',
};
function icon(name) {
  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${icons[name] || icons.document}</svg>`;
}
$$("[data-icon]").forEach((e) => (e.innerHTML = icon(e.dataset.icon)));
const pages = {
  documents: "Documents",
  research: "Research workspace",
  architecture: "Architecture",
  evaluation: "Evaluation",
  recovery: "Recovery",
  retrieval: "Document search",
  limits: "Usage limits",
};
function navigate() {
  const page = Object.hasOwn(pages, location.hash.slice(1))
    ? location.hash.slice(1)
    : "research";
  $$(".page").forEach((s) => (s.hidden = s.id !== page));
  $$("nav a").forEach((a) => {
    if (a.dataset.page === page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("#page-label").textContent = pages[page];
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", navigate);
navigate();
$$("[data-go]").forEach(
  (b) => (b.onclick = () => (location.hash = b.dataset.go)),
);
$("#present").onclick = () => {
  const active = document.body.classList.toggle("presentation");
  $("#present").innerHTML =
    icon("screen") + (active ? "Exit presentation" : "Presentation mode");
  $("#present").setAttribute("aria-pressed", String(active));
};
function fail(error) {
  const e = $("#global-error");
  e.hidden = false;
  e.textContent = error.message || String(error);
}
function clearError() {
  $("#global-error").hidden = true;
}
let publicDemo = false;
let browserDocuments = {};
async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please try again.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {}
    throw Error(message);
  }
  return response.json();
}
const stages = [
  ["Plan", "document"],
  ["Research", "search"],
  ["Check quotes", "quote"],
  ["Review claims", "shield"],
  ["Write", "pen"],
];
let stageStates = stages.map(() => "Waiting");
function renderFlow() {
  $("#flow").innerHTML = stages
    .map(
      ([label, glyph], i) =>
        `<li class="${stageStates[i] === "Complete" ? "complete" : stageStates[i] === "Running" ? "active" : ""}"><span class="stage-title">${icon(glyph)}${label}</span><small>${esc(stageStates[i])}</small></li>`,
    )
    .join("");
}
renderFlow();
let documentCount = 0,
  researchBusy = false,
  providerReady = false;
let claims = [],
  filter = "all",
  source = null,
  runStart = 0,
  terminalReceived = false;
function renderClaims() {
  const visible = claims.filter(
    (c) =>
      filter === "all" ||
      (filter === "verified"
        ? c.outcome === "verified"
        : c.outcome !== "verified"),
  );
  $("#claims").innerHTML = visible.length
    ? visible
        .map((c) => {
          const accepted = c.outcome === "verified";
          const reason = accepted
            ? "Accepted"
            : c.outcome === "rejected_by_prefilter"
              ? "Quote not found"
              : "Review rejected";
          return `<article class="claim"><div class="claim-top"><span>${esc(c.source_id)}</span><span class="decision ${accepted ? "" : "rejected"}">${reason}</span></div><h3>${esc(c.text)}</h3><blockquote>${esc(c.quote)}</blockquote><details><summary>Review details</summary><p>${c.votes.length ? c.votes.map((vote, i) => `${["Support", "Overreach", "Attribution"][i]}: ${vote ? "accept" : "reject"}`).join(" / ") : "Rejected before model review. The quote was not found in its cited source."}</p><p>Claim ID: ${esc(c.id)}</p></details></article>`;
        })
        .join("")
    : `<div class="empty"><h3>${claims.length ? "No claims in this filter" : "No claims returned"}</h3><p>${claims.length ? "Try another filter." : "The run did not produce any claim evidence."}</p></div>`;
}
$$("[data-filter]").forEach(
  (b) =>
    (b.onclick = () => {
      filter = b.dataset.filter;
      $$("[data-filter]").forEach((x) =>
        x.setAttribute("aria-pressed", String(x === b)),
      );
      if (claims.length) renderClaims();
    }),
);
function log(message) {
  const item = document.createElement("li");
  const elapsed = ((Date.now() - runStart) / 1000).toFixed(1);
  item.innerHTML = `<time>${elapsed}s</time>${esc(message)}`;
  $("#activity").prepend(item);
  while ($("#activity").children.length > 9) $("#activity").lastChild.remove();
}
function finishButton() {
  $("#provider-mode").disabled = false;
  const b = $("#run-button");
  researchBusy = false;
  b.disabled = !providerReady || !documentCount;
  b.textContent = "Run research";
}
function updateEvent(event) {
  if (event.event === "node") {
    let index =
      event.role === "planner"
        ? 0
        : event.role === "researcher"
          ? 1
          : event.role.startsWith("verifier")
            ? 3
            : 4;
    if (event.status === "running") stageStates[index] = "Running";
    if (index === 0 && event.status === "done") stageStates[0] = "Complete";
    if (event.status === "failed") stageStates[index] = "Failed";
    log(`${event.id}: ${event.status}`);
  } else if (event.event === "prefilter") {
    stageStates[1] = "Complete";
    stageStates[2] = "Complete";
    log(`${event.rejected_free} quotes rejected before review`);
  } else if (event.event === "quorum") {
    stageStates[3] = "Complete";
    log(`${event.verified} claims accepted`);
  } else if (event.event === "done") {
    terminalReceived = true;
    source?.close();
    finishButton();
    const ok = event.status === "done";
    $("#run-state").textContent = ok ? "Run complete" : "Run failed";
    if (ok) stageStates = stages.map(() => "Complete");
    claims = event.claims || [];
    renderClaims();
    const m = event.metrics;
    $("#summary").innerHTML = [
      ["Model calls", m.llm_calls],
      ["Proposed claims", m.claims_proposed ?? 0],
      ["Accepted claims", m.claims_verified ?? 0],
      ["Quotes rejected", m.rejected_by_prefilter ?? 0],
      ["Review rejections", m.rejected_by_quorum ?? 0],
      ["Elapsed time", `${m.wall_s}s`],
    ]
      .map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`)
      .join("");
    $("#report-panel").hidden = false;
    $("#report").textContent =
      event.report?.markdown || "No report was produced.";
    if (!ok)
      fail(Error("The run failed. Inspect its activity, then try again."));
  } else if (event.event === "error") {
    terminalReceived = true;
    source?.close();
    finishButton();
    $("#run-state").textContent = "Run failed";
    fail(Error(event.detail || "Run failed. Please try again."));
  }
  renderFlow();
}
$("#research-form").onsubmit = async (event) => {
  event.preventDefault();
  if (!documentCount)
    return fail(Error("Add documents before starting research."));
  researchBusy = true;
  clearError();
  if (source) source?.close();
  claims = [];
  terminalReceived = false;
  stageStates = stages.map(() => "Waiting");
  renderFlow();
  runStart = Date.now();
  $("#activity").innerHTML = "";
  $("#claims").innerHTML =
    '<div class="busy"><span class="spinner"></span>Collecting evidence and reviewing claims...</div>';
  $("#summary").innerHTML = [
    "Model calls",
    "Proposed claims",
    "Accepted claims",
    "Elapsed time",
  ]
    .map((k) => `<div><dt>${k}</dt><dd>-</dd></div>`)
    .join("");
  $("#report-panel").hidden = true;
  $("#trace").innerHTML = "";
  const live = $("#provider-mode").value === "live";
  $("#provider-mode").disabled = true;
  $("#run-button").disabled = true;
  $("#run-button").textContent = "Researching...";
  $("#run-state").textContent = "Run in progress";
  try {
    const run = await request("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: $("#question").value,
        mock: !live,
        documents: "uploaded",
        ...(publicDemo ? { uploaded_documents: browserDocuments } : {}),
        ...(live
          ? {}
          : {
              org_rpm: 100000,
              org_tpm: 10000000,
              model_rpm: 100000,
              model_tpm: 10000000,
            }),
      }),
    });
    log(
      live
        ? "Live provider connected. Rate limits may cause waiting."
        : "Offline test run started.",
    );
    $("#report-hint").textContent = live
      ? "Generated by the live provider from accepted corpus claims. Review the sources; the final prose is not independently checked."
      : "The offline provider produces placeholder prose to test the workflow. Acceptance does not guarantee factual accuracy.";
    if (run.result) {
      updateEvent(run.result);
      renderTrace(run.trace);
      return;
    }
    source = new EventSource(run.events);
    source.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        updateEvent(data);
        if (data.event === "done")
          request(`/runs/${run.run_id}/trace`).then(renderTrace).catch(fail);
      } catch (err) {
        fail(err);
        source?.close();
        finishButton();
      }
    };
    source.addEventListener("end", () => {
      source?.close();
      if (!terminalReceived) {
        finishButton();
        $("#run-state").textContent = "Stream ended";
        fail(
          Error(
            "The stream ended before a result arrived. Try running the question again.",
          ),
        );
      }
    });
    source.onerror = () => {
      source?.close();
      if (!terminalReceived) {
        finishButton();
        $("#run-state").textContent = "Connection lost";
        fail(
          Error(
            "Connection lost. The run may still be running. Check the local server, then retry.",
          ),
        );
      }
    };
  } catch (err) {
    finishButton();
    $("#run-state").textContent = "Could not start";
    $("#claims").innerHTML =
      '<div class="empty"><h3>Could not start research</h3><p>Check the message above and try again.</p></div>';
    fail(err);
  }
};
function renderTrace(data) {
  $("#trace").innerHTML =
    `<table><thead><tr><th>Task</th><th>Model</th><th>Tokens</th><th>Time</th></tr></thead><tbody>${data.spans.map((s) => `<tr><td>${esc(s.node_id)}</td><td>${esc(s.model)}</td><td>${s.in_tok + s.out_tok}</td><td>${s.ms} ms</td></tr>`).join("")}</tbody></table>`;
}
function renderEvaluation(data) {
  const labels = ["No checking", "One review", "Three reviews"];
  $("#evaluation-results").innerHTML =
    `<div class="score-grid">${data.rows.map((r, i) => `<div class="score-item"><h2>${labels[i]}</h2><strong>${r.f1.toFixed(2)}</strong><span>F1 score</span><div class="meter"><i style="width:${r.f1 * 100}%"></i></div></div>`).join("")}</div><p class="takeaway">${data.rows[1].false_reject - data.rows[2].false_reject} more valid claims kept. The same ${data.rows[2].leaked} mistakes still slip through.</p><div class="panel table-wrap"><table><thead><tr><th>Approach</th><th>Claims tested</th><th>Precision</th><th>Recall</th><th>Valid claims cut</th><th>Review calls</th></tr></thead><tbody>${data.rows.map((r, i) => `<tr><td>${labels[i]}</td><td>${r.proposed}</td><td>${r.precision.toFixed(2)}</td><td>${r.recall.toFixed(2)}</td><td>${r.false_reject}</td><td>${r.verifier_calls}</td></tr>`).join("")}</tbody></table></div><p class="hint">F1 balances keeping valid claims and rejecting mistakes. Both review approaches first check that quotes exist in their sources. Review calls exclude research and writing.</p>`;
}
function renderRetrieval(data) {
  $("#retrieval-results").innerHTML =
    `<div class="score-grid">${[1, 3, 5].map((k) => `<div class="score-item"><h2>First ${k} ${k === 1 ? "result" : "results"}</h2><strong>${(data["recall@" + k] * 100).toFixed(1)}%</strong><span>Average relevant passages found</span><div class="meter"><i style="width:${data["recall@" + k] * 100}%"></i></div></div>`).join("")}</div>${data.rows
      .filter((r) => r["recall@5"] < 1)
      .map(
        (r) =>
          `<p class="miss"><strong>A miss worth investigating</strong><br>${esc(r.query)}<br>Expected: ${r.relevant.map(esc).join(", ")}</p>`,
      )
      .join(
        "",
      )}<div class="panel query-list"><h2>Inspect all ${data.queries} test questions</h2>${data.rows.map((r) => `<details><summary>${esc(r.query)}</summary><p>Relevant: ${r.relevant.map(esc).join(", ")}</p><p>Retrieved: ${r.retrieved.map(esc).join(", ")}</p><p>Recall in the first five: ${(r["recall@5"] * 100).toFixed(0)}%</p></details>`).join("")}</div>`;
}
function renderLimits(data) {
  const rows = data.benchmark.rows.filter(
    (r) => r.engine === "custom" || r.engine === "custom-throttled",
  );
  const max = Math.max(...rows.map((r) => r.wall_s));
  $("#limit-results").innerHTML =
    rows
      .map(
        (r) =>
          `<div class="limit-bar"><span>${r.engine === "custom" ? "Limits lifted" : "Limits enabled"}</span><div class="meter"><i style="width:${Math.max(3, (r.wall_s / max) * 100)}%"></i></div><strong>${r.wall_s.toFixed(2)} s</strong></div>`,
      )
      .join("") +
    `<p>${rows[0].llm_calls} model calls and ${rows[0].tokens_used.toLocaleString()} estimated tokens in each run.</p>`;
}
function action(buttonId, statusId, url, render, loading) {
  const b = $(buttonId);
  b.onclick = async () => {
    clearError();
    const old = b.textContent;
    b.disabled = true;
    b.textContent = "Running...";
    $(statusId).textContent = loading;
    try {
      const result = await request(url, { method: "POST" });
      render(result);
      $(statusId).textContent = publicDemo
        ? "Completed just now on the demo server"
        : "Completed just now on this machine";
    } catch (e) {
      $(statusId).textContent = "Test failed. Please try again.";
      fail(e);
    } finally {
      b.disabled = false;
      b.textContent = old;
    }
  };
}
action(
  "#evaluate-button",
  "#evaluation-status",
  "/demo/evaluate",
  renderEvaluation,
  "Generating one workload and comparing the three review policies...",
);
action(
  "#retrieval-button",
  "#retrieval-status",
  "/demo/retrieval",
  renderRetrieval,
  "Searching the local corpus for all 14 questions...",
);
action(
  "#recover-button",
  "#recovery-status",
  "/demo/recover",
  (data) => {
    $("#recovery-results").innerHTML =
      `<div class="recovery-success"><strong>${data.replayed}</strong><div><h2>Completed calls replayed</h2><p>${data.reused} saved tasks reused. All ${data.completed} tasks finished.</p><p class="hint">At interruption: ${data.before.done || 0} completed, ${data.before.running || 0} running.</p></div></div>`;
  },
  "Starting, interrupting and resuming a real process. This takes a few seconds...",
);
request("/demo/measurements")
  .then((data) => {
    renderEvaluation(data.verification);
    renderRetrieval(data.retrieval);
    renderLimits(data);
  })
  .catch(fail);

let liveLabel = "Groq live";
function updateMode() {
  const live = $("#provider-mode").value === "live";
  $("#provider-status").innerHTML = live
    ? `${esc(liveLabel)}<small>Research connected</small>`
    : "Offline demo<small>No API key required</small>";
  $("#provider-hint").textContent = live
    ? "Real model responses from your local documents. Provider limits apply; a run can take a few minutes."
    : "Repeatable offline responses. Real workflow execution.";
}
$("#provider-mode").onchange = updateMode;
request("/demo/provider")
  .then((data) => {
    publicDemo = Boolean(data.public_demo);
    if (publicDemo) {
      $("#public-notice").hidden = false;
      $("#sample-button").hidden = false;
      $("#document-note").textContent =
        "Your document library stays in this browser tab. Documents are sent to the server for validation and research, then discarded. Closing the tab clears the library. Use sample documents to try the prototype. Benchmarks always use the fixed sample set.";
      $("footer span").textContent = "Quorum / Interactive prototype";
      try {
        browserDocuments = JSON.parse(
          sessionStorage.getItem("quorum-documents") || "{}",
        );
      } catch {
        browserDocuments = {};
      }
      renderDocuments({documents: [], passages: 0});
      if (Object.keys(browserDocuments).length) {
        request("/documents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(browserDocuments),
        })
          .then(renderDocuments)
          .catch(fail);
      }
      if (!location.hash) {
        location.hash = "documents";
      }
    } else {
      request("/documents").then(renderDocuments).catch(fail);
    }
    liveLabel = data.label;
    const option = $("#provider-mode option[value=live]");
    option.textContent = data.live_available
      ? liveLabel
      : `${liveLabel} (key not configured)`;
    option.disabled = !data.live_available;
    $("#provider-mode").value = data.live_available ? "live" : "mock";
    providerReady = true;
    updateMode();
    finishButton();
  })
  .catch((error) => {
    providerReady = true;
    updateMode();
    finishButton();
    fail(error);
  });

function renderDocuments(data) {
  documentCount = data.documents.length;
  $("#document-count").textContent =
    `${documentCount} documents / ${data.passages} passages`;
  $("#clear-documents").disabled = !documentCount;
  $("#documents-continue").disabled = !documentCount;
  $("#research-documents").innerHTML = documentCount
    ? `Research uses your ${documentCount} uploaded documents (${data.passages} passages). <a href="#documents">Manage documents</a>`
    : 'Add your sources first. <a href="#documents">Open Documents</a> to upload the sample files.';
  $("#document-list").innerHTML = documentCount
    ? data.documents
        .map(
          (d) =>
            `<details class="document-item"><summary>${esc(d.name)} <span class="hint">${d.passages} passages / ${(d.bytes / 1000).toFixed(1)} KB</span></summary><pre>${esc(d.text)}</pre></details>`,
        )
        .join("")
    : '<div class="empty"><h3>No documents added yet</h3><p>Choose files above, then click Add documents to make them available for research.</p></div>';
  if (!researchBusy)
    $("#run-button").disabled = !providerReady || !documentCount;
}
$("#upload-form").onsubmit = async (event) => {
  event.preventDefault();
  clearError();
  const button = $("#upload-button");
  button.disabled = true;
  button.textContent = "Adding...";
  $("#upload-status").textContent = "Reading and indexing documents...";
  try {
    const files = [...$("#document-files").files];
    if (!files.length || files.length > 20)
      throw Error("Choose between 1 and 20 files.");
    const documents = Object.create(null);
    for (const file of files) {
      if (file.size > 200000)
        throw Error(`${file.name}: maximum file size is 200 KB.`);
      if (Object.hasOwn(documents, file.name))
        throw Error("Choose files with distinct names.");
      try {
        documents[file.name] = new TextDecoder("utf-8", { fatal: true }).decode(
          await file.arrayBuffer(),
        );
      } catch {
        throw Error(`${file.name}: choose a UTF-8 text file.`);
      }
    }
    await saveDocuments(documents);
    $("#upload-status").textContent =
      `${files.length} documents added. Ready for research.`;
    $("#document-files").value = "";
  } catch (error) {
    $("#upload-status").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Add documents";
  }
};
$("#clear-documents").onclick = async () => {
  if (
    !confirm(
      "Clear the uploaded copies for a fresh demo? Original files and completed runs will stay unchanged.",
    )
  )
    return;
  try {
    if (publicDemo) {
      browserDocuments = {};
      sessionStorage.removeItem("quorum-documents");
      renderDocuments({ documents: [], passages: 0 });
    } else renderDocuments(await request("/documents", { method: "DELETE" }));
    $("#upload-status").textContent =
      "Library cleared. Choose the sample files to start again.";
  } catch (error) {
    fail(error);
  }
};

async function saveDocuments(documents) {
  const merged = publicDemo ? { ...browserDocuments, ...documents } : documents;
  const result = await request("/documents", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(merged),
  });
  if (publicDemo) {
    sessionStorage.setItem("quorum-documents", JSON.stringify(merged));
    browserDocuments = merged;
  }
  renderDocuments(result);
}
$("#sample-button").onclick = async () => {
  const button = $("#sample-button");
  button.disabled = true;
  try {
    await saveDocuments(await request("/demo/samples"));
    $("#upload-status").textContent =
      "Four sample documents added. Continue to research to try the workflow.";
  } catch (error) {
    fail(error);
  } finally {
    button.disabled = false;
  }
};
