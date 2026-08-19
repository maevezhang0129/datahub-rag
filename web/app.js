/* datahub-rag UI.
 *
 * The conversation is the ordinary half. The inspector is the point: it
 * renders the `trace` the API returns for every turn, so the pipeline --
 * guard, interpretation, relevance gate, retrieval, citation audit -- is
 * visible rather than implied.
 *
 * Deliberately not streamed. The citation audit runs over the finished answer:
 * it strips markers that point at sources which were never supplied. Streaming
 * raw tokens would show the reader citations that verification then removes,
 * which is exactly the failure this system exists to prevent.
 */

const $ = (sel) => document.querySelector(sel);

const thread = $("#thread");
const traceEl = $("#trace");
const tookEl = $("#took");
const form = $("#composer");
const input = $("#question");
const sendBtn = $("#send");
const modeSel = $("#mode");

let sessionId = null;
let turnCount = 0;

/* -- helpers ----------------------------------------------------------- */

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const pct = (x) => `${(x * 100).toFixed(0)}%`;

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

/* -- census ------------------------------------------------------------ */

async function loadCensus() {
  try {
    const r = await fetch("/health").then((r) => r.json());
    const c = r.corpus || {};
    $("#census").textContent =
      `${c.documents ?? "?"} docs · ${c.chunks ?? "?"} chunks · ${r.model}`;
  } catch {
    $("#census").textContent = "api unreachable";
  }
}

/* -- rendering: conversation ------------------------------------------- */

function renderAnswer(data) {
  let body = data.answer;
  let notice = "";

  // The guard notice is appended to the answer by the answer stage so it
  // survives the model ignoring the instruction. Split it back out here so it
  // reads as a system notice rather than as something the model wrote.
  const n = data.guard && data.guard.notice;
  if (n && body.endsWith(n)) {
    body = body.slice(0, -n.length).trimEnd();
    notice = n;
  }

  const withCites = esc(body).replace(
    /\[(\d+)\]/g,
    (_, d) => `<button class="cite" data-n="${d}" title="source ${d}">[${d}]</button>`
  );

  return `
    <p class="a${data.refused ? " refused" : ""}">${withCites}</p>
    ${notice ? `<div class="notice">${esc(notice)}</div>` : ""}`;
}

function addTurn(data) {
  const id = ++turnCount;
  const cited = (data.citations && data.citations.cited_sources) || [];

  const node = el(`
    <article class="turn active" data-turn="${id}">
      <p class="q">${esc(data.question)}</p>
      ${renderAnswer(data)}
      <div class="turn-foot">
        <span>${data.trace.mode}</span>
        <span>${data.sources.length} source${data.sources.length === 1 ? "" : "s"}</span>
        <span>${cited.length} cited</span>
        <span>${data.took_ms} ms</span>
        <button type="button" class="show-trace">show trace</button>
      </div>
    </article>`);

  node._data = data;

  document.querySelectorAll(".turn").forEach((t) => t.classList.remove("active"));
  thread.append(node);
  thread.scrollTop = thread.scrollHeight;

  node.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    select(node);
    if (cite) flashSource(Number(cite.dataset.n));
  });

  select(node);
}

function select(node) {
  document.querySelectorAll(".turn").forEach((t) => t.classList.remove("active"));
  node.classList.add("active");
  renderTrace(node._data);
}

function flashSource(n) {
  const card = traceEl.querySelector(`.source[data-n="${n}"]`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1100);
}

/* -- rendering: the inspector ------------------------------------------ */

function stageGuard(g) {
  const on = g.triggered;
  return `
    <div class="stage ${on ? "off" : "on"}">
      <h3>1 · guard</h3>
      ${on
        ? `<p class="line">Regulated advice detected. The answer is constrained
             to information and carries a referral.</p>
           <div>${g.domains.map((d) => `<span class="pill warn">${esc(d)}</span>`).join("")}</div>`
        : `<p class="line dim">No regulated-advice pattern. Keyword-matched
             before retrieval, so it cannot fail open.</p>`}
    </div>`;
}

function stageInterpret(i, trace, question) {
  const rewritten = trace.query && trace.query !== question;
  return `
    <div class="stage on">
      <h3>2 · interpret</h3>
      <p class="line kv">mode <b>${esc(i.mode || "—")}</b> ·
         confidence <b>${i.confidence ?? "—"}</b>${i.is_followup ? " · <b>follow-up</b>" : ""}</p>
      <div>${(i.query_terms || []).map((t) => `<span class="pill">${esc(t)}</span>`).join("")}</div>
      ${(i.entities || []).length
        ? `<p class="line dim" style="margin-top:4px">entities:
             ${i.entities.map((e) => esc(e)).join(", ")}</p>`
        : ""}
      ${rewritten
        ? `<p class="line dim" style="margin-top:6px">query sent to retrieval:</p>
           <div class="rewrite">${esc(trace.query)}</div>`
        : ""}
    </div>`;
}

function stageGate(t) {
  const passed = t.gate_passed;
  const width = Math.max(0, Math.min(1, t.relevance)) * 100;
  return `
    <div class="stage ${passed ? "on" : "off"}">
      <h3>3 · relevance gate</h3>
      <div class="meter">
        <div class="meter-track">
          <div class="meter-fill ${passed ? "" : "under"}" style="width:${width}%"></div>
          <div class="meter-threshold" style="left:${t.threshold * 100}%"
               title="MIN_RELEVANCE = ${t.threshold}"></div>
        </div>
        <div class="meter-legend">
          <span>best cosine <b>${t.relevance.toFixed(3)}</b></span>
          <span>threshold ${t.threshold}</span>
        </div>
      </div>
      <p class="line ${passed ? "dim" : ""}">
        ${passed
          ? "Above threshold — the corpus can speak to this."
          : "Below threshold — refused without generating. Dense retrieval always returns its top-k however weak the match, so the score, not the result count, is what decides."}
      </p>
    </div>`;
}

function stageRetrieve(t) {
  return `
    <div class="stage on">
      <h3>4 · retrieve</h3>
      <p class="line kv"><b>${esc(t.mode)}</b> · over-fetched
         <b>${t.candidates}</b> → kept <b>${t.kept}</b> from
         <b>${t.documents}</b> document${t.documents === 1 ? "" : "s"}</p>
      <p class="line dim">
        ${t.displaced > 0
          ? `The per-document cap displaced <b>${t.displaced}</b> chunk${t.displaced === 1 ? "" : "s"}
             a plain top-${t.top_k} would have kept. No document may contribute
             more than 2, so one long article cannot fill the context and read
             as corroboration.`
          : `A plain top-${t.top_k} would have returned the same set — the
             per-document cap did not bind here.`}
      </p>
    </div>`;
}

function stageAudit(c, sources) {
  const g = c.groundedness ?? 0;
  const bad = (c.invalid_markers || []).length;
  const uncited = c.uncited_count || 0;
  return `
    <div class="stage ${bad ? "off" : "on"}">
      <h3>5 · citation audit</h3>
      <p class="line kv">groundedness <b>${pct(g)}</b>
         (${c.cited_sentences}/${c.total_sentences} sentences) ·
         <b>${(c.cited_sources || []).length}</b>/${sources.length} sources cited</p>
      ${bad
        ? `<p class="line"><span class="pill bad">stripped ${bad} fabricated
             marker${bad === 1 ? "" : "s"}</span> pointing at sources that were
             never supplied: ${c.invalid_markers.join(", ")}</p>`
        : `<p class="line dim">Every marker resolved to a supplied source.</p>`}
      ${uncited
        ? `<p class="line dim">${uncited} substantive sentence${uncited === 1 ? "" : "s"} carried no citation.</p>`
        : ""}
    </div>`;
}

function renderSources(sources) {
  if (!sources.length) return "";
  return `
    <div class="stage on" style="border-left-color:transparent">
      <h3>context shown to the model</h3>
      <div class="sources">
        ${sources.map((s) => `
          <div class="source ${s.cited ? "cited" : "uncited"}" data-n="${s.n}">
            <div class="source-head">
              <span class="source-n">[${s.n}]</span>
              <span class="source-title">${esc(s.title)}</span>
              <span class="source-score">${s.score.toFixed(4)}</span>
            </div>
            <div class="source-meta">
              <span>${s.cited ? "cited" : "not cited"}</span>
              <span>chunk ${s.chunk_id}</span>
              <span>doc ${s.document_id}</span>
              ${s.published_at ? `<span>${esc(s.published_at)}</span>` : ""}
              ${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">source ↗</a>` : ""}
            </div>
            <div class="source-text" title="click to expand">${esc(s.text)}</div>
          </div>`).join("")}
      </div>
    </div>`;
}

function renderTrace(d) {
  tookEl.textContent = `${d.took_ms} ms`;
  const t = d.trace;

  traceEl.innerHTML = [
    stageGuard(d.guard || {}),
    stageInterpret(d.interpretation || {}, t, d.question),
    stageGate(t),
    // Retrieval and everything after it only happen if the gate opened.
    t.gate_passed ? stageRetrieve(t) : "",
    t.gate_passed ? stageAudit(d.citations || {}, d.sources) : "",
    renderSources(d.sources),
  ].join("");

  traceEl.scrollTop = 0;

  traceEl.querySelectorAll(".source-text").forEach((n) =>
    n.addEventListener("click", () => n.classList.toggle("open")));
}

/* -- asking ------------------------------------------------------------ */

async function ask(question) {
  input.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "…";
  const empty = thread.querySelector(".empty");
  if (empty) empty.remove();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId, mode: modeSel.value }),
    });
    if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 200)}`);
    const data = await res.json();
    sessionId = data.session_id;
    addTurn(data);
  } catch (err) {
    thread.append(el(`<div class="err">${esc(err.message)}</div>`));
    thread.scrollTop = thread.scrollHeight;
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Ask";
    input.focus();
  }
}

/* -- wiring ------------------------------------------------------------ */

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q.length >= 2) ask(q);
});

thread.addEventListener("click", (e) => {
  const example = e.target.closest(".examples button");
  if (example) ask(example.dataset.q);
});

$("#reset").addEventListener("click", () => location.reload());

// Mode changes apply to the next question; the ones already answered were
// retrieved under whatever mode was set at the time, which their trace records.
modeSel.addEventListener("change", () => input.focus());

loadCensus();
input.focus();
