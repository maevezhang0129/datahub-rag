/* datahub-rag operational dashboard.
 *
 * The chat inspector explains one answer. This explains the system: whether
 * every chunk is embedded, whether chunking is current, and whether the
 * relevance threshold still sits in the gap it was calibrated to.
 *
 * Nothing here is cached or precomputed -- each panel is a live aggregate over
 * the same tables the pipeline writes, so it cannot show a stale picture.
 */

const $ = (sel) => document.querySelector(sel);

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const num = (n) => Number(n).toLocaleString("en-US");

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function fail(node, err) {
  node.innerHTML = `<div class="err">${esc(err.message)}</div>`;
}

/* -- census ------------------------------------------------------------ */

get("/health")
  .then((r) => {
    const c = r.corpus || {};
    $("#census").textContent =
      `${num(c.documents ?? 0)} docs · ${num(c.chunks ?? 0)} chunks · ${r.model}`;
  })
  .catch(() => { $("#census").textContent = "api unreachable"; });

/* -- corpus ------------------------------------------------------------ */

function renderCorpus({ health, sizes }) {
  const t = health.totals;
  $("#target-tokens").textContent = health.target_tokens;
  $("#corpus-note").innerHTML =
    `${num(t.documents)} documents · ${num(t.chunks)} chunks ·
     ${num(t.tokens)} tokens indexed.`;

  $("#corpus-table").innerHTML = `
    <div class="table-wrap">
      <table class="grid">
        <thead>
          <tr><th>source</th><th>docs</th><th>chunks</th><th>chunks/doc</th>
              <th>median tok</th><th>tok range</th></tr>
        </thead>
        <tbody>
          ${health.sources.map((s) => `
            <tr>
              <td class="k">${esc(s.source)}</td>
              <td>${num(s.documents)}</td>
              <td>${num(s.chunks)}</td>
              <td><b>${s.chunks_per_document}</b></td>
              <td>${s.tokens_median}</td>
              <td class="dim">${s.tokens_min}–${s.tokens_max}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;

  renderSizes(sizes, health.target_tokens,
              Math.max(0, ...health.sources.map((s) => s.tokens_max)));
}

function renderSizes(sizes, target, observedMax) {
  const max = Math.max(1, ...sizes.series.flatMap((s) => s.counts));
  // The top bucket is clamped to the largest chunk that exists: chunking caps
  // at the target, so labelling it "512-575" would advertise a range nothing
  // can fall into.
  const label = (b) => {
    const hi = Math.min(b + sizes.width - 1, observedMax);
    return hi <= b ? `${b}` : `${b}–${hi}`;
  };

  $("#sizes").innerHTML = `
    <div class="bars">
      ${sizes.buckets.map((bucket, i) => `
        <div class="bar-row">
          <span class="bar-label">${label(bucket)}</span>
          <span class="bar-stack">
            ${sizes.series.map((s, si) => s.counts[i] ? `
              <span class="seg seg-${si}" style="width:${(s.counts[i] / max) * 100}%"
                    title="${esc(s.source)}: ${s.counts[i]} chunks"></span>` : "").join("")}
          </span>
          <span class="bar-value">${sizes.series.reduce((a, s) => a + s.counts[i], 0)}</span>
        </div>`).join("")}
    </div>
    <div class="legend">
      ${sizes.series.map((s, si) =>
        `<span class="key"><i class="swatch seg-${si}"></i>${esc(s.source)}</span>`).join("")}
      <span class="key dim">target ${target}</span>
    </div>`;
}

/* -- index ------------------------------------------------------------- */

function renderIndex({ coverage, staleness }) {
  $("#coverage").innerHTML = `
    <div class="table-wrap">
    <table class="grid">
      <thead>
        <tr><th>model</th><th>embedded</th><th>coverage</th></tr>
      </thead>
      <tbody>
        ${coverage.map((m) => `
          <tr class="${m.indexed ? "" : "row-off"}">
            <td class="k">${esc(m.model)}
              <span class="dim">${m.dimensions}d</span>${m.is_default
              ? ' <span class="pill">default</span>' : ""}</td>
            <td title="${num(m.embedded)} of ${num(m.chunks)} chunks">${
              m.indexed ? num(m.embedded) : "&mdash;"}</td>
            <td>${m.indexed
              ? `<span class="meter-inline"><span style="width:${m.coverage * 100}%"></span></span>${pct(m.coverage)}`
              : '<span class="dim">not indexed</span>'}</td>
          </tr>`).join("")}
      </tbody>
    </table>
    </div>
    ${coverage.some((m) => m.indexed && m.missing)
      ? `<p class="card-note bad-text">Chunks are missing embeddings —
           <code>make pipeline</code> has outstanding work.</p>` : ""}`;

  const checks = [
    ["documents never chunked", staleness.documents_never_chunked],
    ["documents awaiting a re-chunk", staleness.documents_stale],
    ["chunks from a superseded revision", staleness.chunks_from_a_superseded_revision],
  ];

  $("#staleness").innerHTML = `
    <ul class="checks">
      ${checks.map(([label, n]) => `
        <li class="${n ? "off" : "on"}">
          <span>${label}</span><b>${num(n)}</b>
        </li>`).join("")}
    </ul>
    <p class="card-note">${staleness.clean
      ? "Every document has chunks cut from its current content."
      : "Run <code>make pipeline</code> — it is idempotent and will do only the outstanding work."}</p>`;
  $("#staleness").classList.add("stack-top");
}

/* -- calibration ------------------------------------------------------- */

function renderCalibration({ calibration, quality }) {
  const c = calibration;

  if (!c.total_turns) {
    $("#calibration").innerHTML = `
      <p class="hint">
        No persisted turns yet, so there is nothing to calibrate against.
        Run <code>make traffic</code> to replay the evaluation questions
        through the chat layer, or ask something in the
        <a href="/ui/">chat UI</a>.
      </p>`;
    return;
  }

  const max = Math.max(1, ...c.answered.counts, ...c.refused.counts);
  const w = 100 / c.bins;

  const columns = c.answered.counts.map((_, i) => {
    const a = c.answered.counts[i];
    const r = c.refused.counts[i];
    const lo = (i / c.bins).toFixed(2);
    return `
      <div class="hcol" style="width:${w}%" title="${lo}–${((i + 1) / c.bins).toFixed(2)}: ${a} answered, ${r} refused">
        <span class="hbar answered" style="height:${(a / max) * 100}%"></span>
        <span class="hbar refused"  style="height:${(r / max) * 100}%"></span>
      </div>`;
  }).join("");

  const marginLine = c.margin === null
    ? `<span class="dim">needs both answered and refused turns to measure</span>`
    : c.margin > 0
      ? `Cleanly separated: the lowest answered score is
         <b>${c.margin.toFixed(3)}</b> above the highest refused one, and the
         threshold sits inside that gap.`
      : `<span class="bad-text">The two groups overlap by
         ${Math.abs(c.margin).toFixed(3)}.</span> No single threshold divides
         them — <code>MIN_RELEVANCE</code> needs recalibrating, not nudging.`;

  $("#calibration").innerHTML = `
    <div class="hist">
      <div class="hist-plot">
        ${columns}
        <div class="hist-threshold" style="left:${c.threshold * 100}%">
          <span>MIN_RELEVANCE ${c.threshold}</span>
        </div>
      </div>
      <div class="hist-axis"><span>0.0</span><span>0.5</span><span>1.0</span></div>
    </div>

    <div class="legend">
      <span class="key"><i class="swatch answered"></i>answered (${c.answered.n})</span>
      <span class="key"><i class="swatch refused"></i>refused (${c.refused.n})</span>
    </div>

    <p class="card-note">${marginLine}</p>

    <div class="table-wrap">
    <table class="grid compact">
      <tbody>
        <tr><td class="k">answered range</td>
            <td>${fmtRange(c.answered)}</td></tr>
        <tr><td class="k">refused range</td>
            <td>${fmtRange(c.refused)}</td></tr>
        <tr><td class="k">refusal rate</td>
            <td>${pct(quality.refusal_rate)} of ${num(quality.turns)} turns</td></tr>
        <tr><td class="k">mean groundedness</td>
            <td>${quality.mean_groundedness === null
              ? '<span class="dim">no answered turns</span>'
              : pct(quality.mean_groundedness)}</td></tr>
        <tr><td class="k">answers with fabricated citations</td>
            <td class="${quality.answers_with_fabricated_citations ? "bad-text" : ""}">
              ${num(quality.answers_with_fabricated_citations)}</td></tr>
      </tbody>
    </table>
    </div>

    <p class="card-note dim">
      Measured over whatever has been asked and persisted — by default the
      evaluation questions, not production traffic. Read it as a calibration
      check, not a usage report.
    </p>`;
}

const fmtRange = (g) =>
  g.n ? `${g.min.toFixed(3)} – ${g.max.toFixed(3)}` : '<span class="dim">none</span>';

/* -- load -------------------------------------------------------------- */

get("/stats/corpus").then(renderCorpus)
  .catch((e) => { fail($("#corpus-table"), e); fail($("#sizes"), e); });
get("/stats/index").then(renderIndex)
  .catch((e) => { fail($("#coverage"), e); fail($("#staleness"), e); });
get("/stats/calibration").then(renderCalibration)
  .catch((e) => fail($("#calibration"), e));
