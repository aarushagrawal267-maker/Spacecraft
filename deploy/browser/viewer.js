/* Heliograph browser demo: a viewer of the stored reference results (built by deploy/browser/build.py). */
"use strict";
(() => {
  const $ = (s) => document.querySelector(s);
  const COLORS = {
    actual: "#199e9e", prediction: "#d9deea", smoothed: "#e6ebf5", threshold: "#9085e9", label: "#c98500",
    detect: "#9085e9", grid: "#1a2238", axis: "#263152", muted: "#8491ab", ink: "#e6ebf5",
  };
  const METHOD_COLOR = {
    lstm_forecaster: "#3987e5", lstm_autoencoder: "#d95926", ridge_ar: "#199e70", persistence: "#c98500",
    rolling_zscore: "#d55181", isolation_forest: "#0ca30c", random: "#8491ab",
  };
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const getJson = async (p) => { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`); return r.json(); };

  let summary = null;
  let scope = "overall";
  const channelCache = new Map();
  const telemetryCache = new Map();

  // ------------------------------------------------------------------ results table
  function renderResults() {
    const rows = summary.methods.map((m) => ({ m, s: scope === "overall" ? m.overall : m.by_spacecraft[scope] }));
    const best = Math.max(...rows.map((r) => r.s?.event?.f0_5 ?? -1));
    $("#results-table").innerHTML =
      "<thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>F0.5</th><th>F1</th><th>TP / FP / FN</th><th>False alarms / 10k</th><th>Mean delay</th><th>Point F1</th><th>Mean AP</th></tr></thead><tbody>" +
      rows.map(({ m, s }) => {
        const e = s?.event || {};
        return `<tr class="${e.f0_5 === best ? "best" : ""}"><td><span class="dot" style="background:${METHOD_COLOR[m.method] || "#888"}"></span>${esc(m.name)}</td>` +
          `<td class="num">${fmt(e.precision)}</td><td class="num">${fmt(e.recall)}</td><td class="num f1">${fmt(e.f0_5)}</td><td class="num">${fmt(e.f1)}</td>` +
          `<td class="num">${e.tp ?? "–"} / ${e.fp ?? "–"} / ${e.fn ?? "–"}</td><td class="num">${fmt(e.false_alarms_per_10k, 2)}</td>` +
          `<td class="num">${fmt(e.mean_delay, 0)}</td><td class="num">${fmt(s?.point?.f1)}</td><td class="num">${fmt(s?.mean_average_precision)}</td></tr>`;
      }).join("") + "</tbody>";
    const n = summary.channels.filter((c) => scope === "overall" || c.spacecraft === scope).length;
    const labels = summary.channels.filter((c) => scope === "overall" || c.spacecraft === scope).reduce((a, c) => a + c.n_labels, 0);
    $("#results-note").textContent = `${n} channels, ${labels} labelled anomaly sequences. Mean delay is in timesteps from the start of a label to the first detected point. Reference results created ${summary.reference_created_at?.slice(0, 10) || "–"}.`;
  }
  document.querySelectorAll("#sc-filter button").forEach((b) => b.addEventListener("click", () => {
    scope = b.dataset.sc;
    document.querySelectorAll("#sc-filter button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    renderResults();
  }));

  // ------------------------------------------------------------------ data loading
  async function loadChannel(chan) {
    if (!channelCache.has(chan)) {
      channelCache.set(chan, (async () => {
        const [meta, buf] = await Promise.all([
          getJson(`data/channels/${encodeURIComponent(chan)}.json`),
          fetch(`data/channels/${encodeURIComponent(chan)}.bin`).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); }),
        ]);
        meta.floats = new Float32Array(buf);
        return meta;
      })());
    }
    return channelCache.get(chan);
  }

  // Minimal .npy reader (little-endian float64/float32, C order) for the benchmark's test streams.
  function parseNpy(buf) {
    const b = new Uint8Array(buf);
    if (b[0] !== 0x93 || String.fromCharCode(...b.slice(1, 6)) !== "NUMPY") throw new Error("not a .npy file");
    const major = b[6];
    const dv = new DataView(buf);
    const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
    const start = major === 1 ? 10 : 12;
    const header = new TextDecoder().decode(b.slice(start, start + hlen));
    const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
    if (/'fortran_order':\s*True/.test(header)) throw new Error("Fortran-ordered .npy is not supported");
    const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
    const off = start + hlen;
    const data = descr === "<f8" ? new Float64Array(buf.slice(off)) : descr === "<f4" ? new Float32Array(buf.slice(off)) : null;
    if (!data) throw new Error("unsupported dtype " + descr);
    return { data, shape };
  }

  async function loadTelemetry(chan) {
    if (!telemetryCache.has(chan)) {
      telemetryCache.set(chan, (async () => {
        const r = await fetch(`${summary.telemetry_base}/test/${encodeURIComponent(chan)}.npy`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const { data, shape } = parseNpy(await r.arrayBuffer());
        const [n, k] = shape.length === 2 ? shape : [shape[0], 1];
        const values = new Float64Array(n);
        for (let i = 0; i < n; i++) values[i] = data[i * k];
        return values;
      })());
      telemetryCache.get(chan).catch(() => telemetryCache.delete(chan));
    }
    return telemetryCache.get(chan);
  }

  // ------------------------------------------------------------------ channel view
  let token = 0;
  async function showChannel() {
    const chan = $("#chan").value, key = $("#method").value;
    const my = ++token;
    try { history.replaceState(null, "", `#${encodeURIComponent(chan)}/${encodeURIComponent(key)}`); } catch { /* ignore */ }
    $("#chan-status").className = "hint";
    $("#chan-status").textContent = `Loading ${chan}…`;
    let meta;
    try { meta = await loadChannel(chan); } catch (e) {
      $("#chan-status").className = "hint error"; $("#chan-status").textContent = `Could not load the stored results for ${chan} (${e.message}).`; return;
    }
    let telemetry = null, telErr = null;
    try { telemetry = await loadTelemetry(chan); } catch (e) { telErr = e.message; }
    if (my !== token) return;
    const run = meta.methods[key];
    const method = summary.methods.find((m) => m.key === key);
    if (!run) { $("#chan-status").textContent = "No stored run for this channel and method."; return; }
    const arr = (name) => (run.arrays[name] === undefined ? null : meta.floats.subarray(run.arrays[name], run.arrays[name] + meta.n));
    if (telemetry && telemetry.length !== meta.n) { telErr = `length ${telemetry.length} does not match the stored ${meta.n}`; telemetry = null; }
    $("#chan-status").className = telErr ? "hint error" : "hint";
    $("#chan-status").textContent = telErr
      ? `The telemetry could not be downloaded from the public benchmark copy (${telErr}); showing the stored scores only.`
      : `${meta.spacecraft} channel ${chan}: ${meta.n.toLocaleString()} test timesteps, scored from timestep ${run.eval_start}. Telemetry downloaded from the public benchmark copy.`;

    const e = run.metrics.event, p = run.metrics.point;
    const cards = [
      ["Labels found", `${e.tp ?? 0} / ${e.n_labels ?? 0}`, "expert-labelled sequences"],
      ["False alarms", `${e.fp ?? 0}`, `${run.detections.length} detected interval${run.detections.length === 1 ? "" : "s"}`],
      ["Event F1", fmt(e.f1), `P ${fmt(e.precision, 2)} · R ${fmt(e.recall, 2)}`],
      ["Point F1", fmt(p.f1), "every timestep"],
      ["Average precision", fmt(run.metrics.average_precision), "anomaly score ranking"],
    ];
    if (run.epochs_run) cards.push(["Training", `${run.epochs_run} epochs`, `best epoch ${run.best_epoch} · ${fmt(run.train_time_s, 0)} s CPU`]);
    $("#cards").innerHTML = cards.map(([k, v, s]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

    const x = Array.from({ length: meta.n }, (_, i) => i);
    const line = meta.n > 5000 ? "scattergl" : "scatter"; // same rendering choice as the full app
    const traces = [];
    if (telemetry) traces.push({ x, y: Array.from(telemetry), name: "telemetry", type: line, mode: "lines", line: { color: COLORS.actual, width: 1.6 }, yaxis: "y" });
    const pred = method?.produces_prediction ? arr("prediction") : null;
    $("#legend-pred").hidden = !pred;
    if (pred) traces.push({ x, y: Array.from(pred), name: "model forecast", type: line, mode: "lines", line: { color: COLORS.prediction, width: 1.1 }, opacity: 0.85, yaxis: "y" });
    const sm = arr("smoothed"), th = arr("threshold_high"), tl = arr("threshold_low");
    if (sm) traces.push({ x, y: Array.from(sm), name: "smoothed error", type: line, mode: "lines", line: { color: COLORS.smoothed, width: 1.5 }, yaxis: "y2" });
    if (th) traces.push({ x, y: Array.from(th), name: "threshold", type: line, mode: "lines", line: { color: COLORS.threshold, width: 1.4, dash: "dash" }, yaxis: "y2" });
    if (tl && tl.some((v) => Number.isFinite(v))) traces.push({ x, y: Array.from(tl), name: "lower threshold", type: line, mode: "lines", line: { color: COLORS.threshold, width: 1, dash: "dot" }, opacity: 0.6, yaxis: "y2" });
    const shapes = [];
    meta.labels.forEach((l) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: l.start, x1: l.end + 1, y0: 0, y1: 1, fillcolor: "rgba(201,133,0,0.22)", line: { width: 0 }, layer: "below" }));
    run.detections.forEach((d) => shapes.push({ type: "rect", xref: "x", yref: "paper", x0: d.start, x1: d.end + 1, y0: 0, y1: 0.44, fillcolor: "rgba(144,133,233,0.4)", line: { width: 0 }, layer: "below" }));
    if (run.eval_start > 0) shapes.push({ type: "line", xref: "x", yref: "paper", x0: run.eval_start, x1: run.eval_start, y0: 0, y1: 1, line: { color: COLORS.muted, width: 1, dash: "dot" } });
    const axis = { gridcolor: COLORS.grid, zerolinecolor: COLORS.axis, linecolor: COLORS.axis, tickfont: { color: COLORS.muted, size: 11 }, titlefont: { color: COLORS.muted, size: 11 } };
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: COLORS.ink, family: "IBM Plex Sans, system-ui, sans-serif" },
      margin: { l: 48, r: 10, t: 10, b: 36 }, showlegend: false, hovermode: "x unified", shapes,
      xaxis: { ...axis, title: { text: "test timestep", font: { size: 11, color: COLORS.muted } }, anchor: "y2" },
      yaxis: { ...axis, domain: [0.5, 1], title: { text: "telemetry", font: { size: 11, color: COLORS.muted } } },
      yaxis2: { ...axis, domain: [0, 0.44], title: { text: method?.family === "forecast" || method?.family === "reconstruction" ? "error" : "score", font: { size: 11, color: COLORS.muted } } },
    };
    window.Plotly.react("plot", traces, layout, { responsive: true, displayModeBar: false });

    $("#labels").innerHTML = (run.per_label.length ? run.per_label : meta.labels).map((l) =>
      `<li><span class="num">${l.start.toLocaleString()}–${l.end.toLocaleString()}</span> · ${esc(l.cls)}` +
      (l.detected === undefined ? "" : l.detected ? ` <span class="badge good">found</span> <span class="hint">delay ${l.delay}, IoU ${fmt(l.iou, 2)}</span>` : ` <span class="badge bad">missed</span>`) + "</li>").join("") || "<li>No labels.</li>";
    $("#dets").innerHTML = run.detections.map((d) =>
      `<li><span class="num">${d.start.toLocaleString()}–${d.end.toLocaleString()}</span> · ${esc(d.direction)} · score ${fmt(d.score, 2)}` +
      (d.outcome === "matches_label" ? ' <span class="badge good">overlaps a label</span>' : ' <span class="badge bad">false alarm</span>') + "</li>").join("") || "<li>No intervals detected.</li>";
  }

  // ------------------------------------------------------------------ start
  async function init() {
    try { summary = await getJson("data/summary.json"); } catch (e) {
      $("#results-note").textContent = "Could not load the reference results: " + e.message; return;
    }
    renderResults();
    const chanSel = $("#chan");
    for (const sc of ["SMAP", "MSL"]) {
      const g = document.createElement("optgroup"); g.label = sc === "MSL" ? "MSL (Curiosity)" : "SMAP";
      summary.channels.filter((c) => c.spacecraft === sc).forEach((c) => {
        const o = document.createElement("option"); o.value = c.chan_id; o.textContent = `${c.chan_id} · ${c.n_labels} label${c.n_labels === 1 ? "" : "s"}`; g.appendChild(o);
      });
      chanSel.appendChild(g);
    }
    summary.methods.forEach((m) => { const o = document.createElement("option"); o.value = m.key; o.textContent = m.name; $("#method").appendChild(o); });
    const [c0, m0] = decodeURIComponent(location.hash.slice(1)).split("/");
    chanSel.value = summary.channels.some((c) => c.chan_id === c0) ? c0 : "T-12";
    $("#method").value = summary.methods.some((m) => m.key === m0) ? m0 : summary.methods[0].key;
    chanSel.addEventListener("change", showChannel);
    $("#method").addEventListener("change", showChannel);
    if (!window.Plotly) await new Promise((res) => window.addEventListener("load", res, { once: true }));
    if (!window.Plotly) { $("#chan-status").textContent = "The chart library could not be loaded."; return; }
    showChannel();
  }
  init();
})();
