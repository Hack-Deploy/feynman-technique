/* One replay viewer for archive examples, catalog selections and live jobs. */
(() => {
  const colors = ["#1B3F8B", "#B7791F", "#2E7D4F", "#A23B1E", "#8064A2", "#168487", "#6B6B67"];
  const fmt = v => v == null ? "–" : Number(Number(v).toPrecision(3)).toString();
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let data = null, at = 0, elapsed = 0, playing = false, last = null;
  let renderedStep = null;
  const duration = 3.4;
  function axes(series, xLabel, yLabel) {
    const xs = series.flatMap(t => t.map(p => p[0])), ys = series.flatMap(t => t.map(p => p[1]));
    let xmin = Math.min(0, ...xs), xmax = Math.max(0, ...xs), ymin = Math.min(0, ...ys), ymax = Math.max(0, ...ys);
    const xp = Math.max(xmax - xmin, .01) * .06, yp = Math.max(ymax - ymin, .01) * .12;
    xmin -= xp; xmax += xp; ymin -= yp; ymax += yp;
    const x = v => 58 + (v - xmin) / (xmax - xmin) * 566;
    const y = v => 336 - (v - ymin) / (ymax - ymin) * 310;
    let svg = "";
    for (let i = 0; i <= 4; i++) {
      const xv = xmin + (xmax - xmin) * i / 4, yv = ymin + (ymax - ymin) * i / 4;
      svg += `<line x1="${x(xv)}" x2="${x(xv)}" y1="14" y2="336" stroke="#E6E6E2"/><text x="${x(xv)}" y="354" text-anchor="middle" font-size="11" fill="#6B6B67">${fmt(xv)}</text>`;
      svg += `<line x1="58" x2="624" y1="${y(yv)}" y2="${y(yv)}" stroke="#E6E6E2"/><text x="50" y="${y(yv) + 4}" text-anchor="end" font-size="11" fill="#6B6B67">${fmt(yv)}</text>`;
    }
    svg += `<text x="340" y="374" text-anchor="middle" font-size="12" fill="#6B6B67">${esc(xLabel)}</text><text transform="translate(14 180) rotate(-90)" text-anchor="middle" font-size="12" fill="#6B6B67">${esc(yLabel)}</text>`;
    return {x, y, svg};
  }
  function line(points, ax, color, dash = "") {
    if (!points.length) return "";
    return `<polyline points="${points.map(p => `${ax.x(p[0])},${ax.y(p[1])}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2.5" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
  }
  function drawMotion(rd, done) {
    const paths = $("#replayPlot").value === "paths";
    const traces = (data.traces || []).filter(t => t.round <= rd.round);
    const series = traces.map(t => paths ? (t.positions || []) : t.times.map((time, i) => [time, t.observed[i]]));
    const ax = axes(series, paths ? "recorded x position" : "time since experiment started", paths ? "recorded y position" : "inward displacement from initial position");
    let svg = ax.svg;
    traces.forEach((t, i) => {
      const points = series[i], color = colors[i % colors.length];
      const visible = playing && !done && t.round === rd.round ? Math.max(1, Math.ceil(points.length * Math.min(1, elapsed / 1.6))) : points.length;
      const shown = points.slice(0, visible);
      svg += line(shown, ax, color);
      shown.forEach((p, k) => { svg += `<circle cx="${ax.x(p[0])}" cy="${ax.y(p[1])}" r="4" fill="${color}" stroke="#fff"><title>${esc(t.label || `Experiment ${i + 1}`)} · t=${fmt(t.times[k])} · ${paths ? `x=${fmt(p[0])}, y=${fmt(p[1])}` : `displacement=${fmt(p[1])}`}</title></circle>`; });
      if (!paths && t.reference?.length) svg += line(t.reference_times.map((time, k) => [time, t.reference[k]]), ax, "#2E7D4F", "6 5");
    });
    if (!traces.length) svg += `<text x="340" y="180" text-anchor="middle" font-size="15" fill="#6B6B67">${data.state === "running" ? "Waiting for recorded experiment measurements…" : "No experiment measurements recorded for this round."}</text>`;
    $("#lc").innerHTML = svg;
    $("#lc").setAttribute("aria-label", paths ? "Individual recorded particle trajectories" : "Individual recorded probe displacement curves over time");
    $("#learn .legend").innerHTML = `<span><i class="dot"></i>Recorded measurements</span><span><i style="border-color:#1B3F8B"></i>Lines join measurements; each color is a probe or experiment</span>${traces.some(t => t.reference?.length) && !paths ? '<span><i style="border-color:#2E7D4F;border-top-style:dashed"></i>Noise-free simulator reference</span>' : ""}`;
  }
  function drawLaw(rd, done) {
    const law = data.law, round = done ? Infinity : rd.round;
    // While playing, the current round's readings appear one by one and its fit sweeps or morphs in.
    const p = playing && !done ? Math.min(1, elapsed / 1.6) : 1, ease = p * p * (3 - 2 * p);
    const now = law.rounds.filter(r => r.round === round).flatMap(r => r.points || []);
    const before = law.rounds.filter(r => r.round < round).flatMap(r => r.points || []);
    const readings = [...before, ...now.slice(0, Math.ceil(now.length * p))];
    const force = g => Array.from({length:111}, (_, i) => { const r = 1.5 + i * .05; return [r, g.a3 * (3 / r) ** g.n]; });
    const toGuess = r => { const e = r?.estimates || {}; return e.n && e.a3 ? {n: e.n.value, a3: e.a3.value} : null; };
    const idx = data.rounds.indexOf(rd), guess = toGuess(rd), prev = idx > 0 ? toGuess(data.rounds[idx - 1]) : null;
    const reference = force(law.true), hypothesis = force(law.expected), final = guess ? force(guess) : [];
    let fit = final;
    if (guess && p < 1) fit = prev ? force({n: prev.n + (guess.n - prev.n) * ease, a3: prev.a3 + (guess.a3 - prev.a3) * ease})
                                   : final.slice(0, Math.max(2, Math.ceil(final.length * ease)));
    const allReadings = [...before, ...now].map(pt => [pt.r, pt.a]);
    const ax = axes([reference, hypothesis, final, prev ? force(prev) : [], allReadings], "distance from the source, r", "pull on the probe");
    let svg = ax.svg + line(reference, ax, "#2E7D4F", "7 5") + line(hypothesis, ax, "#3D63B5", "2 5") + line(fit, ax, "#B7791F");
    readings.forEach((pt, k) => {
      const fresh = p < 1 && k === readings.length - 1 && k >= before.length;
      if (fresh) svg += `<circle cx="${ax.x(pt.r)}" cy="${ax.y(pt.a)}" r="${5 + 14 * ((elapsed * 3) % 1)}" fill="none" stroke="#1B3F8B" opacity="${1 - ((elapsed * 3) % 1)}"/>`;
      svg += `<circle cx="${ax.x(pt.r)}" cy="${ax.y(pt.a)}" r="5" fill="#1B3F8B" stroke="#fff"><title>r=${fmt(pt.r)} · approximate pull=${fmt(pt.a)}</title></circle>`;
    });
    $("#lc").innerHTML = svg;
    $("#lc").setAttribute("aria-label", "Recorded pull readings and fitted force law against hypothesis and simulator reference");
    $("#learn .legend").innerHTML = `<span><i class="dot"></i>Pull derived from recorded positions</span><span><i style="border-color:#B7791F"></i>Recorded numeric fit, when available</span><span><i style="border-color:#2E7D4F;border-top-style:dashed"></i>Simulator reference</span><span><i style="border-color:#3D63B5;border-top-style:dotted"></i>Hypothesis</span>`;
  }
  function drawMetric(idx) {
    const metric = $("#replayMetric").value;
    if (metric === "confidence") { chart(data.rounds.slice(0, idx + 1), "#ln"); $("#ln").setAttribute("aria-label", "Stated chance and credits spent by round"); return; }
    const pts = data.rounds.flatMap((r, i) => r.estimates?.[metric] ? [[i + 1, r.estimates[metric].value]] : []);
    const reference = data.law?.true?.[metric], expected = data.law?.expected?.[metric];
    const all = [...pts, ...[reference, expected].filter(Number.isFinite).map(v => [1, v])];
    const ax = axes([all], "round", `recorded ${metric}`);
    // The same SVG coordinate system is scaled into the sidebar.
    let svg = ax.svg;
    if (reference != null) svg += line([[1, reference], [data.rounds.length, reference]], ax, "#2E7D4F", "7 5");
    if (expected != null) svg += line([[1, expected], [data.rounds.length, expected]], ax, "#3D63B5", "2 5");
    const seen = pts.filter(p => p[0] <= idx + 1);
    svg += line(seen, ax, "#B7791F");
    seen.forEach(p => { svg += `<circle cx="${ax.x(p[0])}" cy="${ax.y(p[1])}" r="5" fill="#B7791F" stroke="#fff"/>`; });
    $("#ln").innerHTML = `<svg viewBox="0 0 640 380">${svg}</svg>`;
    $("#ln").setAttribute("aria-label", `Recorded ${metric} by round, against available reference and hypothesis`);
  }
  function render(force = false) {
    if (!data) return;
    const N = data.rounds.length, done = data.state !== "running" && data.state !== "error" && at >= N;
    const idx = Math.max(0, Math.min(N - 1, at)), rd = data.rounds[idx] || {round: 0, estimates: {}};
    if ($("#replayPlot").value === "law" && data.law) drawLaw(rd, done); else drawMotion(rd, done);
    const step = `${at}:${data.state}`;
    if (!force && renderedStep === step) return;
    renderedStep = step;
    drawMetric(idx);
    $("#lRound").textContent = done ? "Recorded result" : data.state === "error" ? "Run stopped" : `Round ${rd.round} of up to ${data.max_rounds || "?"}`;
    $("#lConf").textContent = rd.p_success == null ? "" : `says ${pct(rd.p_success)} sure`;
    const estimates = Object.entries(rd.estimates || {});
    $("#lGuess").innerHTML = estimates.length ? estimates.map(([key, e]) => `${esc(key)} = ${fmt(e.value)}${e.sigma ? ` ± ${fmt(e.sigma)}` : ""}`).join("<br>") + "<small>Recorded estimates; a fit is not proof of a correct verdict.</small>" : `${pct(rd.p_success)} stated chance<small>No numeric estimate recorded this round.</small>`;
    $("#lQuote").innerHTML = `${rd.assessment ? `“${esc(rd.assessment)}”` : data.state === "running" ? "The model is thinking…" : "No assessment recorded."}<small>Round ${rd.round} · ${esc(ACTION[rd.action] || rd.action || "")}</small>`;
    $("#replayDetails").innerHTML = roundCard({...rd, assessment: null}).replace(/<div class="t">[\s\S]*?<\/div>/, "");
    const v = $("#lVerdict"); v.classList.toggle("show", done || data.state === "error");
    v.style.borderLeftColor = data.passed ? "var(--green)" : "var(--rust)";
    if (data.state === "error") v.innerHTML = `<b>The run stopped.</b> ${esc(data.error)}`;
    else if (done) v.innerHTML = `<b>${data.passed ? "Success: correct result." : "Failure: no prize."}</b> It said ${esc(data.verdict || "no clear verdict")}; the recorded answer was ${esc(data.answer)}. ${data.judge === "answer_key" ? "Scored against the dataset’s answer key." : "Scored by the run’s checker."}${data.law?.caveat ? ` ${esc(data.law.caveat)}` : ""}<div class="nums"><span>Prize paid ${n0(data.prize_paid)}</span><span>Net ${n0(data.profit)} credits</span><span>${n0(data.spent)} credits spent · $${usd(data.usd)} API</span></div>`;
    $("#kRound").textContent = done ? `${N} · done` : rd.round || "–";
    $("#kSpent").textContent = n0(rd.spent_so_far ?? 0);
    $("#kExp").textContent = data.rounds.slice(0, idx + 1).reduce((sum, r) => sum + (r.experiments || r.n_experiments || 0), 0);
    $("#kP").textContent = pct(rd.p_success);
    renderClaim({id: data.hypothesis_id, hypothesis: data.hypothesis, resolution_criteria: data.resolution_criteria, world: data.world, prize: data.prize, model: `${data.agent} · seed ${data.seed}`, answer: done ? data.answer : null});
    $("#lSteps").innerHTML = data.rounds.map((r, i) => `<button data-i="${i}" class="${!done && at === i ? "on" : ""}">Round ${r.round}</button>`).join("") + (data.state === "running" || data.state === "error" ? "" : `<button data-i="${N}" class="${done ? "on" : ""}">Result</button>`);
    $("#lTrail").innerHTML = data.rounds.map((r, i) => {
      const future = i > idx, traces = (data.traces || []).filter(t => t.round === r.round);
      return `<li class="tr${future ? " future" : i === idx && !done ? " now" : ""}" data-i="${i}"><div class="rn"><b>Round ${r.round}</b><p class="act">${future ? "Not played yet" : esc(ACTION[r.action] || r.action)}</p><p class="saw">${future ? "" : `${r.experiments || r.n_experiments || 0} new experiments · ${traces.length} recorded paths`}</p></div><div class="hyp"><div class="law">${future ? "" : Object.entries(r.estimates || {}).map(([key, e]) => `${esc(key)} = ${fmt(e.value)}`).join("<br>") || "No numeric estimate"}</div></div><p class="why">${future ? "" : esc(r.assessment || "No assessment recorded.")}</p><div class="conf">${future ? "–" : pct(r.p_success)}<span>sure</span></div></li>`;
    }).join("") + (done ? `<li class="tr check ${data.passed ? "ok" : "bad"}" data-i="${N}"><div class="rn"><b>Recorded result</b></div><div class="hyp"><div class="law">${data.passed ? "Correct result" : "No prize"}</div></div><p class="why">It said ${esc(data.verdict || "no verdict")}; the recorded answer is ${esc(data.answer)}.</p><div class="conf">${data.passed ? "won" : "lost"}<span>prize</span></div></li>` : "");
    $("#lPlay").disabled = data.state === "running" || !N;
    $("#lPlay").textContent = playing ? "Pause" : done ? "Replay" : "Play";
  }
  window.showRecordedReplay = (d, autoplay = false) => {
    data = d; at = 0; elapsed = 0; playing = autoplay && !reduced; last = null;
    renderedStep = null;
    $("#replayPlot option[value=law]").disabled = !data.law;
    $("#replayPlot option[value=paths]").disabled = !(data.traces || []).some(t => t.positions?.length);
    $("#replayPlot").value = data.law ? "law" : "motion";
    const keys = [...new Set(data.rounds.flatMap(r => Object.keys(r.estimates || {})))];
    $("#replayMetric").innerHTML = '<option value="confidence">Stated chance and spend</option>' + keys.map(k => `<option value="${esc(k)}">Recorded ${esc(k)}</option>`).join("");
    $("#replayMetric").value = data.law && keys.includes("n") ? "n" : "confidence";
    const curves = (data.traces || []).filter(t => t.times.length > 1).length, points = (data.traces || []).reduce((s,t)=>s+t.times.length,0);
    $("#watchLead").innerHTML = `${esc(data.agent)} on ${esc(data.hypothesis_id)}: ${curves} recorded paths with multiple measurements, ${points} measured points. Compare its experiments, estimates and reasoning round by round. <a href="${esc(data.source_url || HF_URL)}" target="_blank" rel="noopener">Source data</a>.`;
    render();
  };
  window.renderMarketJob = (job, claim) => {
    const source = job.replay || {};
    const rounds = (source.rounds || job.rounds || []).map(r => ({...r, experiments: r.experiments ?? r.n_experiments ?? 0, estimates: r.estimates || {}}));
    const r = job.run || {};
    const d = {...source, kind:"recorded", state: job.state, error:job.error, rounds, traces:source.traces || [],
      agent:source.agent || claim.model, model:job.model, hypothesis_id:claim.id, hypothesis:claim.hypothesis,
      resolution_criteria:claim.resolution_criteria, world:claim.world, prize:claim.prize, seed:job.seed ?? 0,
      max_rounds:source.max_rounds || INFO?.live.max_rounds, passed:r.passed, verdict:r.agent_verdict,
      answer:r.answer, prize_paid:r.prize_paid, profit:r.profit, spent:r.spent, usd:source.usd || 0};
    if (source.expected && source.true) d.law = source;
    window.showRecordedReplay(d, false);
    at = job.state === "done" ? rounds.length : Math.max(0,rounds.length-1); render();
  };
  $("#lPlay").addEventListener("click", () => { if (!data) return; if (at >= data.rounds.length) { at = 0; elapsed = 0; } playing = !playing; last = null; render(true); });
  const jump = e => { const b = e.target.closest("[data-i]"); if (!b || !data) return; at = +b.dataset.i; elapsed = 0; playing = false; render(); };
  $("#lSteps").addEventListener("click", jump); $("#lTrail").addEventListener("click", jump);
  $("#replayPlot").addEventListener("change", () => render(true)); $("#replayMetric").addEventListener("change", () => render(true));
  function tick(t) {
    if (playing && last != null && data) {
      elapsed += (t - last) / 1000;
      if (elapsed >= duration) { elapsed = 0; at++; if (at >= data.rounds.length) playing = false; }
      render();
    }
    last = t; requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  window.dispatchEvent(new Event("replay-ready"));
})();
