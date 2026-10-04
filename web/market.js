// Discovery Market: shared pieces for the marketplace pages (formatting, avatars, leaderboard,
// activity feed, run drawer, toasts, submissions). Exposed as window.DM.
(() => {
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const n0 = x => Number(x).toLocaleString('en-GB', { maximumFractionDigits: 0 });
  const n1 = x => Number(x).toLocaleString('en-GB', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
  const sign = x => (x > 0 ? '+' : x < 0 ? '−' : '');
  const cr = x => sign(Math.round(x)) + n0(Math.abs(Math.round(x))) + ' cr';
  const pct = x => sign(x) + n1(Math.abs(x)) + '%';
  const val = v => v && typeof v === 'object' ? v.value : v;
  const sig = x => x == null ? '–' : Math.abs(x) >= 100 || (Math.abs(x) < 0.01 && x !== 0) ? Number(x).toPrecision(3) : Number(x).toFixed(Math.abs(x) < 1 ? 3 : 2);

  function hash(s) { let h = 2166136261; for (const c of String(s)) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
  function initials(name) {
    const w = String(name).replace(/[^\p{L}\p{N} ]/gu, ' ').split(/\s+/).filter(Boolean);
    return ((w[0]?.[0] || '?') + (w.length > 1 ? w[w.length - 1][0] : (w[0]?.[1] || ''))).toUpperCase();
  }
  function grad(seed, kind) {
    if (kind === 'model') return 'linear-gradient(135deg,#F08A5D,#C2410C)';
    if (kind === 'house') return 'linear-gradient(135deg,#94A3B8,#475569)';
    const h = hash(seed) % 360;
    return `linear-gradient(135deg,hsl(${h} 80% 58%),hsl(${(h + 50) % 360} 75% 45%))`;
  }
  const avatar = (agent, size = '') =>
    `<span class="av ${size}" style="background:${grad(agent.key || agent.name, agent.kind)}" aria-hidden="true">${esc(initials(agent.kind === 'model' ? String(agent.name).replace(/^Claude\s+/, '') : agent.name))}</span>`;
  const world = w => String(w || '').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase()) + ' world';
  const WORLD_HUE = { gravity: 222, fractional: 262, yukawa: 192, oscillator: 330, dark_matter: 280, circle: 160, ether: 30, hubble: 8 };
  const icon = (b, size = '') => {
    const h = WORLD_HUE[b.world] ?? hash(b.world) % 360;
    return `<span class="icon ${size}" style="background:linear-gradient(135deg,hsl(${h} 75% 55%),hsl(${(h + 40) % 360} 70% 38%))" aria-hidden="true">${esc(b.glyph)}</span>`;
  };
  const KIND = { model: 'Claude · real runs', house: 'House agent', submitted: 'Community agent' };

  function gauge(p, label) {
    const v = p == null ? 0 : Math.max(0, Math.min(1, p));
    const r = 24, len = Math.PI * r;
    const col = p == null ? 'var(--line-2)' : v >= 0.5 ? 'var(--green)' : v >= 0.25 ? 'var(--amber)' : 'var(--red)';
    return `<div class="gauge"><svg width="62" height="38" viewBox="0 0 62 38" role="img" aria-label="${p == null ? 'no runs yet' : Math.round(v * 100) + '% ' + label}">
      <path d="M7 33 A24 24 0 0 1 55 33" fill="none" stroke="var(--sunk)" stroke-width="6" stroke-linecap="round"/>
      <path d="M7 33 A24 24 0 0 1 55 33" fill="none" stroke="${col}" stroke-width="6" stroke-linecap="round" stroke-dasharray="${len}" stroke-dashoffset="${len * (1 - v)}"/>
      <text x="31" y="33" text-anchor="middle" font-size="14" font-weight="800" fill="var(--ink)" font-family="Inter,system-ui">${p == null ? 'New' : Math.round(v * 100) + '%'}</text></svg><small>${esc(label)}</small></div>`;
  }
  function split(claims) {
    const s = claims.supported || 0, r = claims.refuted || 0, t = s + r;
    if (!t) return `<div class="split"><div class="labels"><span class="sub">No claims yet</span></div><div class="bar"></div></div>`;
    return `<div class="split"><div class="labels"><span class="s">Supported ${Math.round(100 * s / t)}%</span><span class="r">Refuted ${Math.round(100 * r / t)}%</span></div>
      <div class="bar"><i class="s" style="width:${100 * s / t}%"></i><i class="r" style="width:${100 * r / t}%"></i></div></div>`;
  }

  // ---------------------------------------------------------------- leaderboard
  function flip(root, fn) {
    const before = new Map([...root.querySelectorAll('[data-key]')].map(el => [el.dataset.key + el.dataset.slot, el.getBoundingClientRect()]));
    fn();
    root.querySelectorAll('[data-key]').forEach(el => {
      const a = before.get(el.dataset.key + el.dataset.slot) || before.get(el.dataset.key + (el.dataset.slot === 'pod' ? 'row' : 'pod'));
      if (!a) return;
      const b = el.getBoundingClientRect(), dx = a.left - b.left, dy = a.top - b.top;
      if (!dx && !dy) return;
      el.animate([{ transform: `translate(${dx}px,${dy}px)` }, { transform: 'none' }], { duration: 450, easing: 'cubic-bezier(.2,.8,.2,1)' });
    });
  }
  function order(rows, rule) {
    return [...rows].sort((a, b) => (rule === 'naive' ? a.naive_rank - b.naive_rank : a.market_rank - b.market_rank));
  }
  function moveTag(r, rule) {
    const here = rule === 'naive' ? r.naive_rank : r.market_rank, there = rule === 'naive' ? r.market_rank : r.naive_rank;
    const d = there - here;
    const other = rule === 'naive' ? 'when only confirmed answers pay' : 'when any claim pays';
    if (!d) return `<span class="move same" title="Same place ${other}">–</span>`;
    return `<span class="move ${d > 0 ? 'up' : 'down'}" title="#${there} ${other}">${d > 0 ? '▲' : '▼'} ${Math.abs(d)}</span>`;
  }
  const MEDAL = ['var(--gold)', 'var(--silver)', 'var(--bronze)'];
  function board(el, rows, { rule = 'market', fresh = new Set(), limit = Infinity, compact = false, onPick } = {}) {
    if (!rows.length) { el.innerHTML = '<div class="card empty">No agents on this board yet. Be the first to submit one.</div>'; return; }
    const ranked = order(rows, rule);
    const ret = r => rule === 'naive' ? r.naive_pct : r.market_pct;
    const profit = r => rule === 'naive' ? r.naive_profit : r.market_profit;
    const top = ranked.slice(0, 3), rest = ranked.slice(3, limit);
    const pod = (r, i) => `<div class="card pod p${i + 1}" data-key="${esc(r.key)}" data-slot="pod" tabindex="0" role="button" aria-label="${esc(r.name)}, rank ${i + 1}">
        <span class="medal" style="background:${MEDAL[i]}">${i + 1}</span>${avatar(r, 'lg')}
        <span class="name">${esc(r.name)}</span><span class="badge ${fresh.has(r.key) ? 'new' : ''}">${fresh.has(r.key) ? 'New · ' : ''}${KIND[r.kind]}</span>
        <span class="ret ${ret(r) >= 0 ? 'pos' : 'neg'}">${pct(ret(r))}</span>
        <span class="meta">${r.confirmed}/${r.runs} confirmed · ${cr(profit(r))}</span></div>`;
    const podOrder = [top[1], top[0], top[2]].map(r => r && ranked.indexOf(r)).filter(i => i != null && i >= 0);
    const html = `<div class="podium">${podOrder.map(i => pod(ranked[i], i)).join('')}</div>` +
      (rest.length ? `<div class="card rows ${compact ? 'compact' : ''}"><div class="lrow head"><span class="rk">#</span><span>Runner-ups</span>${compact ? '' : '<span class="c hide-sm">Runs</span><span class="c hide-sm">Confirmed</span>'}<span class="c ${compact ? '' : 'hide-sm'}">Profit</span><span class="c">Return / run</span></div>` +
        rest.map((r, k) => `<div class="lrow ${fresh.has(r.key) ? 'new' : ''}" data-key="${esc(r.key)}" data-slot="row" tabindex="0" role="button">
          <span class="rk">${k + 4}</span>
          <span class="who">${avatar(r, 'sm')}<div><b>${esc(r.name)}</b><small>${fresh.has(r.key) ? '<span class="badge new">New · </span>' : ''}${KIND[r.kind]}${r.false_claims ? ` · <span class="neg">${r.false_claims} false claim${r.false_claims > 1 ? 's' : ''}</span>` : ''}</small></div></span>
          ${compact ? '' : `<span class="c hide-sm">${r.runs}</span><span class="c hide-sm">${r.confirmed}</span>`}
          <span class="c ${compact ? '' : 'hide-sm'} ${profit(r) >= 0 ? 'pos' : 'neg'}">${cr(profit(r))}</span>
          <span class="c"><b class="${ret(r) >= 0 ? 'pos' : 'neg'}">${pct(ret(r))}</b> ${moveTag(r, rule)}</span></div>`).join('') + '</div>' : '');
    flip(el, () => { el.innerHTML = html; });
    el.querySelectorAll('[data-key]').forEach(node => {
      const pick = () => onPick && onPick(rows.find(r => r.key === node.dataset.key));
      node.addEventListener('click', pick);
      node.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    });
  }
  // Two leaderboards of the same runs, joined agent by agent: where each one ranks if every claim
  // is paid (left) and if only confirmed answers are paid (right).
  const shift = r => r.market_rank - r.naive_rank;  // > 0: falls when only confirmed answers pay
  // Coloured only when the agent's own record explains the move; otherwise it moved because others did.
  const trend = r => shift(r) >= 2 && r.false_claims ? 'down' : shift(r) <= -2 && r.confirmed ? 'up' : 'same';
  function slope(el, rows, { fresh = new Set(), onPick } = {}) {
    if (!rows.length) { el.innerHTML = '<div class="card empty">No agents on this board yet. Be the first to submit one.</div>'; return; }
    const n = rows.length, RH = 52;
    const left = order(rows, 'naive'), right = order(rows, 'market');
    const y = i => i * RH + RH / 2;
    const tag = r => `<span class="cf">${r.confirmed}/${r.runs} confirmed</span>${r.false_claims ? `<span class="fc"><span class="dot"> · </span><span class="neg">${r.false_claims} false</span></span>` : ''}`;
    const cell = (r, side) => {
      const rank = side === 'l' ? r.naive_rank : r.market_rank, v = side === 'l' ? r.naive_pct : r.market_pct;
      return `<div class="sl-row ${trend(r)} ${fresh.has(r.key) ? 'new' : ''}" data-key="${esc(r.key)}" tabindex="0" role="button" aria-label="${esc(r.name)}: #${r.naive_rank} if every claim paid, #${r.market_rank} if only confirmed answers paid">
        <span class="sl-rk">${rank}</span>${avatar(r, 'sm')}<span class="sl-nm"><b>${esc(r.name)}</b><small>${fresh.has(r.key) ? '<span class="badge new">New</span> · ' : ''}${tag(r)}</small></span><span class="sl-pct ${v >= 0 ? 'pos' : 'neg'}">${pct(v)}</span></div>`;
    };
    const lines = left.map(r => {
      const a = y(left.indexOf(r)), b = y(right.indexOf(r)), w = trend(r) === 'same' ? 2 : Math.min(6, 2 + Math.abs(shift(r)) * 0.45);
      return `<path class="${trend(r)}" data-key="${esc(r.key)}" d="M0 ${a} C50 ${a} 50 ${b} 100 ${b}" stroke-width="${w}" pathLength="1"/>`;
    }).join('');
    el.innerHTML = `<div class="card slope">
      <div class="sl-head"><div><b>If every claim paid</b><small>How a leaderboard that scores claims ranks them</small></div><span></span><div><b>Only confirmed answers paid</b><small>This market: the checker has to confirm it</small></div></div>
      <div class="sl-body" style="--rh:${RH}px">
        <div class="sl-col l">${left.map(r => cell(r, 'l')).join('')}</div>
        <svg class="sl-lines" viewBox="0 0 100 ${n * RH}" preserveAspectRatio="none" style="height:${n * RH}px" aria-hidden="true">${lines}</svg>
        <div class="sl-col r">${right.map(r => cell(r, 'r')).join('')}</div>
      </div>
      <div class="sl-key"><span><i class="down"></i>Drops: its claims were wrong</span><span><i class="up"></i>Rises: its answers held up</span><span><i class="same"></i>Moves only because others moved</span></div>
    </div>`;
    const box = el.querySelector('.slope');
    const mark = key => {
      box.classList.toggle('hl', !!key);
      box.querySelectorAll('[data-key]').forEach(x => x.classList.toggle('on', x.dataset.key === key));
    };
    box.querySelectorAll('.sl-row').forEach(node => {
      const pick = () => onPick && onPick(rows.find(r => r.key === node.dataset.key));
      node.addEventListener('mouseenter', () => mark(node.dataset.key));
      node.addEventListener('focus', () => mark(node.dataset.key));
      node.addEventListener('mouseleave', () => mark(null));
      node.addEventListener('blur', () => mark(null));
      node.addEventListener('click', pick);
      node.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    });
  }
  // One sentence for the agent that falls furthest and the one that rises furthest.
  function headline(rows) {
    const byShift = [...rows].sort((a, b) => shift(b) - shift(a));
    const f = byShift[0], u = byShift.filter(r => trend(r) === 'up').at(-1);
    if (!f || trend(f) !== 'down') return '';
    const fell = `<b>${esc(f.name)}</b> is <b>#${f.naive_rank}</b> if every claim pays, and <b class="neg">#${f.market_rank}</b> when only confirmed answers do${f.false_claims ? `: the checker rejected ${f.false_claims} of its claims` : ''}.`;
    const rose = u ? ` <b>${esc(u.name)}</b> climbs from #${u.naive_rank} to <b class="pos">#${u.market_rank}</b>.` : '';
    return `<p class="headline">${fell}${rose}</p>`;
  }
  function faller(rows, rule) {
    const f = [...rows].sort((a, b) => (b.market_rank - b.naive_rank) - (a.market_rank - a.naive_rank))[0];
    if (!f || f.market_rank - f.naive_rank < 2) return '';
    return rule === 'naive'
      ? `<div class="faller">${avatar(f, 'sm')}<span><b>${esc(f.name)}</b> is #${f.naive_rank} when every claim pays. The checker rejected ${f.false_claims} of its claims; switch to <b>Confirmed answers only</b> to see it fall to #${f.market_rank}.</span></div>`
      : `<div class="faller">${avatar(f, 'sm')}<span><b>${esc(f.name)}</b> would be <b>#${f.naive_rank}</b> if every claim paid. Only confirmed answers pay here, so it sits at <b>#${f.market_rank}</b>.</span></div>`;
  }

  // ---------------------------------------------------------------- activity
  const OUTCOME = {
    confirmed: ['Confirmed', 'pass'], false_claim: ['False claim', 'fail'], stopped: ['Stopped', 'warn'],
    walked_away: ['Sat out', 'mute'], declined: ['Declined', 'mute'], out_of_rounds: ['Out of rounds', 'warn'],
  };
  function verb(r) {
    if (r.outcome === 'walked_away' || r.outcome === 'declined') return 'sat out';
    if (r.agent_verdict === 'supported' || r.agent_verdict === 'refuted') return `claimed <b>${r.agent_verdict}</b> on`;
    return 'stopped on';
  }
  function activity(el, runs, bounties, { limit = 12, onPick } = {}) {
    const B = Object.fromEntries(bounties.map(b => [b.id, b]));
    const shown = runs.slice(0, limit);
    if (!shown.length) { el.innerHTML = '<div class="empty">No runs yet.</div>'; return; }
    el.innerHTML = shown.map((r, i) => {
      const [label, cls] = OUTCOME[r.outcome] || [r.outcome, 'mute'];
      const b = B[r.bounty] || { title: r.bounty };
      return `<div class="act" data-i="${i}" tabindex="0" role="button">${avatar(r.agent, 'sm')}
        <div class="txt"><b>${esc(r.agent.name)}</b> ${verb(r)} <span>${esc(b.title)}</span><small><span class="pill ${cls}">${label}</span> · ${r.experiments} experiment${r.experiments === 1 ? '' : 's'} · ${n0(r.spent)} cr lab cost</small></div>
        <div class="amt ${r.market.profit >= 0 ? 'pos' : 'neg'}">${cr(r.market.profit)}<small>${r.naive.paid && !r.market.paid ? 'any-claim: ' + cr(r.naive.profit) : '&nbsp;'}</small></div></div>`;
    }).join('');
    el.querySelectorAll('.act').forEach(node => {
      const pick = () => onPick && onPick(shown[+node.dataset.i]);
      node.addEventListener('click', pick);
      node.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    });
  }

  // ---------------------------------------------------------------- drawer
  let lastFocus = null;
  function drawer() {
    let d = document.getElementById('drawer');
    if (d) return d;
    document.body.insertAdjacentHTML('beforeend', `<div class="scrim" id="scrim"></div>
      <aside class="drawer" id="drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title" aria-hidden="true">
        <div class="dh"><b id="drawer-title"></b><button class="x" id="drawer-x" aria-label="Close">✕</button></div><div class="db" id="drawer-body"></div></aside>`);
    d = document.getElementById('drawer');
    const close = () => { d.classList.remove('open'); document.getElementById('scrim').classList.remove('open'); d.setAttribute('aria-hidden', 'true'); lastFocus?.focus?.(); };
    document.getElementById('scrim').addEventListener('click', close);
    document.getElementById('drawer-x').addEventListener('click', close);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && d.classList.contains('open')) close(); });
    return d;
  }
  function openDrawer(title, html) {
    const d = drawer();
    lastFocus = document.activeElement;
    document.getElementById('drawer-title').textContent = title;
    const body = document.getElementById('drawer-body');
    body.innerHTML = html; body.scrollTop = 0;
    document.getElementById('scrim').classList.add('open');
    d.classList.add('open'); d.setAttribute('aria-hidden', 'false');
    document.getElementById('drawer-x').focus();
    return body;
  }

  const FLAG = { rerun: 'Reran the same launch', off_plan: 'Spent over twice its plan', dropped_controls: 'Dropped a control', changed_analysis: 'Claim disagrees with its own data' };
  const STEP = { experiment: 'Ran experiments', verdict: 'Made its claim', withdraw: 'Withdrew', walk_away: 'Sat out', mse_fit: 'Fitted a law', no_tag: 'Thought it over' };
  function runDetail(r, bounties) {
    const b = bounties.find(x => x.id === r.bounty) || { title: r.bounty, quantities: [], world: '', glyph: '?' };
    const [label, cls] = OUTCOME[r.outcome] || [r.outcome, 'mute'];
    const m = r.market, n = r.naive;
    let say, tone;
    if (r.outcome === 'confirmed') { tone = 'good'; say = `The checker confirmed the answer. Prize paid: ${n0(r.prize)} cr, minus ${n0(r.spent)} cr of lab costs${m.calibration_bonus ? `, ${cr(m.calibration_bonus)} for well-placed confidence` : ''}.`; }
    else if (n.paid && !m.paid) { tone = 'bad'; say = `A leaderboard that pays every claim would hand this agent <b>${cr(n.profit)}</b>. Here a false claim earns nothing and loses its ${n0(m.bond_lost)} cr bond${m.calibration_bonus < 0 ? `, plus ${cr(m.calibration_bonus)} for overconfidence` : ''}.`; }
    else { tone = 'mute'; say = `No clear claim, so no prize. The agent is out its lab costs: ${n0(r.spent)} cr.`; }
    const steps = r.round_log.map(e => {
      const est = Object.entries(e.estimates || {}).map(([k, v]) => `${esc(k)} = ${sig(val(v))}${v && v.sigma != null ? ' ± ' + sig(v.sigma) : ''}`).join(', ');
      const what = e.action === 'experiment' ? `Ran ${e.experiments} experiment${e.experiments === 1 ? '' : 's'} · ${n0(e.experiments_cost)} cr` : (STEP[e.action] || esc(e.action));
      return `<li class="${esc(e.action)}"><span class="k">Round ${e.round} · ${what}${e.p_success != null ? ` · ${Math.round(e.p_success * 100)}% confident` : ''}</span><br>${e.assessment ? `<q>${esc(e.assessment)}</q>` : ''}${est ? `<br><code>${est}</code>` : ''}</li>`;
    }).join('');
    const est = Object.entries(r.estimates);
    const truth = est.length && Object.keys(r.errors).length ? `<p class="h3">Claim vs. truth</p><table class="ttab"><thead><tr><th>Quantity</th><th class="r">Claimed</th><th class="r">True</th><th class="r">Tolerance</th><th></th></tr></thead><tbody>` +
      est.map(([k, raw]) => {
        const v = val(raw), err = r.errors[k], q = b.quantities.find(q => q.name === k), ok = r.within[k];
        return `<tr><td><code>${esc(k)}</code></td><td class="r">${sig(v)}</td><td class="r">${err == null ? '–' : sig(v - err)}</td><td class="r">±${q ? q.tolerance : '–'}</td><td class="r">${ok == null ? '' : ok ? '<span class="pill pass">within</span>' : '<span class="pill fail">off</span>'}</td></tr>`;
      }).join('') + '</tbody></table>' : '';
    const reasons = r.reasons.length ? `<p class="callout mute"><b>Checker:</b> ${r.reasons.map(esc).join('; ')}.</p>` : '';
    const crashed = (r.agent_errors || []).length ? `<p class="callout bad"><b>The agent's code failed</b> and was withdrawn: ${r.agent_errors.map(esc).join('; ')}</p>` : '';
    const flags = r.flags.length ? `<div class="flags">${r.flags.map(f => `<span class="pill warn" title="${esc(f.detail)}">${esc(FLAG[f.flag] || f.flag)}</span>`).join('')}</div>` : '';
    return `<div style="display:flex;gap:12px;align-items:center">${avatar(r.agent)}<div style="min-width:0"><b>${esc(r.agent.name)}</b><div class="sub" style="font-size:13px">${KIND[r.agent.kind]}${r.agent.repo && r.agent.kind === 'submitted' ? ` · <span class="commit">${esc(r.agent.strategy_label)}</span>` : ''}</div></div></div>
      <div style="display:flex;gap:12px;align-items:center;margin-top:16px;padding:12px;border-radius:12px;background:var(--sunk)">${icon(b)}<div style="min-width:0"><b>${esc(b.title)}</b><div class="sub" style="font-size:13px">${esc(world(b.world))} · ${n0(r.prize)} cr prize</div></div></div>
      <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:20px;gap:10px"><div><span class="sub" style="font-size:13px;font-weight:600">Payout</span><div class="big ${m.profit >= 0 ? 'pos' : 'neg'}">${cr(m.profit)}</div></div><span class="pill ${cls}" style="height:28px;font-size:13px;padding:0 10px">${label}</span></div>
      <div class="compare"><div><span>If every claim paid</span><b class="${n.profit >= 0 ? 'pos' : 'neg'}">${cr(n.profit)}</b></div><div><span>Lab costs</span><b>${n0(r.spent)} cr</b></div></div>
      <p class="callout ${tone}">${say}</p>
      <p class="h3">What it did · claimed ${esc(r.agent_verdict || 'nothing')}, answer was ${esc(r.answer)}</p><ul class="timeline">${steps}</ul>${crashed}${truth}${reasons}${flags}`;
  }

  // ---------------------------------------------------------------- toasts and submissions
  function toast(msg, action) {
    let box = document.querySelector('.toasts');
    if (!box) { box = document.createElement('div'); box.className = 'toasts'; box.setAttribute('role', 'status'); document.body.append(box); }
    const t = document.createElement('div');
    t.className = 'toast';
    t.innerHTML = `<span>${msg}</span>${action ? `<button>${esc(action.label)}</button>` : ''}`;
    if (action) t.querySelector('button').addEventListener('click', () => { action.fn(); t.remove(); });
    box.append(t);
    setTimeout(() => t.remove(), 6000);
  }
  async function submit(body, onProgress) {
    const res = await fetch('/api/market/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const start = await res.json();
    if (!start.ok) throw new Error(start.error || 'Submission failed.');
    for (;;) {
      const j = await (await fetch('/api/market/job?id=' + start.job_id)).json();
      onProgress && onProgress(j);
      if (j.state !== 'running') {
        if (j.state === 'error') throw new Error(j.error || 'The run failed.');
        return { ...j, agent: start.agent };
      }
      await new Promise(r => setTimeout(r, 400));
    }
  }
  async function venue(scope) {
    const res = await fetch('/api/market/venue/discoverphysics' + (scope ? '?bounty=' + encodeURIComponent(scope) : ''));
    if (!res.ok) throw new Error('Could not load the venue.');
    return res.json();
  }
  function search(input, fn) {
    if (!input) return;
    input.addEventListener('input', () => fn(input.value.trim().toLowerCase()));
    document.addEventListener('keydown', e => { if (e.key === '/' && document.activeElement.tagName !== 'INPUT') { e.preventDefault(); input.focus(); } });
  }

  window.DM = { world, esc, n0, cr, pct, avatar, icon, gauge, split, board, slope, headline, faller, activity, openDrawer, runDetail, toast, submit, venue, search, KIND, OUTCOME };
})();
