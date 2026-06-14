// AppWorld trajectory comparison player.
let manifest = [];
let timer = null;
let paused = false;
let stepIndex = 0;
let current = null; // { vanilla: doc, serpo: doc, maxSteps }
const emptyRun = { vanilla: 0, serpo: 0 }; // count of pending collapsed empty steps

async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  return r.json();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

async function initSelector() {
  manifest = await loadJSON('data/manifest.json');
  const wrap = document.getElementById('task-cards');
  wrap.innerHTML = '';
  for (const t of manifest) {
    const card = document.createElement('div');
    card.className = 'task-card';
    card.innerHTML = `
      <div class="app">${escapeHtml(t.app)}</div>
      <div class="instr">${escapeHtml(t.instruction)}</div>
      <div class="badge">Vanilla <span class="fail">✗ fail</span>
        &nbsp;→&nbsp; SeRPO <span class="pass">✓ pass</span></div>`;
    card.onclick = () => startTask(t.task_id);
    wrap.appendChild(card);
  }
}

async function startTask(taskId) {
  if (timer) { clearTimeout(timer); timer = null; }
  const vanilla = await loadJSON(`data/${taskId}.vanilla.json`);
  const serpo = await loadJSON(`data/${taskId}.serpo.json`);
  current = { vanilla, serpo, maxSteps: Math.max(vanilla.num_steps, serpo.num_steps) };
  document.getElementById('selector').hidden = true;
  document.getElementById('stage').hidden = false;
  document.getElementById('task-instruction').textContent = vanilla.instruction;
  for (const p of document.querySelectorAll('.panel')) {
    p.querySelector('[data-role=chat]').innerHTML = '';
    p.querySelector('[data-role=appui]').innerHTML = '';
  }
  stepIndex = 0;
  emptyRun.vanilla = 0; emptyRun.serpo = 0;
  paused = false;
  document.getElementById('playpause').textContent = '⏸ Pause';
  scheduleNext();
}

function panelFor(model) { return document.querySelector(`.panel.${model}`); }

// Flush any accumulated empty-code steps as a single muted note bubble.
// Called when a non-empty step arrives (before rendering it) or when the
// trajectory ends (so trailing empty steps also get reported).
function flushEmptyRun(model) {
  if (emptyRun[model] > 0) {
    const chat = panelFor(model).querySelector('[data-role=chat]');
    const note = document.createElement('div');
    note.className = 'bubble note';
    note.textContent = `… ${emptyRun[model]} empty step${emptyRun[model] > 1 ? 's' : ''}`;
    chat.appendChild(note);
    chat.scrollTop = chat.scrollHeight;
    emptyRun[model] = 0;
  }
}

// Render one step for one model panel.
// Empty-code steps: increment emptyRun counter; still update app UI; do NOT
// add agent/env bubbles.  Non-empty steps: flush any pending emptyRun first,
// then add bubbles normally.
function renderStep(model, doc) {
  if (stepIndex >= doc.num_steps) {
    // Trajectory is done — flush any trailing empty-step run, then banner.
    flushEmptyRun(model);
    maybeDoneBanner(model, doc);
    return;
  }
  const step = doc.steps[stepIndex];
  const isEmpty = !step.code || !step.code.trim();

  if (isEmpty) {
    // Accumulate into the current empty-step run.
    emptyRun[model]++;
    // App UI still reflects this step's state (usually idle/unchanged).
    renderAppUI(model, step, doc.app);
    return;
  }

  // Non-empty step: flush any prior empty-step run before showing this bubble.
  flushEmptyRun(model);

  const panel = panelFor(model);
  const chat = panel.querySelector('[data-role=chat]');

  // Chat shows ONLY the agent's code. AppWorld's response is shown as a GUI
  // visualization in the right panel (renderAppUI), not as a text bubble.
  const agent = document.createElement('div');
  agent.className = 'bubble agent' + (step.is_error ? ' errored' : '');
  agent.innerHTML = `<div class="who">🤖 Agent — step ${escapeHtml(step.step)}</div>` +
                    `<code class="mono">${escapeHtml(step.code)}</code>`;
  chat.appendChild(agent);

  // Mark a runtime dead-end inline so the audience sees the agent stumble.
  if (step.is_error) {
    const marker = document.createElement('div');
    marker.className = 'bubble error-marker';
    marker.textContent = `⚠️ ${step.error || 'Execution failed'}`;
    chat.appendChild(marker);
  }
  chat.scrollTop = chat.scrollHeight;

  renderAppUI(model, step, doc.app);
}

// Brand + display name for the device frame, by app.
const APP_BRAND = {
  venmo: { cls: 'venmo', name: 'venmo' },
  phone: { cls: 'phone', name: '⏰ Alarms' },
};

function deviceFrame(app, bodyHtml) {
  const brand = APP_BRAND[app] || { cls: 'neutral', name: 'AppWorld' };
  return `<div class="device ${brand.cls}">
    <div class="device-head">
      <div class="clock">9:41</div>
      <div class="brand">${escapeHtml(brand.name)}</div>
    </div>
    <div class="device-body">${bodyHtml}</div>
  </div>`;
}

// Centered status screen used for plain steps (login / docs / generic calls).
function statusScreen(emoji, msg) {
  return `<div class="status-screen"><div class="big">${emoji}</div>
          <div class="msg">${escapeHtml(msg)}</div></div>`;
}

// A one-line, human-readable summary of what a step's call RETURNED.
function resultLine(step) {
  if (step.is_error) return step.error || 'Execution failed';
  // Don't surface credentials in the result line.
  if (step.api === 'show_account_passwords') return 'credentials retrieved';
  const ui = step.ui_state || {};
  switch (ui.kind) {
    case 'docs': return 'API list / schema returned';
    case 'login': return 'access token received';
    case 'phone_alarms':
      if (step.api === 'update_alarm') {
        const ch = (ui.rows || []).find(r => r.changed);
        return ch ? `alarm ${ch.time} → snooze ${ch.snooze_minutes}m` : 'alarm updated';
      }
      return `${(ui.rows || []).length} alarms returned`;
    case 'venmo_transactions': return 'transactions scanned';
    case 'result':
      return ui.answer != null ? `answer submitted: ${ui.answer}` : 'task submitted (no answer)';
    default: {
      // Fall back to a trimmed first line of the raw output.
      const first = (step.output || '').trim().split('\n')[0];
      return first.length > 70 ? first.slice(0, 70) + '…' : (first || 'done');
    }
  }
}

// The "what the agent tried → what happened" strip shown atop every screen.
function actionStrip(step) {
  const call = (step.app && step.api) ? `${step.app}.${step.api}()` : 'compute';
  const cls = step.is_error ? 'result-line err' : 'result-line';
  const icon = step.is_error ? '⚠️' : '↳';
  return `<div class="action-strip">
    <div class="tried-line">▶ ${escapeHtml(call)}</div>
    <div class="${cls}">${icon} ${escapeHtml(resultLine(step))}</div>
  </div>`;
}

function renderAppUI(model, step, app) {
  const appui = panelFor(model).querySelector('[data-role=appui]');
  const ui = step && step.ui_state;
  if (!ui) return;
  let body;
  if (ui.kind === 'docs') {
    body = statusScreen('📖', 'Reading API documentation…');
  } else if (ui.kind === 'login') {
    body = statusScreen('🔓', ui.title || 'Signed in');
  } else if (ui.kind === 'venmo_transactions') {
    // The agent computed the total in code; no row list comes back.
    body = `<div class="status-screen"><div class="big">💳</div>
            <div class="msg">Scanning sent transactions…</div></div>`;
  } else if (ui.kind === 'phone_alarms') {
    body = (!ui.rows || ui.rows.length === 0)
      ? statusScreen('⏰', 'Loading alarms…')
      : (ui.rows || []).map(r => `
      <div class="alarm ${r.changed ? 'changed' : ''}">
        <div>
          <div class="t">${escapeHtml(r.time)}</div>
          <div class="sub">${escapeHtml(r.label)} · snooze ${escapeHtml(r.snooze_minutes)}m${r.changed ? ' ⟲' : ''}</div>
        </div>
        <div class="state">${r.enabled ? 'on' : 'off'}</div>
      </div>`).join('');
  } else if (ui.kind === 'result') {
    const ans = ui.answer != null
      ? `<div class="cap">ELECTRICITY THIS YEAR</div>
         <div class="total">$${escapeHtml(ui.answer)}</div>
         <div class="note">computed from your sent transactions</div>
         <div class="answer">✓ Answer submitted: ${escapeHtml(ui.answer)}</div>`
      : `<div class="answer">✓ Task submitted</div>`;
    body = `<div class="venmo-hero">${ans}</div>`;
  } else if (ui.kind === 'api_call') {
    body = statusScreen('⚙️', ui.title || 'API call');
  } else { // idle / unknown
    body = statusScreen('•', ui.title || 'Working…');
  }
  // On a runtime error, show WHY prominently as the screen content.
  if (step.is_error) {
    body = `<div class="error-screen">
        <div class="big">⚠️</div>
        <div class="er-title">Call failed</div>
        <div class="er-why">${escapeHtml(step.error || 'Execution failed')}</div>
      </div>`;
  }
  // Every screen leads with the "tried → result" strip.
  appui.innerHTML = deviceFrame(app, actionStrip(step) + body);
}

function maybeDoneBanner(model, doc) {
  const appui = panelFor(model).querySelector('[data-role=appui]');
  if (appui.querySelector('.done-banner')) return;
  const banner = document.createElement('div');
  const pass = doc.passed === true;
  banner.className = `done-banner ${pass ? 'pass' : 'fail'}`;
  if (pass) {
    banner.textContent = '✓ Task passed';
  } else {
    // For a failed run, show WHY it failed when we have a reason.
    banner.innerHTML = `<div class="bn-title">✗ Task failed</div>` +
      (doc.fail_reason ? `<div class="bn-why">${escapeHtml(doc.fail_reason)}</div>` : '');
  }
  appui.appendChild(banner);
}

function tick() {
  timer = null;
  renderStep('vanilla', current.vanilla);
  renderStep('serpo', current.serpo);
  stepIndex++;
  if (stepIndex >= current.maxSteps) {
    // One final pass to flush trailing empty runs and show done banners.
    renderStep('vanilla', current.vanilla);
    renderStep('serpo', current.serpo);
    timer = null;
    return;
  }
  scheduleNext();
}

function scheduleNext() {
  if (paused) return;
  const delay = parseInt(document.getElementById('speed').value, 10);
  timer = setTimeout(tick, delay);
}

function wireControls() {
  document.getElementById('back').onclick = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    document.getElementById('stage').hidden = true;
    document.getElementById('selector').hidden = false;
  };
  document.getElementById('playpause').onclick = (e) => {
    paused = !paused;
    e.target.textContent = paused ? '▶ Play' : '⏸ Pause';
    if (paused) {
      if (timer) { clearTimeout(timer); timer = null; }
    } else if (!timer) {
      scheduleNext();
    }
  };
  document.getElementById('restart').onclick = () => {
    if (current) startTask(current.vanilla.task_id);
  };
}

wireControls();
initSelector().catch(err => {
  document.getElementById('task-cards').textContent = 'Failed to load data: ' + err.message;
});
