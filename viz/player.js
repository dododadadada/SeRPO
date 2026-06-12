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
    renderAppUI(model, step.ui_state);
    return;
  }

  // Non-empty step: flush any prior empty-step run before showing this bubble.
  flushEmptyRun(model);

  const panel = panelFor(model);
  const chat = panel.querySelector('[data-role=chat]');

  const agent = document.createElement('div');
  agent.className = 'bubble agent';
  agent.innerHTML = `<div class="who">🤖 Agent — step ${step.step}</div>` +
                    `<code class="mono">${escapeHtml(step.code)}</code>`;
  chat.appendChild(agent);

  const env = document.createElement('div');
  env.className = 'bubble env';
  env.innerHTML = `<div class="who">🌐 AppWorld</div>` +
                  `<div class="out">${escapeHtml(step.output)}</div>`;
  chat.appendChild(env);
  chat.scrollTop = chat.scrollHeight;

  renderAppUI(model, step.ui_state);
}

function renderAppUI(model, ui) {
  const appui = panelFor(model).querySelector('[data-role=appui]');
  if (!ui) return;
  let html = `<h3>${escapeHtml(ui.title || '')}</h3>`;
  if (ui.kind === 'docs') {
    html += `<div class="docs-note">Looking up available APIs…</div>`;
  } else if (ui.kind === 'login') {
    html += `<div class="app-row">🔓 Signed in</div>`;
  } else if (ui.kind === 'venmo_transactions') {
    for (const r of ui.rows) {
      const hit = /electric|power/i.test(r.description || '');
      html += `<div class="app-row ${hit ? 'hit' : ''}">
        <span>${escapeHtml(r.sender)} → ${escapeHtml(r.receiver)}: ${escapeHtml(r.description)}</span>
        <b>$${escapeHtml(r.amount)}</b></div>`;
    }
  } else if (ui.kind === 'phone_alarms') {
    for (const r of ui.rows) {
      html += `<div class="app-row ${r.changed ? 'changed' : ''}">
        <span>⏰ ${escapeHtml(r.label)} (${escapeHtml(r.time)})</span>
        <b>snooze ${escapeHtml(r.snooze_minutes)}m</b></div>`;
    }
  } else if (ui.kind === 'result') {
    html += `<div class="result-card">Task submitted</div>`;
  } else if (ui.kind === 'api_call') {
    html += `<div class="app-row">${escapeHtml(ui.title)}</div>`;
  }
  appui.innerHTML = html;
}

function maybeDoneBanner(model, doc) {
  const appui = panelFor(model).querySelector('[data-role=appui]');
  if (appui.querySelector('.done-banner')) return;
  const banner = document.createElement('div');
  const pass = doc.passed === true;
  banner.className = `done-banner ${pass ? 'pass' : 'fail'}`;
  banner.textContent = pass ? '✓ Task passed' : '✗ Task failed';
  appui.appendChild(banner);
}

function tick() {
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
    if (!paused && !timer) scheduleNext();
  };
  document.getElementById('restart').onclick = () => {
    if (current) startTask(current.vanilla.task_id);
  };
}

wireControls();
initSelector().catch(err => {
  document.getElementById('task-cards').textContent = 'Failed to load data: ' + err.message;
});
