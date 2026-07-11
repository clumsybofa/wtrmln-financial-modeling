// wtrmln operator UI.
// Rendering rule: anything that originated from the model, a provider page,
// or the database is inserted with textContent — never innerHTML.

let currentSession = null;
let awaiting = false;
let evtSource = null;

// Session tokens are held per-browser; only sessions started (or previously
// watched) from this browser can be controlled.
function tokenFor(sessionId) {
  return localStorage.getItem('wtrmln_token_' + sessionId) || '';
}
function rememberToken(sessionId, token) {
  localStorage.setItem('wtrmln_token_' + sessionId, token);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadConnectors() {
  const list = await api('/api/connectors');
  const root = document.getElementById('connectors');
  root.replaceChildren();
  for (const c of list) {
    const card = el('div', 'card');
    const row = el('div', 'row');
    row.append(el('span', 'icon', c.icon || '🔌'),
               el('span', 'name', c.name),
               el('span', 'spacer'));
    const btn = el('button', null, 'Connect');
    btn.addEventListener('click', () => connect(c.slug));
    row.append(btn);
    card.append(row, el('div', 'desc', c.description || ''));
    root.append(card);
  }
}

async function loadConnections() {
  const list = await api('/api/connections');
  const root = document.getElementById('connections');
  root.replaceChildren();
  if (!list.length) {
    root.append(el('div', 'conn', 'None yet.'));
    return;
  }
  for (const c of list) {
    const card = el('div', 'card conn');
    const head = el('div');
    head.append(el('b', null, c.connector), document.createTextNode(' — '),
                el('span', 'status ' + c.status, c.status.replace('_', ' ')));
    card.append(head);
    if (c.summary) card.append(el('div', 'desc', c.summary));
    if (c.session_id && tokenFor(c.session_id)) {
      const wrap = el('div');
      wrap.style.marginTop = '6px';
      const btn = el('button', 'secondary', 'Watch');
      btn.addEventListener('click', () => watch(c.session_id));
      wrap.append(btn);
      card.append(wrap);
    }
    root.append(card);
  }
}

async function connect(slug) {
  try {
    const r = await api('/api/connections', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({connector: slug}),
    });
    rememberToken(r.session_id, r.session_token);
    watch(r.session_id);
    loadConnections();
  } catch (e) {
    alert('Could not start session: ' + e.message);
  }
}

function watch(sessionId) {
  currentSession = sessionId;
  document.getElementById('feed').replaceChildren();
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('screen').style.display = 'block';
  if (evtSource) evtSource.close();
  // EventSource cannot set headers, so the token rides as a query parameter.
  evtSource = new EventSource(
    `/api/sessions/${encodeURIComponent(sessionId)}/events?token=` +
    encodeURIComponent(tokenFor(sessionId)));
  evtSource.onmessage = (m) => handleEvent(JSON.parse(m.data));
}

function feed(text, cls) {
  const d = el('div', 'ev' + (cls ? ' ' + cls : ''), text);
  const f = document.getElementById('feed');
  f.append(d);
  f.scrollTop = f.scrollHeight;
}

function feedStatus(status, summary) {
  const d = el('div', 'ev');
  d.append(el('b', null, 'Status: '),
           el('span', 'status ' + status, status.replace('_', ' ')));
  if (summary) d.append(el('div', null, summary));
  const f = document.getElementById('feed');
  f.append(d);
  f.scrollTop = f.scrollHeight;
}

function handleEvent(e) {
  if (e.type === 'screen') refreshScreen();
  else if (e.type === 'agent_message') feed(e.text, 'agent');
  else if (e.type === 'log') feed(e.text);
  else if (e.type === 'action') feed('▸ ' + e.action + ' ' + JSON.stringify(e.detail || {}), 'action');
  else if (e.type === 'credential_saved') feed('🔐 Saved "' + e.name + '" to encrypted vault', 'cred');
  else if (e.type === 'status') {
    setAwaiting(e.status === 'awaiting_user', e.reason);
    feedStatus(e.status, e.summary);
    loadConnections();
  }
}

function setAwaiting(on, reason) {
  awaiting = on;
  document.getElementById('banner').className = on ? 'show' : '';
  document.getElementById('typebar').className = on ? 'show' : '';
  document.getElementById('screen').className = on ? 'interactive' : '';
  if (on) {
    document.getElementById('bannertext').textContent =
      '🔒 ' + (reason || 'Please log in.') +
      ' Your input goes directly to the site — the AI never sees it.';
  }
}

function refreshScreen() {
  if (!currentSession) return;
  document.getElementById('screen').src =
    `/api/sessions/${encodeURIComponent(currentSession)}/screen?token=` +
    encodeURIComponent(tokenFor(currentSession)) + '&ts=' + Date.now();
}

async function sessionPost(path, body) {
  if (!currentSession) return;
  await fetch(`/api/sessions/${encodeURIComponent(currentSession)}/${path}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json',
              'X-Session-Token': tokenFor(currentSession)},
    body: body ? JSON.stringify(body) : null,
  });
}

document.getElementById('screen').addEventListener('click', (ev) => {
  if (!awaiting) return;
  const rect = ev.target.getBoundingClientRect();
  sessionPost('input', {
    kind: 'click',
    x: (ev.clientX - rect.left) * (1280 / rect.width),
    y: (ev.clientY - rect.top) * (800 / rect.height),
  });
});

document.getElementById('typeinput').addEventListener('keydown', async (ev) => {
  if (!awaiting) return;
  if (ev.key === 'Enter') {
    const val = ev.target.value;
    ev.target.value = '';
    if (val) await sessionPost('input', {kind: 'type', text: val});
    await sessionPost('input', {kind: 'key', text: 'Return'});
  }
});

document.getElementById('tabbtn').addEventListener('click',
  () => sessionPost('input', {kind: 'key', text: 'Tab'}));
document.getElementById('resumebtn').addEventListener('click',
  () => sessionPost('resume'));
document.getElementById('abortbtn').addEventListener('click',
  () => sessionPost('abort'));

loadConnectors();
loadConnections();
setInterval(loadConnections, 5000);
