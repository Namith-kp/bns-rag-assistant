/* ── BNS 2023 RAG Frontend ────────────────────────────────────────────────── */

// ── DOM Refs ─────────────────────────────────────────────────────────────────
const $messages      = document.getElementById('messages');
const $queryInput    = document.getElementById('query-input');
const $sendBtn       = document.getElementById('send-btn');
const $inputForm     = document.getElementById('input-form');
const $healthBadge   = document.getElementById('health-badge');
const $healthDot     = document.getElementById('health-dot');
const $healthText    = document.getElementById('health-text');
const $errorBanner   = document.getElementById('error-banner');
const $errorMsg      = document.getElementById('error-msg');
const $errorDismiss  = document.getElementById('error-dismiss');
const $inspectorToggle = document.getElementById('inspector-toggle-btn');
const $inspector       = document.getElementById('inspector');
const $inspectorIdle   = document.getElementById('inspector-idle');
const $pipelineWrap    = document.getElementById('pipeline-wrap');
const $tokenCounter    = document.getElementById('token-counter');
const $tokenStream     = document.getElementById('token-stream');
const $tokenDisplay    = document.getElementById('token-display');
const $emptyState      = document.getElementById('empty-state');

// Stage DOM nodes
const stages = {
  embedding:   document.getElementById('stage-embedding'),
  retrieval:   document.getElementById('stage-retrieval'),
  prompt_built: document.getElementById('stage-prompt'),
  generating:  document.getElementById('stage-generating'),
  done:        document.getElementById('stage-done'),
};

const stageBadges = {
  embedding:   document.getElementById('badge-embedding'),
  retrieval:   document.getElementById('badge-retrieval'),
  prompt_built: document.getElementById('badge-prompt'),
  generating:  document.getElementById('badge-generating'),
  done:        document.getElementById('badge-done'),
};

const stageDetails = {
  retrieval:   document.getElementById('detail-retrieval'),
  prompt_built: document.getElementById('detail-prompt'),
};

// Architecture diagram nodes (by stage)
const archNodes = {
  embedding:   document.getElementById('arch-embed'),
  retrieval:   document.getElementById('arch-search'),
  prompt_built: document.getElementById('arch-context'),
  generating:  document.getElementById('arch-llm'),
  done:        document.getElementById('arch-answer'),
};

// ── State ─────────────────────────────────────────────────────────────────────
let isStreaming = false;
let currentAssistantMsg = null;
let currentSources = [];
let tokenCount = 0;
let tokenStart = null;
let chatHistory = [];

// ── Utility ────────────────────────────────────────────────────────────────────
const timeNow = () => new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});

function showError(msg) {
  $errorMsg.textContent = msg;
  $errorBanner.classList.add('visible');
}

function hideError() {
  $errorBanner.classList.remove('visible');
}

function setLocked(locked) {
  isStreaming = locked;
  $sendBtn.disabled = locked;
  $queryInput.disabled = locked;
}

function scrollToBottom() {
  $messages.scrollTop = $messages.scrollHeight;
}

// ── Health Check ───────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.status === 'ok') {
      $healthBadge.className = 'ok';
      $healthDot.className = 'health-dot';
      $healthText.textContent = 'Models OK';
    } else if (data.status === 'degraded') {
      $healthBadge.className = 'warn';
      $healthDot.className = 'health-dot';
      const missing = [];
      if (!data.chat_ok)  missing.push('chat');
      if (!data.embed_ok) missing.push('embed');
      $healthText.textContent = `Missing: ${missing.join(', ')}`;
    } else {
      throw new Error(data.detail || 'unknown error');
    }
  } catch (e) {
    $healthBadge.className = 'error';
    $healthDot.className = 'health-dot';
    $healthText.textContent = 'LM Studio offline';
  }
}

// ── Architecture Diagram Highlight ─────────────────────────────────────────────
function highlightArch(stageKey) {
  Object.entries(archNodes).forEach(([key, node]) => {
    if (!node) return;
    node.classList.remove('arch-active', 'arch-done');
  });
  // Mark everything before current as done
  const order = ['embedding','retrieval','prompt_built','generating','done'];
  const idx = order.indexOf(stageKey);
  order.forEach((key, i) => {
    if (!archNodes[key]) return;
    if (i < idx)         archNodes[key].classList.add('arch-done');
    else if (i === idx)  archNodes[key].classList.add('arch-active');
  });
}

// ── Stage Timeline ─────────────────────────────────────────────────────────────
function resetPipeline() {
  Object.values(stages).forEach(el => {
    if (!el) return;
    el.classList.remove('active','done');
  });
  Object.values(stageBadges).forEach(el => {
    if (!el) return;
    el.textContent = '';
  });
  if (stageDetails.retrieval) stageDetails.retrieval.innerHTML = '';
  if (stageDetails.prompt_built) stageDetails.prompt_built.innerHTML = '';
  if ($tokenStream) $tokenStream.textContent = '';
  if ($tokenCounter) $tokenCounter.textContent = '0 tokens';
  if ($tokenDisplay) $tokenDisplay.style.display = 'none';
  $inspectorIdle.style.display = 'none';
  $pipelineWrap.style.display = 'flex';
  Object.values(archNodes).forEach(n => n && n.classList.remove('arch-active','arch-done'));
}

function activateStage(key) {
  const el = stages[key];
  if (!el) return;
  // Deactivate all, mark previous ones done if needed
  Object.entries(stages).forEach(([k, s]) => {
    if (!s) return;
    s.classList.remove('active');
  });
  el.classList.add('active');
  highlightArch(key);
}

function completeStage(key, badgeText) {
  const el = stages[key];
  if (!el) return;
  el.classList.remove('active');
  el.classList.add('done');
  const badge = stageBadges[key];
  if (badge && badgeText) badge.textContent = badgeText;
}

// ── Chunk Cards ────────────────────────────────────────────────────────────────
function buildChunkCard(chunk) {
  const card = document.createElement('div');
  card.className = 'chunk-card fade-in';

  const pct = Math.round(chunk.score * 100);
  card.innerHTML = `
    <div class="chunk-card-header">
      <span class="chunk-rank">#${chunk.rank}</span>
      <span class="chunk-score-label mono">${pct}% match</span>
    </div>
    <div class="score-bar-wrap">
      <div class="score-bar-fill" style="width:${pct}%"></div>
    </div>
    <div class="chunk-preview">${escHtml(chunk.preview)}</div>
    <button class="chunk-expand-btn" aria-expanded="false">
      + show full text
    </button>
    <div class="chunk-full-text">${escHtml(chunk.full_text)}</div>
  `;

  const expandBtn  = card.querySelector('.chunk-expand-btn');
  const fullText   = card.querySelector('.chunk-full-text');
  expandBtn.addEventListener('click', () => {
    const open = fullText.classList.toggle('open');
    expandBtn.textContent = open ? '- hide full text' : '+ show full text';
    expandBtn.setAttribute('aria-expanded', open);
  });

  return card;
}

// ── Chat Messages ──────────────────────────────────────────────────────────────
function escHtml(str) {
  return str
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function appendUserMessage(query) {
  if ($emptyState) $emptyState.remove();

  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-body">
      <div class="msg-meta">${timeNow()}</div>
      <div class="msg-content">${escHtml(query)}</div>
    </div>
  `;
  $messages.appendChild(el);
  scrollToBottom();
}

function createAssistantMessage() {
  const el = document.createElement('div');
  el.className = 'msg assistant fade-in';
  el.innerHTML = `
    <div class="msg-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
      </svg>
    </div>
    <div class="msg-body">
      <div class="msg-meta">BNS RAG · ${timeNow()}</div>
      <div class="msg-content" id="streaming-content">
        <span class="streaming-cursor"></span>
      </div>
    </div>
  `;
  $messages.appendChild(el);
  scrollToBottom();
  return el;
}

function finalizeAssistantMessage(msgEl, answer, sources) {
  const content = msgEl.querySelector('#streaming-content');
  content.id = '';

  // Build sources button
  const sourcesHtml = sources.length ? `
    <button class="sources-toggle" aria-expanded="false">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
      View ${sources.length} source${sources.length > 1 ? 's' : ''}
    </button>
    <div class="sources-list hidden">
      ${sources.map((s,i) => `
        <div class="source-chip">
          <span class="chip-score">[${Math.round(s.score*100)}%]</span>
          ${escHtml(s.text.replace(/\n/g,' ').slice(0,160))}${s.text.length > 160 ? '…' : ''}
        </div>
      `).join('')}
    </div>
  ` : '';

  content.innerHTML = `
    <p>${escHtml(answer)}</p>
    ${sourcesHtml}
  `;

  // Toggle sources
  const toggle = content.querySelector('.sources-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const list = content.querySelector('.sources-list');
      const open = list.classList.toggle('hidden');
      toggle.classList.toggle('open', !open);
      toggle.setAttribute('aria-expanded', !open);
    });
  }

  scrollToBottom();
}

// ── SSE Event Handlers ─────────────────────────────────────────────────────────
function handleEmbedding(data) {
  if (data.status === 'running') {
    activateStage('embedding');
  } else if (data.status === 'done') {
    completeStage('embedding', `${data.ms}ms`);
  }
}

function handleRetrieval(data) {
  if (data.status === 'running') {
    activateStage('retrieval');
  } else if (data.status === 'done') {
    completeStage('retrieval', `${data.chunks.length} chunks · ${data.ms}ms`);
    currentSources = data.chunks;

    // Render chunk cards
    const detail = stageDetails.retrieval;
    detail.innerHTML = `<p class="text-muted" style="font-size:12px;margin-bottom:8px">
      Searched ${data.total_candidates} candidates, kept top ${data.chunks.length}
    </p>`;
    data.chunks.forEach(chunk => {
      detail.appendChild(buildChunkCard(chunk));
    });
  }
}

function handlePromptBuilt(data) {
  activateStage('prompt_built');
  completeStage('prompt_built', `~${data.estimated_tokens} tokens`);

  const detail = stageDetails.prompt_built;
  detail.innerHTML = `
    <div class="prompt-info-box">
      <div class="prompt-stat">
        <span class="prompt-stat-label">Est. Tokens</span>
        <span class="prompt-stat-value mono">${data.estimated_tokens}</span>
      </div>
      <div class="prompt-stat">
        <span class="prompt-stat-label">Context Chars</span>
        <span class="prompt-stat-value mono">${data.context_chars}</span>
      </div>
    </div>
  `;
}

function handleGenerating(data) {
  if (data.status === 'running') {
    activateStage('generating');
    tokenCount = 0;
    tokenStart = Date.now();
    $tokenDisplay.style.display = 'block';
    $tokenStream.textContent = '';
    $tokenCounter.textContent = '0 tokens';
  } else if (data.status === 'streaming') {
    $tokenStream.textContent += data.token;
    $tokenStream.scrollTop = $tokenStream.scrollHeight;
    tokenCount = data.token_count;

    const elapsedSec = (Date.now() - tokenStart) / 1000;
    const tps = elapsedSec > 0.5 ? ` · ${(tokenCount/elapsedSec).toFixed(1)} tok/s` : '';
    $tokenCounter.textContent = `${tokenCount} tokens${tps}`;

    // Live update chat bubble
    if (currentAssistantMsg) {
      const content = currentAssistantMsg.querySelector('#streaming-content');
      if (content) {
        const cursor = content.querySelector('.streaming-cursor');
        if (cursor) cursor.remove();
        content.textContent = $tokenStream.textContent;
        const newCursor = document.createElement('span');
        newCursor.className = 'streaming-cursor';
        content.appendChild(newCursor);
      }
    }
    scrollToBottom();
  }
}

function handleDone(data) {
  completeStage('generating', `${data.total_tokens} tokens`);
  activateStage('done');
  completeStage('done', 'complete');

  // Finalize chat bubble
  if (currentAssistantMsg) {
    finalizeAssistantMessage(currentAssistantMsg, data.answer, data.sources);
    currentAssistantMsg = null;
  }

  setLocked(false);
}

function handleError(data) {
  showError(`Pipeline error: ${data.detail}`);
  if (currentAssistantMsg) {
    const content = currentAssistantMsg.querySelector('#streaming-content') ||
                    currentAssistantMsg.querySelector('.msg-content');
    if (content) {
      content.innerHTML = `<span class="text-err">Error: ${escHtml(data.detail)}</span>`;
    }
    currentAssistantMsg = null;
  }
  setLocked(false);
}

// ── Send Query ─────────────────────────────────────────────────────────────────
async function sendQuery(query) {
  if (!query.trim() || isStreaming) return;

  hideError();
  setLocked(true);
  currentSources = [];

  appendUserMessage(query);
  currentAssistantMsg = createAssistantMessage();

  resetPipeline();

  try {
    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, history: chatHistory })
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep the last incomplete line

      for (let line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
          const dataStr = line.substring(6).trim();
          if (currentEvent && dataStr) {
            try {
              const data = JSON.parse(dataStr);
              if (currentEvent === 'embedding') handleEmbedding(data);
              else if (currentEvent === 'retrieval') handleRetrieval(data);
              else if (currentEvent === 'prompt_built') handlePromptBuilt(data);
              else if (currentEvent === 'generating') handleGenerating(data);
              else if (currentEvent === 'done') {
                handleDone(data);
                chatHistory.push({ role: 'user', content: query });
                chatHistory.push({ role: 'assistant', content: data.answer });
              }
              else if (currentEvent === 'error') {
                handleError(data);
              }
            } catch (e) {
              if (currentEvent === 'error') handleError({detail: dataStr});
            }
          }
          currentEvent = null;
        }
      }
    }
  } catch (err) {
    if (isStreaming) {
      handleError({ detail: 'Connection lost or error: ' + err.message });
    }
  }
}

// ── Form Submit ────────────────────────────────────────────────────────────────
$inputForm.addEventListener('submit', e => {
  e.preventDefault();
  const q = $queryInput.value.trim();
  if (q) {
    sendQuery(q);
    $queryInput.value = '';
    $queryInput.style.height = 'auto';
  }
});

$queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $inputForm.dispatchEvent(new Event('submit'));
  }
});

$queryInput.addEventListener('input', () => {
  // Auto-resize
  $queryInput.style.height = 'auto';
  $queryInput.style.height = Math.min($queryInput.scrollHeight, 120) + 'px';
  // Enable/disable send button
  $sendBtn.disabled = $queryInput.value.trim().length === 0;
});

// ── Voice Dictation ─────────────────────────────────────────────────────────────
const $micBtn = document.getElementById('mic-btn');
if ($micBtn) {
  let recognition = null;
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    
    let originalValue = "";

    recognition.onstart = () => {
      $micBtn.classList.add('recording');
      originalValue = $queryInput.value;
    };

    recognition.onresult = (e) => {
      let transcript = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        transcript += e.results[i][0].transcript;
      }
      $queryInput.value = (originalValue ? originalValue + ' ' : '') + transcript;
      // Auto-resize input
      $queryInput.style.height = 'auto';
      $queryInput.style.height = ($queryInput.scrollHeight) + 'px';
    };

    recognition.onerror = (e) => {
      console.error('Speech recognition error', e);
      $micBtn.classList.remove('recording');
    };

    recognition.onend = () => {
      $micBtn.classList.remove('recording');
    };

    $micBtn.addEventListener('click', () => {
      if ($micBtn.classList.contains('recording')) {
        recognition.stop();
      } else {
        recognition.start();
      }
    });
  } else {
    $micBtn.style.display = 'none';
  }
}

// ── Inspector Toggle ──────────────────────────────────────────────────
const $app = document.getElementById('app');
$inspectorToggle.addEventListener('click', () => {
  $app.classList.toggle('inspector-open');
  const open = $app.classList.contains('inspector-open');
  $inspectorToggle.querySelector('span').textContent = open ? 'Close Inspector' : 'Inspector';
});

// Click outside drawer to close (mobile only)
document.addEventListener('click', e => {
  if (window.innerWidth <= 900 &&
      $app.classList.contains('inspector-open') &&
      !$inspector.contains(e.target) &&
      !$inspectorToggle.contains(e.target)) {
    $app.classList.remove('inspector-open');
    $inspectorToggle.querySelector('span').textContent = 'Inspector';
  }
});

// ── Error Dismiss ─────────────────────────────────────────────────────────────
$errorDismiss.addEventListener('click', hideError);

// ── Init ──────────────────────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30_000);

// Show idle state on inspector initially
$inspectorIdle.style.display = 'flex';
$pipelineWrap.style.display = 'none';
