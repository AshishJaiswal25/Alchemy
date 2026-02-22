/* ─── Alchemy — Frontend Application ───────────────────────────────────────── */

const API_BASE = '';

// ── Tab Switching ─────────────────────────────────────────────────────────────
function switchTab(tabName) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    document.querySelector(`.nav-tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

// ── Theme Toggle ──────────────────────────────────────────────────────────────
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('alchemy-theme', next);
}

// Restore saved theme
const savedTheme = localStorage.getItem('alchemy-theme');
if (savedTheme) document.documentElement.setAttribute('data-theme', savedTheme);

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
    const indicator = document.getElementById('serverStatus');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        indicator.className = 'status-indicator online';
        indicator.querySelector('.status-text').textContent = `Online · ${data.models_loaded.length} models`;
    } catch {
        indicator.className = 'status-indicator offline';
        indicator.querySelector('.status-text').textContent = 'Offline';
    }
}

checkHealth();
setInterval(checkHealth, 15000);

// ── Toast Notifications ───────────────────────────────────────────────────────
let toastContainer = null;

function showToast(message, type = 'info') {
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type]}</span> ${message}`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(24px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ── File Upload / Drag-and-Drop ───────────────────────────────────────────────
function setupDropZone(zoneId, fileInputId, handler) {
    const zone = document.getElementById(zoneId);
    const fileInput = document.getElementById(fileInputId);

    zone.addEventListener('click', () => fileInput.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handler(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handler(fileInput.files);
        }
        fileInput.value = '';
    });
}

// ── Loading / Result Rendering ────────────────────────────────────────────────
function showLoading(containerId, filename) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="loading-indicator">
            <div class="spinner"></div>
            <span class="loading-text">Processing ${filename}...</span>
            <span class="loading-subtext">This may take a moment for large files</span>
            <div class="progress-bar-container">
                <div class="progress-bar indeterminate"></div>
            </div>
        </div>
    `;
}

function renderResult(containerId, data, filename) {
    const container = document.getElementById(containerId);
    const result = data.result || {};
    const hasChunks = result.chunks && result.chunks.length > 0;
    const hasRaw = result.raw != null;

    // Determine what to display based on content available
    let bodyHtml = '';
    let rawForCopy = '';

    if (hasChunks && containerId === 'results-documents' && document.getElementById('opt-format')?.value === 'chunks') {
        // Render chunks view
        rawForCopy = JSON.stringify(result.chunks, null, 2);
        bodyHtml = `
            <div class="chunks-summary" style="padding:12px 16px;font-size:13px;color:var(--text-secondary);border-bottom:1px solid var(--border);">
                📦 ${result.chunks.length} chunks · ~${result.chunks.reduce((a,c) => a + (c.tokens||0), 0)} tokens total
            </div>
            <div class="chunks-list" style="display:flex;flex-direction:column;gap:8px;padding:12px 16px;">
                ${result.chunks.map(c => `
                    <div class="chunk-card" style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:13px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:11px;color:var(--text-tertiary);">
                            <span>Chunk #${c.index}${c.section ? ` · ${escapeHtml(c.section)}` : ''}</span>
                            <span>${c.tokens || '?'} tokens</span>
                        </div>
                        <pre style="white-space:pre-wrap;word-break:break-word;margin:0;font-size:12px;line-height:1.5;color:var(--text-primary);">${escapeHtml(c.text)}</pre>
                    </div>
                `).join('')}
            </div>
        `;
    } else if (hasRaw && containerId === 'results-documents' && document.getElementById('opt-format')?.value === 'json') {
        // Render raw JSON view
        rawForCopy = JSON.stringify(result.raw, null, 2);
        bodyHtml = `<pre style="max-height:600px;overflow:auto;">${escapeHtml(rawForCopy)}</pre>`;
    } else {
        // Default: render markdown
        const content = result.markdown || result.text || JSON.stringify(result, null, 2);
        rawForCopy = content;
        const isMarkdown = !!result.markdown;
        bodyHtml = isMarkdown
            ? `<div class="markdown-content">${simpleMarkdown(content)}</div>`
            : `<pre>${escapeHtml(content)}</pre>`;
    }

    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
        <div class="result-header">
            <div class="result-header-left">
                <span class="result-filename">${filename}</span>
                <span class="result-meta">Job: ${data.job_id?.slice(0, 8)}... · ${data.status}</span>
            </div>
            <div class="result-actions">
                <button class="btn-icon" onclick="copyResult(this)" title="Copy to clipboard">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                </button>
                <button class="btn-icon" onclick="downloadResult(this)" title="Download">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                </button>
            </div>
        </div>
        <div class="result-body">
            ${bodyHtml}
        </div>
    `;
    card.dataset.raw = rawForCopy;
    container.innerHTML = '';
    container.appendChild(card);
}

function renderError(containerId, error) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="result-card" style="border-color: var(--error)">
            <div class="result-header" style="background: var(--error-bg)">
                <span class="result-filename" style="color: var(--error)">⚠ Error</span>
            </div>
            <div class="result-body">
                <pre style="color: var(--error)">${escapeHtml(typeof error === 'string' ? error : JSON.stringify(error, null, 2))}</pre>
            </div>
        </div>
    `;
}

// ── Helpers for API calls ─────────────────────────────────────────────────────
function getErrorMessage(err) {
    if (err instanceof TypeError && err.message === 'Failed to fetch') {
        return 'Cannot connect to server. Make sure Alchemy is running (python server.py --all)';
    }
    return err.detail || err.message || (typeof err === 'string' ? err : JSON.stringify(err));
}

async function throwIfError(res) {
    if (!res.ok) {
        const text = await res.text();
        let parsed;
        try { parsed = JSON.parse(text); } catch { parsed = { detail: text || `Server error ${res.status}` }; }
        throw parsed;
    }
}

// ── Document Upload Handler ───────────────────────────────────────────────────
async function handleDocumentUpload(files) {
    for (const file of files) {
        showLoading('results-documents', file.name);
        showToast(`Uploading ${file.name}...`, 'info');

        const form = new FormData();
        form.append('file', file);
        form.append('extract_tables', document.getElementById('opt-tables').checked);
        form.append('extract_images', document.getElementById('opt-images').checked);
        form.append('output_format', document.getElementById('opt-format').value);

        try {
            const res = await fetch(`${API_BASE}/parse/document`, { method: 'POST', body: form });
            await throwIfError(res);
            const data = await res.json();
            renderResult('results-documents', data, file.name);
            showToast(`${file.name} parsed successfully!`, 'success');
        } catch (err) {
            renderError('results-documents', getErrorMessage(err));
            showToast(`Failed to parse ${file.name}`, 'error');
        }
    }
}

// ── Image Upload Handler ──────────────────────────────────────────────────────
async function handleImageUpload(files) {
    const file = files[0];

    // Show preview
    const preview = document.getElementById('image-preview');
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.innerHTML = `<img src="${e.target.result}" alt="${file.name}">`;
    };
    reader.readAsDataURL(file);

    showLoading('results-images', file.name);
    showToast(`Analyzing ${file.name}...`, 'info');

    const form = new FormData();
    form.append('file', file);
    form.append('task', document.getElementById('opt-image-task').value);
    const prompt = document.getElementById('opt-image-prompt').value;
    if (prompt) form.append('prompt', prompt);

    try {
        const res = await fetch(`${API_BASE}/parse/image`, { method: 'POST', body: form });
        await throwIfError(res);
        const data = await res.json();
        renderResult('results-images', data, file.name);
        showToast('Image analyzed!', 'success');
    } catch (err) {
        renderError('results-images', getErrorMessage(err));
        showToast('Image analysis failed', 'error');
    }
}

// ── Audio Upload Handler ──────────────────────────────────────────────────────
async function handleAudioUpload(files) {
    const file = files[0];
    showLoading('results-audio', file.name);
    showToast(`Transcribing ${file.name}...`, 'info');

    const form = new FormData();
    form.append('file', file);
    const lang = document.getElementById('opt-audio-lang').value;
    if (lang) form.append('language', lang);
    form.append('diarize', document.getElementById('opt-diarize').checked);

    try {
        const res = await fetch(`${API_BASE}/parse/audio`, { method: 'POST', body: form });
        await throwIfError(res);
        const data = await res.json();
        renderResult('results-audio', data, file.name);
        showToast('Transcription complete!', 'success');
    } catch (err) {
        renderError('results-audio', getErrorMessage(err));
        showToast('Transcription failed', 'error');
    }
}

// ── Video Upload Handler ──────────────────────────────────────────────────────
async function handleVideoUpload(files) {
    const file = files[0];
    showLoading('results-video', file.name);
    showToast(`Processing ${file.name}... this may take a while`, 'info');

    const form = new FormData();
    form.append('file', file);
    const lang = document.getElementById('opt-video-lang').value;
    if (lang) form.append('language', lang);
    form.append('diarize', document.getElementById('opt-video-diarize').checked);
    form.append('extract_frames', document.getElementById('opt-video-frames').checked);
    form.append('async_mode', false);

    try {
        const res = await fetch(`${API_BASE}/parse/video`, { method: 'POST', body: form });
        await throwIfError(res);
        const data = await res.json();
        renderResult('results-video', data, file.name);
        showToast('Video processed!', 'success');
    } catch (err) {
        renderError('results-video', getErrorMessage(err));
        showToast('Video processing failed', 'error');
    }
}

// ── Web Crawl Handler ─────────────────────────────────────────────────────────
async function parseWeb() {
    const url = document.getElementById('web-url').value.trim();
    if (!url) {
        showToast('Please enter a URL', 'error');
        return;
    }

    showLoading('results-web', url);
    showToast(`Crawling ${url}...`, 'info');

    const payload = {
        url: url,
        max_depth: parseInt(document.getElementById('opt-web-depth').value),
        async_mode: false,
    };

    const selector = document.getElementById('opt-web-selector').value.trim();
    if (selector) payload.css_selector = selector;

    try {
        const res = await fetch(`${API_BASE}/parse/web`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        await throwIfError(res);
        const data = await res.json();
        renderResult('results-web', data, url);
        showToast('Page crawled successfully!', 'success');
    } catch (err) {
        renderError('results-web', getErrorMessage(err));
        showToast('Crawl failed', 'error');
    }
}

// ── Copy / Download ───────────────────────────────────────────────────────────
function copyResult(btn) {
    const card = btn.closest('.result-card');
    const raw = card.dataset.raw;
    navigator.clipboard.writeText(raw).then(() => {
        showToast('Copied to clipboard!', 'success');
    });
}

function downloadResult(btn) {
    const card = btn.closest('.result-card');
    const raw = card.dataset.raw;
    const filename = card.querySelector('.result-filename').textContent;
    const blob = new Blob([raw], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${filename.replace(/[^a-zA-Z0-9._-]/g, '_')}_result.md`;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ── Simple Markdown → HTML ────────────────────────────────────────────────────
function simpleMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Bold / Italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Inline code
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');
    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    return `<p>${html}</p>`;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Image task toggle prompt field ────────────────────────────────────────────
document.getElementById('opt-image-task').addEventListener('change', function () {
    const promptInput = document.getElementById('opt-image-prompt');
    promptInput.style.display = this.value === 'qa' ? 'block' : 'none';
});

// ── Web URL enter key ─────────────────────────────────────────────────────────
document.getElementById('web-url').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') parseWeb();
});

// ── Initialize Drop Zones ─────────────────────────────────────────────────────
setupDropZone('dropzone-documents', 'file-documents', handleDocumentUpload);
setupDropZone('dropzone-images', 'file-images', handleImageUpload);
setupDropZone('dropzone-audio', 'file-audio', handleAudioUpload);
setupDropZone('dropzone-video', 'file-video', handleVideoUpload);

// ── Keyboard Shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey) {
        const tabs = ['documents', 'images', 'audio', 'video', 'web'];
        const num = parseInt(e.key);
        if (num >= 1 && num <= 5) {
            e.preventDefault();
            switchTab(tabs[num - 1]);
        }
    }
});

console.log('⚗️ Alchemy UI loaded');
