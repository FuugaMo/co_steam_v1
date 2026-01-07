"""
Control Pad - Web界面控制所有服务参数
访问 http://localhost:5560
"""

import argparse
import asyncio
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Message, PORTS

# Import SLM inference to get system prompt
try:
    from slm.inference import get_system_prompt
except ImportError:
    def get_system_prompt():
        return "SLM module not available"

ROOT = Path(__file__).resolve().parent.parent

# 当前参数状态
CONFIG = {
    "asr": {
        "chunk_sec": 3.0,
        "context_sec": 60.0,
        "language": "en",
        "model": "faster-whisper-small",
        "min_chars": 5,
    },
    "slm": {
        "timeout": 10.0,
        "workers": 2,
        "chunk_interval": 1,
        "temperature": 0.3,
        "num_predict": 80,
        "max_turns": 20,
    },
    "t2i": {
        "style": "",
        "staff_suffix": "",
        "staff_negative": "",
        "reference_images": [],
        "version_tag": "0.0.1",
        "enabled": True
    },
    "bridge": {
        "log_level": "info",
    }
}

# 持久化配置文件
STATE_FILE = ROOT / "config" / "control_pad_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_config_from_disk():
    """Load CONFIG from persistent file if present"""
    global CONFIG
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open(encoding="utf-8") as f:
                disk_cfg = json.load(f)
            if isinstance(disk_cfg, dict):
                CONFIG = disk_cfg
                t2i_cfg = CONFIG.get("t2i", {})
                t2i_cfg.setdefault("reference_images", [])
                print(f"Loaded Control Pad config from {STATE_FILE}")
        except Exception as e:
            print(f"Failed to load {STATE_FILE}: {e}")


def save_config_to_disk(cfg: dict):
    """Persist CONFIG to disk"""
    try:
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save {STATE_FILE}: {e}")


load_config_from_disk()

# 服务状态
STATUS = {
    "asr": {"connected": False, "chunks": 0},
    "slm": {"connected": False, "processed": 0, "queued": 0},
    "t2i": {"connected": False, "generated": 0, "queued": 0},
    "bridge": {"connected": False, "messages": 0},
}

# 消息日志
LOG_BUFFER = []
MAX_LOG_SIZE = 100

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Control Pad</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #0f0; margin-bottom: 20px; }
        .toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        .panel { background: #16213e; border-radius: 10px; padding: 20px; }
        .panel h2 { margin-top: 0; color: #00d9ff; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status.on { background: #0f0; }
        .status.off { background: #f00; }
        .param { margin: 10px 0; }
        .param label { display: block; font-size: 12px; color: #888; margin-bottom: 4px; }
        .param input, .param select {
            width: 100%; padding: 8px; background: #0f0f23; border: 1px solid #333;
            color: #fff; border-radius: 4px;
        }
        .param input[type=range] { padding: 0; }
        .param .value { font-size: 12px; color: #0f0; float: right; }
        .param textarea {
            width: 100%; min-height: 60px; padding: 8px; background: #0f0f23;
            border: 1px solid #333; color: #fff; border-radius: 4px; font-family: monospace;
            font-size: 11px; resize: vertical;
        }
        .readonly { background: #1a1a2e !important; color: #666 !important; cursor: not-allowed; }
        button {
            background: #0f0; color: #000; border: none; padding: 10px 20px;
            border-radius: 4px; cursor: pointer; font-weight: bold; margin: 5px;
        }
        button:hover { background: #0c0; }
        button.danger { background: #f00; color: #fff; }
        .log-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 15px;
            height: 600px;
            grid-column: span 4;
        }
        .log-panel {
            background: #0a0a15;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            overflow-y: auto;
        }
        .log-panel-header {
            font-size: 13px;
            font-weight: bold;
            padding-bottom: 8px;
            margin-bottom: 8px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
            background: #0a0a15;
            z-index: 10;
        }
        .log-panel-content {
            font-family: monospace;
            font-size: 11px;
            line-height: 1.4;
        }
        .log-panel-content > div {
            word-wrap: break-word;
            white-space: pre-wrap;
            margin-bottom: 3px;
            padding: 2px 0;
        }
        .log-asr { color: #0ff; }
        .log-slm { color: #888; }
        .log-slm-ism { color: #f90; font-weight: bold; }  /* Keywords → ISM (orange) */
        .log-slm-user { color: #0f0; }  /* Agent → User (green) */
        .log-t2i { color: #f0f; }  /* T2I (magenta) */
        .log-bridge { color: #666; font-size: 11px; }
        .log-error { color: #f00; }
        .preview-box {
            background: #0f0f23; border: 1px solid #333; border-radius: 4px;
            padding: 8px; margin-top: 8px; font-size: 10px; color: #888;
            font-family: monospace; max-height: 120px; overflow-y: auto;
        }
        .stats { font-size: 24px; color: #0f0; }
        .snapshot-bar {
            background: #16213e;
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .snapshot-bar label { color: #888; font-size: 14px; }
        .snapshot-bar input {
            padding: 8px 12px;
            background: #0f0f23;
            border: 1px solid #333;
            color: #0f0;
            border-radius: 4px;
            font-size: 14px;
            width: 150px;
        }
        .snapshot-bar button {
            background: #00d9ff;
            color: #000;
        }
        .snapshot-bar button:hover { background: #00b8d4; }
        .snapshot-bar .snapshot-status {
            color: #888;
            font-size: 12px;
            margin-left: auto;
        }
    </style>
</head>
<body>
    <h1>🎛️ Pipeline Control Pad</h1>

    <div class="snapshot-bar">
        <label>Version:</label>
        <input type="text" id="version-tag" value="0.0.1" placeholder="e.g., 1.0.0" onchange="handleVersionChange()">
        <div class="toolbar">
            <button onclick="applyConfig()">⚡ Apply</button>
            <button onclick="createSnapshot()">📸 Create Snapshot</button>
            <button onclick="resetState()">♻️ Reset</button>
        </div>
        <span class="snapshot-status" id="snapshot-status"></span>
    </div>

    <div class="grid">
        <div class="panel">
            <h2><span class="status" id="asr-status"></span>ASR Service :5551</h2>
            <div class="stats" id="asr-chunks">0</div>
            <div style="color:#888">chunks processed</div>

            <div class="param">
                <label>Chunk Size (sec) <span class="value" id="chunk-sec-val">3.0</span></label>
                <input type="range" id="chunk-sec" min="1" max="10" step="0.5" value="3.0"
                       onchange="updateParam('asr', 'chunk_sec', this.value)">
            </div>
            <div class="param">
                <label>Context Window (sec) <span class="value" id="context-sec-val">60</span></label>
                <input type="range" id="context-sec" min="0" max="300" step="10" value="60"
                       onchange="updateParam('asr', 'context_sec', this.value)">
            </div>
            <div class="param">
                <label>Language</label>
                <select id="language" onchange="updateParam('asr', 'language', this.value)">
                    <option value="en">English</option>
                    <option value="zh">Chinese</option>
                    <option value="ja">Japanese</option>
                </select>
            </div>
            <div class="param">
                <label>ASR Model</label>
                <select id="asr-model" onchange="updateParam('asr', 'model', this.value)">
                    <option value="faster-whisper-tiny">tiny (75M)</option>
                    <option value="faster-whisper-base">base (142M)</option>
                    <option value="faster-whisper-small" selected>small (466M)</option>
                    <option value="faster-whisper-medium">medium (1.5B)</option>
                    <option value="faster-whisper-large-v3">large-v3 (3B)</option>
                </select>
            </div>
        </div>

        <div class="panel">
            <h2><span class="status" id="slm-status"></span>SLM Service :5552</h2>
            <div class="stats" id="slm-processed">0</div>
            <div style="color:#888">intents classified</div>

            <div class="param">
                <label>⚡ Chunk Interval <span class="value" id="chunk-interval-val">1</span></label>
                <input type="range" id="chunk-interval" min="1" max="5" step="1" value="1"
                       onchange="updateParam('slm', 'chunk_interval', parseFloat(this.value))">
                <small style="color:#666; font-size:10px;">Process every Nth chunk (1=all, 2=half)</small>
            </div>
            <div class="param">
                <label>Workers <span class="value" id="workers-val">2</span></label>
                <input type="range" id="workers" min="1" max="5" step="1" value="2"
                       onchange="updateParam('slm', 'workers', parseFloat(this.value))">
            </div>
            <div class="param">
                <label>Timeout (sec) <span class="value" id="timeout-val">10</span></label>
                <input type="range" id="timeout" min="3" max="30" step="1" value="10"
                       onchange="updateParam('slm', 'timeout', parseFloat(this.value))">
            </div>
            <div class="param">
                <label>🌡️ Temperature <span class="value" id="temperature-val">0.3</span></label>
                <input type="range" id="temperature" min="0.0" max="1.0" step="0.1" value="0.3"
                       onchange="updateParam('slm', 'temperature', parseFloat(this.value))">
                <small style="color:#666; font-size:10px;">Creativity (0=deterministic, 1=creative)</small>
            </div>
            <div class="param">
                <label>📏 Max Tokens <span class="value" id="num-predict-val">80</span></label>
                <input type="range" id="num-predict" min="20" max="150" step="10" value="80"
                       onchange="updateParam('slm', 'num_predict', parseFloat(this.value))">
                <small style="color:#666; font-size:10px;">Response length limit</small>
            </div>
            <div class="param">
                <label>🧠 History Turns <span class="value" id="max-turns-val">20</span></label>
                <input type="range" id="max-turns" min="5" max="50" step="5" value="20"
                       onchange="updateParam('slm', 'max_turns', parseFloat(this.value))">
                <small style="color:#666; font-size:10px;">Conversation memory depth</small>
            </div>
            <div class="param">
                <label>📝 System Prompt (Read-Only)</label>
                <textarea id="slm-prompt" class="readonly" readonly style="min-height:150px; max-height:250px;">[Loading...]</textarea>
                <small style="color:#666; font-size:10px;">Current SLM system prompt (saved in snapshots)</small>
            </div>
        </div>

        <div class="panel">
            <h2><span class="status" id="t2i-status"></span>T2I Service :5554</h2>
            <div class="stats" id="t2i-generated">0</div>
            <div style="color:#888">images generated</div>

            <div class="param">
                <label>SLM Concepts (Read-Only)</label>
                <input type="text" id="t2i-concepts" value="[waiting...]" class="readonly" readonly>
                <small style="color:#666; font-size:10px;">Concepts from SLM when image_trigger=true</small>
            </div>

            <div class="param">
                <label>Style (free text)</label>
                <textarea id="style-text" placeholder="e.g., Educational diagram, clear structure, pedagogical visualization"
                          onchange="updateParam('t2i', 'style', this.value)"></textarea>
            </div>

            <div class="param">
                <label>Staff Prompt Suffix</label>
                <textarea id="staff-suffix" placeholder="e.g., with labels, arrows, process steps..."
                          onchange="updateParam('t2i', 'staff_suffix', this.value)"></textarea>
                <small style="color:#666; font-size:10px;">Additional positive prompt (appended to base)</small>
            </div>

            <div class="param">
                <label>Negative Prompt (Override)</label>
                <textarea id="staff-negative" placeholder="Leave empty to omit negative prompt"
                          onchange="updateParam('t2i', 'staff_negative', this.value)"></textarea>
                <small style="color:#666; font-size:10px;">Empty = no negative prompt; non-empty = full override</small>
            </div>

            <div class="param">
                <label>Style Reference Images (t2i/references)</label>
                <div id="reference-list" style="background:#0f0f23; border:1px solid #333; padding:6px; max-height:120px; overflow-y:auto; font-size:11px;"></div>
                <small style="color:#666; font-size:10px;">Select one or more style reference images</small>
            </div>

            <div class="param">
                <label>Last Prompt Preview</label>
                <div class="preview-box" id="prompt-preview">
                    <div style="color:#0f0;">+ <span id="preview-positive">N/A</span></div>
                    <div style="color:#f00; margin-top:4px;">- <span id="preview-negative">N/A</span></div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2><span class="status" id="bridge-status"></span>Bridge :5555</h2>
            <div class="stats" id="bridge-messages">0</div>
            <div style="color:#888">messages forwarded</div>
            <br>
            <button onclick="restartService('asr')">Restart ASR</button>
            <button onclick="restartService('slm')">Restart SLM</button>
            <button onclick="restartService('t2i')">Restart T2I</button>
            <button class="danger" onclick="stopAll()">Stop All</button>
            <br><br>
            <button onclick="clearAllLogs()">Clear All Logs</button>
            <button onclick="location.reload()">Refresh</button>
        </div>

        <!-- 四宫格日志面板 -->
        <div class="log-grid">
            <!-- 左上：ASR -->
            <div class="log-panel">
                <div class="log-panel-header" style="color: #0ff;">🎤 ASR (Speech Recognition)</div>
                <div class="log-panel-content" id="log-asr"></div>
            </div>

            <!-- 右上：SLM → ISM Keywords -->
            <div class="log-panel">
                <div class="log-panel-header" style="color: #f90;">🔑 SLM → ISM (Keywords)</div>
                <div class="log-panel-content" id="log-ism"></div>
            </div>

            <!-- 左下：SLM → User -->
            <div class="log-panel">
                <div class="log-panel-header" style="color: #0f0;">💬 SLM → User (Agent)</div>
                <div class="log-panel-content" id="log-user"></div>
            </div>

            <!-- 右下：T2I (Image Generation) -->
            <div class="log-panel">
                <div class="log-panel-header" style="color: #f0f;">🎨 T2I (Image Generation)</div>
                <div class="log-panel-content" id="log-t2i"></div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let config = %CONFIG%;
        let imagePrompts = {};  // Track each image's prompt data for this session
        let pendingConfig = JSON.parse(JSON.stringify(config)); // 临时态
        let appliedConfig = JSON.parse(JSON.stringify(config)); // 已应用态
        const STORAGE_KEY = 'control_pad_state_v1';

        function addLog(type, text) {
            // 简单日志，避免未定义报错
            console.log(`[${type}] ${text}`);
        }

        function normalizeVersion(raw) {
            let v = (raw || '').trim();
            v = v.replace(/^[vV]/, '');  // drop leading v
            if (!v) return "0.0.1";

            let parts = v.split('.').filter(Boolean);
            if (parts.length === 1) parts = [parts[0], "0", "0"];
            if (parts.length === 2) parts.push("0");
            if (parts.length > 3) parts = parts.slice(0, 3);

            const valid = parts.every(p => /^\d+$/.test(p));
            return valid ? parts.join('.') : "0.0.1";
        }

        function handleVersionChange() {
            const input = document.getElementById('version-tag');
            const version = normalizeVersion(input.value);
            input.value = version;
            pendingConfig.t2i.version_tag = version;
        }

        function renderReferenceList(files) {
            const listEl = document.getElementById('reference-list');
            listEl.innerHTML = '';
            files.forEach(name => {
                const id = `ref-${name.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
                const div = document.createElement('div');
                div.innerHTML = `<label><input type="checkbox" id="${id}" value="${name}" onchange="toggleReference('${name}', this.checked)"> ${name}</label>`;
                listEl.appendChild(div);
            });
            // restore selections
            (pendingConfig.t2i.reference_images || []).forEach(name => {
                const id = `ref-${name.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
                const el = document.getElementById(id);
                if (el) el.checked = true;
            });
        }

        function toggleReference(name, checked) {
            const list = new Set(pendingConfig.t2i.reference_images || []);
            if (checked) list.add(name); else list.delete(name);
            pendingConfig.t2i.reference_images = Array.from(list);
        }

        function loadReferences() {
            fetch('/api/references')
                .then(r => r.json())
                .then(files => {
                    renderReferenceList(files || []);
                })
                .catch(() => {});
        }

        function loadSLMPrompt() {
            fetch('/api/slm_prompt')
                .then(r => r.json())
                .then(data => {
                    const promptEl = document.getElementById('slm-prompt');
                    if (promptEl && data.prompt) {
                        promptEl.value = data.prompt;
                    }
                })
                .catch(e => {
                    const promptEl = document.getElementById('slm-prompt');
                    if (promptEl) {
                        promptEl.value = '[Failed to load SLM prompt]';
                    }
                });
        }

        function loadState() {
            try {
                const raw = localStorage.getItem(STORAGE_KEY);
                if (!raw) return;
                const saved = JSON.parse(raw);
                config = saved.config || config;
                pendingConfig = JSON.parse(JSON.stringify(config));
                appliedConfig = JSON.parse(JSON.stringify(config));
                // 回填 UI
                document.getElementById('version-tag').value = config.t2i.version_tag || '0.0.1';
                document.getElementById('style-text').value = config.t2i.style || '';
                document.getElementById('staff-suffix').value = config.t2i.staff_suffix;
                document.getElementById('staff-negative').value = config.t2i.staff_negative;
                document.getElementById('chunk-sec').value = config.asr.chunk_sec;
                document.getElementById('context-sec').value = config.asr.context_sec;
                document.getElementById('language').value = config.asr.language;
                document.getElementById('asr-model').value = config.asr.model;
                document.getElementById('chunk-interval').value = config.slm.chunk_interval;
                document.getElementById('workers').value = config.slm.workers;
                document.getElementById('timeout').value = config.slm.timeout;
                document.getElementById('temperature').value = config.slm.temperature;
                document.getElementById('num-predict').value = config.slm.num_predict;
                document.getElementById('max-turns').value = config.slm.max_turns;
                // 触发显示更新
                document.querySelectorAll('input[type=range]').forEach(el => {
                    const valSpan = document.getElementById(el.id + '-val');
                    if (valSpan) valSpan.innerText = el.value;
                });
            } catch (e) {
                console.warn('Load state failed', e);
            }
        }

        function saveState() {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify({ config }));
            } catch (e) {
                console.warn('Save state failed', e);
            }
        }

        function resetState() {
            localStorage.removeItem(STORAGE_KEY);
            location.reload();
        }

        function connect() {
            ws = new WebSocket('ws://localhost:5555');
            ws.onopen = () => {
                document.getElementById('bridge-status').className = 'status on';
                addLog('bridge', 'Connected to Bridge');
                // Send current version and pending config on connect
                handleVersionChange();
                applyConfig(true);
            };
            ws.onclose = () => {
                document.getElementById('bridge-status').className = 'status off';
                addLog('error', 'Disconnected from Bridge');
                setTimeout(connect, 3000);
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            const source = msg.source || 'unknown';
            const type = msg.type || '';

            // Update status indicators and dispatch logs
            if (source === 'asr') {
                document.getElementById('asr-status').className = 'status on';
                if (type === 'asr_text') {
                    const count = parseInt(document.getElementById('asr-chunks').innerText) + 1;
                    document.getElementById('asr-chunks').innerText = count;
                    const text = msg.data.text || '';
                    const versionTag = config.t2i.version_tag || '0.0.1';

                    // 发送到ASR面板
                    addLogToPanel('log-asr', `[ver ${versionTag}] [#${msg.data.chunk_id}] ${text}`, 'log-asr');
                }
            } else if (source === 'slm') {
                document.getElementById('slm-status').className = 'status on';
                if (type === 'keywords') {
                    const count = parseInt(document.getElementById('slm-processed').innerText) + 1;
                    document.getElementById('slm-processed').innerText = count;

                    const keywords = msg.data.keywords || [];
                    const agentResp = msg.data.agent_response || '';
                    const latency = msg.data.latency_ms || 0;
                    const history = msg.data.history_length || 0;
                    const versionTag = config.t2i.version_tag || '0.0.1';
                    const input = msg.data.original_text || '';

                    // 发送Keywords到ISM面板
                    if (keywords && keywords.length > 0) {
                        addLogToPanel('log-ism',
                            `[ver ${versionTag}] ${JSON.stringify(keywords)} (${latency}ms, ${history} turns)<br><span style="color:#666; font-size:10px;">Input: ${input}</span>`,
                            'log-slm-ism');
                    }

                    // 发送Agent response到User面板
                    if (agentResp) {
                        addLogToPanel('log-user',
                            `[ver ${versionTag}] ${agentResp}<br><span style="color:#666; font-size:10px;">Input: ${input}</span>`,
                            'log-slm-user');
                    }

                    // T2I trigger detection - update concepts field
                    const imageTrigger = msg.data.image_trigger || false;
                    const imageKeywords = msg.data.image_keywords || [];
                    const topicScore = msg.data.topic_change_score || 0.0;

                    if (imageTrigger && imageKeywords.length > 0) {
                        document.getElementById('t2i-concepts').value = imageKeywords.join(', ');
                        addLogToPanel('log-t2i',
                            `🎨 TRIGGER: ${JSON.stringify(imageKeywords)} (topic_score=${topicScore.toFixed(2)})<br><span style="color:#666; font-size:10px;">Input: ${input}</span>`,
                            'log-t2i');
                    }
                }
            } else if (source === 't2i') {
                document.getElementById('t2i-status').className = 'status on';

                if (type === 't2i_start') {
                    const requestId = msg.data.request_id || '';
                    const keywords = msg.data.keywords || [];
                    const versionTag = msg.data.version_tag || config.t2i.version_tag || '0.0.1';
                    addLogToPanel('log-t2i',
                        `[ver ${versionTag}] ⏳ START: ${requestId}<br><span style="color:#666; font-size:10px;">Keywords: ${JSON.stringify(keywords)}</span>`,
                        'log-t2i');
                } else if (type === 't2i_complete') {
                    const count = parseInt(document.getElementById('t2i-generated').innerText) + 1;
                    document.getElementById('t2i-generated').innerText = count;

                    const imagePath = msg.data.image_path || '';
                    const versionTag = msg.data.version_tag || (msg.data.structure || {}).version_tag || config.t2i.version_tag || '0.0.1';
                    const prompt = msg.data.prompt || '';
                    const negativePrompt = msg.data.negative_prompt || '';
                    const structure = msg.data.structure || {};
                    const requestId = msg.data.request_id || '';
                    const keywords = msg.data.keywords || [];
                    const refs = structure.reference_images || [];

                    // Extract filename from path and save full prompt data
                    // 拆出文件名，正则匹配正/反斜杠
                    const filename = imagePath.split(/[\\\\\\\\/]/).pop();
                    if (filename) {
                        imagePrompts[filename] = {
                            prompt: prompt,
                            negative_prompt: negativePrompt,
                            keywords: keywords,
                            structure: structure,
                            request_id: requestId,
                            timestamp: new Date().toISOString(),
                            version_tag: versionTag
                        };
                    }

                    // Update prompt preview
                    document.getElementById('preview-positive').innerText = prompt.substring(0, 200) + (prompt.length > 200 ? '...' : '');
                    document.getElementById('preview-negative').innerText = negativePrompt.substring(0, 100) + (negativePrompt.length > 100 ? '...' : '');

                    addLogToPanel('log-t2i',
                        `[ver ${versionTag}] ✅ DONE: ${requestId}<br><span style="color:#0f0; font-size:10px;">Image: ${imagePath}</span><br><span style="color:#666; font-size:10px;">${versionTag ? `Ver: ${versionTag}` : ''}${refs.length ? `<br>Ref: ${refs.join(', ')}` : ''}</span>`,
                        'log-t2i');
                } else if (type === 't2i_error') {
                    const requestId = msg.data.request_id || '';
                    const error = msg.data.error || 'Unknown error';
                    const versionTag = msg.data.version_tag || config.t2i.version_tag || '0.0.1';
                    addLogToPanel('log-t2i',
                        `[ver ${versionTag}] ❌ ERROR: ${requestId}<br><span style="color:#f00; font-size:10px;">${error}</span>`,
                        'log-error');
                }
            }

            // Update bridge message count
            const bridgeCount = parseInt(document.getElementById('bridge-messages').innerText) + 1;
            document.getElementById('bridge-messages').innerText = bridgeCount;
        }

        function addLogToPanel(panelId, text, cssClass) {
            const panel = document.getElementById(panelId);
            if (!panel) return;

            const time = new Date().toLocaleTimeString();
            const div = document.createElement('div');
            if (cssClass) div.className = cssClass;
            div.innerHTML = `[${time}] ${text}`;
            panel.appendChild(div);
            panel.scrollTop = panel.scrollHeight;
        }

        function clearAllLogs() {
            document.getElementById('log-asr').innerHTML = '';
            document.getElementById('log-ism').innerHTML = '';
            document.getElementById('log-user').innerHTML = '';
            document.getElementById('log-t2i').innerHTML = '';
        }

        function updateParam(service, param, value) {
            // Update display
            const valSpan = document.getElementById(param.replace('_', '-') + '-val');
            if (valSpan) valSpan.innerText = value;

            // Update pending config only (deferred apply)
            pendingConfig[service][param] = value;
            if (service === 't2i' && param === 'version_tag') {
                document.getElementById('version-tag').value = value;
            }
        }

        function applyConfig(forceAll = false) {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                addLog('error', 'WebSocket not connected, cannot apply config');
                return;
            }
            // Compute diff and send
            ['asr', 'slm', 't2i'].forEach(svc => {
                const curr = pendingConfig[svc];
                const prev = appliedConfig[svc] || {};
                Object.keys(curr).forEach(key => {
                    const shouldSend = forceAll || (curr[key] !== prev[key]);
                    if (shouldSend) {
                        ws.send(JSON.stringify({
                            type: 'config_update',
                            source: 'control_pad',
                            data: { service: svc, param: key, value: curr[key] }
                        }));
                        appliedConfig[svc][key] = curr[key];
                        config[svc][key] = curr[key];
                        addLog('bridge', `Config applied: ${svc}.${key} = ${JSON.stringify(curr[key])}`);
                    }
                });
            });
            // Persist to server and local
            fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ config: config })
            }).catch(() => {});
            saveState();
        }

        function restartService(service) {
            addLog('bridge', `Restarting ${service}...`);
            fetch(`/api/restart/${service}`)
                .then(r => r.json())
                .then(data => {
                    addLog('bridge', `${service}: ${data.message || data.status}`);
                    // Reset status indicator
                    document.getElementById(`${service}-status`).className = 'status off';
                })
                .catch(e => addLog('error', `Restart failed: ${e}`));
        }

        function stopAll() {
            if (!confirm('Stop all services?')) return;
            addLog('bridge', 'Stopping all services...');
            fetch('/api/stop-all')
                .then(r => r.json())
                .then(data => {
                    addLog('bridge', data.message || 'Services stopped');
                    document.getElementById('asr-status').className = 'status off';
                    document.getElementById('slm-status').className = 'status off';
                    document.getElementById('bridge-status').className = 'status off';
                })
                .catch(e => addLog('error', `Stop failed: ${e}`));
        }

        function createSnapshot() {
            const versionInput = document.getElementById('version-tag');
            const version = normalizeVersion(versionInput.value);
            versionInput.value = version;  // show normalized 3-part version
            const statusEl = document.getElementById('snapshot-status');
            statusEl.style.color = '#ff0';
            statusEl.innerText = 'Creating snapshot...';

            // Collect current log content
            const logs = {
                asr: document.getElementById('log-asr').innerText,
                ism: document.getElementById('log-ism').innerText,
                user: document.getElementById('log-user').innerText,
                t2i: document.getElementById('log-t2i').innerText
            };

            fetch('/api/snapshot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    version: normalizeVersion(version),
                    config: config,
                    logs: logs,
                    imagePrompts: imagePrompts,  // Per-image prompt mapping for this session
                    stats: {
                        asr_chunks: document.getElementById('asr-chunks').innerText,
                        slm_processed: document.getElementById('slm-processed').innerText,
                        t2i_generated: document.getElementById('t2i-generated').innerText,
                        bridge_messages: document.getElementById('bridge-messages').innerText
                    }
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'ok') {
                    statusEl.style.color = '#0f0';
                    statusEl.innerText = '✓ Snapshot saved: ' + data.path;
                } else {
                    statusEl.style.color = '#f00';
                    statusEl.innerText = '✗ Error: ' + data.message;
                }
            })
            .catch(e => {
                statusEl.style.color = '#f00';
                statusEl.innerText = '✗ Failed: ' + e;
            });
        }

        // Initialize
        loadState();
        loadReferences();
        loadSLMPrompt();
        connect();

        // Update slider displays
        document.querySelectorAll('input[type=range]').forEach(el => {
            el.dispatchEvent(new Event('change'));
        });
    </script>
</body>
</html>
""".replace('%CONFIG%', json.dumps(CONFIG))


class ControlPadHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/api/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(CONFIG).encode())
        elif self.path == '/api/references':
            refs_dir = ROOT / "t2i" / "references"
            files = []
            if refs_dir.exists():
                files = [p.name for p in refs_dir.iterdir() if p.is_file()]
                files.sort()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(files).encode())
        elif self.path == '/api/slm_prompt':
            prompt = get_system_prompt()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"prompt": prompt}, ensure_ascii=False).encode('utf-8'))
        elif self.path == '/api/images':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(load_image_metadata(), ensure_ascii=False).encode())
        elif self.path.startswith('/api/restart/'):
            service = self.path.split('/')[-1]
            result = restart_service(service)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path == '/api/stop-all':
            result = stop_all_services()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/config':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            # 支持整份配置或单个字段更新
            if 'config' in data and isinstance(data['config'], dict):
                global CONFIG
                CONFIG = data['config']
                save_config_to_disk(CONFIG)
            else:
                service = data.get('service')
                param = data.get('param')
                value = data.get('value')

                if service in CONFIG and param in CONFIG[service]:
                    CONFIG[service][param] = value
                    save_config_to_disk(CONFIG)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

        elif self.path == '/api/snapshot':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            result = create_snapshot(data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        elif self.path.startswith('/api/restart/'):
            service = self.path.split('/')[-1]
            result = restart_service(service)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress default logging


import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Snapshot directories
SNAPSHOT_DIR = ROOT / "snapshots"
IMAGES_DIR = ROOT / "data" / "generated_images"


def load_image_metadata(limit: int | None = None) -> dict:
    """Load image metadata from sidecar JSONs; fallback to bare PNG files."""
    records: dict[str, dict] = {}

    # Sidecar JSONs
    for meta_file in IMAGES_DIR.glob("*.json"):
        try:
            with meta_file.open(encoding="utf-8") as f:
                data = json.load(f)
            filename = data.get("filename") or f"{meta_file.stem}.png"
            data.setdefault("image_path", str(IMAGES_DIR / filename))
            data.setdefault("created_at", datetime.fromtimestamp(meta_file.stat().st_mtime).isoformat())
            records[filename] = data
        except Exception:
            continue

    # Fallback: png without metadata
    for png in IMAGES_DIR.glob("*.png"):
        fname = png.name
        if fname not in records:
            records[fname] = {
                "filename": fname,
                "image_path": str(png),
                "prompt": "",
                "negative_prompt": "",
                "keywords": [],
                "structure": {},
                "request_id": "",
                "created_at": datetime.fromtimestamp(png.stat().st_mtime).isoformat()
            }

    # Sort by file mtime desc for deterministic trimming
    items = list(records.items())
    items.sort(key=lambda kv: Path(kv[1].get("image_path", "")).stat().st_mtime if Path(kv[1].get("image_path", "")).exists() else 0, reverse=True)
    if limit:
        items = items[:limit]
    return {k: v for k, v in items}


def create_snapshot(data: dict) -> dict:
    """Create a version snapshot with config, logs, and images"""
    try:
        version = data.get('version', '0.0.1')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_name = f"{version}_{timestamp}"
        snapshot_path = SNAPSHOT_DIR / snapshot_name

        # Create snapshot directories
        snapshot_path.mkdir(parents=True, exist_ok=True)
        (snapshot_path / "logs").mkdir(exist_ok=True)
        (snapshot_path / "images").mkdir(exist_ok=True)

        # 1. Save config.json
        config_data = data.get('config', CONFIG)
        with open(snapshot_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        # 2. 合并前端上报与磁盘元数据
        frontend_prompts = data.get('imagePrompts', {}) or {}
        disk_prompts = load_image_metadata()
        image_prompts = disk_prompts.copy()
        for fname, meta in frontend_prompts.items():
            if fname not in image_prompts:
                image_prompts[fname] = {}
            image_prompts[fname].update(meta or {})
            image_prompts[fname].setdefault("filename", fname)
        # 仅保留当前版本
        image_prompts = {
            k: v for k, v in image_prompts.items()
            if v.get("version_tag") == version
        }

        # 3. Save image_prompts.json (each image's full prompt data)
        with open(snapshot_path / "image_prompts.json", 'w', encoding='utf-8') as f:
            json.dump(image_prompts, f, indent=2, ensure_ascii=False)

        # 4. Save logs from browser
        logs = data.get('logs', {})
        for log_name, log_content in logs.items():
            if log_content:
                with open(snapshot_path / "logs" / f"{log_name}.log", 'w', encoding='utf-8') as f:
                    f.write(log_content)

        # 5. Copy ONLY images referenced in image_prompts (fallback: sidecar merge covers all)
        image_count = 0
        for filename in image_prompts.keys():
            img_path_str = image_prompts[filename].get("image_path", "")
            src_path = Path(img_path_str) if img_path_str else (IMAGES_DIR / filename)
            if not src_path.exists():
                src_path = IMAGES_DIR / filename
            if src_path.exists():
                shutil.copy2(src_path, snapshot_path / "images" / filename)
                image_count += 1
                meta_src = src_path.with_suffix(".json")
                if meta_src.exists():
                    shutil.copy2(meta_src, snapshot_path / "images" / f"{Path(filename).stem}.json")

        # 6. Save SLM system prompt
        slm_prompt = get_system_prompt()
        with open(snapshot_path / "slm_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(slm_prompt)

        # 7. Create manifest.json
        manifest = {
            "version": version,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "stats": data.get('stats', {}),
            "image_count": image_count,
            "config_snapshot": True,
            "logs_snapshot": True,
            "slm_prompt_snapshot": True
        }
        with open(snapshot_path / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return {
            "status": "ok",
            "path": snapshot_name,
            "image_count": image_count
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


CONDA = r"D:\Miniconda3\condabin\conda.bat"

SERVICE_COMMANDS = {
    "asr": f'start "ASR-5551" cmd /c "title ASR :5551 && {CONDA} run -n asr python {BASE_DIR}\\asr\\service.py"',
    "slm": f'start "SLM-5552" cmd /c "title SLM :5552 && {CONDA} run -n asr python {BASE_DIR}\\slm\\service.py --workers 2"',
    "t2i": f'start "T2I-5554" cmd /c "title T2I :5554 && {CONDA} run -n asr python {BASE_DIR}\\t2i\\service.py"',
    "bridge": f'start "Bridge-5555" cmd /c "title Bridge :5555 && {CONDA} run -n asr python {BASE_DIR}\\bridge\\service.py"',
}

def restart_service(service: str) -> dict:
    """Restart a specific service"""
    try:
        # Kill existing
        if service == "asr":
            subprocess.run('taskkill /fi "WINDOWTITLE eq ASR*" /f', shell=True, capture_output=True)
        elif service == "slm":
            subprocess.run('taskkill /fi "WINDOWTITLE eq SLM*" /f', shell=True, capture_output=True)
        elif service == "t2i":
            subprocess.run('taskkill /fi "WINDOWTITLE eq T2I*" /f', shell=True, capture_output=True)
        elif service == "bridge":
            subprocess.run('taskkill /fi "WINDOWTITLE eq Bridge*" /f', shell=True, capture_output=True)

        # Wait a moment
        import time
        time.sleep(1)

        # Start new
        if service in SERVICE_COMMANDS:
            subprocess.Popen(SERVICE_COMMANDS[service], shell=True)
            return {"status": "ok", "message": f"{service} restarted"}
        else:
            return {"status": "error", "message": f"Unknown service: {service}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def stop_all_services() -> dict:
    """Stop all services"""
    try:
        subprocess.run('taskkill /fi "WINDOWTITLE eq ASR*" /f', shell=True, capture_output=True)
        subprocess.run('taskkill /fi "WINDOWTITLE eq SLM*" /f', shell=True, capture_output=True)
        subprocess.run('taskkill /fi "WINDOWTITLE eq T2I*" /f', shell=True, capture_output=True)
        subprocess.run('taskkill /fi "WINDOWTITLE eq Bridge*" /f', shell=True, capture_output=True)
        return {"status": "ok", "message": "All services stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_http_server(port):
    server = HTTPServer(('0.0.0.0', port), ControlPadHandler)
    print(f"Control Pad: http://localhost:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description='Control Pad Web Interface')
    parser.add_argument('--port', type=int, default=5560, help='HTTP port')
    args = parser.parse_args()

    print("=" * 50)
    print("Pipeline Control Pad")
    print("=" * 50)
    print(f"Open in browser: http://localhost:{args.port}")
    print("=" * 50)

    run_http_server(args.port)


if __name__ == "__main__":
    main()
