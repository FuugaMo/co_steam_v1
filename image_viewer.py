"""
T2I Image Viewer Service
Real-time display of generated images
Port: 5565
"""

import asyncio
import os
import sys
import json
import logging
import traceback
from pathlib import Path
from aiohttp import web
import websockets

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import Message, MessageType, PORTS

# Setup logging FIRST - with immediate flush
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create handlers with immediate flush
file_handler = logging.FileHandler(LOG_DIR / "image_viewer.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger = logging.getLogger("ImageViewer")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Also redirect print to flush immediately
print = lambda *args, **kwargs: (sys.stdout.write(' '.join(map(str, args)) + '\n'), sys.stdout.flush())

# 图片目录
IMAGE_DIR = ROOT / "data" / "generated_images"

# 最新图片信息
latest_image = {
    "path": None,
    "keywords": [],
    "prompt": "",
    "timestamp": 0
}

# WebSocket clients for real-time updates
ws_clients = set()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>T2I Image Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .status {
            background: #16213e;
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .status.connected { border-left: 4px solid #00ff88; }
        .status.disconnected { border-left: 4px solid #ff4444; }
        .status.generating { border-left: 4px solid #ffaa00; animation: pulse 1s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .image-container {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            max-width: 800px;
            width: 100%;
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 512px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .placeholder {
            width: 512px;
            height: 512px;
            max-width: 100%;
            background: #0f0f23;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 18px;
            margin: 0 auto;
        }
        .info {
            margin-top: 15px;
            text-align: left;
            background: #0f0f23;
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
        }
        .info-row {
            margin: 8px 0;
            display: flex;
        }
        .info-label {
            color: #888;
            width: 80px;
            flex-shrink: 0;
        }
        .info-value {
            color: #00d4ff;
            word-break: break-all;
        }
        .keywords {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .keyword {
            background: #00d4ff22;
            color: #00d4ff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .history {
            margin-top: 20px;
            max-width: 800px;
            width: 100%;
        }
        .history h3 {
            color: #888;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
        }
        .history-item {
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .history-item:hover {
            transform: scale(1.05);
        }
        .history-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
    </style>
</head>
<body>
    <h1>T2I Image Viewer</h1>
    <div id="status" class="status disconnected">Connecting...</div>

    <div class="image-container">
        <div id="imageDisplay">
            <div class="placeholder">Waiting for images...</div>
        </div>
        <div id="imageInfo" class="info" style="display:none;">
            <div class="info-row">
                <span class="info-label">Keywords:</span>
                <span id="keywords" class="info-value keywords"></span>
            </div>
            <div class="info-row">
                <span class="info-label">Prompt:</span>
                <span id="prompt" class="info-value"></span>
            </div>
            <div class="info-row">
                <span class="info-label">Time:</span>
                <span id="time" class="info-value"></span>
            </div>
        </div>
    </div>

    <div class="history">
        <h3>History</h3>
        <div id="historyGrid" class="history-grid"></div>
    </div>

    <script>
        let ws;

        function connect() {
            ws = new WebSocket('ws://' + location.host + '/ws');

            ws.onopen = () => {
                document.getElementById('status').className = 'status connected';
                document.getElementById('status').textContent = 'Connected - Waiting for images';
            };

            ws.onclose = () => {
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').textContent = 'Disconnected - Reconnecting...';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                console.log('Received:', data);

                if (data.type === 'generating') {
                    document.getElementById('status').className = 'status generating';
                    document.getElementById('status').textContent = 'Generating: ' + (data.keywords || []).join(', ');
                }
                else if (data.type === 'image') {
                    document.getElementById('status').className = 'status connected';
                    document.getElementById('status').textContent = 'Connected - Last update: ' + new Date().toLocaleTimeString();

                    // Update main image with cache bust
                    const imgUrl = data.url + '?t=' + Date.now();
                    document.getElementById('imageDisplay').innerHTML =
                        '<img src="' + imgUrl + '" alt="Generated Image">';

                    // Update info
                    document.getElementById('imageInfo').style.display = 'block';
                    document.getElementById('keywords').innerHTML =
                        (data.keywords || []).map(k => '<span class="keyword">' + k + '</span>').join('');
                    document.getElementById('prompt').textContent = data.prompt || '-';
                    document.getElementById('time').textContent = new Date().toLocaleString();

                    // Add to history
                    addToHistory(data.url, data.keywords || []);
                }
                else if (data.type === 'history') {
                    // Load initial history
                    (data.images || []).forEach(img => addToHistory(img.url, img.keywords || [], false));
                }
            };
        }

        function addToHistory(url, keywords, prepend = true) {
            const grid = document.getElementById('historyGrid');
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = '<img src="' + url + '" title="' + (keywords || []).join(', ') + '">';
            item.onclick = () => {
                document.getElementById('imageDisplay').innerHTML = '<img src="' + url + '?t=' + Date.now() + '">';
            };

            if (prepend) {
                grid.insertBefore(item, grid.firstChild);
                // Keep only last 20
                while (grid.children.length > 20) {
                    grid.removeChild(grid.lastChild);
                }
            } else {
                grid.appendChild(item);
            }
        }

        connect();
    </script>
</body>
</html>
"""


async def handle_index(request):
    return web.Response(text=HTML_PAGE, content_type='text/html')


async def handle_test(request):
    """Test endpoint - manually broadcast latest image to all browsers"""
    logger.info("=== TEST ENDPOINT CALLED ===")

    # Find the latest image
    if IMAGE_DIR.exists():
        images = sorted(IMAGE_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
        if images:
            latest = images[0]
            broadcast_data = {
                "type": "image",
                "url": f"/images/{latest.name}",
                "keywords": ["test"],
                "prompt": "Manual test broadcast"
            }
            logger.info(f"Test broadcasting: {broadcast_data}")
            await broadcast_to_clients(broadcast_data)
            return web.Response(text=f"Broadcasted: {latest.name} to {len(ws_clients)} clients")

    return web.Response(text="No images found", status=404)


async def handle_image(request):
    """Serve image files"""
    filename = request.match_info['filename']
    filepath = IMAGE_DIR / filename

    logger.info(f"Image request: {filename}")

    if filepath.exists():
        file_size = filepath.stat().st_size
        logger.info(f"  Serving: {filepath} ({file_size} bytes)")
        return web.FileResponse(filepath)

    logger.warning(f"  File NOT found: {filepath}")
    return web.Response(status=404, text=f"Image not found: {filename}")


async def handle_websocket(request):
    """WebSocket handler for real-time updates from browser"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    ws_clients.add(ws)
    sys.stdout.write(f"[BROWSER] ✓ Connected! Total browsers: {len(ws_clients)}\n")
    sys.stdout.flush()

    # Send current history
    history_images = []
    if IMAGE_DIR.exists():
        for img_path in sorted(IMAGE_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)[:20]:
            history_images.append({
                "url": f"/images/{img_path.name}",
                "keywords": []
            })

    await ws.send_json({"type": "history", "images": history_images})
    logger.info(f"Sent {len(history_images)} history images to new browser")

    # Send latest if exists
    if latest_image["path"]:
        latest_path = Path(latest_image["path"])
        if latest_path.exists():
            msg = {
                "type": "image",
                "url": f"/images/{latest_path.name}",
                "keywords": latest_image["keywords"],
                "prompt": latest_image["prompt"]
            }
            await ws.send_json(msg)
            logger.info(f"Sent latest image to new browser: {latest_path.name}")

    try:
        async for msg in ws:
            pass  # Keep connection alive
    finally:
        ws_clients.discard(ws)
        sys.stdout.write(f"[BROWSER] Disconnected. Remaining: {len(ws_clients)}\n")
        sys.stdout.flush()

    return ws


async def broadcast_to_clients(data: dict):
    """Send update to all connected browser clients"""
    sys.stdout.write(f"[BROADCAST] Called with type='{data.get('type')}', clients={len(ws_clients)}\n")
    sys.stdout.flush()

    if not ws_clients:
        sys.stdout.write(f"[BROADCAST] WARNING: No browser clients!\n")
        sys.stdout.flush()
        return

    success_count = 0
    for ws in list(ws_clients):
        try:
            await ws.send_json(data)
            success_count += 1
        except Exception as e:
            sys.stdout.write(f"[BROADCAST] ERROR sending: {e}\n")
            sys.stdout.flush()
            ws_clients.discard(ws)

    sys.stdout.write(f"[BROADCAST] Done: {success_count}/{len(ws_clients)} received\n")
    sys.stdout.flush()


async def t2i_listener():
    """Listen to T2I service for new images via WebSocket"""
    t2i_port = PORTS['t2i']
    t2i_url = f"ws://localhost:{t2i_port}"

    sys.stdout.write(f"[T2I_LISTENER] Starting, will connect to {t2i_url}\n")
    sys.stdout.flush()

    while True:
        try:
            sys.stdout.write(f"[T2I_LISTENER] Connecting to {t2i_url}...\n")
            sys.stdout.flush()

            async with websockets.connect(t2i_url) as ws:
                sys.stdout.write(f"[T2I_LISTENER] ✓ CONNECTED to T2I!\n")
                sys.stdout.flush()

                async for raw in ws:
                    try:
                        sys.stdout.write(f"[T2I_LISTENER] >>> RAW MESSAGE: {raw[:150]}...\n")
                        sys.stdout.flush()

                        msg = Message.from_json(raw)
                        msg_type = msg.type

                        sys.stdout.write(f"[T2I_LISTENER] Parsed type: {msg_type}\n")
                        sys.stdout.flush()

                        # Use string comparison (msg.type is string after from_json)
                        if msg_type == MessageType.T2I_START.value or msg_type == MessageType.T2I_START:
                            keywords = msg.data.get("keywords", [])
                            sys.stdout.write(f"[T2I_LISTENER] === T2I_START === keywords={keywords}\n")
                            sys.stdout.flush()
                            await broadcast_to_clients({
                                "type": "generating",
                                "keywords": keywords
                            })

                        elif msg_type == MessageType.T2I_COMPLETE.value or msg_type == MessageType.T2I_COMPLETE:
                            global latest_image
                            image_path = msg.data.get("image_path", "")
                            keywords = msg.data.get("keywords", [])
                            prompt = msg.data.get("prompt", "")

                            sys.stdout.write(f"[T2I_LISTENER] === T2I_COMPLETE ===\n")
                            sys.stdout.write(f"[T2I_LISTENER]   path: {image_path}\n")
                            sys.stdout.write(f"[T2I_LISTENER]   keywords: {keywords}\n")
                            sys.stdout.flush()

                            # Extract filename
                            if image_path:
                                image_name = Path(image_path).name
                                file_exists = Path(image_path).exists()
                                sys.stdout.write(f"[T2I_LISTENER]   filename: {image_name}, exists: {file_exists}\n")
                                sys.stdout.flush()

                                # Update latest_image
                                latest_image = {
                                    "path": image_path,
                                    "keywords": keywords,
                                    "prompt": prompt,
                                    "timestamp": asyncio.get_event_loop().time()
                                }

                                # Broadcast to browser clients
                                broadcast_data = {
                                    "type": "image",
                                    "url": f"/images/{image_name}",
                                    "keywords": keywords,
                                    "prompt": prompt
                                }
                                sys.stdout.write(f"[T2I_LISTENER] Broadcasting to browsers: {broadcast_data}\n")
                                sys.stdout.flush()
                                await broadcast_to_clients(broadcast_data)
                                sys.stdout.write(f"[T2I_LISTENER] ✓ Broadcast done!\n")
                                sys.stdout.flush()
                            else:
                                sys.stdout.write(f"[T2I_LISTENER] WARNING: no image_path in T2I_COMPLETE!\n")
                                sys.stdout.flush()
                        else:
                            sys.stdout.write(f"[T2I_LISTENER] Ignoring type: {msg_type}\n")
                            sys.stdout.flush()

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        logger.error(f"Raw data: {raw[:500]}")
                    except Exception as e:
                        logger.error(f"Error processing T2I message: {e}")
                        logger.error(traceback.format_exc())

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"T2I connection closed: {e}, reconnecting in 2s...")
            await asyncio.sleep(2)
        except ConnectionRefusedError:
            logger.warning(f"T2I service not available at {t2i_url}, retrying in 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            logger.error(f"T2I connection error: {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(3)


async def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("  T2I Image Viewer Starting...")
    logger.info("=" * 50)

    # Create image directory
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Image directory: {IMAGE_DIR}")

    # Setup web app
    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/ws', handle_websocket)
    app.router.add_get('/images/{filename}', handle_image)
    app.router.add_get('/test', handle_test)  # Manual test endpoint

    # Start web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 5565)
    await site.start()

    logger.info("=" * 50)
    logger.info("  T2I Image Viewer READY")
    logger.info("=" * 50)
    logger.info(f"  Web UI: http://localhost:5565")
    logger.info(f"  Images: {IMAGE_DIR}")
    logger.info(f"  T2I port: {PORTS['t2i']}")
    logger.info("=" * 50)

    # Start T2I listener as a task so web server keeps running
    t2i_task = asyncio.create_task(t2i_listener())

    # Keep the server running
    try:
        await t2i_task
    except asyncio.CancelledError:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
