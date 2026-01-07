"""
ComfyUI API Client
Handles workflow submission, progress monitoring, image retrieval
"""

import requests
import json
import time
import websocket
import uuid
import io
from PIL import Image
from typing import Optional, Dict, List


class ComfyUIClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        self.ws: Optional[websocket.WebSocket] = None

    def connect_ws(self):
        """Connect to ComfyUI WebSocket for progress updates"""
        ws_url = self.server_url.replace("http", "ws") + f"/ws?clientId={self.client_id}"
        self.ws = websocket.create_connection(ws_url)

    def queue_prompt(self, workflow: dict) -> str:
        """Submit workflow to ComfyUI queue"""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        response = requests.post(f"{self.server_url}/prompt", json=payload)
        if response.status_code != 200:
            error_detail = response.text
            print(f"| ComfyUI Error Response: {error_detail}")
            response.raise_for_status()
        return response.json()["prompt_id"]

    def get_progress(self) -> Optional[dict]:
        """Get generation progress from WebSocket"""
        if not self.ws:
            return None
        try:
            self.ws.settimeout(1.0)
            message = self.ws.recv()
            data = json.loads(message)
            print(f"ComfyUI WS: {data.get('type', 'unknown')}")
            return data
        except websocket.WebSocketTimeoutException:
            return None
        except Exception as e:
            print(f"ComfyUI WS error: {e}")
            return None

    def wait_for_completion(self, prompt_id: str, timeout: float = 120.0) -> bool:
        """Wait for prompt execution to complete"""
        import time
        start = time.time()

        while time.time() - start < timeout:
            progress = self.get_progress()
            if progress is None:
                time.sleep(0.2)
                continue

            msg_type = progress.get("type", "")

            # Check for execution complete
            if msg_type == "executing":
                data = progress.get("data", {})
                # When node is None and prompt_id matches, execution is done
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    print(f"ComfyUI: Execution complete for {prompt_id}")
                    return True

            elif msg_type == "executed":
                data = progress.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    print(f"ComfyUI: Node executed for {prompt_id}")
                    # Continue waiting for final completion

        print(f"ComfyUI: Timeout waiting for {prompt_id}")
        return False

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        """Download generated image"""
        url = f"{self.server_url}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    def get_history(self, prompt_id: str) -> dict:
        """Get workflow execution history"""
        response = requests.get(f"{self.server_url}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    def get_system_stats(self) -> dict:
        """Get ComfyUI system statistics"""
        response = requests.get(f"{self.server_url}/system_stats")
        response.raise_for_status()
        return response.json()
