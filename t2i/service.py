"""
T2I WebSocket Service
Text-to-Image generation using ComfyUI
Receives image triggers from SLM, generates educational diagrams
"""

import argparse
import asyncio
import os
import sys
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import WSServer, WSClient, Message, MessageType, Source, PORTS
from t2i.comfyui_client import ComfyUIClient
from t2i.prompt_builder import build_prompt


class T2IService:
    def __init__(self, args):
        self.args = args
        self.server = WSServer("t2i", args.port)
        self.slm_client = WSClient("t2i", "slm", args.slm_host)  # 订阅SLM
        self.comfyui = ComfyUIClient(args.comfyui_url)
        self.executor = ThreadPoolExecutor(max_workers=1)  # 单GPU worker
        self.queue = asyncio.Queue()
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.version_tag = args.version_tag
        # Prompt controls (from args / Control Pad)
        self.style = getattr(args, "style", "")
        self.staff_suffix = getattr(args, "staff_suffix", "")
        self.staff_negative = getattr(args, "staff_negative", "")
        self.reference_images: list[str] = []
        # 去抖与排队上限（单线程防堆积）
        self.max_queue = 1
        self.debounce_sec = 2.0
        self._last_enqueue_ts = 0.0
        # ComfyUI paths (for style reference images / controlnet models)
        self.comfy_input_dir = ROOT / "ComfyUI_cu126" / "ComfyUI_windows_portable" / "ComfyUI" / "input"
        self.comfy_controlnet_dir = ROOT / "ComfyUI_cu126" / "ComfyUI_windows_portable" / "ComfyUI" / "models" / "controlnet"
        self.comfy_t2iadapter_dir = ROOT / "ComfyUI_cu126" / "ComfyUI_windows_portable" / "ComfyUI" / "models" / "t2iadapter"

    def _style_model_ok(self) -> tuple[bool, str]:
        """Check style adapter model exists and has valid size"""
        candidate = self.comfy_t2iadapter_dir / "t2iadapter_style_sd14v1.pth"
        if not candidate.exists():
            return False, f"Style model missing: {candidate}"
        # Just check file size (>100MB is reasonable for T2I adapter)
        size_mb = candidate.stat().st_size / (1024 * 1024)
        if size_mb < 100:
            return False, f"Style model too small: {size_mb:.1f}MB"
        return True, candidate.name

    async def handle_slm_message(self, msg: Message):
        """Handle KEYWORDS message from SLM"""
        if msg.type != MessageType.KEYWORDS:
            return

        # 检查是否需要生成图像
        image_trigger = msg.data.get("image_trigger", False)
        if not image_trigger:
            return

        image_keywords = msg.data.get("image_keywords", [])
        topic_score = msg.data.get("topic_change_score", 0.0)
        original_text = msg.data.get("original_text", "")

        if not image_keywords:
            print(f"T2I: Skipped - no image keywords")
            return

        now = time.time()
        if (now - self._last_enqueue_ts) < self.debounce_sec:
            print(f"T2I: Skipped - debounced ({now - self._last_enqueue_ts:.2f}s since last enqueue)")
            return
        if self.queue.qsize() >= self.max_queue:
            print(f"T2I: Skipped - queue backlog {self.queue.qsize()} >= {self.max_queue}")
            return

        # 生成请求ID
        request_id = f"t2i_{int(time.time() * 1000)}"

        # 入队处理
        await self.queue.put({
            "request_id": request_id,
            "image_keywords": image_keywords,
            "original_text": original_text,
            "topic_score": topic_score
        })
        self._last_enqueue_ts = now

        print(f"T2I: Queued {request_id} - keywords={image_keywords}, score={topic_score:.2f}")

    async def generation_worker(self):
        """Worker that processes generation queue"""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            request_id = request["request_id"]
            image_keywords = request["image_keywords"]

            print(f"=== T2I [{request_id}] ===")
            print(f"| Keywords: {image_keywords}")
            print(f"| Version: {self.version_tag}")

            # 发送开始消息
            start_msg = Message(
                type=MessageType.T2I_START,
                source=Source.T2I,
                data={
                    "request_id": request_id,
                    "keywords": image_keywords,
                    "version_tag": self.version_tag
                }
            )
            print(f"| Broadcasting T2I_START to {len(self.server.clients)} clients...")
            await self.server.broadcast(start_msg)
            print(f"| T2I_START sent!")

            try:
                # 在线程池中生成（阻塞操作）
                result = await loop.run_in_executor(
                    self.executor,
                    self.generate_image,
                    request_id,
                    image_keywords,
                    request
                )

                # 发送完成消息
                complete_msg = Message.t2i_complete(
                    image_path=result["image_path"],
                    prompt=result["full_prompt"],
                    negative_prompt=result["negative_prompt"],
                    structure=result["structure"]
                )
                complete_msg.data["request_id"] = request_id
                complete_msg.data["keywords"] = image_keywords
                complete_msg.data["filename"] = result.get("filename", "")
                complete_msg.data["version_tag"] = self.version_tag

                print(f"| Broadcasting T2I_COMPLETE to {len(self.server.clients)} clients...")
                await self.server.broadcast(complete_msg)
                print(f"| [OK] T2I_COMPLETE sent! path={result['image_path']}")
                print(f"=== {result['elapsed']:.1f}s ===")

            except Exception as e:
                # 发送错误消息
                error_msg = Message(
                    type=MessageType.T2I_ERROR,
                    source=Source.T2I,
                    data={
                        "request_id": request_id,
                        "error": str(e),
                        "version_tag": self.version_tag
                    }
                )
                await self.server.broadcast(error_msg)
                print(f"| [ERR] Error: {e}")
                print(f"===================")

    def generate_image(self, request_id: str, image_keywords: list, request: dict) -> dict:
        """Generate image using ComfyUI (blocking)"""
        start_time = time.time()

        # 使用显式可控的提示词构建（无隐式推断）
        prompt_data = build_prompt(
            concept_keywords=image_keywords,        # SLM提供的概念（只读）
            style=self.style,                       # Control Pad控制
            staff_suffix=self.staff_suffix,         # Control Pad控制
            staff_negative=self.staff_negative,     # Control Pad控制（可完全覆盖）
            reference_images=self.reference_images
        )

        full_prompt = prompt_data["positive"]
        negative_prompt = prompt_data["negative"]
        structure = prompt_data["structure"]
        structure["version_tag"] = self.version_tag
        structure["reference_images"] = self.reference_images
        structure["style_reference_images"] = self.reference_images

        # 日志输出（显示结构化信息）
        print(f"| Style: {structure['style']} | Version: {self.version_tag}")
        print(f"| Concepts: {structure['concept_keywords']}")
        print(f"| Positive: {full_prompt[:80]}...")
        if structure['staff_suffix']:
            print(f"| Staff Suffix: {structure['staff_suffix'][:50]}...")
        if self.reference_images:
            print(f"| Style Reference: {self.reference_images}")

        # 加载workflow模板（有风格参考图则使用 style adapter 工作流）
        workflow_name = self.args.workflow
        selected_style_model = None
        if self.reference_images and workflow_name == "sd15_fast":
            ok, info = self._style_model_ok()
            if ok:
                workflow_name = "sd15_style"
                selected_style_model = info
                print(f"| Using style workflow: {workflow_name} ({selected_style_model})")
            else:
                print(f"| Style adapter unavailable: {info}; fallback to {workflow_name}")
        workflow = self.load_workflow(workflow_name)

        # 注入提示词/参考图
        workflow = self.inject_prompts(workflow, full_prompt, negative_prompt)
        if self.reference_images:
            workflow = self.inject_reference_image(workflow)
            # Debug: print LoadImage node after injection
            for node_id, node in workflow.items():
                if node.get("class_type") == "LoadImage":
                    print(f"| DEBUG LoadImage[{node_id}]: image={node['inputs'].get('image')}")

        # 连接ComfyUI WebSocket
        self.comfyui.connect_ws()

        # 提交workflow
        prompt_id = self.comfyui.queue_prompt(workflow)
        print(f"| Submitted to ComfyUI: {prompt_id}")

        # 等待完成
        self.comfyui.wait_for_completion(prompt_id, timeout=120.0)

        # 获取生成的图像
        history = self.comfyui.get_history(prompt_id)
        output_images = self.extract_images_from_history(history, prompt_id)

        # 保存图像
        image_path = self.output_dir / f"{request_id}.png"
        filename = image_path.name
        if output_images:
            image = self.comfyui.get_image(
                output_images[0]["filename"],
                output_images[0].get("subfolder", ""),
                output_images[0].get("type", "output")
            )
            image.save(image_path)
            # 保存同名元数据，便于快照关联
            metadata = {
                "filename": filename,
                "image_path": str(image_path),
                "prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "keywords": image_keywords,
                "structure": structure,
                "request_id": request_id,
                "prompt_id": prompt_id,
                "original_text": request.get("original_text", ""),
                "topic_change_score": request.get("topic_score", 0.0),
                "workflow": self.args.workflow,
                "style": self.style,
                "staff_suffix": self.staff_suffix,
                "staff_negative": self.staff_negative or "",
                "reference_images": self.reference_images,
                "style_reference_images": self.reference_images,
                "style_model": selected_style_model or "",
                "version_tag": self.version_tag,
                "created_at": datetime.now().isoformat()
            }
            with open(image_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time

        return {
            "image_path": str(image_path),
            "filename": filename,
            "full_prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "structure": structure,  # 包含结构化信息
            "prompt_id": prompt_id,
            "reference_images": self.reference_images,
            "elapsed": elapsed
        }

    def inject_prompts(self, workflow: dict, positive: str, negative: str) -> dict:
        """Replace prompt placeholders in workflow"""
        for node_id, node in workflow.items():
            if node["class_type"] == "CLIPTextEncode":
                text = node["inputs"].get("text", "")
                if "POSITIVE_PROMPT_PLACEHOLDER" in text:
                    node["inputs"]["text"] = positive
                elif "NEGATIVE_PROMPT_PLACEHOLDER" in text:
                    node["inputs"]["text"] = negative
        return workflow

    def inject_reference_image(self, workflow: dict) -> dict:
        """Inject style reference image into LoadImage node (first image only)"""
        if not self.reference_images:
            return workflow
        ref_rel = self.reference_images[0]
        ref_path = (ROOT / "t2i" / "references" / ref_rel).resolve()
        if not ref_path.exists():
            print(f"| Reference not found: {ref_path}")
            return workflow
        # Ensure ComfyUI input has the file
        if self.comfy_input_dir.exists():
            dest = self.comfy_input_dir / ref_path.name
            try:
                if dest.resolve() != ref_path.resolve():
                    dest.write_bytes(ref_path.read_bytes())
            except Exception as e:
                print(f"| Reference copy failed: {e}")
        ref_name = ref_path.name

        for node in workflow.values():
            if node.get("class_type") == "LoadImage":
                if node["inputs"].get("image") in ["REFERENCE_IMAGE_PLACEHOLDER", "STYLE_IMAGE_PLACEHOLDER"]:
                    node["inputs"]["image"] = ref_name
        return workflow

    def load_workflow(self, name: str = None) -> dict:
        """Load ComfyUI workflow JSON"""
        wf = name or self.args.workflow
        workflow_path = Path(__file__).parent / "workflows" / f"{wf}.json"
        with open(workflow_path) as f:
            return json.load(f)

    def extract_images_from_history(self, history: dict, prompt_id: str) -> list:
        """Extract output images from ComfyUI history"""
        if prompt_id not in history:
            return []

        outputs = history[prompt_id].get("outputs", {})
        images = []

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                images.extend(node_output["images"])

        return images

    async def run(self):
        """Run the service"""
        self.running = True

        # 动态参数（可从Control Pad实时完全覆盖）
        self.style = self.args.style                        # 自由文本风格
        self.staff_suffix = self.args.staff_suffix          # 工作人员正向后缀
        self.staff_negative = self.args.staff_negative or None  # 工作人员负向提示词（None=不注入）
        self.version_tag = getattr(self.args, "version_tag", "0.0.1")

        # 注册SLM消息处理器
        @self.slm_client.on_message
        async def on_slm(msg):
            await self.handle_slm_message(msg)

        # 处理来自Control Pad的配置更新
        @self.server.on_message
        async def on_config(msg: Message):
            msg_type = msg.type.value if hasattr(msg.type, "value") else str(msg.type)
            if msg_type == MessageType.CONFIG_UPDATE.value:
                param = msg.data.get("param")
                value = msg.data.get("value")
                print(f"T2I: CONFIG_UPDATE {param}={value}")

                if param == "style":
                    self.style = value
                    print(f"T2I: Style → {value}")
                elif param == "staff_suffix":
                    self.staff_suffix = value
                    print(f"T2I: Staff suffix → '{value[:50]}...'")
                elif param == "staff_negative":
                    self.staff_negative = value if (value and value.strip()) else None
                    status = "CUSTOM" if (value and value.strip()) else "EMPTY"
                    print(f"T2I: Negative prompt → {status} '{(value or '')[:50]}...'")
                elif param == "version_tag":
                    self.version_tag = value or "0.0.1"
                    print(f"T2I: Version tag → {self.version_tag}")
                elif param == "reference_images":
                    self.reference_images = value or []
                    print(f"T2I: Reference images → {self.reference_images}")

        # 启动WebSocket服务器
        await self.server.start()

        # 发送状态消息
        status_msg = Message.status(Source.T2I, "ready", {
            "comfyui_url": self.args.comfyui_url,
            "workflow": self.args.workflow,
            "style": self.style,
            "version_tag": self.version_tag
        })
        await self.server.broadcast(status_msg)

        print(f"T2I service ready")
        print(f"ComfyUI: {self.args.comfyui_url}")
        print(f"Output: {self.output_dir}")
        print(f"Style: {self.style}")
        print(f"Version: {self.version_tag}")

        # 启动worker
        worker_task = asyncio.create_task(self.generation_worker())

        # 启动SLM客户端连接
        slm_task = asyncio.create_task(self.slm_client.run_forever())

        await asyncio.gather(slm_task, worker_task)


def main():
    print("=" * 50)
    print("  T2I Service Starting...")
    print("=" * 50)
    import sys
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description='T2I WebSocket Service')
    parser.add_argument('--port', type=int, default=PORTS["t2i"])
    parser.add_argument('--slm-host', default='localhost')
    parser.add_argument('--comfyui-url', default='http://127.0.0.1:8188')
    parser.add_argument('--workflow', default='sd15_fast')
    parser.add_argument('--output-dir', default=str(ROOT / 'data' / 'generated_images'))
    parser.add_argument('--version-tag', default='0.0.1',
                        help='Version tag for image metadata and snapshots (e.g., 1.0.0)')

    # 显式可控的提示词参数（默认值，可被Control Pad完全覆盖）
    parser.add_argument('--style', default='',
                        help='Free-text style description')
    parser.add_argument('--staff-suffix', default='',
                        help='Staff-controlled positive prompt suffix')
    parser.add_argument('--staff-negative', default='',
                        help='Staff-controlled negative prompt (can fully override default)')

    parser.add_argument('--vram-mode', default='8gb', choices=['8gb', '12gb'])
    args = parser.parse_args()

    service = T2IService(args)
    asyncio.run(service.run())


if __name__ == "__main__":
    main()
