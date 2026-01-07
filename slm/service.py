"""
SLM WebSocket Service
Intent classification and keyword extraction
Subscribes to ASR, broadcasts to State/Bridge
"""

import argparse
import asyncio
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WSServer, WSClient, Message, MessageType, Source, PORTS
from slm.inference import route, set_max_turns


class SLMService:
    def __init__(self, args):
        self.args = args
        self.server = WSServer("slm", args.port)
        self.asr_client = WSClient("slm", "asr", args.asr_host)
        self.bridge_client = WSClient("slm", "bridge", args.bridge_host)
        self.executor = ThreadPoolExecutor(max_workers=args.workers)
        self.running = False
        self.queue = asyncio.Queue()  # Queue for pending chunks
        self.processed_count = 0
        self.queue_size = 0
        self.chunk_interval = args.chunk_interval
        self.temperature = args.temperature
        self.num_predict = args.num_predict
        self.max_turns = args.max_turns
        self.last_image_keywords = None  # Track keywords from last generated image

        # Chunk accumulation for interval processing
        self.accumulated_chunks = []  # Buffer for accumulating chunks

    def process_text(self, text: str, context: list = None) -> dict:
        """Process text through SLM (single call: intent + topics + response)"""
        result = route(
            text,
            timeout=self.args.timeout,
            temperature=self.temperature,
            num_predict=self.num_predict,
            last_image_keywords=self.last_image_keywords
        )
        result['data']['current_chunk'] = text
        return result

    async def handle_bridge_message(self, msg: Message):
        """Handle messages from Bridge (T2I_COMPLETE and CONFIG_UPDATE)"""
        if msg.type == MessageType.T2I_COMPLETE:
            # Update last image keywords when new image is generated
            keywords = msg.data.get("keywords", [])
            if keywords:
                self.last_image_keywords = keywords
                print(f"SLM: Updated last_image_keywords = {keywords}")

        elif msg.type == MessageType.CONFIG_UPDATE:
            # Handle config updates from Control Pad
            service = msg.data.get("service")
            if service == "slm":
                param = msg.data.get("param")
                value = msg.data.get("value")

                # Update the parameter
                if param == "chunk_interval":
                    old_value = self.chunk_interval
                    self.chunk_interval = int(value)
                    # Clear accumulated chunks when interval changes
                    self.accumulated_chunks = []
                    print(f"SLM: chunk_interval updated {old_value} → {self.chunk_interval}")
                elif param == "temperature":
                    self.temperature = float(value)
                    print(f"SLM: temperature updated → {self.temperature}")
                elif param == "num_predict":
                    self.num_predict = int(value)
                    print(f"SLM: num_predict updated → {self.num_predict}")
                elif param == "max_turns":
                    old_value = self.max_turns
                    self.max_turns = int(value)
                    set_max_turns(self.max_turns)
                    print(f"SLM: max_turns updated {old_value} → {self.max_turns}")

    async def handle_asr_message(self, msg: Message):
        """Handle incoming ASR message - accumulate and process by interval"""
        if msg.type != MessageType.ASR_TEXT:
            return

        text = msg.data.get("text", "")
        context = msg.data.get("context", [])
        chunk_id = msg.data.get("chunk_id", 0)

        if not text:
            return

        # Accumulate chunks
        self.accumulated_chunks.append(text)
        print(f"SLM [{chunk_id}] accumulated ({len(self.accumulated_chunks)}/{self.chunk_interval}): {text[:40]}...")

        # Process when we've accumulated enough chunks
        if len(self.accumulated_chunks) >= self.chunk_interval:
            # Merge accumulated chunks
            merged_text = " ".join(self.accumulated_chunks)

            # Add to queue for processing
            await self.queue.put({
                "text": merged_text,
                "context": context,
                "chunk_id": chunk_id
            })
            self.queue_size = self.queue.qsize()
            print(f"SLM [{chunk_id}] queued merged text ({len(self.accumulated_chunks)} chunks, queue: {self.queue_size})")

            # Clear accumulation buffer
            self.accumulated_chunks = []

    async def process_worker(self):
        """Worker that processes queued chunks"""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                # Get from queue with timeout
                item = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            text = item["text"]
            context = item["context"]
            chunk_id = item["chunk_id"]
            self.queue_size = self.queue.qsize()

            print(f"SLM [{chunk_id}] processing: {text[:40]}... (queue: {self.queue_size})")

            try:
                # Process in thread pool
                result = await loop.run_in_executor(
                    self.executor,
                    self.process_text,
                    text,
                    context
                )

                self.processed_count += 1
                latency = result['data'].get('latency_ms', 0)

                # Create and broadcast message
                keywords = result["data"].get("keywords", [])
                response = result["data"].get("response", "")
                history_len = result["data"].get("history_length", 0)

                # Unified message format: keywords + agent response + T2I trigger
                out_msg = Message(
                    type=MessageType.KEYWORDS,
                    source=Source.SLM,
                    data={
                        "keywords": keywords,  # → ISM
                        "agent_response": response,  # → User
                        "image_trigger": result["data"].get("image_trigger", False),  # → T2I
                        "image_keywords": result["data"].get("image_keywords", []),  # → T2I
                        "topic_change_score": result["data"].get("topic_change_score", 0.0),  # → T2I
                        "original_text": text,
                        "history_length": history_len,
                        "latency_ms": latency,
                        "queue_size": self.queue_size
                    }
                )

                # Readable console output (ASCII-safe for Windows GBK)
                image_status = "[IMG]" if result["data"].get("image_trigger") else "[TXT]"
                image_kw = result["data"].get("image_keywords", [])
                topic_score = result["data"].get("topic_change_score", 0.0)

                print(f"=== SLM [{chunk_id}] === {latency}ms === {image_status}")
                print(f"| Input: {text[:60]}{'...' if len(text) > 60 else ''}")
                print(f"| Keywords (->ISM): {keywords}")
                print(f"| Agent (->User): {response}")
                if result["data"].get("image_trigger"):
                    print(f"| [IMG] T2I Keywords: {image_kw} (topic_change={topic_score:.2f})")
                print(f"=== History: {history_len} turns | Queue: {self.queue_size} ===")

                await self.server.broadcast(out_msg)

            except Exception as e:
                print(f"SLM [{chunk_id}] error: {e}")
                error_msg = Message.error(Source.SLM, str(e))
                await self.server.broadcast(error_msg)

    async def run(self):
        """Run the service"""
        self.running = True

        # Initialize conversation history settings
        set_max_turns(self.max_turns)

        # Setup ASR client handler
        @self.asr_client.on_message
        async def on_asr(msg):
            await self.handle_asr_message(msg)

        # Setup Bridge client handler (to receive T2I_COMPLETE)
        @self.bridge_client.on_message
        async def on_bridge(msg):
            await self.handle_bridge_message(msg)

        # Setup SLM server handler (to receive CONFIG_UPDATE from Bridge)
        @self.server.on_message
        async def on_server(msg):
            # Handle CONFIG_UPDATE messages sent by Bridge to SLM server
            if msg.type == MessageType.CONFIG_UPDATE:
                await self.handle_bridge_message(msg)

        # Start server
        await self.server.start()

        # Send status
        status_msg = Message.status(Source.SLM, "ready", {
            "model": "ministral-3:3b-instruct-2512-q4_K_M",
            "workers": self.args.workers,
            "chunk_interval": self.chunk_interval,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "max_turns": self.max_turns
        })
        await self.server.broadcast(status_msg)

        print(f"SLM service ready (workers={self.args.workers})")
        print(f"Connecting to ASR at {self.asr_client.url}")
        print(f"Connecting to Bridge at {self.bridge_client.url}")

        # Start worker tasks
        workers = [
            asyncio.create_task(self.process_worker())
            for _ in range(self.args.workers)
        ]

        # Run ASR and Bridge clients (with auto-reconnect)
        asr_task = asyncio.create_task(self.asr_client.run_forever())
        bridge_task = asyncio.create_task(self.bridge_client.run_forever())

        # Wait for all tasks
        await asyncio.gather(asr_task, bridge_task, *workers)

    def start(self):
        """Blocking start"""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            print("\nSLM service stopped")
            self.running = False
            self.executor.shutdown(wait=False)


def main():
    parser = argparse.ArgumentParser(description='SLM WebSocket Service')
    parser.add_argument('--port', type=int, default=PORTS["slm"])
    parser.add_argument('--asr-host', default='localhost', help='ASR service host')
    parser.add_argument('--bridge-host', default='localhost', help='Bridge service host')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers')
    parser.add_argument('--timeout', type=float, default=5.0, help='Ollama timeout (sec)')
    parser.add_argument('--chunk-interval', type=int, default=1,
                        help='Process every Nth chunk (1=all, 2=every other, 3=every third)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='LLM temperature (0.0-1.0, creativity)')
    parser.add_argument('--num-predict', type=int, default=80,
                        help='Max tokens to generate')
    parser.add_argument('--max-turns', type=int, default=20,
                        help='Conversation history depth (turns)')
    args = parser.parse_args()

    service = SLMService(args)
    service.start()


if __name__ == "__main__":
    main()
