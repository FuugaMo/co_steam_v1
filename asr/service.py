"""
ASR WebSocket Service
Realtime speech recognition, broadcasts transcribed text
"""

import argparse
import asyncio
import queue
import sys
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Thread

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import WSServer, Message, MessageType, Source, PORTS


@dataclass
class ChunkRecord:
    text: str
    timestamp: float


class ContextWindow:
    """Rolling window of past ASR chunks"""

    def __init__(self, window_seconds: float, chunk_seconds: float):
        self.window_seconds = window_seconds
        self.chunk_seconds = chunk_seconds
        self.max_chunks = max(1, int(window_seconds / chunk_seconds))
        self.chunks: deque[ChunkRecord] = deque(maxlen=self.max_chunks)

    def add(self, text: str):
        self.chunks.append(ChunkRecord(text=text, timestamp=time.time()))

    def get_context(self) -> list:
        """Get list of recent chunks"""
        now = time.time()
        return [c.text for c in self.chunks if (now - c.timestamp) <= self.window_seconds]

    def get_context_text(self) -> str:
        """Get concatenated context"""
        return " ".join(self.get_context())


def find_input_device(name_substring):
    if not name_substring:
        return None
    name_substring = name_substring.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get('max_input_channels', 0) > 0:
            if name_substring in dev.get('name', '').lower():
                return idx
    return None


def list_input_devices():
    """Return list of (index, name) for input-capable devices"""
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get('max_input_channels', 0) > 0:
            devices.append((idx, dev.get('name', '')))
    return devices


def prompt_device_selection():
    """Interactively ask user to pick an input device index"""
    devices = list_input_devices()
    if not devices:
        print("No input-capable devices found.")
        return None

    print("Available input devices (index: name):")
    for idx, name in devices:
        print(f"{idx}: {name}")

    valid_indices = {idx for idx, _ in devices}
    while True:
        choice = input("Select input device index (or 'q' to skip): ").strip()
        if choice.lower() == 'q':
            return None
        try:
            choice_idx = int(choice)
        except ValueError:
            print("Please enter a number from the list.")
            continue
        if choice_idx in valid_indices:
            return choice_idx
        print("Index not in the list, try again.")


class ASRService:
    def __init__(self, args):
        self.args = args
        self.server = WSServer("asr", PORTS["asr"])
        self.model = None
        self.context = None
        self.audio_queue = queue.Queue()
        self.text_queue = asyncio.Queue()
        self.chunk_id = 0
        self.running = False

    def setup_model(self):
        """Initialize whisper model"""
        print(f"Loading ASR model: {self.args.model}")
        self.model = WhisperModel(
            self.args.model,
            device='cpu',
            compute_type='int8'
        )
        print("ASR model loaded")

        if self.args.context_sec > 0:
            self.context = ContextWindow(
                window_seconds=self.args.context_sec,
                chunk_seconds=self.args.chunk_sec
            )
            print(f"Context window: {self.args.context_sec}s ({self.context.max_chunks} chunks)")

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block"""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def run_asr_loop(self):
        """ASR processing loop (runs in separate thread)"""
        if self.args.device_index is not None:
            device_index = self.args.device_index
            print(f'Input device (by index): {device_index}')
        else:
            device_index = find_input_device(self.args.device)
            if device_index is None:
                print('Input device not found, using default.', file=sys.stderr)
            else:
                dev_name = sd.query_devices(device_index)['name']
                print(f'Input device: {dev_name}')

        blocksize = int(self.args.sample_rate * 0.5)
        chunk_samples = int(self.args.sample_rate * self.args.chunk_sec)
        overlap_samples = int(self.args.sample_rate * self.args.overlap_sec)
        stride_samples = max(1, chunk_samples - overlap_samples)

        buffer = np.empty((0,), dtype=np.int16)

        with sd.RawInputStream(
            samplerate=self.args.sample_rate,
            blocksize=blocksize,
            dtype='int16',
            channels=1,
            device=device_index,
            callback=self.audio_callback,
        ):
            print("ASR listening...")
            while self.running:
                try:
                    indata = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                data = np.frombuffer(indata, dtype=np.int16)
                if data.size == 0:
                    continue
                buffer = np.concatenate([buffer, data])

                while buffer.size >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[stride_samples:]
                    audio = chunk.astype(np.float32) / 32768.0

                    segments, _ = self.model.transcribe(
                        audio,
                        language=self.args.language,
                        vad_filter=True,
                        beam_size=1,
                    )

                    for seg in segments:
                        text = seg.text.strip()
                        if text and len(text) >= self.args.min_chars:
                            self.chunk_id += 1

                            # Add to context
                            context_list = []
                            if self.context:
                                self.context.add(text)
                                context_list = self.context.get_context()

                            # Queue for async broadcast
                            asyncio.run_coroutine_threadsafe(
                                self.text_queue.put({
                                    "text": text,
                                    "chunk_id": self.chunk_id,
                                    "context": context_list
                                }),
                                self.loop
                            )
                            # Readable console output
                            context_info = f"{len(context_list)} chunks" if context_list else "no context"
                            print(f"╔═══ ASR [{self.chunk_id}] ═══")
                            print(f"║ Text: {text}")
                            print(f"╚═══ Context: {context_info} ═══")

    async def broadcast_loop(self):
        """Broadcast ASR results to WebSocket clients"""
        while self.running:
            try:
                data = await asyncio.wait_for(self.text_queue.get(), timeout=0.5)
                msg = Message.asr_text(
                    text=data["text"],
                    chunk_id=data["chunk_id"],
                    context=data["context"]
                )
                await self.server.broadcast(msg)
            except asyncio.TimeoutError:
                continue

    async def run(self):
        """Run the service"""
        self.setup_model()
        self.running = True
        self.loop = asyncio.get_event_loop()

        # Start ASR in separate thread
        asr_thread = Thread(target=self.run_asr_loop, daemon=True)
        asr_thread.start()

        # Start WebSocket server
        await self.server.start()

        # Send status message
        status_msg = Message.status(Source.ASR, "ready", {
            "model": self.args.model,
            "language": self.args.language,
            "context_sec": self.args.context_sec
        })
        await self.server.broadcast(status_msg)

        # Run broadcast loop
        await self.broadcast_loop()

    def start(self):
        """Blocking start"""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            print("\nASR service stopped")
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='ASR WebSocket Service')
    parser.add_argument('--device', default='Yeti X', help='Input device name (substring match)')
    parser.add_argument('--device-index', type=int, default=None, help='Input device index (overrides --device)')
    parser.add_argument('--list-devices', action='store_true', help='List available input devices and exit')
    parser.add_argument('--no-prompt-device', action='store_true',
                        help='Skip interactive device selection (use flags/defaults)')
    parser.add_argument('--model', default='D:/co_steam_v1/models/faster-whisper-small',
                        help='ASR model path')
    parser.add_argument('--language', default='en', help='Language code')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--chunk-sec', type=float, default=3.0)
    parser.add_argument('--overlap-sec', type=float, default=0.2)
    parser.add_argument('--min-chars', type=int, default=5)
    parser.add_argument('--context-sec', type=float, default=60.0,
                        help='Context window seconds (0=disable)')
    parser.add_argument('--port', type=int, default=PORTS["asr"])
    args = parser.parse_args()

    if args.list_devices:
        print("Available input devices (index: name):")
        for idx, name in list_input_devices():
            print(f"{idx}: {name}")
        return

    # Interactive device selection (default) unless explicitly disabled
    if not args.no_prompt_device and args.device_index is None:
        selection = prompt_device_selection()
        if selection is not None:
            args.device_index = selection

    service = ASRService(args)
    service.start()


if __name__ == "__main__":
    main()
