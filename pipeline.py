"""
ASR → SLM Pipeline
Realtime speech recognition with intent classification and routing
"""

import argparse
import queue
import sys
import time
import os
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pathlib import Path

# Add parent directory to path for slm import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.inference import route, Intent

ROOT = Path(__file__).resolve().parent.parent

def find_input_device(name_substring):
    if not name_substring:
        return None
    name_substring = name_substring.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get('max_input_channels', 0) > 0:
            if name_substring in dev.get('name', '').lower():
                return idx
    return None


@dataclass
class ChunkRecord:
    text: str
    timestamp: float


class ContextWindow:
    """Rolling window of past ASR chunks"""

    def __init__(self, window_seconds: float, chunk_seconds: float):
        """
        Args:
            window_seconds: How many seconds of context to keep
            chunk_seconds: Duration of each ASR chunk
        """
        self.window_seconds = window_seconds
        self.chunk_seconds = chunk_seconds
        # Calculate max chunks: window / chunk_duration
        self.max_chunks = max(1, int(window_seconds / chunk_seconds))
        self.chunks: deque[ChunkRecord] = deque(maxlen=self.max_chunks)

    def add(self, text: str):
        """Add a new chunk"""
        self.chunks.append(ChunkRecord(text=text, timestamp=time.time()))

    def get_context(self, include_current: bool = True) -> str:
        """Get concatenated context from past chunks"""
        now = time.time()
        # Filter chunks within window
        valid = [c for c in self.chunks if (now - c.timestamp) <= self.window_seconds]
        if not valid:
            return ""
        if include_current:
            return " ".join(c.text for c in valid)
        else:
            # Exclude the most recent chunk
            return " ".join(c.text for c in valid[:-1]) if len(valid) > 1 else ""

    def get_current(self) -> str:
        """Get only the current (most recent) chunk"""
        return self.chunks[-1].text if self.chunks else ""

    def get_stats(self) -> dict:
        """Get context window stats"""
        return {
            "chunks": len(self.chunks),
            "max_chunks": self.max_chunks,
            "window_sec": self.window_seconds,
            "total_chars": sum(len(c.text) for c in self.chunks)
        }


class Pipeline:
    def __init__(self, context_window: ContextWindow = None,
                 on_image=None, on_conversation=None):
        """
        Args:
            context_window: ContextWindow for accumulating past chunks
            on_image: callback for image intent
            on_conversation: callback for conversation intent
        """
        self.context = context_window
        self.on_image = on_image or self._default_image_handler
        self.on_conversation = on_conversation or self._default_conversation_handler

    def _default_image_handler(self, data):
        print(f"\n[IMAGE] → State Machine")
        print(f"  Prompt: {data['prompt']}")
        print(f"  Confidence: {data['confidence']}")
        if 'context_stats' in data:
            stats = data['context_stats']
            print(f"  Context: {stats['chunks']}/{stats['max_chunks']} chunks, {stats['total_chars']} chars")

    def _default_conversation_handler(self, data):
        print(f"\n[CONVERSATION] → Agent")
        print(f"  Topics: {data.get('topics', [])}")
        print(f"  Questions: {data.get('questions', [])}")
        print(f"  Sentiment: {data.get('sentiment', 'neutral')}")
        if 'context_stats' in data:
            stats = data['context_stats']
            print(f"  Context: {stats['chunks']}/{stats['max_chunks']} chunks, {stats['total_chars']} chars")

    def process(self, text: str):
        """Process ASR text through SLM and route"""
        if not text or len(text.strip()) < 2:
            return

        # Add to context window
        if self.context:
            self.context.add(text)
            # Use full context for SLM
            full_context = self.context.get_context()
            result = route(full_context)
            # Store original current chunk in result
            result['data']['current_chunk'] = text
            result['data']['context_stats'] = self.context.get_stats()
        else:
            result = route(text)

        if result['intent'] == 'image':
            self.on_image(result['data'])
        else:
            self.on_conversation(result['data'])


def main():
    parser = argparse.ArgumentParser(description='ASR → SLM Pipeline')
    parser.add_argument('--device', default='Yeti X', help='Input device name substring')
    parser.add_argument('--model', default=str(ROOT / 'models' / 'faster-whisper-base'), help='ASR model path')
    parser.add_argument('--language', default='en', help='Language code')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--chunk-sec', type=float, default=3.0)
    parser.add_argument('--overlap-sec', type=float, default=0.2)
    parser.add_argument('--min-chars', type=int, default=5, help='Minimum chars to process')
    parser.add_argument('--context-sec', type=float, default=60.0,
                        help='Context window in seconds (accumulate past chunks, 0=disable)')
    args = parser.parse_args()

    device_index = find_input_device(args.device)
    if device_index is None:
        print('Input device not found, using default.', file=sys.stderr)
    else:
        dev_name = sd.query_devices(device_index)['name']
        print(f'Input: {dev_name}')

    print(f'ASR model: {args.model}')
    model = WhisperModel(args.model, device='cpu', compute_type='int8')

    # Setup context window
    context_window = None
    if args.context_sec > 0:
        context_window = ContextWindow(
            window_seconds=args.context_sec,
            chunk_seconds=args.chunk_sec
        )
        print(f'Context: {args.context_sec}s window ({context_window.max_chunks} chunks)')
    else:
        print('Context: disabled (single chunk mode)')

    pipeline = Pipeline(context_window=context_window)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata)

    blocksize = int(args.sample_rate * 0.5)
    chunk_samples = int(args.sample_rate * args.chunk_sec)
    overlap_samples = int(args.sample_rate * args.overlap_sec)
    stride_samples = max(1, chunk_samples - overlap_samples)

    buffer = np.empty((0,), dtype=np.int16)

    with sd.RawInputStream(
        samplerate=args.sample_rate,
        blocksize=blocksize,
        dtype='int16',
        channels=1,
        device=device_index,
        callback=callback,
    ):
        print('Pipeline ready. Listening...')
        print('-' * 50)
        try:
            while True:
                indata = audio_q.get()
                data = np.frombuffer(indata, dtype=np.int16)
                if data.size == 0:
                    continue
                buffer = np.concatenate([buffer, data])

                while buffer.size >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[stride_samples:]
                    audio = chunk.astype(np.float32) / 32768.0

                    segments, _info = model.transcribe(
                        audio,
                        language=args.language,
                        vad_filter=True,
                        beam_size=1,
                    )

                    for seg in segments:
                        text = seg.text.strip()
                        if text and len(text) >= args.min_chars:
                            print(f"ASR: {text}")
                            pipeline.process(text)

        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()
