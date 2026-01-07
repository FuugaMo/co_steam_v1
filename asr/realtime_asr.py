import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def find_input_device(name_substring):
    if not name_substring:
        return None
    name_substring = name_substring.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get('max_input_channels', 0) > 0:
            if name_substring in dev.get('name', '').lower():
                return idx
    return None


def main():
    parser = argparse.ArgumentParser(description='Realtime ASR with faster-whisper')
    parser.add_argument('--device', default='Yeti X', help='Input device name substring')
    parser.add_argument('--model', default='D:/co_steam_v1/models/faster-whisper-base', help='Model size or path')
    parser.add_argument('--language', default='en', help='Language code, e.g. en, zh, ja')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--chunk-sec', type=float, default=3.0)
    parser.add_argument('--overlap-sec', type=float, default=0.2)
    args = parser.parse_args()

    device_index = find_input_device(args.device)
    if device_index is None:
        print('Input device not found, using default input.', file=sys.stderr)
    else:
        dev_name = sd.query_devices(device_index)['name']
        print(f'Using input device: {dev_name} (index {device_index})')

    model = WhisperModel(args.model, device='cpu', compute_type='int8')

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
    base_time = 0.0

    with sd.RawInputStream(
        samplerate=args.sample_rate,
        blocksize=blocksize,
        dtype='int16',
        channels=1,
        device=device_index,
        callback=callback,
    ):
        print('Listening... Press Ctrl+C to stop.')
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
                        start = base_time + seg.start
                        end = base_time + seg.end
                        text = seg.text.strip()
                        if text:
                            print(f'{start:7.2f}-{end:7.2f} {text}')
                    base_time += stride_samples / args.sample_rate
        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()

