"""
WebSocket messaging protocol
共享的消息格式和工具
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional
from enum import Enum


class MessageType(str, Enum):
    # ASR events
    ASR_TEXT = "asr_text"           # ASR transcribed text
    ASR_STATUS = "asr_status"       # ASR status (started, stopped, error)

    # SLM events
    INTENT = "intent"               # Intent classification result
    KEYWORDS = "keywords"           # Extracted keywords

    # State machine events
    STATE_CHANGE = "state_change"   # State transition
    STATE_ACTION = "state_action"   # Action to execute

    # T2I events
    T2I_START = "t2i_start"         # Generation started
    T2I_PROGRESS = "t2i_progress"   # Generation progress
    T2I_COMPLETE = "t2i_complete"   # Image ready
    T2I_ERROR = "t2i_error"         # Generation failed

    # System events
    STATUS = "status"               # Service status
    ERROR = "error"                 # Error message
    CONFIG_UPDATE = "config_update" # Parameter update from Control Pad
    PING = "ping"                   # Health check
    PONG = "pong"                   # Health check response


class Source(str, Enum):
    ASR = "asr"
    SLM = "slm"
    STATE = "state"
    T2I = "t2i"
    BRIDGE = "bridge"
    CLIENT = "client"


@dataclass
class Message:
    type: str
    source: str
    data: dict
    timestamp: float = None
    id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.id is None:
            self.id = f"{self.source}_{int(self.timestamp * 1000)}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "Message":
        d = json.loads(data)
        return cls(**d)

    @classmethod
    def asr_text(cls, text: str, chunk_id: int = 0, context: list = None):
        return cls(
            type=MessageType.ASR_TEXT,
            source=Source.ASR,
            data={
                "text": text,
                "chunk_id": chunk_id,
                "context": context or []
            }
        )

    @classmethod
    def intent(cls, intent: str, confidence: str, prompt: str, context_stats: dict = None):
        return cls(
            type=MessageType.INTENT,
            source=Source.SLM,
            data={
                "intent": intent,
                "confidence": confidence,
                "prompt": prompt,
                "context_stats": context_stats
            }
        )

    @classmethod
    def keywords(cls, topics: list, questions: list, sentiment: str, original: str):
        return cls(
            type=MessageType.KEYWORDS,
            source=Source.SLM,
            data={
                "topics": topics,
                "questions": questions,
                "sentiment": sentiment,
                "original": original
            }
        )

    @classmethod
    def t2i_complete(cls, image_path: str, prompt: str, negative_prompt: str = "", structure: dict = None):
        return cls(
            type=MessageType.T2I_COMPLETE,
            source=Source.T2I,
            data={
                "image_path": image_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "structure": structure or {}
            }
        )

    @classmethod
    def status(cls, source: str, status: str, info: dict = None):
        return cls(
            type=MessageType.STATUS,
            source=source,
            data={
                "status": status,
                "info": info or {}
            }
        )

    @classmethod
    def error(cls, source: str, error: str, details: dict = None):
        return cls(
            type=MessageType.ERROR,
            source=source,
            data={
                "error": error,
                "details": details or {}
            }
        )


# Default ports
PORTS = {
    "asr": 5551,
    "slm": 5552,
    "state": 5553,
    "t2i": 5554,
    "bridge": 5555,
}

def get_ws_url(service: str, host: str = "localhost") -> str:
    """Get WebSocket URL for a service"""
    return f"ws://{host}:{PORTS[service]}"
