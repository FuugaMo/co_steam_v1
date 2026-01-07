"""
Conversation Agent - maintains dialogue history and generates minimal responses
"""

import requests
import json
from dataclasses import dataclass, field
from typing import List
from collections import deque

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "ministral-3:3b-instruct-2512-q4_K_M"

SYSTEM_PROMPT = """Minimal AI collaborator. Reflect key point + one question. STRICT: Max 15 words total. No extra text."""


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationAgent:
    """Maintains conversation history and generates responses"""

    max_history: int = 20  # Max conversation turns to keep
    history: deque = field(default_factory=lambda: deque(maxlen=20))

    def add_user_message(self, content: str):
        """Add user message to history"""
        self.history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str):
        """Add assistant response to history"""
        self.history.append(Message(role="assistant", content=content))

    def get_messages(self) -> List[dict]:
        """Get formatted messages for Ollama API"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def generate_response(self, user_input: str, timeout: float = 10.0) -> dict:
        """Generate response based on conversation history"""
        import time
        start = time.time()

        # Add user input to history
        self.add_user_message(user_input)

        try:
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": MODEL,
                    "messages": self.get_messages(),
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 30,  # ~15 words max
                    }
                },
                timeout=timeout
            )
            resp.raise_for_status()

            result = resp.json()
            response_text = result.get("message", {}).get("content", "").strip()
            latency = int((time.time() - start) * 1000)

            # Add response to history
            self.add_assistant_message(response_text)

            return {
                "response": response_text,
                "latency_ms": latency,
                "history_length": len(self.history),
                "success": True
            }

        except Exception as e:
            latency = int((time.time() - start) * 1000)
            return {
                "response": "",
                "error": str(e),
                "latency_ms": latency,
                "history_length": len(self.history),
                "success": False
            }

    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()

    def get_history_text(self) -> str:
        """Get conversation history as text"""
        lines = []
        for msg in self.history:
            prefix = "User" if msg.role == "user" else "AI"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)


# Global agent instance
_agent = None

def get_agent() -> ConversationAgent:
    """Get or create global agent instance"""
    global _agent
    if _agent is None:
        _agent = ConversationAgent()
    return _agent


def chat(user_input: str, timeout: float = 10.0) -> dict:
    """Convenience function to chat with the agent"""
    agent = get_agent()
    return agent.generate_response(user_input, timeout)


def clear_history():
    """Clear the agent's conversation history"""
    agent = get_agent()
    agent.clear_history()


# CLI test
if __name__ == "__main__":
    import sys

    print(f"Model: {MODEL}")
    print(f"System: {SYSTEM_PROMPT[:50]}...")
    print("-" * 50)

    if len(sys.argv) > 1:
        # Single input from command line
        text = " ".join(sys.argv[1:])
        result = chat(text)
        print(f"User: {text}")
        print(f"AI: {result['response']}")
        print(f"Latency: {result['latency_ms']}ms")
    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit, 'clear' to reset history.")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == 'quit':
                    break
                if user_input.lower() == 'clear':
                    clear_history()
                    print("History cleared.")
                    continue

                result = chat(user_input)
                print(f"AI: {result['response']}")
                print(f"   [{result['latency_ms']}ms, {result['history_length']} turns]")
                print()

            except KeyboardInterrupt:
                break

        print("\nConversation history:")
        print(get_agent().get_history_text())
