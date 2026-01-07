"""
SLM inference module - single LLM call for intent + topics + response
Uses Ollama with Ministral models
"""

import requests
import json
from collections import deque
from dataclasses import dataclass
import time

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "ministral-3:3b-instruct-2512-q4_K_M"

SYSTEM_PROMPT = """You are a minimal AI collaborator with visual awareness. For each user input:

1. Extract 2-3 topic keywords (for state machine)
2. Create a concise agent response in format: "keyword → question?"
   - Example: "photosynthesis → how visualize?"
3. Detect topic change by comparing with LAST IMAGE KEYWORDS (if provided):
   - If last image keywords are given, compare current topic with them
   - If no last image keywords, compare with recent conversation
   - Assess if topic changed significantly (0.0-1.0 score)
4. Decide if NEW image generation is needed:
   - Compare current topic with last image keywords
   - Topic change score > 0.6 (significant difference from last image) AND
   - Topic is STEAM-related (science/tech/engineering/art/math) AND
   - Topic benefits from visual diagram
   - Don't generate if current topic is same/similar to last image
5. If image needed, provide NEW image keywords (3-5 words describing the NEW concept)

IMPORTANT: You will receive "Last image keywords: [...]" before user input if an image was recently generated. Use these to avoid generating duplicate images on the same topic.

Reply JSON only:
{
  "keywords": ["k1", "k2", "k3"],
  "response": "keyword → question?",
  "image_trigger": true/false,
  "image_keywords": ["concept1", "concept2"],
  "topic_change_score": 0.8
}

Examples:
- Input: "show me how photosynthesis works"
  Last image: None
  → image_trigger=true, image_keywords=["photosynthesis", "plant", "chloroplast", "light"]

- Input: "what about the light reaction?"
  Last image: ["photosynthesis", "plant", "chloroplast", "light"]
  → image_trigger=false (same topic as last image)

- Input: "now explain cellular respiration"
  Last image: ["photosynthesis", "plant", "chloroplast", "light"]
  → image_trigger=true, image_keywords=["cellular respiration", "mitochondria", "ATP", "glucose"]"""


@dataclass
class ConversationHistory:
    """Maintains conversation history for context"""
    max_turns: int = 20
    messages: deque = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = deque(maxlen=self.max_turns * 2)  # user + assistant pairs

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list:
        """Get messages for API call"""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs.extend(list(self.messages))
        return msgs

    def clear(self):
        self.messages.clear()


# Global conversation history
_history = ConversationHistory()


def route(text: str, timeout: float = 8.0, temperature: float = 0.3, num_predict: int = 80,
          last_image_keywords: list = None) -> dict:
    """
    Single LLM call: classify intent + extract topics + generate response
    Args:
        text: User input text
        last_image_keywords: Keywords from the last generated image (for topic change detection)
    Returns: {"data": {...}}
    """
    global _history
    start = time.time()

    # Prepend last image keywords info if available
    user_input = text
    if last_image_keywords:
        keywords_str = ", ".join(last_image_keywords)
        user_input = f"Last image keywords: [{keywords_str}]\n\nUser: {text}"

    # Add user message to history
    _history.add_user(user_input)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": _history.get_messages(),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                }
            },
            timeout=timeout
        )
        resp.raise_for_status()

        result = resp.json()
        response_text = result.get("message", {}).get("content", "").strip()
        latency = int((time.time() - start) * 1000)

        # Add assistant response to history
        _history.add_assistant(response_text)

        # Clean markdown code blocks
        clean_text = response_text
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            clean_text = "\n".join(lines).strip()

        # Parse JSON
        try:
            data = json.loads(clean_text)
            keywords = data.get("keywords", data.get("topics", []))  # Support both field names
            agent_response = data.get("response", "")

            return {
                "data": {
                    "keywords": keywords,  # → ISM
                    "response": agent_response,  # → User
                    "image_trigger": data.get("image_trigger", False),  # → T2I
                    "image_keywords": data.get("image_keywords", []),  # → T2I
                    "topic_change_score": data.get("topic_change_score", 0.0),  # → T2I
                    "original": text,
                    "latency_ms": latency,
                    "history_length": len(_history.messages) // 2
                }
            }

        except json.JSONDecodeError:
            # Fallback: use raw response
            return {
                "data": {
                    "keywords": [],
                    "response": response_text[:50],  # Use raw response as fallback
                    "original": text,
                    "latency_ms": latency,
                    "history_length": len(_history.messages) // 2
                }
            }

    except Exception as e:
        latency = int((time.time() - start) * 1000)
        return {
            "data": {
                "keywords": [],
                "response": "",
                "error": str(e),
                "original": text,
                "latency_ms": latency
            }
        }


def clear_history():
    """Clear conversation history"""
    global _history
    _history.clear()


def get_history_length() -> int:
    """Get current history length"""
    return len(_history.messages) // 2


def set_max_turns(max_turns: int):
    """Update max turns for conversation history"""
    global _history
    _history.max_turns = max_turns
    _history.messages = deque(_history.messages, maxlen=max_turns * 2)


def get_system_prompt() -> str:
    """Get the current system prompt"""
    return SYSTEM_PROMPT


# CLI test
if __name__ == "__main__":
    import sys

    print(f"Model: {MODEL}")
    print("-" * 50)

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = route(text)
        print(f"Input: {text}")
        print(f"Keywords (→ISM): {result['data'].get('keywords', [])}")
        print(f"Response (→User): {result['data'].get('response', '')}")
        print(f"Latency: {result['data'].get('latency_ms', 0)}ms")
        print(f"History: {result['data'].get('history_length', 0)} turns")
    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit, 'clear' to reset.")
        while True:
            try:
                text = input("\nYou: ").strip()
                if not text:
                    continue
                if text.lower() == 'quit':
                    break
                if text.lower() == 'clear':
                    clear_history()
                    print("History cleared.")
                    continue

                result = route(text)
                print(f"Keywords (→ISM): {result['data'].get('keywords', [])}")
                print(f"Agent (→User): {result['data'].get('response', '')}")
                print(f"[{result['data'].get('latency_ms', 0)}ms, {result['data'].get('history_length', 0)} turns]")

            except KeyboardInterrupt:
                break
