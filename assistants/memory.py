from typing import List, Dict


class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Keep only the last max_turns pairs
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2):]

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def get_turn_count(self) -> int:
        return len(self.messages) // 2

    def get_summary(self) -> str:
        turns = self.get_turn_count()
        return f"{turns} turn{'s' if turns != 1 else ''} in memory"

    def clear(self):
        self.messages = []
