"""
Conversation memory layer.
Maintains a bounded rolling window of prior turns so the agent can
handle follow-up questions coherently.
"""

from typing import List, Dict

from utils.constants import MEMORY_WINDOW_SIZE


class ConversationMemory:
    """Stores and formats recent conversation turns."""

    def __init__(self, window_size: int = MEMORY_WINDOW_SIZE):
        self.window_size = window_size
        self._turns: List[Dict[str, str]] = []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Record a completed conversational turn.

        Args:
            user_message: The user's message text.
            assistant_message: The assistant's response text.
        """
        self._turns.append({"user": user_message, "assistant": assistant_message})
        if len(self._turns) > self.window_size:
            self._turns = self._turns[-self.window_size:]

    def get_history_text(self) -> str:
        """
        Return the conversation history formatted as plain text.

        Returns:
            A formatted string of recent turns, or a placeholder if empty.
        """
        if not self._turns:
            return "No previous conversation."

        lines = []
        for turn in self._turns:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored conversation turns."""
        self._turns = []

    def get_turns(self) -> List[Dict[str, str]]:
        """Return the raw list of stored turns."""
        return list(self._turns)
