class ScratchMemory:

    def __init__(self, max_turns=5):

        self.max_turns = max_turns
        self.messages = []

    def save(self, user, assistant):

        self.messages.append({
            "role": "user",
            "content": user
        })

        self.messages.append({
            "role": "assistant",
            "content": assistant
        })

        self.messages = self.messages[-self.max_turns * 2:]

    def get_history(self):

        return "\n".join(
            f"{m['role']}: {m['content']}"
            for m in self.messages
        )

    def clear(self):

        self.messages = []


memory = ScratchMemory()