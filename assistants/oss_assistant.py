import time
from huggingface_hub import InferenceClient
from .memory import ConversationMemory
from .guardrails import check_input_safety, check_output_safety

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "If you don't know something, say so rather than guessing."
)


class OSSAssistant:
    def __init__(self, hf_token: str, model: str = MODEL_ID):
        self.model = model
        self.client = InferenceClient(token=hf_token)
        self.memory = ConversationMemory(max_turns=10)

    def chat(self, user_message: str) -> dict:
        start = time.time()

        is_safe, reason = check_input_safety(user_message)
        if not is_safe:
            return {
                "response": f"[BLOCKED by guardrail] {reason}",
                "latency": 0.0,
                "model": self.model,
                "blocked": True,
                "memory_summary": self.memory.get_summary(),
            }

        self.memory.add_message("user", user_message)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.memory.get_messages()

        try:
            completion = self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=512,
                temperature=0.7,
            )
            reply = completion.choices[0].message.content.strip()

            is_out_safe, out_reason = check_output_safety(reply)
            if not is_out_safe:
                reply = "[FILTERED] Response removed for safety reasons."

            self.memory.add_message("assistant", reply)
            return {
                "response": reply,
                "latency": round(time.time() - start, 3),
                "model": self.model,
                "blocked": False,
                "memory_summary": self.memory.get_summary(),
            }
        except Exception as e:
            self.memory.messages.pop()  # remove the user message we added
            return {
                "response": f"Error: {e}",
                "latency": round(time.time() - start, 3),
                "model": self.model,
                "blocked": False,
                "error": True,
                "memory_summary": self.memory.get_summary(),
            }

    def clear_memory(self):
        self.memory.clear()
