import json
import time
from openai import OpenAI
from .memory import ConversationMemory
from .guardrails import check_input_safety, check_output_safety
from .tools import OPENAI_TOOLS, execute_tool

MODEL_ID = "anthropic/claude-haiku-4-5-20251001"
BASE_URL = "https://ai-gateway.vercel.sh/v1"

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer questions accurately and concisely. "
    "You have access to tools: get_current_datetime, calculate, and get_weather. "
    "Use them when relevant."
)


class FrontierAssistant:
    def __init__(self, api_key: str, model: str = MODEL_ID):
        self.model_name = model
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
        self.memory = ConversationMemory(max_turns=10)

    def chat(self, user_message: str) -> dict:
        start = time.time()

        is_safe, reason = check_input_safety(user_message)
        if not is_safe:
            return {
                "response": f"[BLOCKED by guardrail] {reason}",
                "latency": 0.0,
                "model": self.model_name,
                "blocked": True,
                "memory_summary": self.memory.get_summary(),
                "tool_used": None,
            }

        self.memory.add_message("user", user_message)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.memory.get_messages()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
            )

            tool_used = None
            msg = response.choices[0].message

            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                tool_used = fn_name
                tool_result = execute_tool(fn_name, fn_args)

                messages.append(msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                reply = response.choices[0].message.content or ""
            else:
                reply = msg.content or ""

            reply = reply.strip()
            if not reply:
                reply = "I'm sorry, I couldn't generate a response."

            is_out_safe, _ = check_output_safety(reply)
            if not is_out_safe:
                reply = "[FILTERED] Response removed for safety reasons."

            self.memory.add_message("assistant", reply)
            return {
                "response": reply,
                "latency": round(time.time() - start, 3),
                "model": self.model_name,
                "blocked": False,
                "memory_summary": self.memory.get_summary(),
                "tool_used": tool_used,
            }
        except Exception as e:
            if self.memory.messages and self.memory.messages[-1]["role"] == "user":
                self.memory.messages.pop()
            return {
                "response": f"Error: {e}",
                "latency": round(time.time() - start, 3),
                "model": self.model_name,
                "blocked": False,
                "error": True,
                "memory_summary": self.memory.get_summary(),
                "tool_used": None,
            }

    def clear_memory(self):
        self.memory.clear()
