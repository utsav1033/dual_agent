"""Tool definitions and handlers — Gemini (google-genai) and OpenAI-compatible formats."""
import math
from datetime import datetime
from google.genai import types


def _make_declarations() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="get_current_datetime",
            description="Returns the current date and time.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluates a safe mathematical expression and returns the result. "
                "Supports +, -, *, /, **, sqrt, abs, round, floor, ceil, sin, cos, tan, log, pi, e."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="A math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
                    )
                },
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_weather",
            description="Returns mock weather information for a city (demo only).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="City name to get weather for.",
                    )
                },
                required=["city"],
            ),
        ),
    ]


GEMINI_TOOL = types.Tool(function_declarations=_make_declarations())

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates a safe mathematical expression and returns the result. "
                "Supports +, -, *, /, **, sqrt, abs, round, floor, ceil, sin, cos, tan, log, pi, e."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns mock weather information for a city (demo only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to get weather for.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]

SAFE_MATH = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "pi": math.pi, "e": math.e,
}

MOCK_WEATHER = {
    "london":   {"temp_c": 14, "condition": "Overcast",     "humidity": 78},
    "new york": {"temp_c": 18, "condition": "Sunny",        "humidity": 45},
    "tokyo":    {"temp_c": 25, "condition": "Clear",        "humidity": 55},
    "mumbai":   {"temp_c": 32, "condition": "Humid",        "humidity": 85},
    "paris":    {"temp_c": 16, "condition": "Rainy",        "humidity": 72},
    "default":  {"temp_c": 22, "condition": "Partly cloudy","humidity": 60},
}


def execute_tool(name: str, args: dict) -> str:
    if name == "get_current_datetime":
        return datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S")

    if name == "calculate":
        expr = args.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, SAFE_MATH)
            return f"{expr} = {result}"
        except Exception as ex:
            return f"Error evaluating '{expr}': {ex}"

    if name == "get_weather":
        city = args.get("city", "")
        data = MOCK_WEATHER.get(city.lower(), MOCK_WEATHER["default"])
        return (
            f"Weather in {city}: {data['condition']}, "
            f"{data['temp_c']}°C, humidity {data['humidity']}%."
        )

    return f"Unknown tool: {name}"
