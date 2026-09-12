from pathlib import Path
from typing import TypedDict, Annotated, Any

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

import ast
import math
import operator
import os
import sqlite3
import requests


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from your .env file.")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing from your .env file.")


# --------------------------------------------------
# OpenAI model
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_retries=2,
)


# --------------------------------------------------
# Tool 1: Tavily search
# --------------------------------------------------

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)


# --------------------------------------------------
# Tool 2: Calculator
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Supports arithmetic and selected math functions.
    Examples: 2 + 2, math.sqrt(16), sum([1, 2, 3]).
    """
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "math.sqrt": math.sqrt,
        "math.sin": math.sin,
        "math.cos": math.cos,
        "math.tan": math.tan,
        "math.log": math.log,
        "math.log10": math.log10,
        "math.ceil": math.ceil,
        "math.floor": math.floor,
    }

    constants = {
        "math.pi": math.pi,
        "math.e": math.e,
        "math.tau": math.tau,
    }

    def get_name(node):
        if isinstance(node, ast.Name):
            return node.id

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "math"
        ):
            return f"math.{node.attr}"

        raise ValueError("Unsupported function or name.")

    def checked_number(value):
        if type(value) not in (int, float):
            raise ValueError("A real number is required.")

        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Number is outside the supported range.")

        return value

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return checked_number(node.value)

        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(item) for item in node.elts]

        if isinstance(node, ast.BinOp):
            operation = binary_ops.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator.")

            left = checked_number(evaluate(node.left))
            right = checked_number(evaluate(node.right))

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            return checked_number(operation(left, right))

        if isinstance(node, ast.UnaryOp):
            operation = unary_ops.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported unary operator.")

            value = checked_number(evaluate(node.operand))
            return checked_number(operation(value))

        if isinstance(node, ast.Call):
            function = functions.get(get_name(node.func))

            if function is None or node.keywords:
                raise ValueError("Unsupported function or arguments.")

            result = function(*(evaluate(arg) for arg in node.args))
            return checked_number(result)

        if isinstance(node, (ast.Name, ast.Attribute)):
            name = get_name(node)

            if name in constants:
                return constants[name]

        raise ValueError("Unsupported expression.")

    try:
        if len(expression) > 500:
            return "Calculation error: expression is too long."

        tree = ast.parse(expression, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > 150:
            return "Calculation error: expression is too complex."

        return str(checked_number(evaluate(tree.body)))

    except Exception as error:
        return f"Calculation error: {error}"


# --------------------------------------------------
# Tool 3: Stock quote — key included directly
# --------------------------------------------------

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available Alpha Vantage stock quote
    for a symbol such as AAPL or TSLA.
    """
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": "9MZO2JUBR7IFNTOI",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            return {"error": "Unexpected stock API response."}

        if data.get("Global Quote"):
            return data["Global Quote"]

        return {
            "error": (
                data.get("Error Message")
                or data.get("Note")
                or data.get("Information")
                or "No stock quote found. Check the symbol and API access."
            )
        }

    except requests.Timeout:
        return {"error": "The stock service timed out."}

    except requests.HTTPError as error:
        status = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )
        return {"error": f"Stock API returned HTTP {status}."}

    except requests.RequestException:
        return {"error": "Could not connect to the stock service."}

    except ValueError:
        return {"error": "The stock service returned invalid JSON."}


# --------------------------------------------------
# Tool 4: Current weather
# --------------------------------------------------

@tool
def get_current_weather(location: str) -> str:
    """
    Get current weather for a city.
    Examples: Chennai,IN; London,GB; New York,US.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Set OPENWEATHER_API_KEY in your .env file."

    try:
        # Convert the city name into coordinates.
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={
                "q": location,
                "limit": 1,
                "appid": api_key,
            },
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        place = locations[0]

        # Fetch current weather using coordinates.
        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": place["lat"],
                "lon": place["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather = weather_response.json()

        visibility = weather.get("visibility")
        visibility_km = (
            round(visibility / 1000, 1)
            if visibility is not None
            else "N/A"
        )

        display_location = ", ".join(
            part
            for part in [
                place.get("name", location),
                place.get("state", ""),
                place.get("country", ""),
            ]
            if part
        )

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {weather['weather'][0]['description'].title()}\n"
            f"- Temperature: {weather['main']['temp']}°C\n"
            f"- Feels like: {weather['main']['feels_like']}°C\n"
            f"- Humidity: {weather['main']['humidity']}%\n"
            f"- Pressure: {weather['main']['pressure']} hPa\n"
            f"- Wind speed: {weather.get('wind', {}).get('speed', 'N/A')} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service timed out. Please try again."

    except requests.HTTPError as error:
        status = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )

        if status == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned HTTP {status}."

    except requests.RequestException:
        return "Could not connect to the weather service."

    except (KeyError, IndexError, TypeError, ValueError):
        return "The weather service returned an unexpected response."


# --------------------------------------------------
# Bind tools to the model
# --------------------------------------------------

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# Conversation state
# --------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------
# Graph nodes
# --------------------------------------------------

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


# --------------------------------------------------
# SQLite conversation memory
# --------------------------------------------------

conn = sqlite3.connect(
    database=str(BASE_DIR / "chatbot.db"),
    check_same_thread=False,
)

checkpoint = SqliteSaver(conn)


# --------------------------------------------------
# Build graph
# --------------------------------------------------

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


# --------------------------------------------------
# Helper for the Streamlit frontend
# --------------------------------------------------

def get_all_threads():
    all_threads = set()

    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"].get("thread_id")

        if thread_id is not None:
            all_threads.add(thread_id)

    return sorted(all_threads)