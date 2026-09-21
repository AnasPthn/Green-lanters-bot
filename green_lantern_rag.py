import os
import json
from typing import TypedDict

from huggingface_hub import InferenceClient

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. QWEN SETUP
# ============================================================

client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_TOKEN")
)

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"


# ============================================================
# 2. LOAD GREEN LANTERN KNOWLEDGE
# ============================================================

loader = TextLoader("lanterns.txt")

documents = loader.load()

print("Documents loaded:", len(documents))


# ============================================================
# 3. SPLIT DOCUMENT
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings created")


# ============================================================
# 5. CREATE CHROMA VECTOR DATABASE
# ============================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Chroma database created")


# ============================================================
# 6. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)

print("Retriever ready")


# ============================================================
# 7. GREEN LANTERN SEARCH TOOL
# ============================================================

def search_green_lantern(query: str) -> str:

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 8. CALCULATOR TOOL
# ============================================================

def calculator(a: int, b: int) -> int:
    """Add two numbers."""

    return a + b


# ============================================================
# 9. TOOL DESCRIPTIONS FOR QWEN
# ============================================================

tools = [

    {
        "type": "function",

        "function": {

            "name": "search_green_lantern",

            "description": (
                "Search the Green Lantern knowledge base "
                "for information about Green Lantern characters, "
                "powers, locations, history, and the Green Lantern Corps."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "The information you want to find "
                            "in the Green Lantern knowledge base."
                        )
                    }

                },

                "required": ["query"]
            }
        }
    },


    {
        "type": "function",

        "function": {

            "name": "calculator",

            "description": "Add two numbers together.",

            "parameters": {

                "type": "object",

                "properties": {

                    "a": {
                        "type": "integer",
                        "description": "First number"
                    },

                    "b": {
                        "type": "integer",
                        "description": "Second number"
                    }

                },

                "required": ["a", "b"]
            }
        }
    }

]


# ============================================================
# 10. LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    messages: list


# ============================================================
# 11. LLM NODE
# ============================================================

def call_llm(state: State):

    response = client.chat.completions.create(

        model=MODEL,

        messages=state["messages"],

        tools=tools,

        tool_choice="auto",

        max_tokens=300
    )

    assistant_message = response.choices[0].message


    # Convert Qwen response into normal dictionary

    message = {

        "role": "assistant",

        "content": assistant_message.content
    }


    # If Qwen requested tools

    if assistant_message.tool_calls:

        message["tool_calls"] = [

            {

                "id": tool_call.id,

                "type": "function",

                "function": {

                    "name": tool_call.function.name,

                    "arguments": tool_call.function.arguments
                }

            }

            for tool_call in assistant_message.tool_calls

        ]


    return {

        "messages": state["messages"] + [message]

    }


# ============================================================
# 12. TOOL NODE
# ============================================================

def run_tool(state: State):

    last_message = state["messages"][-1]

    tool_calls = last_message["tool_calls"]

    tool_messages = []


    # Process every tool call Qwen requested

    for tool_call in tool_calls:

        tool_name = tool_call["function"]["name"]

        arguments = json.loads(
            tool_call["function"]["arguments"]
        )


        # -------------------------
        # Green Lantern Search
        # -------------------------

        if tool_name == "search_green_lantern":

            result = search_green_lantern(
                arguments["query"]
            )


        # -------------------------
        # Calculator
        # -------------------------

        elif tool_name == "calculator":

            result = calculator(
                arguments["a"],
                arguments["b"]
            )


        # -------------------------
        # Unknown Tool
        # -------------------------

        else:

            result = "Unknown tool"


        print("\n==============================")
        print("Tool executed:", tool_name)
        print("Tool result:", result)
        print("==============================")


        # Create tool message

        tool_message = {

            "role": "tool",

            "tool_call_id": tool_call["id"],

            "name": tool_name,

            "content": str(result)

        }


        tool_messages.append(tool_message)


    return {

        "messages": state["messages"] + tool_messages

    }


# ============================================================
# 13. ROUTER
# ============================================================

def should_continue(state: State):

    last_message = state["messages"][-1]


    # Qwen requested a tool

    if "tool_calls" in last_message:

        return "tool"


    # Qwen produced final answer

    return "end"


# ============================================================
# 14. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)


graph.add_node(
    "llm",
    call_llm
)

graph.add_node(
    "tool",
    run_tool
)


# START → LLM

graph.add_edge(
    START,
    "llm"
)


# LLM → TOOL or END

graph.add_conditional_edges(

    "llm",

    should_continue,

    {

        "tool": "tool",

        "end": END

    }

)


# TOOL → LLM

graph.add_edge(
    "tool",
    "llm"
)


# Compile

app = graph.compile()


# ============================================================
# 15. RUN TEST
# ============================================================

question = input(
    "\nAsk the Green Lantern Agent: "
)


result = app.invoke({

    "messages": [

        {

            "role": "user",

            "content": question

        }

    ]

})


# ============================================================
# 16. FINAL ANSWER
# ============================================================

print("\n")
print("======================================")
print("FINAL ANSWER")
print("======================================")

print(
    result["messages"][-1]["content"]
)