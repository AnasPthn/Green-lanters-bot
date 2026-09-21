# 🟢 Green Lantern RAG Bot

An Agentic RAG chatbot built with **Python, LangGraph, Hugging Face Qwen, ChromaDB, and embeddings**.

The project allows an AI agent to search a Green Lantern knowledge base when necessary and use a calculator tool for mathematical questions.

## 🚀 Features

* 🧠 Qwen 2.5 7B Instruct LLM
* 🔎 Retrieval-Augmented Generation (RAG)
* 🗄️ Chroma vector database
* 🔢 Hugging Face sentence-transformer embeddings
* 🛠️ Custom tool calling
* 🧮 Calculator tool
* 🟢 Green Lantern knowledge search tool
* 🔄 LangGraph agent loop
* 🔀 Conditional routing
* 🤖 Automatic tool selection by the LLM

## 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │    QWEN     │
                    │     LLM     │
                    └──────┬──────┘
                           │
                    Tool required?
                     /            \
                   YES             NO
                    │               │
                    ▼               ▼
             ┌──────────────┐    ANSWER
             │  TOOL NODE   │
             └──────┬───────┘
                    │
             ┌──────┴──────────┐
             │                 │
             ▼                 ▼
      Green Lantern        Calculator
        Search Tool           Tool
             │                 │
             ▼                 ▼
          Chroma              Math
             │                 │
             └────────┬────────┘
                      ▼
                     QWEN
                      │
                      ▼
                   ANSWER
```

## 📚 How RAG Works

The Green Lantern knowledge file is converted into smaller chunks.

```text
knowledge.txt
      ↓
Document Loader
      ↓
Text Splitter
      ↓
Text Chunks
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Retriever
```

When the agent needs Green Lantern information:

```text
User Question
      ↓
Qwen
      ↓
search_green_lantern()
      ↓
ChromaDB
      ↓
Relevant Information
      ↓
Qwen
      ↓
Final Answer
```

## 🛠️ Technologies

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Programming language                |
| LangGraph             | Agent workflow and state management |
| Hugging Face          | LLM inference                       |
| Qwen 2.5 7B           | Language model                      |
| ChromaDB              | Vector database                     |
| Sentence Transformers | Text embeddings                     |
| LangChain             | Document loading and retrieval      |

## 📁 Project Structure

```text
Green-lanters-bot/
│
├── knowledge.txt
│
├── green_lantern_agent.py
│
├── README.md
│
└── ...
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnasPthn/Green-lanters-bot.git
```

### 2. Enter the project

```bash
cd Green-lanters-bot
```

### 3. Install dependencies

```bash
pip install langgraph langchain langchain-community langchain-huggingface langchain-chroma huggingface-hub sentence-transformers
```

### 4. Configure Hugging Face

The application expects the Hugging Face token to be available through the environment variable:

```text
HF_TOKEN
```

The token should **never be hardcoded into the source code**.

## ▶️ Run

From the project directory:

```powershell
python green_lantern_agent.py
```

The agent will ask:

```text
Ask the Green Lantern Agent:
```

Example:

```text
Who is John Stewart?
```

The agent can use the Green Lantern search tool and retrieve relevant information from ChromaDB.

Another example:

```text
What is 25 + 40?
```

The agent can select the calculator tool.

## 🧪 Example Flow

### Green Lantern question

```text
User:
Who is Hal Jordan?

        ↓

Qwen

        ↓

search_green_lantern()

        ↓

ChromaDB

        ↓

Retrieved Context

        ↓

Qwen

        ↓

Final Answer
```

### Mathematical question

```text
User:
What is 25 + 40?

        ↓

Qwen

        ↓

calculator()

        ↓

65

        ↓

Qwen

        ↓

Final Answer
```

## 🧠 What I Learned From This Project

This project was built while learning LangGraph and Agentic AI concepts.

Key concepts practiced:

* Graph fundamentals
* State
* Nodes and edges
* Conditional routing
* LLM integration
* Tool calling
* Agent loops
* ReAct-style workflows
* Retrieval-Augmented Generation
* Vector databases
* Embeddings
* Agentic RAG
* Multiple tools
* LangGraph state transitions

## 🔄 Agent Loop

The core agent loop is:

```text
START
  ↓
LLM
  ↓
Tool required?
  ├── NO → END
  │
  └── YES
       ↓
     TOOL
       ↓
      LLM
       ↓
Tool required?
  ├── YES → TOOL
  │
  └── NO → END
```

This allows the agent to repeatedly reason and use tools until it can produce a final answer.

## 🔮 Future Improvements

Planned improvements include:

* Better retrieval quality
* More Green Lantern source documents
* Web search
* Conversation memory
* Persistent checkpointing
* Human-in-the-loop approval
* Adaptive RAG
* Research Agent
* Multi-agent architecture
* Production deployment

## 👨‍💻 Author

**Anas Pathan**

MCA Student | AI & Agentic AI Developer

Currently learning and building with:

```text
Python
LangChain
LangGraph
RAG
LLMs
Agentic AI
```

## ⭐ Project

If you find this project useful, consider giving the repository a star.
