# Agentic AI

A practical collection of Agentic AI experiments and implementations using **LangChain, LangGraph, Streamlit, RAG, Mem0, Chroma, Groq, and Mistral embeddings**.

This repository explores how modern LLM applications can be designed using workflows, retrieval systems, persistent memory, and multi-step agent architectures.

---

## 🚀 What This Project Contains

The repository is organized into three major areas:

### 1. Personalized AI Chatbot

The `chat/` directory contains a personalized chatbot implementation with:

- Groq-powered LLM
- Multiple AI personas
- Streamlit chat interface
- Mem0 long-term memory
- User-specific memory retrieval
- Persistent local chat history using SQLite
- Recent conversation context
- Personalized responses based on previously stored information

The chatbot can remember information associated with a specific user ID and retrieve relevant memories when answering future questions.

---

### 2. Document RAG System

The `workflow/conditional_rag.py` file demonstrates a conditional Retrieval-Augmented Generation workflow.

The system works with two PDF documents:

- `academics_handbook.pdf`
- `fee_structure.pdf`

It uses:

- `PyPDFLoader` for document loading
- `RecursiveCharacterTextSplitter` for chunking
- `MistralAIEmbeddings` for embeddings
- `Chroma` as the vector database
- `LangGraph` for workflow orchestration
- `Groq` for classification and response generation

The workflow first classifies a question and then routes it to the appropriate retrieval path.

Example routing:

```text
User Query
    ↓
Classifier
    ↓
 ┌───────────────┬───────────────┐
 ↓               ↓               ↓
Academic         Fee           General
 ↓               ↓               ↓
Academic RAG     Fee RAG       Direct LLM
 └───────────────┴───────────────┘
                 ↓
             Final Response
```

---

### 3. LangGraph Workflows

The `workflow/` directory also contains examples of different workflow patterns.

#### Sequential Workflow

`sequencial.py`

Demonstrates a multi-stage sequential pipeline:

```text
Raw Text
   ↓
Editor
   ↓
Scriptwriter
   ↓
Hinglish Translator
   ↓
Final Output
```

Each stage receives the output of the previous stage and performs a specific transformation.

---

#### Parallel Workflow

`parallel.py`

Demonstrates parallel execution using LangGraph.

A single piece of text is analyzed simultaneously for:

- Toxicity
- Copyright risk
- Cultural sensitivity

The results are combined into a shared safety score.

```text
                 Input
                   ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
    Toxicity   Copyright   Culture
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
             Safety Scores
```

---

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core development |
| LangChain | LLM application development |
| LangGraph | Agent/workflow orchestration |
| Groq | Fast LLM inference |
| Mem0 | Long-term user memory |
| Streamlit | Chatbot UI |
| Mistral AI | Embeddings |
| Chroma | Vector database |
| SQLite | Local chat history |
| PyPDFLoader | PDF document loading |
| dotenv | Environment variable management |

---

## 📁 Project Structure

```text
Agentic-AI/
│
├── academics_handbook.pdf
├── fee_structure.pdf
│
├── chat/
│   ├── chatmodel.py
│   └── ui_chat.py
│
├── workflow/
│   ├── conditional_rag.py
│   ├── parallel.py
│   └── sequencial.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

> `venv/` and `__pycache__/` should not be committed to GitHub.

---

## 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
MEM0_API_KEY=your_mem0_api_key
```

Never commit the `.env` file or expose API keys publicly.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ritik4407gupta/Chatting_Bot_.git
cd Agentic-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit Chatbot

Run:

```bash
streamlit run chat/ui_chat.py
```

The application provides a conversational interface with:

- Funny persona
- Sad persona
- Angry persona
- Mem0-powered long-term memory
- Persistent local conversation history

---

## 💾 How Memory Works

The chatbot uses two different kinds of memory.

### Short-Term Conversation Memory

The current conversation is maintained during the session and recent messages are passed to the LLM.

### Long-Term Memory

Mem0 stores meaningful user/assistant interactions.

When a new message is received:

```text
User Message
     ↓
Mem0 Search
     ↓
Relevant Memories
     ↓
Persona + Memories + Recent Conversation
     ↓
LLM
     ↓
Assistant Response
     ↓
Mem0
     ↓
Stored Memory
```

This allows the chatbot to provide personalized responses across conversations.

---

## 🔎 RAG Architecture

The conditional RAG workflow creates separate retrievers for the academic handbook and fee structure documents.

```text
PDF Documents
      ↓
Document Loading
      ↓
Text Chunking
      ↓
Mistral Embeddings
      ↓
Chroma Vector Store
      ↓
Retriever
      ↓
LangGraph Router
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Final Answer
```

---

## 🧩 Agentic Workflow Patterns

This repository demonstrates three important workflow patterns:

### Sequential

Tasks execute one after another.

```text
Task A → Task B → Task C
```

### Parallel

Independent tasks execute simultaneously.

```text
          → Task A →
Input →   → Task B →   → Results
          → Task C →
```

### Conditional

The next step depends on the result of a classifier/router.

```text
Input
  ↓
Decision
 ├── Path A
 ├── Path B
 └── Path C
```

These patterns are useful when designing more advanced AI agents and multi-step LLM systems.

---

## 🎯 Learning Goals

This repository is intended to explore:

- LLM application development
- Prompt engineering
- LangChain
- LangGraph
- Agent workflows
- Retrieval-Augmented Generation
- Vector databases
- Embeddings
- Persistent AI memory
- Personalized AI assistants
- Sequential and parallel execution
- Conditional routing
- Streamlit-based AI interfaces

---

## 🔮 Future Improvements

Possible future extensions include:

- Multi-agent architectures
- Tool calling
- Web search agents
- Agent planning and reasoning
- Better memory management
- Authentication and user management
- Cloud vector databases
- Streaming responses
- Evaluation pipelines
- Production deployment
- Observability and tracing

---

## 👨‍💻 Author

**Ritik Gupta**

Computer Science Engineering

Interested in:

- AI & Generative AI
- Agentic AI
- Machine Learning
- Full-Stack Development
- LLM Applications

---

## ⭐ Project

This repository is a collection of hands-on implementations built while exploring modern AI engineering concepts and agent architectures.

If you find the project useful, consider giving the repository a ⭐.
