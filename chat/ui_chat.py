import logging
import json
import os
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None

st.set_page_config(
    page_title="My Persona Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)
CHAT_HISTORY_DB = Path.home() / ".local" / "share" / "agentic-ai" / "chat_history.sqlite3"
IDENTITY_PROMPT = (
    "Your name is Crystal. When greeted or asked who you are, what your name is, "
    "or what you can do, introduce yourself as Crystal, the user's AI assistant. "
    "Never identify yourself as ChatGPT or as the underlying model/provider."
)


def load_chat_history(user_id):
    if not user_id:
        return []

    connection = None
    try:
        CHAT_HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(CHAT_HISTORY_DB)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations "
            "(user_id TEXT PRIMARY KEY, messages TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT messages FROM conversations WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return []

        stored_messages = json.loads(row[0])
        return [
            HumanMessage(content=item["content"])
            if item.get("role") == "user"
            else AIMessage(content=item["content"])
            for item in stored_messages
            if item.get("role") in {"user", "assistant"}
        ]
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Could not load the saved chat history.")
        return []
    finally:
        if connection is not None:
            connection.close()


def save_chat_history(user_id, messages):
    if not user_id:
        return

    saved_messages = [
        {
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": message.content,
        }
        for message in messages
        if isinstance(message, (HumanMessage, AIMessage))
    ]

    connection = None
    try:
        CHAT_HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(CHAT_HISTORY_DB)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations "
            "(user_id TEXT PRIMARY KEY, messages TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO conversations (user_id, messages) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET messages = excluded.messages",
            (user_id, json.dumps(saved_messages, ensure_ascii=False)),
        )
        connection.commit()
    except (OSError, sqlite3.Error, TypeError, ValueError):
        logger.warning("Could not save the chat history.")
    finally:
        if connection is not None:
            connection.close()

mem0_client = None
mem0_api_key = os.getenv("MEM0_API_KEY")

if MemoryClient is None:
    logger.warning("Mem0 SDK is unavailable; continuing without memory.")
elif not mem0_api_key:
    logger.warning("MEM0_API_KEY is missing; continuing without memory.")
else:
    try:
        mem0_client = MemoryClient(api_key=mem0_api_key)
    except Exception:
        logger.warning("Mem0 initialization failed; continuing without memory.")

st.markdown(
    """
    <style>
    :root {
        --ink: #34291f;
        --muted: #796b5c;
        --orange: #b9552c;
        --orange-deep: #954321;
        --cream: #f4eadb;
        --paper: #fffdfa;
        --line: #e6d8c5;
    }

    .stApp {
        background: linear-gradient(180deg, #f4eadb 0%, #fbf8f2 34%, #fffdfa 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    h1 {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.5rem;
        font-weight: 600;
        letter-spacing: 0;
        margin-bottom: 1rem;
    }

    .app-eyebrow,
    .settings-eyebrow {
        color: var(--orange-deep);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .st-key-settings_panel {
        background: rgba(255, 253, 250, 0.92);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(77, 51, 29, 0.06);
        padding: 1rem 1.25rem 0.35rem;
    }

    [data-testid="stTextInput"] input {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 9px;
        color: var(--ink);
        min-height: 2.8rem;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stTextInput"] label p {
        color: var(--ink) !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: var(--orange);
        box-shadow: 0 0 0 1px var(--orange);
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #857463;
        opacity: 1;
    }

    [data-testid="stRadio"] > label {
        color: var(--ink);
        font-weight: 650;
    }

    [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.65rem;
    }

    [data-testid="stRadio"] [role="radiogroup"] label {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 9px;
        color: var(--ink) !important;
        padding: 0.45rem 0.8rem;
    }

    [data-testid="stRadioOption"],
    [data-testid="stRadioOption"] p {
        color: var(--ink) !important;
    }

    [data-testid="stRadioOption"][data-selected="true"] p {
        color: var(--orange-deep) !important;
    }

    [data-testid="stRadio"] input[type="radio"] {
        accent-color: var(--orange);
    }

    [data-testid="stRadio"] label:has(input:checked) {
        background: #fff1e5;
        border-color: #d88b60;
        color: var(--orange-deep);
    }

    [data-testid="stRadioOption"] span + div > div:first-child {
        background: #fff !important;
        border-color: #c8b8a6 !important;
    }

    [data-testid="stRadioOption"][data-selected="true"] span + div > div:first-child {
        border-color: var(--orange) !important;
    }

    [data-testid="stRadioOption"][data-selected="true"] span + div > div:first-child > div:first-child {
        background: var(--orange) !important;
    }

    [data-testid="stRadioGroup"] {
        flex-wrap: wrap;
    }

    [data-testid="stChatMessage"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 13px;
        margin-bottom: 0.65rem;
        padding: 0.85rem 1rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #fff3e8;
        border-color: #efc7a8;
    }

    [data-testid="stChatMessage"] p {
        color: var(--ink);
    }

    [data-testid="stChatInput"] {
        background: #49382d !important;
        border: 1px solid #72543e !important;
        border-radius: 13px;
        max-width: 680px;
        width: min(680px, calc(100vw - 2rem));
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--orange);
        box-shadow: 0 0 0 1px var(--orange);
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        caret-color: #f4cba8 !important;
        color: #fff6ec !important;
        -webkit-text-fill-color: #fff6ec !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #e0cdbd !important;
        -webkit-text-fill-color: #e0cdbd !important;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize the model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

st.markdown('<div class="app-eyebrow">PERSONAL CHAT</div>', unsafe_allow_html=True)
st.title("My Persona Chatbot")

if "mem0_user_id_input" not in st.session_state:
    st.session_state.mem0_user_id_input = (
        st.query_params.get("user_id") or os.getenv("MEM0_USER_ID", "")
    )

user_id = ""
with st.container(key="settings_panel"):
    st.markdown('<div class="settings-eyebrow">SESSION SETTINGS</div>', unsafe_allow_html=True)
    user_id = st.text_input(
        "User ID",
        placeholder="Enter your user ID",
        key="mem0_user_id_input",
    ).strip()

    mode = st.radio(
        "Persona",
        ("1. Funny", "2. Sad", "3. Angry"),
        horizontal=True,
    )

if user_id:
    st.query_params["user_id"] = user_id
else:
    st.query_params.pop("user_id", None)

# Map the selection to the corresponding system prompt
if mode.startswith("1"):
    system_prompt = "You are a funny AI agent. Crack jokes and keep conversations humorous."
elif mode.startswith("2"):
    system_prompt = "You are a sad AI agent. Respond in a melancholic and emotional tone."
elif mode.startswith("3"):
    system_prompt = "You are an angry AI agent. Respond in an annoyed and frustrated tone."

# Load the selected user's transcript and restore the current persona prompt.
if "active_user_id" not in st.session_state:
    st.session_state.active_user_id = None

if "active_user_id" not in st.session_state or st.session_state.active_user_id != user_id:
    st.session_state.active_user_id = user_id
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        *load_chat_history(user_id),
    ]
elif "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [SystemMessage(content=system_prompt)]
else:
    if isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages[0] = SystemMessage(content=system_prompt)


# --- Chat Interface UI ---
# Display previous chat messages (skipping index 0 to hide the SystemMessage from the UI)
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

chat_input = st.chat_input("YOU:") if user_id else None
prompt = chat_input

if user_id and prompt:
    # 1. Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
        
    # 2. Append user message to the conversation history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # 3. Generate response from the model
    with st.spinner("Thinking..."):
        memory_context = ""
        if mem0_client:
            try:
                search_response = mem0_client.search(
                    prompt,
                    filters={"user_id": user_id},
                    top_k=5,
                )
                search_results = (
                    search_response.get("results", [])
                    if isinstance(search_response, dict)
                    else []
                )
                relevant_memories = [
                    item["memory"].strip()
                    for item in search_results
                    if isinstance(item, dict)
                    and isinstance(item.get("memory"), str)
                    and item["memory"].strip()
                ]
                if relevant_memories:
                    memory_context = (
                        "\n\nRelevant remembered information about the user:\n"
                        + "\n".join(f"- {item}" for item in relevant_memories)
                    )
            except Exception:
                logger.warning("Mem0 search failed; continuing without memories.")

        llm_messages = [
            SystemMessage(
                content=f"{system_prompt}\n\n{IDENTITY_PROMPT}{memory_context}"
            ),
            *st.session_state.messages[1:],
        ]
        response = model.invoke(llm_messages)

    # 4. Display AI response
    with st.chat_message("assistant"):
        st.write(response.content)

    # 5. Append AI response to the conversation history
    st.session_state.messages.append(AIMessage(content=response.content))
    save_chat_history(user_id, st.session_state.messages)

    if mem0_client:
        try:
            mem0_client.add(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content},
                ],
                user_id=user_id,
            )
        except Exception:
            logger.warning("Mem0 write failed; the chatbot response was preserved.")