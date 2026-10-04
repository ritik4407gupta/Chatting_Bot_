import os

from dotenv import load_dotenv
from mem0 import MemoryClient
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
mem0_api_key = os.getenv("MEM0_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not mem0_api_key:
    raise ValueError("MEM0_API_KEY is missing from .env")


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

memory = MemoryClient(
    api_key=mem0_api_key
)


PERSONALITIES = {
    "1": "You are a friendly and humorous AI assistant. Keep conversations light and engaging when appropriate.",
    "2": "You are a calm and empathetic AI assistant. Respond with warmth, patience, and emotional awareness.",
    "3": "You are a direct and energetic AI assistant. Be confident, sharp, and expressive without being disrespectful."
}


print("\n==============================")
print("       AI MEMORY CHATBOT")
print("==============================\n")

try:
    user_id = input("Enter your user ID(can be any number ,futher u can login to that only\nOtherwise it wont treat u like it knows u ): ").strip()
except KeyboardInterrupt:
    print("\n\nChatbot stopped.")
    raise SystemExit(0)

if not user_id:
    print("User ID cannot be empty.")
    exit()

print("\nChoose your assistant style:")
print("1. Funny")
print("2. Calm")
print("3. Energetic")

mode = input("\nEnter the digit: ").strip()

if mode not in PERSONALITIES:
    print("Invalid option.")
    exit()

system_prompt = PERSONALITIES[mode]

conversation = []

MAX_HISTORY = 12

print("\n--------------------------------")
print("Chat started")
print("Type 'exit' to close the chatbot")
print("--------------------------------\n")


while True:
    try:
        user_input = input("YOU: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        memory_response = memory.search(
            user_input,
            filters={
                "user_id": user_id
            }
        )

        if isinstance(memory_response, dict):
            memory_results = memory_response.get("results", [])
        else:
            memory_results = memory_response

        relevant_memories = []

        for item in memory_results:
            if isinstance(item, dict):
                memory_text = item.get("memory")

                if memory_text:
                    relevant_memories.append(memory_text)

        memory_context = "\n".join(
            f"- {item}"
            for item in relevant_memories[:5]
        )

        system_content = f"""
{system_prompt}

You are a personalized AI assistant.

Use the user's remembered information when it is relevant to the current request.

Do not mention the memory system, database, retrieval process, or internal context to the user.

Do not assume that every stored memory is relevant.

If the user gives newer information that conflicts with an older memory, prefer the newer information.

Answer naturally, clearly, and helpfully.

When user asks Hello or Hi, respond with a friendly greeting and ask how you can assist them today.

when user asks for your name, respond with "I am your AI assistant,you can call me Crystal. How can I help you today?"
"""

        if memory_context:
            system_content += f"""

Relevant remembered information:

{memory_context}
"""

        llm_messages = [
            SystemMessage(content=system_content)
        ]

        llm_messages.extend(
            conversation[-MAX_HISTORY:]
        )

        llm_messages.append(
            HumanMessage(content=user_input)
        )

        response = model.invoke(llm_messages)

        assistant_response = response.content

        conversation.append(
            HumanMessage(content=user_input)
        )

        conversation.append(
            AIMessage(content=assistant_response)
        )

        memory.add(
            [
                {
                    "role": "user",
                    "content": user_input
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            user_id=user_id
        )

        print(f"\nAI: {assistant_response}\n")

    except KeyboardInterrupt:
        print("\n\nChatbot stopped.")
        break

    except Exception as e:
        print(f"\nChatbot Stopped.\n")


print("\nChat ended.")