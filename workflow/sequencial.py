import os
from typing import TypedDict

#state
class Project(TypedDict):
    raw_data : str
    edited_data : str
    scripted_data : str
    final_output : str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

def editor_node(state: Project) -> dict:
    """
    Stage 1: Cleans grammar, spelling, readability, and overall writing quality.
    """

    print("\n---[Stage 1] Executing Editor Node---")

    prompt = f"""
You are a world-class professional copy editor.

Your responsibility is to refine the given raw text while preserving the author's original meaning, intent, and factual accuracy.

### Your Tasks
- Correct all grammar mistakes.
- Fix spelling and punctuation errors.
- Improve sentence structure.
- Improve readability and clarity.
- Remove awkward phrasing.
- Improve logical flow between sentences.
- Eliminate unnecessary repetition.
- Make the writing sound natural and fluent.
- Maintain a consistent tone throughout.
- Preserve names, numbers, facts, quotations, and technical terms exactly unless they contain obvious typographical errors.
- Do NOT summarize.
- Do NOT shorten the content unnecessarily.
- Do NOT add new information.
- Do NOT change the intended message.
- Keep approximately the same length as the original.

### Output Requirements
- Return ONLY the edited text.
- Do not include explanations.
- Do not use markdown.
- Do not add headings.
- Do not include notes.

Raw Text:
{state["raw_data"]}
"""

    res = llm.invoke(prompt)

    return {
        "edited_data": res.content.strip()
    }


def scriptwriter_node(state: Project) -> dict:
    """
    Stage 2: Converts the edited text into an engaging video script.
    """

    print("\n---[Stage 2] Executing Scriptwriter Node---")

    prompt = f"""
You are an expert YouTube and short-form video scriptwriter.

Your task is to transform the following edited text into a highly engaging,
natural-sounding video script.

Instructions:
- Keep the original meaning completely intact.
- Make the script conversational and easy to listen to.
- Begin with a strong hook that grabs attention.
- Organize the content logically.
- Use smooth transitions between ideas.
- Keep the audience engaged throughout.
- End with a memorable conclusion.
- Do NOT add information that is not present in the text.
- Return ONLY the script.

Edited Text:
{state["edited_data"]}
"""

    res = llm.invoke(prompt)

    return {
        "scripted_data": res.content.strip()
    }


def translator_node(state: Project) -> dict:
    """
    Stage 3: Translates the script into natural spoken Hinglish.
    """

    print("\n---[Stage 3] Executing Translator Node---")

    prompt = f"""
You are an expert translator, script localizer, and Hindi content writer.

Your task is to convert the following English video script into highly natural, conversational Hinglish that sounds like a professional YouTuber or storyteller speaking directly to an Indian audience.

### Guidelines

- Preserve the complete meaning of every sentence.
- Do NOT summarize.
- Do NOT omit any information.
- Do NOT add new information.
- Maintain the same sequence of ideas.
- Keep the storytelling flow smooth.
- Make the dialogue sound natural when spoken aloud.
- Use everyday Hindi mixed naturally with English.
- Keep technical terms in English whenever appropriate.
- Translate emotions, not just words.
- Avoid literal word-for-word translation.
- Avoid overly formal Hindi.
- Avoid robotic language.
- Ensure smooth transitions between sentences.
- Preserve the hook and ending.
- The output should sound like a real human narrator speaking naturally.

### Style

Imagine explaining the topic to a friend on YouTube.

Example:

Instead of:
"Yeh ek bahut hi mahatvapurna vishay hai."

Write:
"Ye topic kaafi interesting hai aur iske baare mein jaana zaroori hai."

Instead of:
"Hum iske baare mein charcha karenge."

Write:
"Chaliye dekhte hain ki actually ye kaam kaise karta hai."

### Output Requirements

- Return ONLY the translated Hinglish script.
- No markdown.
- No explanations.
- No notes.
- No headings.

English Script:

{state["scripted_data"]}
"""

    res = llm.invoke(prompt)

    return {
        "final_output": res.content.strip()
    }

from langgraph.graph import StateGraph,START,END

graph = StateGraph(Project)

graph.add_node("editor", editor_node)
graph.add_node("scripter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scripter")
graph.add_edge("scripter", "translator")
graph.add_edge("translator",END)

app = graph.compile()

result = app.invoke({
    "raw_data" : "I want to get into sscbs college ,even my cuet scores are very good but i am doubtful if i will get the college or not."
})

print("Your result is: \n\n")

print(result['final_output'])