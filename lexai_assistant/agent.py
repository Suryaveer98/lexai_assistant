
import os
import json
import re
import math
from datetime import datetime, date
from typing import TypedDict, List, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

GROQ_MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=1024,
)

from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("lexai_kb")

from knowledge_base.legal_docs import DOCUMENTS

_ids   = [d["id"]   for d in DOCUMENTS]
_texts = [d["text"] for d in DOCUMENTS]
_metas = [{"topic": d["topic"]} for d in DOCUMENTS]
_embs  = embedder.encode(_texts).tolist()
collection.add(documents=_texts, embeddings=_embs, ids=_ids, metadatas=_metas)
print(f"[LexAI] Knowledge base loaded: {len(DOCUMENTS)} documents.")


_probe = "What are the rights of an accused in a criminal trial?"
_probe_emb = embedder.encode([_probe]).tolist()
_probe_res  = collection.query(query_embeddings=_probe_emb, n_results=1)
print(f"[LexAI] Retrieval smoke test → {_probe_res['metadatas'][0][0]['topic']}")


PROFILE_DIR = Path("user_profiles")
PROFILE_DIR.mkdir(exist_ok=True)

def load_profile(user_id: str) -> dict:
    path = PROFILE_DIR / f"{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"user_id": user_id, "name": "", "queries": 0, "topics_asked": []}

def save_profile(user_id: str, profile: dict):
    path = PROFILE_DIR / f"{user_id}.json"
    path.write_text(json.dumps(profile, indent=2))


class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    user_name:    str
    user_id:      str
    user_profile: dict


FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2
SLIDING_WINDOW   = 6   # messages to keep in prompt context


def memory_node(state: CapstoneState) -> dict:
    msgs = list(state.get("messages", []))
    user_name = state.get("user_name", "")
    user_id   = state.get("user_id", "anon")

    profile = load_profile(user_id)
    if not user_name and profile.get("name"):
        user_name = profile["name"]

    match = re.search(r"my name is ([A-Za-z]+)", state["question"], re.IGNORECASE)
    if match:
        user_name = match.group(1).capitalize()
        profile["name"] = user_name
        save_profile(user_id, profile)

    msgs.append({"role": "user", "content": state["question"]})

    msgs = msgs[-SLIDING_WINDOW:]

    profile["queries"] = profile.get("queries", 0) + 1
    save_profile(user_id, profile)

    return {
        "messages":     msgs,
        "user_name":    user_name,
        "user_profile": profile,
        "eval_retries": 0,
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "faithfulness": 1.0,
    }


# ─── Node 2: router_node ──────────────────────────────────────────────────────
ROUTER_PROMPT = """\
You are a router for a Legal AI assistant. Your job is to decide which route handles the user question best.

Routes:
- retrieve   : The question asks about legal concepts, rights, procedures, acts, case law, or
               anything the legal knowledge base covers.
- tool       : The question asks for today's date, current time, a specific statute number lookup,
               or any live/current information that the KB cannot answer.
- memory_only: The question is purely conversational (greetings, asks the user's own name back,
               follow-ups that need no new legal knowledge).

Reply with EXACTLY ONE WORD: retrieve, tool, or memory_only
"""

def router_node(state: CapstoneState) -> dict:
    prompt = f"{ROUTER_PROMPT}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    route = response.content.strip().lower().split()[0]
    if route not in ("retrieve", "tool", "memory_only"):
        route = "retrieve"
    return {"route": route}


# ─── Node 3: retrieval_node ───────────────────────────────────────────────────
def retrieval_node(state: CapstoneState) -> dict:
    emb = embedder.encode([state["question"]]).tolist()
    res = collection.query(query_embeddings=emb, n_results=3)

    chunks  = res["documents"][0]
    topics  = [m["topic"] for m in res["metadatas"][0]]
    context = "\n\n".join(f"[{t}]\n{c}" for t, c in zip(topics, chunks))

    # Track topic
    profile = state.get("user_profile", {})
    for t in topics:
        asked = profile.get("topics_asked", [])
        if t not in asked:
            asked.append(t)
            profile["topics_asked"] = asked
    save_profile(state.get("user_id", "anon"), profile)

    return {
        "retrieved": context,
        "sources":   topics,
        "user_profile": profile,
    }


# ─── Node 4: skip_retrieval_node ──────────────────────────────────────────────
def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


# ─── Node 5: tool_node ────────────────────────────────────────────────────────
STATUTES = {
    "ipc": "Indian Penal Code, 1860 — the main criminal code of India.",
    "crpc": "Code of Criminal Procedure, 1973 — procedural law for criminal matters in India.",
    "cpc": "Code of Civil Procedure, 1908 — procedural law for civil suits in India.",
    "constitution": "Constitution of India, 1950 — the supreme law of India.",
    "contract act": "Indian Contract Act, 1872 — governs contract law in India.",
    "evidence act": "Indian Evidence Act, 1872 — rules of evidence in Indian courts.",
    "it act": "Information Technology Act, 2000 — covers cyber crimes and e-commerce in India.",
    "pocso": "Protection of Children from Sexual Offences Act, 2012.",
    "motor vehicles act": "Motor Vehicles Act, 1988 — governs road transport.",
    "rte": "Right to Education Act, 2009 — free and compulsory education for children 6-14.",
}

def tool_node(state: CapstoneState) -> dict:
    """Handles datetime and statute-lookup queries. Never raises exceptions."""
    try:
        q = state["question"].lower()

        # Datetime
        if any(w in q for w in ("today", "date", "time", "day", "year", "month")):
            now = datetime.now()
            result = (
                f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
                f"Current time: {now.strftime('%I:%M %p')}\n"
                f"Day of week: {now.strftime('%A')}"
            )
            return {"tool_result": result}

        # Statute lookup
        for key, desc in STATUTES.items():
            if key in q:
                return {"tool_result": f"Statute: {desc}"}

        # Fallback
        return {"tool_result": "No specific tool result available for this query. Please check a legal database."}

    except Exception as e:
        return {"tool_result": f"Tool error: {str(e)} — please try rephrasing."}


# ─── Node 6: answer_node ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are LexAI, an intelligent legal assistant for paralegals and junior lawyers in India.

STRICT RULES:
1. Answer ONLY using the provided Context and Tool Result. Do not fabricate legal information.
2. If the Context and Tool Result do not contain enough information to answer, say:
   "I don't have specific information on that in my knowledge base. Please consult a licensed advocate or refer to official legal databases (e.g., Manupatra, SCC Online)."
3. Never give medical, financial, or personal advice unrelated to legal matters.
4. If the question involves an emergency legal situation (arrest, custody), immediately provide:
   National Legal Services Authority helpline: 15100.
5. Cite which legal act, section, or concept your answer is based on, when possible.
6. Keep answers clear, structured, and professional.
7. Address the user by name if known.
{retry_instruction}
"""

def answer_node(state: CapstoneState) -> dict:
    name = state.get("user_name", "")
    greeting = f"The user's name is {name}. Address them as {name}." if name else ""

    retry_instruction = ""
    if state.get("eval_retries", 0) > 0:
        retry_instruction = (
            "\n⚠️ Previous answer scored below faithfulness threshold. "
            "Be MORE grounded — cite context explicitly. "
            "If context is insufficient, admit it rather than guessing."
        )

    system = SYSTEM_PROMPT.format(retry_instruction=retry_instruction)
    if greeting:
        system += f"\n{greeting}"

    history = "\n".join(
        f"{'User' if m['role']=='user' else 'LexAI'}: {m['content']}"
        for m in state.get("messages", [])[-4:]
    )

    context_section = ""
    if state.get("retrieved"):
        context_section = f"\n\n=== KNOWLEDGE BASE CONTEXT ===\n{state['retrieved']}"
    if state.get("tool_result"):
        context_section += f"\n\n=== TOOL RESULT ===\n{state['tool_result']}"

    full_prompt = (
        f"{system}\n\n"
        f"=== CONVERSATION HISTORY ===\n{history}\n"
        f"{context_section}\n\n"
        f"=== CURRENT QUESTION ===\n{state['question']}\n\n"
        f"Answer:"
    )

    response = llm.invoke(full_prompt)
    return {"answer": response.content.strip()}


# ─── Node 7: eval_node ────────────────────────────────────────────────────────
EVAL_PROMPT = """\
You are a legal QA evaluator. Rate the faithfulness of the Answer to the Context on a scale of 0.0 to 1.0.

Faithfulness = does the answer rely ONLY on the context, without adding unsupported legal claims?

Context:
{context}

Question: {question}
Answer: {answer}

Rules:
- 1.0 = every claim in the answer is directly supported by context.
- 0.7–0.9 = mostly supported, minor extrapolation.
- 0.4–0.6 = some claims unsupported or vague.
- 0.0–0.3 = fabricated or contradicts context.
- If context is empty (tool or memory-only), return 1.0 automatically.

Reply with ONLY a float between 0.0 and 1.0. Nothing else.
"""

def eval_node(state: CapstoneState) -> dict:
    # Skip eval for tool and memory-only routes
    if not state.get("retrieved"):
        return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0)}

    prompt = EVAL_PROMPT.format(
        context=state.get("retrieved", "")[:1500],
        question=state["question"],
        answer=state.get("answer", ""),
    )
    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip().split()[0])
        score = max(0.0, min(1.0, score))
    except:
        score = 0.8

    retries = state.get("eval_retries", 0) + 1
    print(f"[eval_node] faithfulness={score:.2f} | retries={retries}")
    return {"faithfulness": score, "eval_retries": retries}


# ─── Node 8: save_node ────────────────────────────────────────────────────────
def save_node(state: CapstoneState) -> dict:
    msgs = list(state.get("messages", []))
    msgs.append({"role": "assistant", "content": state.get("answer", "")})
    return {"messages": msgs}


# ─── Conditional Edge Functions ──────────────────────────────────────────────
def route_decision(state: CapstoneState) -> str:
    r = state.get("route", "retrieve")
    if r == "tool":
        return "tool"
    if r == "memory_only":
        return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    faith   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if faith < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
        return "answer"   # retry
    return "save"


# ─── Graph Assembly ───────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(CapstoneState)
graph.add_node("memory",   memory_node)
graph.add_node("router",   router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip",     skip_retrieval_node)
graph.add_node("tool",     tool_node)
graph.add_node("answer",   answer_node)
graph.add_node("eval",     eval_node)
graph.add_node("save",     save_node)

graph.set_entry_point("memory")
graph.add_edge("memory", "router")
graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "skip":     "skip",
    "tool":     "tool",
})
graph.add_edge("retrieve", "answer")
graph.add_edge("skip",     "answer")
graph.add_edge("tool",     "answer")
graph.add_edge("answer",   "eval")
graph.add_conditional_edges("eval", eval_decision, {
    "answer": "answer",
    "save":   "save",
})
graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())
print("[LexAI] Graph compiled successfully.")


# ─── Public ask() helper ─────────────────────────────────────────────────────
def ask(question: str, thread_id: str = "default", user_id: str = "anon") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "question":     question,
            "messages":     [],
            "route":        "",
            "retrieved":    "",
            "sources":      [],
            "tool_result":  "",
            "answer":       "",
            "faithfulness": 1.0,
            "eval_retries": 0,
            "user_name":    "",
            "user_id":      user_id,
            "user_profile": {},
        },
        config=config,
    )
    return result
