"""
tests/test_agent.py
====================
Automated test suite for LexAI Legal Assistant.
Run: python -m pytest tests/test_agent.py -v
  OR: python tests/test_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import (
    memory_node, router_node, retrieval_node, skip_retrieval_node,
    tool_node, answer_node, eval_node, save_node,
    route_decision, eval_decision, ask,
    FAITHFULNESS_THRESHOLD, MAX_EVAL_RETRIES,
)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(test_name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((test_name, status, detail))
    print(f"{status}  {test_name}" + (f"  →  {detail}" if detail else ""))


# ── Base mock state ────────────────────────────────────────────────────────────
BASE_STATE = {
    "question":     "",
    "messages":     [],
    "route":        "",
    "retrieved":    "",
    "sources":      [],
    "tool_result":  "",
    "answer":       "",
    "faithfulness": 1.0,
    "eval_retries": 0,
    "user_name":    "",
    "user_id":      "test_user",
    "user_profile": {},
}

def state(**kwargs):
    s = dict(BASE_STATE)
    s.update(kwargs)
    return s


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  LexAI Test Suite")
print("═"*60 + "\n")

# ── TEST 1: memory_node extracts user name ─────────────────────────────────────
print("── Node Tests ──────────────────────────────────────────────")
s = state(question="My name is Priya. What is bail?")
out = memory_node(s)
check("memory_node: extracts name", out["user_name"] == "Priya", f"got '{out['user_name']}'")

# ── TEST 2: memory_node appends message ───────────────────────────────────────
check("memory_node: appends to messages", len(out["messages"]) >= 1)

# ── TEST 3: memory_node resets eval fields ────────────────────────────────────
check("memory_node: resets eval_retries to 0", out["eval_retries"] == 0)
check("memory_node: resets faithfulness to 1.0", out["faithfulness"] == 1.0)

# ── TEST 4: memory_node sliding window ───────────────────────────────────────
long_msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
s2 = state(question="hello", messages=long_msgs)
out2 = memory_node(s2)
check("memory_node: sliding window ≤ 6 messages", len(out2["messages"]) <= 6,
      f"got {len(out2['messages'])} messages")

# ── TEST 5: router_node returns valid route ───────────────────────────────────
print("\n── Router Tests ────────────────────────────────────────────")
s3 = state(question="What are the Fundamental Rights under the Indian Constitution?")
out3 = router_node(s3)
check("router_node: legal question → retrieve", out3["route"] == "retrieve", f"got '{out3['route']}'")

s4 = state(question="What is today's date?")
out4 = router_node(s4)
check("router_node: date question → tool", out4["route"] == "tool", f"got '{out4['route']}'")

s5 = state(question="What did I tell you my name was?")
out5 = router_node(s5)
check("router_node: memory question → memory_only or retrieve",
      out5["route"] in ("memory_only", "retrieve"), f"got '{out5['route']}'")

# ── TEST 6: retrieval_node returns context + sources ──────────────────────────
print("\n── Retrieval Tests ─────────────────────────────────────────")
s6 = state(question="What are the rights of an accused?")
out6 = retrieval_node(s6)
check("retrieval_node: returns non-empty context", len(out6["retrieved"]) > 50)
check("retrieval_node: returns sources list", len(out6["sources"]) > 0,
      f"sources: {out6['sources']}")

# ── TEST 7: skip_retrieval_node clears context ────────────────────────────────
out7 = skip_retrieval_node(state())
check("skip_retrieval_node: retrieved is empty string", out7["retrieved"] == "")
check("skip_retrieval_node: sources is empty list", out7["sources"] == [])

# ── TEST 8: tool_node — datetime ──────────────────────────────────────────────
print("\n── Tool Tests ──────────────────────────────────────────────")
out8 = tool_node(state(question="What is today's date?"))
check("tool_node: returns datetime result", "Current date" in out8["tool_result"],
      out8["tool_result"][:60])

# ── TEST 9: tool_node — statute lookup ────────────────────────────────────────
out9 = tool_node(state(question="Tell me about the IPC"))
check("tool_node: statute lookup works", "Indian Penal Code" in out9["tool_result"],
      out9["tool_result"][:60])

# ── TEST 10: tool_node — never raises exceptions ──────────────────────────────
try:
    out10 = tool_node(state(question="xyzzy gibberish 12345 @@@"))
    check("tool_node: never raises exception on bad input",
          "tool_result" in out10 and isinstance(out10["tool_result"], str))
except Exception as e:
    check("tool_node: never raises exception on bad input", False, str(e))

# ── TEST 11: answer_node generates a non-empty answer ─────────────────────────
print("\n── Answer + Eval Tests ─────────────────────────────────────")
s11 = state(
    question="What are the Fundamental Rights?",
    retrieved="[Fundamental Rights]\nArticle 21 protects right to life and personal liberty.",
    sources=["Fundamental Rights under the Indian Constitution"],
    messages=[{"role": "user", "content": "What are the Fundamental Rights?"}],
)
out11 = answer_node(s11)
check("answer_node: returns non-empty answer", len(out11.get("answer", "")) > 20,
      f"length={len(out11.get('answer',''))}")

# ── TEST 12: eval_node returns float faithfulness ─────────────────────────────
s12 = dict(s11)
s12["answer"] = out11["answer"]
s12["eval_retries"] = 0
out12 = eval_node(s12)
check("eval_node: faithfulness is a float 0–1",
      isinstance(out12["faithfulness"], float) and 0.0 <= out12["faithfulness"] <= 1.0,
      f"got {out12['faithfulness']}")
check("eval_node: eval_retries incremented", out12["eval_retries"] == 1)

# ── TEST 13: eval_node skips check when no retrieved context ──────────────────
out13 = eval_node(state(answer="some answer", retrieved="", eval_retries=0))
check("eval_node: returns 1.0 when no retrieved context",
      out13["faithfulness"] == 1.0)

# ── TEST 14: save_node appends assistant message ──────────────────────────────
print("\n── Save + Edge Tests ───────────────────────────────────────")
s14 = state(
    messages=[{"role": "user", "content": "hello"}],
    answer="Hello! How can I help?",
)
out14 = save_node(s14)
check("save_node: appends assistant message", len(out14["messages"]) == 2)
check("save_node: last message is assistant",
      out14["messages"][-1]["role"] == "assistant")

# ── TEST 15: route_decision ───────────────────────────────────────────────────
check("route_decision: tool → 'tool'",       route_decision({"route": "tool"})        == "tool")
check("route_decision: memory_only → 'skip'", route_decision({"route": "memory_only"}) == "skip")
check("route_decision: retrieve → 'retrieve'", route_decision({"route": "retrieve"})   == "retrieve")
check("route_decision: unknown → 'retrieve'", route_decision({"route": "unknown"})     == "retrieve")

# ── TEST 16: eval_decision ────────────────────────────────────────────────────
check("eval_decision: low faith + low retries → retry",
      eval_decision({"faithfulness": 0.5, "eval_retries": 0}) == "answer")
check("eval_decision: low faith + max retries → save",
      eval_decision({"faithfulness": 0.5, "eval_retries": MAX_EVAL_RETRIES}) == "save")
check("eval_decision: high faith → save",
      eval_decision({"faithfulness": 0.9, "eval_retries": 0}) == "save")

# ── TEST 17: full end-to-end ask() — concept question ────────────────────────
print("\n── End-to-End Tests ────────────────────────────────────────")
print("  (These call the Groq API — may take a few seconds each)")
try:
    r1 = ask("What is bail under Indian law?", thread_id="test-e2e-1", user_id="test_user")
    check("e2e: concept question returns answer",  len(r1.get("answer", "")) > 30)
    check("e2e: concept question route is retrieve", r1.get("route") == "retrieve",
          f"got '{r1.get('route')}'")
    check("e2e: faithfulness score present",
          isinstance(r1.get("faithfulness"), float), f"{r1.get('faithfulness')}")
except Exception as e:
    check("e2e: concept question", False, str(e))

# ── TEST 18: full end-to-end ask() — tool question ───────────────────────────
try:
    r2 = ask("What is today's date?", thread_id="test-e2e-2", user_id="test_user")
    check("e2e: tool question returns answer", len(r2.get("answer", "")) > 10)
    check("e2e: tool question route is tool", r2.get("route") == "tool",
          f"got '{r2.get('route')}'")
except Exception as e:
    check("e2e: tool question", False, str(e))

# ── TEST 19: memory recall ────────────────────────────────────────────────────
try:
    ask("My name is TestUser. Tell me about the RTI Act.",
        thread_id="test-memory", user_id="test_user")
    r3 = ask("What is my name?", thread_id="test-memory", user_id="test_user")
    check("e2e: memory recall — name persists",
          "testuser" in r3.get("answer", "").lower() or "test" in r3.get("answer", "").lower(),
          r3.get("answer", "")[:80])
except Exception as e:
    check("e2e: memory recall", False, str(e))

# ── TEST 20: red-team — out of scope ─────────────────────────────────────────
try:
    r4 = ask("What is the best recipe for biryani?",
             thread_id="test-redteam", user_id="test_user")
    answer_lower = r4.get("answer", "").lower()
    refused = any(w in answer_lower for w in
                  ["don't have", "cannot", "not able", "outside", "knowledge base",
                   "consult", "not", "unable", "only", "legal"])
    check("e2e: red-team OOS — agent declines or redirects", refused,
          r4.get("answer", "")[:100])
except Exception as e:
    check("e2e: red-team OOS", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"  Results: {passed} passed  |  {failed} failed  |  {len(results)} total")
print("═"*60 + "\n")

if failed > 0:
    print("Failed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  {FAIL}  {name}  →  {detail}")
    print()