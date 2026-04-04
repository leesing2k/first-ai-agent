from openai import OpenAI
import json
import numpy as np
import re

vector_store = []

MAX_HISTORY = 10          # keep last N messages
SUMMARY_THRESHOLD = 20    # when to summarize
max_reflection_loops = 3
MAX_MEMORY = 1000

client = OpenAI()

# -------- Tool --------
tools = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Use this to perform arithmetic operations like add, subtract, multiply. Useful for multi-step calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply"]
                },
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "type": "function",
        "name": "explain",
        "description": "Explain a result in simple terms for the user",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }
    }
]

def create_plan(goal):
    plan_response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a planner.

Rules:
- If the task is simple (can be solved in 1–2 steps), DO NOT create a long plan
- Keep plans minimal and practical
- Do NOT overthink

Format:
1. ...
2. ...
"""
            },
            {"role": "user", "content": goal}
        ]
    )

    return plan_response.output_text

def reflect(goal, answer):
    reflection = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a strict evaluator.

Rules:
- ONLY use facts explicitly present in:
    a. the conversation
    b. retrieved memory (if provided)
- Do NOT claim missing evidence if it exists
- If the answer matches known facts, mark COMPLETE

Reply ONLY in this format:

STATUS: COMPLETE or INCOMPLETE
REASON: short explanation
"""
            },
            {
                "role": "user",
                "content": f"""
GOAL:
{goal}

ANSWER:
{answer}
"""
            }
        ]
    )

    return reflection.output_text

def print_vector_store():
    print("\n===== VECTOR STORE =====")
    for i, item in enumerate(vector_store):
        print(f"{i}: {item['text'][:80]}...")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def fast_importance(fact):
    score = 5

    if len(fact) > 40:
        score += 2
    if any(k in fact.lower() for k in ["error", "bug", "fail"]):
        score += 2
    if any(k in fact.lower() for k in ["user", "goal", "plan"]):
        score += 1

    return min(score, 10)

def score_importance(fact):
    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Rate importance of this memory (1-10):

{fact}

Return ONLY a number.
"""
    )

    text = response.output_text.strip()
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    return 5  # fallback

def keyword_score(query, text):
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    return len(q_words & t_words) / (len(q_words) + 1e-5)

def retrieve_memory(query, top_k=5):
    query_emb = get_embedding(query)

    scored = []
    for item in vector_store:
        if item.get("importance", 0) < 6:
            continue
        semantic = cosine_similarity(query_emb, item["embedding"])
        keyword = keyword_score(query, item["text"])
        recency = 1 / (1 + (len(vector_store) - item["timestamp"]))

        score = (
            0.7 * semantic +
            0.2 * keyword +
            0.1 * recency
        )

        scored.append((score, item["text"]))

    scored.sort(reverse=True)

    return [text for _, text in scored[:top_k]]

def rerank(query, candidates):
    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Select the 3 most relevant memories for answering the query.

Prioritize:
- direct relevance
- specificity
- usefulness for reasoning

Query:
{query}

Candidates:
{candidates}

Return ONLY the selected items, one per line.
"""
    )

    lines = response.output_text.strip().split("\n")
    return [l.strip("- ").strip() for l in lines if l.strip()]

def rewrite_query(query):
    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Rewrite this into an optimized retrieval query.

Rules:
- Expand vague terms
- Add missing technical context
- Keep concise

Query:
{query}
"""
    )
    return response.output_text.strip()

def extract_facts(text, role="user"):
    subject_rule = (
        "Start every sentence with 'The user ...'"
        if role == "user"
        else "Start every sentence with 'The assistant replied...'"
    )
    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Extract useful long-term facts from the text.

Rules:
- Convert facts into FULL natural sentences
- {subject_rule}
- Make facts self-contained (no labels like "Name:")
- Keep them clear and specific
- Max 5 facts
- DO NOT extract procedural steps or actions
- DO NOT include calculations or temporary results
- ONLY extract stable, long-term facts about the user or environment

Good examples:
- The user's name is John.
- The user was born on 23 June 1980.
- The assistant calculated the result as 1533.

Bad examples:
- Name: John
- Birthday: 23 June
- John is the name

Text:
{text}

Output format:
- sentence 1
- sentence 2

If no useful facts, return NOTHING.
"""
    )

    facts = response.output_text.strip().split("\n")

    cleaned = []
    for f in facts:
        f = f.replace("- ", "").strip()

        if not f:
            continue
        if len(f) < 10:
            continue

        cleaned.append(f)

    return cleaned

def explain(text):
    return f"Explanation: {text}"

def calculator(operation, a, b):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b

def summarize_conversation(conversation):
    # Only keep human-readable messages
    filtered = [
        msg for msg in conversation
        if msg.get("role") in ["user", "assistant"]
    ]

    summary_response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
            Summarize the conversation focusing on:
            - key facts
            - user preferences
            - important context

            Conversation:
            {filtered}
            """
    )
    return summary_response.output_text

def optimize_memory(conversation, last_summary):
    # Trigger summary
    if len(conversation) > SUMMARY_THRESHOLD and last_summary is None:
        last_summary = summarize_conversation(conversation)

    system_msgs = [m for m in conversation if m["role"] == "system"]
    others = [m for m in conversation if m["role"] != "system"]

    # ✅ DEFINE optimized FIRST
    optimized = system_msgs[:1]

    # ✅ Always keep GOAL and PLAN
    important = []
    for msg in others:
        if (
            msg.get("tag") == "goal" 
            or msg.get("tag") == "plan"
        ):
            important.append(msg)

    # Recent messages
    recent = others[-MAX_HISTORY:]

    # Merge (avoid duplicates)
    seen = set()
    merged = []

    for msg in important + recent:
        key = (msg.get("content"), msg.get("tag"))
        if key not in seen:
            merged.append(msg)
            seen.add(key)

    optimized += merged

    return optimized, last_summary

def clean_conversation(conv):
    cleaned = []

    for msg in conv:
        role = msg.get("role")

        # Keep only allowed fields
        if role in ["system", "user", "assistant"] and "content" in msg:
            cleaned.append({
                "role": role,
                "content": msg["content"]
            })

        elif role == "tool":
            cleaned.append(msg)

    return cleaned

# -------- Conversation --------
conversation = [
    {
    "role": "system",
    "content": """
You are an autonomous AI agent.

You will:
1. First follow the given plan
2. Execute step by step
3. Use tools when needed
4. Reflect and improve if necessary

Rules:
- Follow the plan unless correction is needed
- Do not skip steps
- Only say DONE when everything is complete
After a tool result is obtained:
- Do NOT repeat the same tool call
- Continue to the next step
- Do NOT redo completed steps
You already have a plan. Do NOT recreate the plan unless necessary.
If all steps in the plan are completed, output the final answer and say DONE.
"""
}
]

last_summary = None

while True:
    goal = input("Enter GOAL: ")

    if goal == "exit":
        break

    conversation.append({
        "role": "user",
        "content": goal,
        "tag": "goal"
    })

    conversation = [
        msg for msg in conversation
        if msg.get("tag") != "plan"
    ]
    plan = create_plan(goal)
    print("\n===== PLAN =====")
    print(plan)

    conversation.append({
        "role": "assistant",
        "content": plan,
        "tag": "plan"
    })

    tool_calls_count = 0
    reflection_loops = 0
    
    for step in range(10):  # more steps for autonomy
        current_step = step + 1
        print(f"---- STEP {step+1} ----")
        optimized_conversation, last_summary = optimize_memory(
            conversation,
            last_summary
        )
        print("\n===== OPTIMIZED MEMORY =====")
        for msg in optimized_conversation:
            print(msg)

        better_query = rewrite_query(goal)
        candidates = retrieve_memory(better_query, top_k=10)
        if not candidates:
            relevant_memory = []
        else:
            relevant_memory = rerank(better_query, candidates)
        print("\n===== RAG MEMORY =====")
        for m in relevant_memory:
            print("-", m)
        rag_context = "\n".join(relevant_memory)

        enhanced_input = [optimized_conversation[0]]
        if rag_context:
            enhanced_input.append({
            "role": "system",
            "content": f"""
            You have access to retrieved memory.

            Rules:
            - Use memory ONLY if relevant
            - If used, ground your answer in it
            - Do NOT hallucinate beyond it

            Memory:
            {rag_context}
            """
            })

        enhanced_input += optimized_conversation[1:]
        print("\n===== LLM INPUT =====")
        for msg in enhanced_input:
            print(msg)
        
        clean_input = clean_conversation(enhanced_input)

        response = client.responses.create(
            model="gpt-5",
            input=clean_input,
            tools=tools
        )
        print("\n===== RAW RESPONSE =====")
        print(response.output)

        tool_call = None

        for item in response.output:
            if item.type == "function_call":
                tool_call = item
                break

        # -------- If tool is called --------
        if tool_call:
            tool_calls_count += 1
            if tool_calls_count > 5:
                print("⚠️ Too many tool calls, forcing stop")
                break
            print("TOOL CALL:", tool_call.name, tool_call.arguments)
            args = json.loads(tool_call.arguments)

            if tool_call.name == "calculator":
                result = calculator(
                    args["operation"],
                    args["a"],
                    args["b"]
                )

            elif tool_call.name == "explain":
                result = explain(
                    args["text"]
                )

            else:
                result = "Unknown tool"

            # Tool result
            conversation.append({
                "role": "assistant",
                "content": f"Step {current_step} completed with result: {result}",
                "tag": "step_result"
            })

            # Continue loop → let LLM decide next step
            continue

        # -------- No tool call → final answer --------
        else:
            reply = response.output_text
            print("AGENT:", reply)

            conversation.append({"role": "assistant", "content": reply})

            # Store only meaningful facts (simple heuristic)
            facts = extract_facts(goal, role="user")
            for fact in facts:
                if any(fact == item["text"] for item in vector_store):
                    continue
                vector_store.append({
                    "text": fact,
                    "embedding": get_embedding(fact),
                    "type": "fact",
                    "source": "user",
                    "timestamp": len(vector_store),
                    "importance": fast_importance(fact)
                })

            # Only store short + useful replies
            facts = extract_facts(reply, role="assistant")
            for fact in facts:
                if any(fact == item["text"] for item in vector_store):
                    continue
                vector_store.append({
                    "text": fact,
                    "embedding": get_embedding(fact),
                    "type": "fact",
                    "source": "assistant",
                    "timestamp": len(vector_store),
                    "importance": fast_importance(fact)
                })
            
            print_vector_store()

            # -------- SELF REFLECTION --------
            reflection = reflect(
                    goal,
                    f"""
                Conversation:
                {conversation}

                Answer:
                {reply}
                """
                )
            print("REFLECTION:", reflection)

            conversation = [
                msg for msg in conversation
                if msg.get("tag") != "reflection"
            ]
            conversation.append({
                "role": "assistant",
                "content": reflection,
                "tag": "reflection"
            })

            if "STATUS: COMPLETE" in reflection:
                print("✅ Goal completed")
                break
            else:
                if reflection_loops > max_reflection_loops:
                    print("⚠️ Reflection loop detected, stopping")
                    break

                reflection_loops += 1

                conversation = [
                    msg for msg in conversation
                    if msg.get("tag") != "plan"
                ]
                plan = create_plan(goal)
                print("\n===== PLAN =====")
                print(plan)

                conversation.append({
                    "role": "assistant",
                    "content": plan,
                    "tag": "plan"
                })
                print("🔁 Continuing... improving answer")