from colorama import Fore, Style
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

AGENTS = {
    "planner": {
        "model": "gpt-5-mini",
        "system_prompt": """
You are a planner.

Rules:
- If the task is simple (can be solved in 1-2 steps), DO NOT create a long plan
- Keep plans minimal and practical
- Do NOT overthink
- Return ONLY valid JSON
- Use this schema exactly:
  {"steps": ["step 1", "step 2"]}
- Keep between 1 and 5 steps
- Each step must be short and actionable
"""
    },
    "executor": {
        "model": "gpt-5",
        "system_prompt": """
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
    },
    "reviewer": {
        "model": "gpt-5-mini",
        "system_prompt": """
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
    }
}

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

def run_agent(agent_name, input_data, tools=None):
    agent = AGENTS[agent_name]
    request = {
        "model": agent["model"],
        "input": input_data
    }

    if tools is not None:
        request["tools"] = tools

    return client.responses.create(**request)

def build_planner_input(state, goal):
    planner_history = [
        msg for msg in state["planner_history"]
        if msg.get("tag") != "plan"
    ]
    return clean_conversation(planner_history)

def build_reviewer_input(state):
    return clean_conversation(state["reviewer_history"])

def parse_plan_response(plan_text):
    try:
        parsed = json.loads(plan_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        steps = parsed.get("steps", [])
        if isinstance(steps, list):
            cleaned_steps = [
                str(step).strip()
                for step in steps
                if str(step).strip()
            ]
            if cleaned_steps:
                return {
                    "tasks": [
                        {
                            "id": index,
                            "description": step,
                            "status": "pending",
                            "result": None
                        }
                        for index, step in enumerate(cleaned_steps, start=1)
                    ],
                    "raw_text": plan_text
                }

    fallback_steps = [
        line.strip()
        for line in plan_text.splitlines()
        if line.strip()
    ]

    return {
        "tasks": [
            {
                "id": index,
                "description": step,
                "status": "pending",
                "result": None
            }
            for index, step in enumerate(
                fallback_steps or ["Think through the goal and respond."],
                start=1
            )
        ],
        "raw_text": plan_text
    }

def render_plan(plan_data):
    return "\n".join(
        f'{task["id"]}. [{task["status"].upper()}] {task["description"]}'
        for task in plan_data["tasks"]
    )

def build_plan_system_message(plan_data):
    if not plan_data or not plan_data.get("tasks"):
        return None

    rendered_plan = render_plan(plan_data)
    current_task = get_current_task(plan_data)
    current_task_text = "No pending task."
    if current_task is not None:
        current_task_text = (
            f'{current_task["id"]}. {current_task["description"]}'
        )
    return {
        "role": "system",
        "content": f"""
You have a structured plan to follow.

Current plan:
{rendered_plan}

Current task:
{current_task_text}
"""
    }

def get_current_task(plan_data):
    if not plan_data:
        return None

    for task in plan_data["tasks"]:
        if task["status"] == "pending":
            return task

    return None

def has_pending_tasks(plan_data):
    return get_current_task(plan_data) is not None

def update_current_task(plan_data, status, result_text=None, review_reason=None):
    current_task = get_current_task(plan_data)
    if current_task is None:
        return

    current_task["status"] = status
    current_task["result"] = result_text
    current_task["review_reason"] = review_reason

def mark_current_task_completed(plan_data, result_text, review_reason=None):
    update_current_task(
        plan_data,
        "completed",
        result_text=result_text,
        review_reason=review_reason
    )

def build_task_review_input(state, goal, task, result_text):
    plan_text = "No plan available."
    if state["current_plan"] is not None:
        plan_text = render_plan(state["current_plan"])

    return [
        {
            "role": "system",
            "content": AGENTS["reviewer"]["system_prompt"]
        },
        {
            "role": "system",
            "content": f"""
Review this single task result, not the whole goal.

Allowed statuses:
- completed
- blocked
- needs_replan

Return ONLY valid JSON using this schema:
{{"status":"completed|blocked|needs_replan","reason":"short explanation"}}

Current plan:
{plan_text}
"""
        },
        {
            "role": "user",
            "content": f"""
GOAL:
{goal}

TASK:
{task["id"]}. {task["description"]}

TASK RESULT:
{result_text}
"""
        }
    ]

def parse_task_review_response(review_text):
    try:
        parsed = json.loads(review_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        status = str(parsed.get("status", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()
        if status in ["completed", "blocked", "needs_replan"]:
            return {
                "status": status,
                "reason": reason or "No reason provided."
            }

    lowered = review_text.lower()
    if "needs_replan" in lowered:
        status = "needs_replan"
    elif "blocked" in lowered:
        status = "blocked"
    else:
        status = "completed"

    return {
        "status": status,
        "reason": review_text.strip() or "No reason provided."
    }

def review_current_task(state, goal, result_text):
    task = get_current_task(state["current_plan"])
    if task is None:
        return {
            "status": "completed",
            "reason": "No pending task to review."
        }

    review_response = run_agent(
        "reviewer",
        build_task_review_input(state, goal, task, result_text)
    )
    return parse_task_review_response(review_response.output_text)

def finalize_goal(state, goal, reply):
    add_executor_event(state, "assistant", reply)
    add_shared_event(state, "assistant", reply)

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

    reflection = reflect(
            state,
            goal,
            f"""
        Conversation:
        {state["shared_memory"]}

        Answer:
        {reply}
        """
        )
    print(Fore.BLUE + "REFLECTION:", reflection + Style.RESET_ALL)

    state["shared_memory"] = [
        msg for msg in state["shared_memory"]
        if msg.get("tag") != "reflection"
    ]
    state["executor_history"] = [
        msg for msg in state["executor_history"]
        if msg.get("tag") != "reflection"
    ]
    add_executor_event(
        state,
        "assistant",
        reflection,
        tag="reflection"
    )
    add_shared_event(
        state,
        "assistant",
        reflection,
        tag="reflection"
    )

    return reflection

def create_plan(state, goal):
    plan_response = run_agent(
        "planner",
        build_planner_input(state, goal)
    )

    return parse_plan_response(plan_response.output_text)

def refresh_plan(state, goal):
    state["planner_history"] = [
        msg for msg in state["planner_history"]
        if msg.get("tag") != "plan"
    ]
    state["executor_history"] = [
        msg for msg in state["executor_history"]
        if msg.get("tag") != "plan"
    ]
    state["shared_memory"] = [
        msg for msg in state["shared_memory"]
        if msg.get("tag") != "plan"
    ]

    plan = create_plan(state, goal)
    state["current_plan"] = plan
    rendered_plan = render_plan(plan)
    print(Fore.BLUE + "\n===== PLAN =====" + Style.RESET_ALL)
    print(rendered_plan)

    add_planner_event(state, "assistant", rendered_plan, tag="plan")
    add_executor_event(state, "assistant", rendered_plan, tag="plan")
    add_shared_event(state, "assistant", rendered_plan, tag="plan")

    return state

def reflect(state, goal, answer):
    set_reviewer_history(state, goal, answer)
    reflection = run_agent(
        "reviewer",
        build_reviewer_input(state)
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

def create_agent_state():
    return {
        "shared_memory": [],
        "planner_history": [
            {
                "role": "system",
                "content": AGENTS["planner"]["system_prompt"]
            }
        ],
        "executor_history": [
            {
                "role": "system",
                "content": AGENTS["executor"]["system_prompt"]
            }
        ],
        "reviewer_history": [
            {
                "role": "system",
                "content": AGENTS["reviewer"]["system_prompt"]
            }
        ],
        "executor_last_summary": None,
        "current_plan": None
    }

def add_shared_event(state, role, content, tag=None):
    event = {
        "role": role,
        "content": content
    }
    if tag is not None:
        event["tag"] = tag
    state["shared_memory"].append(event)

def add_planner_event(state, role, content, tag=None):
    event = {
        "role": role,
        "content": content
    }
    if tag is not None:
        event["tag"] = tag
    state["planner_history"].append(event)

def add_executor_event(state, role, content, tag=None):
    event = {
        "role": role,
        "content": content
    }
    if tag is not None:
        event["tag"] = tag
    state["executor_history"].append(event)

def set_reviewer_history(state, goal, answer):
    plan_text = "No plan available."
    if state["current_plan"] is not None:
        plan_text = render_plan(state["current_plan"])

    state["reviewer_history"] = [
        {
            "role": "system",
            "content": AGENTS["reviewer"]["system_prompt"]
        },
        {
            "role": "system",
            "content": f"""
Structured plan:
{plan_text}
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

def build_executor_input(state, goal):
    conversation = state["executor_history"]
    last_summary = state["executor_last_summary"]
    current_task = get_current_task(state["current_plan"])

    optimized_conversation, last_summary = optimize_memory(
        conversation,
        last_summary
    )
    state["executor_last_summary"] = last_summary

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
    for memory_item in relevant_memory:
        print("-", memory_item)

    rag_context = "\n".join(relevant_memory)

    enhanced_input = [optimized_conversation[0]]

    plan_message = build_plan_system_message(state["current_plan"])
    if plan_message:
        enhanced_input.append(plan_message)

    if current_task is not None:
        enhanced_input.append({
            "role": "system",
            "content": f"""
Focus on this one task now:
{current_task["id"]}. {current_task["description"]}

Complete this task before moving on to any other task.
If a tool is needed, use it.
If this is the final pending task, provide the final answer after completing it.
"""
        })

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

    print(Fore.BLUE + "\n===== LLM INPUT =====" + Style.RESET_ALL)
    for msg in enhanced_input:
        print(msg)

    return clean_conversation(enhanced_input)

# -------- Conversation --------
state = create_agent_state()

while True:
    goal = input("Enter GOAL: ")

    if goal == "exit":
        break

    add_shared_event(state, "user", goal, tag="goal")
    add_planner_event(state, "user", goal, tag="goal")
    add_executor_event(state, "user", goal, tag="goal")

    state = refresh_plan(state, goal)

    tool_calls_count = 0
    reflection_loops = 0
    
    for step in range(10):  # more steps for autonomy
        current_step = step + 1
        print(Fore.GREEN + f"---- STEP {step+1} ----" + Style.RESET_ALL)
        current_task = get_current_task(state["current_plan"])
        if current_task is None:
            break
        print(
            Fore.CYAN
            + f'TASK {current_task["id"]}: {current_task["description"]}'
            + Style.RESET_ALL
        )
        clean_input = build_executor_input(state, goal)

        response = run_agent(
            "executor",
            clean_input,
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
                print(Fore.RED + "⚠️ Too many tool calls, forcing stop" + Style.RESET_ALL)
                break
            print(Fore.MAGENTA + "TOOL CALL:", tool_call.name, tool_call.arguments + Style.RESET_ALL)
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
            add_executor_event(
                state,
                "assistant",
                f"Step {current_step} completed with result: {result}",
                tag="step_result"
            )
            add_shared_event(
                state,
                "assistant",
                f"Step {current_step} completed with result: {result}",
                tag="step_result"
            )
            task_result_text = f"Step {current_step} completed with result: {result}"
            task_review = review_current_task(
                state,
                goal,
                task_result_text
            )
            print(
                Fore.BLUE
                + f'TASK REVIEW: {task_review["status"].upper()} - {task_review["reason"]}'
                + Style.RESET_ALL
            )

            if task_review["status"] == "completed":
                mark_current_task_completed(
                    state["current_plan"],
                    task_result_text,
                    review_reason=task_review["reason"]
                )
            elif task_review["status"] == "blocked":
                update_current_task(
                    state["current_plan"],
                    "blocked",
                    result_text=task_result_text,
                    review_reason=task_review["reason"]
                )
                print(Fore.RED + "Task is blocked. Stopping execution." + Style.RESET_ALL)
                break
            else:
                update_current_task(
                    state["current_plan"],
                    "needs_replan",
                    result_text=task_result_text,
                    review_reason=task_review["reason"]
                )
                add_planner_event(
                    state,
                    "assistant",
                    f'Task needs replan: {task_review["reason"]}',
                    tag="task_review"
                )
                add_shared_event(
                    state,
                    "assistant",
                    f'Task needs replan: {task_review["reason"]}',
                    tag="task_review"
                )
                state = refresh_plan(state, goal)
                print(Fore.YELLOW + "Task review requested replanning." + Style.RESET_ALL)
                continue

            if has_pending_tasks(state["current_plan"]):
                print(
                    Fore.YELLOW
                    + "Task completed via tool. Moving to the next planned task."
                    + Style.RESET_ALL
                )
                continue

            reply = f"The final answer is {result}."
            print(Fore.GREEN + "AGENT:", reply + Style.RESET_ALL)
            reflection = finalize_goal(state, goal, reply)

            if "STATUS: COMPLETE" in reflection:
                print(Fore.GREEN + "✅ Goal completed" + Style.RESET_ALL)
            else:
                print(Fore.RED + "⚠️ Goal finished but review marked it incomplete." + Style.RESET_ALL)
            break

        # -------- No tool call → final answer --------
        else:
            reply = response.output_text
            print(Fore.GREEN + "AGENT:", reply + Style.RESET_ALL)
            task_review = review_current_task(
                state,
                goal,
                reply
            )
            print(
                Fore.BLUE
                + f'TASK REVIEW: {task_review["status"].upper()} - {task_review["reason"]}'
                + Style.RESET_ALL
            )

            if task_review["status"] == "completed":
                mark_current_task_completed(
                    state["current_plan"],
                    reply,
                    review_reason=task_review["reason"]
                )
            elif task_review["status"] == "blocked":
                update_current_task(
                    state["current_plan"],
                    "blocked",
                    result_text=reply,
                    review_reason=task_review["reason"]
                )
                add_executor_event(state, "assistant", reply)
                add_shared_event(state, "assistant", reply)
                print(Fore.RED + "Task is blocked. Stopping execution." + Style.RESET_ALL)
                break
            else:
                update_current_task(
                    state["current_plan"],
                    "needs_replan",
                    result_text=reply,
                    review_reason=task_review["reason"]
                )
                add_executor_event(state, "assistant", reply)
                add_shared_event(state, "assistant", reply)
                add_planner_event(
                    state,
                    "assistant",
                    f'Task needs replan: {task_review["reason"]}',
                    tag="task_review"
                )
                add_shared_event(
                    state,
                    "assistant",
                    f'Task needs replan: {task_review["reason"]}',
                    tag="task_review"
                )
                state = refresh_plan(state, goal)
                print(Fore.YELLOW + "Task review requested replanning." + Style.RESET_ALL)
                continue

            if has_pending_tasks(state["current_plan"]):
                add_executor_event(state, "assistant", reply)
                add_shared_event(state, "assistant", reply)
                print(
                    Fore.YELLOW
                    + "Task completed. Moving to the next planned task."
                    + Style.RESET_ALL
                )
                continue

            reflection = finalize_goal(state, goal, reply)

            if "STATUS: COMPLETE" in reflection:
                print(Fore.GREEN + "✅ Goal completed" + Style.RESET_ALL)
                break
            else:
                if reflection_loops > max_reflection_loops:
                    print(Fore.RED + "⚠️ Reflection loop detected, stopping" + Style.RESET_ALL)
                    break

                reflection_loops += 1

                state = refresh_plan(state, goal)
                print(Fore.YELLOW + "🔁 Continuing... improving answer" + Style.RESET_ALL)
