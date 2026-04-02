from openai import OpenAI
import json
import numpy as np

vector_store = []

MAX_HISTORY = 10          # keep last N messages
SUMMARY_THRESHOLD = 20    # when to summarize

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
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": """
You are a planner.

Given a goal:
- Break it into clear, ordered steps
- Be concise
- Do NOT solve the problem
- Only output the plan

Format:
1. ...
2. ...
3. ...
"""
            },
            {
                "role": "user",
                "content": f"GOAL: {goal}"
            }
        ]
    )

    return plan_response.output_text

def reflect(goal, answer):
    reflection = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": """
You are a strict evaluator.

Your job:
- Check if the answer fully completes the goal
- Check correctness
- Check if anything is missing

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

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_memory(query, top_k=3):
    query_emb = get_embedding(query)

    scored = []
    for item in vector_store:
        score = cosine_similarity(query_emb, item["embedding"])
        scored.append((score, item["text"]))

    scored.sort(reverse=True)

    return [text for _, text in scored[:top_k]]

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
        model="gpt-5",
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
    system_msg = conversation[0]

    # If too long and no summary yet → create one
    if len(conversation) > SUMMARY_THRESHOLD and last_summary is None:
        last_summary = summarize_conversation(conversation)

    # Build memory
    optimized = [system_msg]

    if last_summary:
        optimized.append({
            "role": "assistant",
            "content": f"Summary of previous conversation: {last_summary}"
        })

    optimized += conversation[-MAX_HISTORY:]

    return optimized, last_summary

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
"""
}
]

last_summary = None

while True:
    goal = input("Enter GOAL: ")

    if goal == "exit":
        break

    conversation.append({"role": "user", "content": f"GOAL: {goal}"})

    plan = create_plan(goal)
    print("\n===== PLAN =====")
    print(plan)

    vector_store.append({
        "text": f"Plan: {plan}",
        "embedding": get_embedding(plan)
    })

    conversation.append({
        "role": "system",
        "content": f"Execution plan:\n{plan}"
    })

    for step in range(10):  # more steps for autonomy
        print(f"---- STEP {step+1} ----")
        conversation.append({
            "role": "system",
            "content": f"Current step: {step+1}"
        })
        optimized_conversation, last_summary = optimize_memory(
            conversation,
            last_summary
        )
        print("\n===== OPTIMIZED MEMORY =====")
        for msg in optimized_conversation:
            print(msg)

        relevant_memory = retrieve_memory(goal)
        print("\n===== RAG MEMORY =====")
        for m in relevant_memory:
            print("-", m)
        rag_context = "\n".join(relevant_memory)
        enhanced_input = [
            optimized_conversation[0],  # system
            {
                "role": "system",
                "content": f"Relevant past information:\n{rag_context}"
            }
        ] + optimized_conversation[1:]
        print("\n===== LLM INPUT =====")
        for msg in enhanced_input:
            print(msg)
        
        response = client.responses.create(
            model="gpt-5",
            input=enhanced_input,
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

            # 1. Add the function call to conversation
            conversation.append({
                "type": "function_call",
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments
            })

            # 2. Then add the result
            conversation.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result)
            })

            # Continue loop → let LLM decide next step
            continue

        # -------- No tool call → final answer --------
        else:
            reply = response.output_text
            print("AGENT:", reply)

            conversation.append({"role": "assistant", "content": reply})

            # store user goal
            vector_store.append({
                "text": goal,
                "embedding": get_embedding(goal)
            })

            # store assistant reply
            vector_store.append({
                "text": reply,
                "embedding": get_embedding(reply)
            })
            
            # -------- SELF REFLECTION --------
            reflection = reflect(goal, reply)
            print("REFLECTION:", reflection)

            conversation.append({
                "role": "system",
                "content": f"Reflection feedback:\n{reflection}"
            })

            if "STATUS: COMPLETE" in reflection:
                print("✅ Goal completed")
                break
            else:
                plan = create_plan(goal)
                print("\n===== PLAN =====")
                print(plan)

                vector_store.append({
                    "text": f"Plan: {plan}",
                    "embedding": get_embedding(plan)
                })

                conversation.append({
                    "role": "system",
                    "content": f"Execution plan:\n{plan}"
                })
                print("🔁 Continuing... improving answer")