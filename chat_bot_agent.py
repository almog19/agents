from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama

from typing import Annotated, TypedDict
from enum import Enum
import json
#langgraph keys:
#node → returns dict
#router → returns "node_name"

# ---------------- CONFIGURATE LLM ----------------
llm = ChatOllama(model="qwen2.5:14b", temperature=0)

# ---------------- DEFINE TOOLS ----------------
@tool
def weather_in_NYC():
    """
    return the current temperature in new york city(NYC)
    OUTPUT:
    integer in Fahrenheit
    """
    return 76

@tool
def send_mail(mail_address: str, content: str):
    """
    send a mail to the mail address with the specified content, will return error and error message if something went wrong
    input:
    mail address -> address of the arriving mail
    content -> the text the mail will included
    output:
    success code(0) if mail has delivired correctly, error if message has failed
    """
    return {"error" : "failed to send mail"}

@tool
def backup_mail(mail_address: str, content: str):
    """
    IMPORTANT: only use if there is no other way to send the mail
    send a mail to the mail address with the specified content, will return error and error message if something went wrong
    input:
    mail address -> address of the arriving mail
    content -> the text the mail will included
    output:
    success code(0) if mail has delivired correctly, error if message has failed
    """
    return {"success code" : "0"}


#bind LLM with tools
tools = [weather_in_NYC,send_mail,backup_mail]
llm_with_tools = llm.bind_tools(tools)

sensitive_tools = ["send_mail","backup_mail"]

class Permission(Enum):
    ALLOW = "allow"
    DENY = "deny"
# ---------------- STATE ----------------
class AgentState(TypedDict):
    # 'add_messages' ensures that new LLM responses are appended, not replaced.
    messages: Annotated[list, add_messages]
    
    pending_tool: list[dict]
    awaiting_permission: bool
    permission_decision: Permission | None

    failures: int
    max_failures: int
    
    permission_cache: dict[str, Permission]
# ---------------- NODES ----------------
#the doing notes:
#each node takes a state as input and return an update to the state
#   - tool node -> execute a function
#   - Agent node -> calls the LLM to decide what to do
def call_model(state: AgentState):
    if state["failures"] >= state["max_failures"]:
        return {
            AIMessage(content="Unable to complete this request after severals failed attempts")
        }

    system_prompt = SystemMessage(content="""
You are a chat agent assistant. 
1. LANGUAGE: You MUST use English as your ONLY LANGUAGE!(ALWAYS ALWAYS ALWAYS follow this important rule!). 
2. TOOLS: You must use the your tools.
3. LIMITS: If the tool fails, try again untill max failures reach. Never hallucinate results!.
4. After using a tool, you MUST respond with a natural language explanation of the result.
* Never end a turn with only a tool call.

If you have already retrieved data from a tool, use that data from the conversation history.
If an error is getting thrown by a tool try to bypass it using another way, if not possible try again untill max failure reached. 
""")
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

#the decision maker:
# rounter function that looks at the state and decides what to do next.
#   - if LLM output contain tool call -> call that tool
#   - if LLM output is final answer -> Go to END
def route(state: AgentState):
    last_message = state["messages"][-1]

    if state.get("failures") >= state.get("max_failures"):
        return END

    if state.get("awaiting_permission"):
        return "handle_permission"
    
    # Check if the LLM wants to call a tool
    if last_message.tool_calls:
        sensitive = []
        non_sensitive = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in sensitive_tools:
                fp = tool_fingerprint(tool_call)
                cached = state['permission_cache'].get(fp)

                if cached == Permission.ALLOW:
                    non_sensitive.append(tool_call)
                elif cached == Permission.DENY:
                    return "deny_tool"
                else:
                    sensitive.append(tool_call)
            else:
                non_sensitive.append(tool_call)
        return "request_permission" if sensitive else "tools"

    return END

# --------- STOP AFTER FAILURE ---------
def observe_tool_result(state: AgentState):
    last_msg = state["messages"][-1]

    if isinstance(last_msg, ToolMessage):
        content = str(last_msg.content).lower()
        if "error" in content or "failed" in content:
            print("[INCREASING FAILURE]! to:", (state['failures']+1))
            return {
                "failures": state["failures"] + 1
            }

    return {}

def after_tool_router(state: AgentState):
    if state["failures"] >= state["max_failures"]:
        return END
    return "agent"

def tool_fingerprint(tool_call: dict) -> str:
    return f"{tool_call["name"]}:{json.dumps(tool_call['args'], sort_keys=True)}"

# --------- STOP LOOP FOR PERMISSION ---------
POSITIVE_KEYWORDS = {
    "yes", "yeah", "sure", "ok", "okay", "go ahead",
    "do it", "approved", "fine", "sounds good"
}
NEGATIVE_KEYWORDS = {
    "no", "nope", "don't", "do not", "stop", "cancel",
    "deny", "not allowed", "never"
}

def llm_sentiment(text: str) -> Permission:
    t = text.lower()
    if any(k in t for k in POSITIVE_KEYWORDS):
        print("--- fast possitive")
        return Permission.ALLOW

    if any(k in t for k in NEGATIVE_KEYWORDS):
        print("--- fast niggas")
        return Permission.DENY
    
    prompt = AIMessage(content="""
You are a sentiment classifier.

Classify the user's intent.
Respond with ONLY one word:
ALLOW or DENY
""")

    response = llm.invoke([
        prompt,
        HumanMessage(content=text)
    ])
    print("[DESICION RESPONSE]\n",response, "\n\n\n",response.content)
    decision = response.content.upper()

    if "ALLOW" in decision or "YES" in decision:
        return Permission.ALLOW

    return Permission.DENY

#permission node(does not execute anything, just ask the user for permission)
def request_permission(state: AgentState):
    last_message = state["messages"][-1]
    
    sensitive_calls = [tc for tc in last_message.tool_calls if tc["name"] in sensitive_tools]
    tool_list = "\n".join(f"-{tc["name"]}\t({tc["args"]})" for tc in sensitive_calls)
    permission_prompt = AIMessage(content=f"""I need your permission to proceed with the following actions:

{tool_list}

Do you allow all of these actions?
""")
    return {
        "messages": [permission_prompt],
        "pending_tool": sensitive_calls,
        "awaiting_permission": True
    }

def handle_permission(state: AgentState):
    debug_state(state, "handle_permission")
    print("[FP CACHE]", state["permission_cache"])
    cache = dict(state.get("permission_cache", {}))


    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break
    user_text = msg.content
    desicion = llm_sentiment(user_text)
    for tc in state["pending_tool"]:
        print("found ", tc['name'])
        fp = tool_fingerprint(tc)
        cache[fp] = desicion
    print("[FP CACHE] after update", cache)
    return {
        "permission_decision" : desicion,
        "permission_cache" : cache,
        "awaiting_permission" : False,
    }

def permission_router(state: AgentState):
    debug_state(state, "permission_router")
    if state["permission_decision"] == Permission.ALLOW:
        return "tools_from_permission"
    return "deny_tool"

def tools_from_permission(state: AgentState):
    debug_state(state, "tools_from_permission")
    return {
        "messages": [
            AIMessage(
                content="Permission granted. Proceeding with the requested action.",
                tool_calls=state["pending_tool"]
            )
        ],
        "pending_tool": [],
        "awaiting_permission": False,
        "permission_decision" : None,
    }

def deny_tool(state: AgentState):
    debug_state(state, "DENY")
    return {
        "messages": AIMessage(content="User denied permission. I will not use these actions."),
        "pending_tool" : [],
        "awaiting_permission" : False,
        "permission_decision" : None,
    }

tool_node = ToolNode(tools)

# ---------------- GRAPH ----------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node) # Assume tool_node is defined
workflow.add_node("request_permission", request_permission)
workflow.add_node("tools_from_permission",tools_from_permission)
workflow.add_node("handle_permission",handle_permission)
workflow.add_node("deny_tool",deny_tool)
workflow.add_node("observe_tool",observe_tool_result)

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "observe_tool")

workflow.add_conditional_edges("agent", route)
workflow.add_conditional_edges("handle_permission", permission_router)
workflow.add_conditional_edges("observe_tool", after_tool_router)

workflow.add_edge("tools_from_permission", "tools")
workflow.add_edge("deny_tool", "agent") # This closes the LOOP
workflow.add_edge("tools", "agent") # This closes the LOOP

# ---------------- PERSISTENCE ----------------
# 1) Short term memory(Threads) -> when compiling the graph(SqliteSaver) Langgraph save a "screenshot" of the state after every code
# 2) Threads IDS -> by profividing a thread id, you can resume a conversation where it left off
#   *changing the THREAD ID will reset the memory, "test_1" -> "test_2"
# 3) Time Travel -> you can rewind the agent to a previous state, modify a message and try it again.(debugging complex client)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
# ---------------- RAN ----------------

config = {"configurable" : {"thread_id" : "permission_test"}}

print(app.get_graph().draw_ascii())


def debug_state(state: AgentState, label=""):
    print(f"\n[DEBUG {label}]")
    print("awaiting_permission:", state.get("awaiting_permission"))
    print("permission_decision:", state.get("permission_decision"))
    print("pending_tool:", state.get("pending_tool"))
flag = False
while(True):
    user_message = input("rub my belly and ask for a wish!\n")
    if(flag == False):
        result = app.invoke({
            "messages": [
                HumanMessage(content=user_message)],
            "failures" : 0,
            "max_failures" : 3,
            "permission_cache" : {},
        }, config)
        flag = True
    else:
        result = app.invoke({
            "messages": [
                HumanMessage(content=user_message)],
            "failures" : 0,
            "max_failures" : 3,
        }, config)


    # Get all steps for this thread
    history = list(app.get_state_history(config))

    #print(f"\n--- GRAPH STEP HISTORY (Total Steps: {len(history)}) ---")
    #for i, state in enumerate(reversed(history)):
        # state.next tells you which node was about to run
        # state.metadata tells you which node just finished
    #    finished_node = state.metadata.get("source", "START")
    #    print(f"Graph Step {i}: Just finished Node [{finished_node}]")

    def print_agent_trajectory(result):
        print("\n--- AGENT TRAJECTORY REPORT ---")
        step = 1
        for msg in result["messages"]:
            # Check if the AI decided to call a tool
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"Step {step}: [ACTION] Calling Tool '{tc['name']}' with {tc['args']}")
                    step += 1
            
            # Check if this message is the actual output from a tool
            if msg.type == "tool":
                print(f"Step {step}: [OBSERVATION] Tool returned: {msg.content}")
                step += 1
                
            # The final answer
            if msg.type == "ai" and not msg.tool_calls:
                print(f"Step {step}: [FINAL RESPONSE] {msg.content}")

    # Use it after app.invoke
    print_agent_trajectory(result)

    #for i, msg in enumerate(result["messages"]):
    #    print(f"{i} -> {msg.content}")

    # Fetch the current state for that specific thread
    current_state = app.get_state(config)

    # Access the 'messages' list within the state values
    messages_in_history = current_state.values.get("messages", [])
    
    history = list(app.get_state_history(config))
    print("\n--- ROUTE TRACE ---")
    for i, state in enumerate(reversed(history)):
        source = state.metadata.get("source", "START")
        next_node = state.next
        print(f"{i:02d}: {source}  →  {next_node}")

    #print(f"\n--- CHECKPOINT AUDIT ---")
    #print(f"Thread ID: {config['configurable']['thread_id']}")
    #print(f"Number of messages saved: {len(messages_in_history)}")

    # If you want to see the roles to make sure they are correct:
    #for msg in messages_in_history:
    #    print(f"Role: {type(msg).__name__} | Content: {msg.content[:50]}...")
    
    print(F"\n--- AI Message ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)
