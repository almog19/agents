"""
Content Engine
Agent
    1) The researcher
        finds current news about a topic
    2) The copyrighter
        takes the research and drafts a LinkedIn post in a specific "voice"
    3) The publisher
        Sensitive agent which ask for permission, it will send emails/post
"""
"""
manager promt:
You are a project manager
Current Research: {state['research_data']}
Current Draft: {state['draft_content']}

Decide the next step:
- If there is no research, return 'RESEARCHER'.
- If there is research but no draft, return 'WRITER'.
- If the draft is complete, return 'FINISH'.
"""
# ---------------- Imports ----------------
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama

#tools
from ddgs import DDGS
from newspaper import Article
# ---------------- Tools ----------------
# -------- Researcher Tools --------
@tool
def research_topic(topic: str) -> str:
    """
    Searches the web for a topic and return a summary of the topic artickles, used to get real time data
    input:
        topic(string)
    output:
        topic real time data summary
    """
    print("\t[@TOOL] -> RESEARCH_TOPIC")
    with DDGS() as ddgs:
        search_results = [r for r in ddgs.text(topic, max_results=3)]
    
    final_report = ""
    print(f"\t\t\tfound {len(search_results)} results!")
    for r in search_results:
        try:
            url = r['href']
            article = Article(url)
            article.download()
            article.parse()
            content = article.text[:1000]
            date = article.publish_date
            final_report += f"\nSource -> {url}\nContent({date}):\n {content}\n"
        except:
            print(f"\t\t\t[FAILED OPENING] -> {url}")
            continue
    return final_report

# -------- Define LLM --------
llm = ChatOllama(model="qwen2.5:14b", temperature=0.0)

researcher_llm = llm.bind_tools([research_topic])
researcher_tools = ToolNode([research_topic])


# ---------------- Define State ----------------
"""
tracks the data(Research & Draft) seperate each one so agent will not get confuse
"""
class TeamState(TypedDict):
    messages : Annotated[list, add_messages]
    research_Summary: str
    #draft_post: str
    next_agent: str
    permission: bool

    loop_counter: int

# ---------------- Agents ----------------

# -------- Manager Agent --------
def manager_agent(state: TeamState):
    print("[MANAGER]:")
    """
    check result with the user
        Input:
            draft post
        Output:
            permission(yes/no)
    """
    research_summary = state.get("research_Summary", "").strip()
    current_loop = state.get("loop_counter", 0)
    print(f"\t[RESEARCH_SUMMARY] -> {research_summary[:100]}")
    print(f"\t\t[CURRENT_LOOP] -> {current_loop}")
    system_prompt = SystemMessage(content = f"""
You are a Project Manager. Respond EXCLUSIVELY in English. 
Do not use Cyrillic or any other alphabet.

TASK:
Does the 'Current Research' below contain useful facts about the topic?
- If it is empty, returns 'RESEARCHER'.
- If it has facts, return 'FINISH'.

CURRENT RESEARCH:
{research_summary if research_summary else "None"}

OUTPUT RULE:
Return ONLY the word 'RESEARCHER' or 'FINISH'. 
No explanations. No other languages.
""")
    response = llm.invoke([system_prompt])
    decision = response.content.strip().upper()
    if "FINISH" in decision:
        final_decision = "FINISH"
    else:
        final_decision = "RESEARCHER"
    print(f"\t[NEXT_AGENT] -> {final_decision}")
    return {"next_agent" : final_decision, "loop_counter": current_loop + 1}


def route_manager(state: TeamState):
    print("[ROUTE_MANAGER]")
    loops = state.get("loop_counter", 0)
    if loops > 3:
        print("[SAFTEY BRAKE], max loops reached")
        return END
    
    if "RESEARCHER" in state['next_agent']:
        return "researcher"
    #elif "WRITER" in state['next_agent']:
    #    return "writer"
    return END

# -------- Researcher Agent --------
def researcher_agent(state: TeamState) -> str:
    print("[RESEARCH_AGENT]")
    """
    Search news & summarize key points
        Input:
            Topic
        Output:
            Research summary
    tools:
    search and find tool
    fetching data tool
    summarization LLM
    """
    system_prompt = SystemMessage(content = """
You are a Researcher. 
1. Use your tools to find facts, DO NOT MAKE THEM UP OR USE YOU PRETRAIN DATA.
2. After using the tool, summarize the findings(DO NOT HALLUCINATE DATA )
3. STRICT RULE: You must speak and summarize EXCLUSIVELY in English. 
4. DO NOT use any non-English script!.
""")
    messages = [system_prompt] + state['messages']
    response = researcher_llm.invoke(messages) #return AImessage witch might contain tool 
    if not response.tool_calls and len(response.content) > 10:
        return { "messages" : [response], "research_Summary" : response.content}
    return {"messages" : [response]}

def router_researcher(state: TeamState):
    print("[ROUTER_RESEARCHER]")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "researcher_tools"
    return "manager"

# -------- Writer Agent --------
"""
Turn research into a post(writes 3 Variations), dont have any tools
    Input:
        Research summary
    Output:
        Draft post
"""




# ---------------- Graph ----------------
workflow = StateGraph(TeamState)

workflow.add_node("manager", manager_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("researcher_tools", researcher_tools)

workflow.add_edge(START,"manager") #entry point

workflow.add_conditional_edges("manager", route_manager)
workflow.add_conditional_edges("researcher", router_researcher, {"researcher_tools": "researcher_tools","manager": "manager"})

workflow.add_edge("researcher_tools", "researcher")#tools always go back to the researcher to explain(interpert) the results

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ---------------- Main ----------------

config = {"configurable" : {"thread_id" : "permission_test"}}

print(app.get_graph().draw_ascii())


while(True):
    user_message = input("rub my belly and ask for a wish!\n")
    result = app.invoke({
        "messages": [
            HumanMessage(content=user_message)],
        "loop_counter" : 0,
    }, config)

    # Get all steps for this thread
    history = list(app.get_state_history(config))

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
                print(f"Step {step}: [OBSERVATION] Tool returned: {msg.content[:50]}")
                step += 1
                
            # The final answer
            if msg.type == "ai" and not msg.tool_calls:
                print(f"Step {step}: [FINAL RESPONSE] {msg.content[:50]}")

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
