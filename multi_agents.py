#pip install langgraph langchain langchain_ollama ddgs newspaper3k lxml_html_clean grandalf
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
# ---------------- Imports ----------------
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
import json
from datetime import date
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
llm = ChatOllama(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.0)

researcher_llm = llm.bind_tools([research_topic])
researcher_tools = ToolNode([research_topic])


# ---------------- Define State ----------------
"""
tracks the data(Research & Draft) seperate each one so agent will not get confuse
"""
class TeamState(TypedDict):
    messages : Annotated[list, add_messages]
    research_Summary: str
    draft_posts: str
    next_agent: str
    permission: bool

    loop_counter: int

    feed_back: str

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
    research_summary = state.get("research_Summary", "EMPTY")
    draft_posts = state.get("draft_posts", "EMPTY")
    current_loop = state.get("loop_counter", 0)
    last_msg = state["messages"][-1]
    user_input = last_msg.content if isinstance(last_msg, HumanMessage) else "No new feedback"

    print(f"\t\t[CURRENT_LOOP] -> {current_loop}")
    system_prompt = SystemMessage(content = f"""
You are Manager agent

DECISION NEXT STEP:
1. If 'Research' is EMPTY -> 'RESEARCHER'.
2. If 'Research' exists but 'Draft Post' is EMPTY -> 'WRITER'.
3. If User Input is feedback/edits -> 'WRITER'.
4. otherwise -> 'FINISH'.

DATA:
1. Research: {research_summary}
2.  Draft Post: {draft_posts}
3.  User Input: {user_input}

OUTPUT VALID JSON ONLY:
{{
decision: "RESEARCHER | WRITER | FINISH",
reason: "short explaining",
}}
""")
    response = llm.invoke([system_prompt])
    
    try: 
        data = json.loads(response.content)
        print(f"\t\t[RESPONSE] -> {data}")
        decision = data['decision'].upper()

    except:
        print("couldnt open as json")
        decision = "FINISH"

    print(f"\t[NEXT_AGENT] -> {decision}")
    return {
        "next_agent" : decision,
        "loop_counter": current_loop + 1,
        "feedback": user_input
    
    }


def route_manager(state: TeamState):
    print("[ROUTE_MANAGER]")
    loops = state.get("loop_counter", 0)
    if loops > 3:
        print("[SAFTEY BRAKE], max loops reached")
        return END
    
    if "RESEARCHER" in state['next_agent']:
        return "researcher"
    elif "WRITER" in state['next_agent']:
        return "writer"
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
    # Filter messages: Keep only Human, AI Tool Calls, and Tool Results.
    # Exclude the Manager's JSON decisions to save tokens.
    filtered_messages = [
        m for m in state["messages"] 
        if isinstance(m, (HumanMessage, ToolMessage)) or (isinstance(m, AIMessage) and m.tool_calls)
    ]
    research_summary = state.get("research_Summary", "EMPTY")
    feedback = state.get("feedback", "EMPTY")
    system_prompt = SystemMessage(content = f"""
You are a Researcher Agent

ROLE:
- find and distill real-time information.
- Use your tools to find facts about the provided topic.
- Extract key statistics, names. and current events.
- Summarize the finding into a structured list of facts

STRICT RULES:
- ENGLISH ONLY. Never use any other language or script.
- Do NOT write the social media post. Only provide the facts.
- If the tool returns no data, admit it; do NOT hallucinate facts.
- Provide source URLs for every major fact found.

DATA:
1. Facts: {research_summary}
2. current data: {date.today()}
3. user feedback: {feedback}

FEEDBACK CASE:
if user requested more information try to fetch more information, if you couldnt find more information return "could not find more data".

OUTPUT FORMAT:
if you have facts return a JSON VALID OUTPUT
{{
summary : "list of facts",
sources : "the sources that have been used",
relativity : "how relevant the facts are for today, a double number between 0 - 1"
}}
else use your tools to get the facts
""")
    messages = [system_prompt] + state['messages']
    response = researcher_llm.invoke(messages) #return AImessage which might contain tool
    if not response.tool_calls:
        try:
            json_file = json.loads(response.content)
            print(f"\t\t[SUMMARY] -> {len(json_file['summary'])}\n\t\t[SOURCES] -> {json_file['sources']}\n\t\t[RELATIVITY] -> {json_file['relativity']}")
            return { "messages" : [response], "research_Summary" : json_file}
        except:
            print(f"(could not open as json!)\n\t\t[SUMMARY] -> {response.content}")
            return {"messages": [response], "research_Summary": response.content}

    return {"messages" : [response]}

def router_researcher(state: TeamState):
    print("[ROUTER_RESEARCHER]")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "researcher_tools"
    return "manager"

# -------- Writer Agent --------
def writer_agent(state: TeamState) -> str:
    """
    Turn research into a post(writes 3 Variations), dont have any tools
        Input:
            Research summary
        Output:
            Draft post
    """

    print("[WRITER_AGENT]")
    research_summary = state.get("research_Summary", "EMPTY")
    existing_draft = state.get("draft_posts", "EMPTY")
    feedback = state.get("feedback", "EMPTY")
    system_prompt = SystemMessage(content=f"""
You are a LinkedIn Expert agent. 

TASK:
Create or edit 3 variations(Expert, Contrarian, Storyteller)

DATA:
- Facts: {research_summary}
- Current Draft: {existing_draft}
- User feedback: {feedback}

STRICT RULES:
- English Only.
- Output the full set of 3 variations, including your edits.

OUTPUT VALID JASON ONLY, without any other characther:
{{
  Expert: "Expert version",
  Contrarian: "Contrarian version",
  Storyteller: "Storyteller version",
  satisfaction: satisfied rate base on the feedback
}}
""")
    response = llm.invoke([system_prompt])
    print(response)
    try:
        data = json.loads(response.content)
        print(f"\t\t[satisfaction] -> {data['satisfaction']}")
        return {"messages": [response], "draft_posts": data}
    except:
        print("couldnt open as json")
        return {"messages": [response], "draft_posts": {"Expert": response.content}}

def router_writer(state: TeamState):
    print("[ROUTER_WRITER]")
    return "manager"

# ---------------- Graph ----------------
workflow = StateGraph(TeamState)

workflow.add_node("manager", manager_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("researcher_tools", researcher_tools)
workflow.add_node("writer", writer_agent)

workflow.add_edge(START,"manager") #entry point

workflow.add_conditional_edges("manager", route_manager)
workflow.add_conditional_edges("researcher", router_researcher, {"researcher_tools": "researcher_tools","manager": "manager"})

workflow.add_edge("researcher_tools", "researcher")#tools always go back to the researcher to explain(interpert) the results
workflow.add_edge("writer", "manager")

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
