# pip install faster-whisper sounddevice scipy edge-tts pygame numpy

# ---------------- Imports ----------------
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver

import json
from langchain_ollama import ChatOllama

#
import asyncio
import edge_tts
import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import queue
import io
import pygame


pygame.mixer.init()
audio_queue = queue.Queue()

# ---------------- Audio Input ----------------
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def listen_until_silence(samplerate=16000, silence_threshold=0.01, max_silence_sec=3, chunk_duration=0.1):
    audio = []
    silence_chunks = 0
    max_silence_chunks = int(max_silence_sec / chunk_duration)
    print("🎤 Listening...")

    with sd.InputStream(samplerate=samplerate, channels=1, blocksize=int(samplerate*chunk_duration), callback=audio_callback):
        while True:
            chunk = audio_queue.get()
            chunk = chunk.flatten().astype(np.float32)
            rms = np.sqrt(np.mean(chunk**2))

            if rms < silence_threshold:
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks and len(audio) > 5:
                    break
            else:
                silence_chunks = 0
                audio.append(chunk)

    if not audio:
        return None
    return np.concatenate(audio)


def STT(audio):
    segments, _ = model.transcribe(audio, language="en")
    return "".join(seg.text for seg in segments)

async def TTS(text, voice="en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice=voice)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    if audio_bytes:
        sound_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(20)

async def conversation_loop():
    print("✅ System Ready. Speak now.")

    while True:
        # 1. Listen
        audio = listen_until_silence()
        if audio is None:
            continue

        # 2. Transcribe
        user_text = STT(audio)
        if not user_text.strip():
            continue

        print(f"👤 You: {user_text}")
        
        response = app.invoke({
            "messages": [
                HumanMessage(content=user_text)
            ],
            "exit": False
        })
        step = 1
        for msg in response["messages"]:
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

        if response["exit"]:
            print("🤖 AI: Goodbye!")
            await TTS("Goodbye!")
            break

        AI_text = response["messages"][-1].content
        print(f"🤖 AI: {AI_text}")

        # 4. Speak
        await TTS(AI_text)

# ---------------- Tools ----------------
@tool

# ---------------- Agents ----------------
# -------- Configuration --------
llm = ChatOllama(model="qwen2.5:7b-instruct-q4_K_M", temperature=0.0)
llm_with_tools =llm.bind_tools([])

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    exit : bool

def chat_agent(state: AgentState):
    system_prompt = SystemMessage(content=f"""
You are a chat agent

TASK:
- you will recieve text from a user, and you need to response base on your training data.
- if the user want to end the conversation return "EXIT" and explain why
IMPORANT: Answer in short prases and to the point.
""")
    messages = [system_prompt] + state['messages']
    
    response = llm.invoke(messages)
    print(f"[CHAT BOT]: {response.content}")

    if "EXIT" in response.content:
        return { "messages" : [response], "exit": True}
    return { "messages" : [response]}

def rounter(state: AgentState):
    return END
# -------- Graph --------
workflow = StateGraph(AgentState)

workflow.add_node("chat_agent", chat_agent)

workflow.add_edge(START,"chat_agent")
workflow.add_conditional_edges("chat_agent", rounter)

app = workflow.compile()
# ---------------- Main ----------------
model = WhisperModel("base", device="cpu", compute_type="int8")

print(app.get_graph().draw_ascii())
try:
    asyncio.run(conversation_loop())
except KeyboardInterrupt:
    print("\nStopping...")
