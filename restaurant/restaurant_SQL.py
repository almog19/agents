"""
AI agent puposes:
1) reservation
    - ask for name, phone, people_number, time
    - if not a table avaiable:
        suggest other time when there is a table available
        if user dont want other time
        suggest a smaller number of group that can fit in the restaurant at that table
2) cancel reservation
    - ask for one of the identification marks(name, phone number, time)
    search for that table:
    if found tell the user the identification marks and ask again for cancalation
    if not found search tables base on the identification mark:
        - name: typo, currect first name but not seccond
        - phone: typo
        - time: check tables at different times to see if table has found(base on name or phone)
    if multiple table fits the identification marks ask for more details, if user cant provide more details tell him the identification marks found for the fitting tables.
* Ai agent will return status code, and ask the user to clarify the action with the details
* Ai agent will not have direct access to the SQL for safety reasons.

AGENT STEPS:
at any time, if user change a previous step, llm will behave accordingly
1) action classification:
    - reservation
    - cancalation
    - availability
    - clarification
2) information collection loop(base on the action):
if missing
    - namme
    - phone
    - time
    - people
3) tool execution:
calling the tool
4) tool result:
    - success
    - no_table(suggest alternatives)
    - multiple_matches(ask for more info)
    - not_found(inform user)

"""
# ==================================================
# Imports
# ==================================================
#sql
import sqlite3
from dateutil import parser
from datetime import datetime, timedelta
from enum import Enum
import re

from pydantic import BaseModel

#app
import tkinter as tk
from tkinter import scrolledtext

#langgraph
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from typing import Annotated, TypedDict, List, Optional
from langgraph.checkpoint.memory import MemorySaver

import json
from langchain_ollama import ChatOllama
import random
from collections import defaultdict
import itertools

# ==================================================
# restaurant SQL
# future upgrades:
#   - alternative time suggestion logic
#   - type corrections
#   - race conditional safe locking
# ==================================================

DB_PATH = "restaurant.db"
RESERVATION_DURATION_HOURS = 2
OPEN_HOUR = 10
CLOSE_HOUR = 23


# =========================
# STATUS ENUMS
# =========================

class Status(Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    NO_TABLE_AVAILABLE = "NO_TABLE_AVAILABLE"
    NOT_FOUND = "NOT_FOUND"
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
    ALREADY_EXISTS = "ALREADY_EXISTS"


# =========================
# VALIDATION FIREWALL
# =========================

def validate_reservation_input(name, phone, party_size, start_time_value):
    if not name or len(name.strip()) < 2:
        return False, "Invalid name"

    if not re.match(r"^\+?\d{8,15}$", phone):
        return False, "Invalid phone"

    if party_size <= 0:
        return False, "Invalid party size"

    if isinstance(start_time_value, str):
        start_time = datetime.strptime(start_time_value, "%Y-%m-%d %H:%M")
    elif isinstance(start_time_value, datetime):
        start_time = start_time_value
    else:
        raise ValueError("Invalid start_time format")

    if start_time.hour < OPEN_HOUR or start_time.hour >= CLOSE_HOUR:
        return False, "Outside opening hours"

    return True, None


# =========================
# DB CONNECTION
# =========================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_basic_restaurant():
    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing tables (for clean re-init)
    cursor.execute("DELETE FROM reservation_tables")
    cursor.execute("DELETE FROM reservations")
    cursor.execute("DELETE FROM tables")

    tables_data = [
        # Window (good for small romantic bookings)
        ("W1", 2, "window"),
        ("W2", 2, "window"),
        ("W3", 4, "window"),

        # Main hall
        ("M1", 4, "main"),
        ("M2", 4, "main"),
        ("M3", 6, "main"),
        ("M4", 6, "main"),

        # Patio
        ("P1", 2, "patio"),
        ("P2", 4, "patio"),

        # Private / large
        ("R1", 8, "private"),
        ("R2", 10, "private"),
        ("R3", 12, "private"),
    ]

    cursor.executemany("""
        INSERT INTO tables (name, seats, zone)
        VALUES (?, ?, ?)
    """, tables_data)

    conn.commit()
    conn.close()

    print("✅ Basic restaurant initialized with 12 tables.")

def seed_random_reservations(count=30):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(seats) FROM tables")
    max_capacity = cursor.fetchone()[0] or 20

    names = ["john", "emma", "liam", "olivia", "noah", "ava", "mason", "mia"]
    
    for _ in range(count):
        name = random.choice(names) + str(random.randint(1, 999))
        phone = str(random.randint(1000000000, 9999999999))
        
        party_size = random.randint(2, min(10, max_capacity))

        # Random time within next 2 days
        base = datetime.now()
        random_minutes = random.randint(0, 5 * 60)
        start_time = base + timedelta(minutes=random_minutes)

        create_reservation(name, phone, party_size, start_time)

    conn.close()
    print(f"✅ Seeded {count} random reservations.")
# =========================
# TIME OVERLAP CHECK
# =========================

def time_overlap(start1, end1, start2, end2):
    return start1 < end2 and start2 < end1

def cleanup_expired_reservations():
    conn = get_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    cursor.execute("""
        DELETE FROM reservations
        WHERE end_time < ?
    """, (now_str,))

    conn.commit()
    conn.close()

# =========================
# CHECK TABLE AVAILABILITY
# =========================

def find_available_table(conn, party_size, start_time, end_time):
    cursor = conn.cursor()

    query = """
    SELECT t.id, t.name, t.seats, t.zone
    FROM tables t
    WHERE t.seats >= ?
    AND NOT EXISTS (
        SELECT 1
        FROM reservation_tables rt
        JOIN reservations r ON r.id = rt.reservation_id
        WHERE rt.table_id = t.id
        AND r.start_time < ?
        AND r.end_time   > ?
    )
    ORDER BY t.seats ASC
    LIMIT 1
    """

    cursor.execute(query, (
        party_size,
        end_time.strftime("%Y-%m-%d %H:%M"),
        start_time.strftime("%Y-%m-%d %H:%M")
    ))

    row = cursor.fetchone()

    if row:
        return {
            "id": row[0],
            "name": row[1],
            "seats": row[2],
            "zone": row[3]
        }

    return None
def get_tables_status_at_time(check_time):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            t.id,
            t.name,
            t.seats,
            t.zone,
            CASE 
                WHEN EXISTS (
                    SELECT 1
                    FROM reservation_tables rt
                    JOIN reservations r 
                        ON r.id = rt.reservation_id
                    WHERE rt.table_id = t.id
                    AND r.start_time < ?
                    AND r.end_time > ?
                )
                THEN 1
                ELSE 0
            END AS reserved
        FROM tables t
    """, (check_time, check_time))

    tables = cursor.fetchall()
    conn.close()

    return tables
def check_availability(party_size, start_time):
    conn = get_connection()

    table = find_available_table(conn, party_size, start_time)

    conn.close()

    if table:
        return {
            "status": "available",
            "table": table["name"]
        }
    else:
        return {
            "status": "not_available"
        }
def find_reservations(name=None, phone=None, start_time=None):
    conn = get_connection()
    cursor = conn.cursor()
    if name:
        name = name.lower()

    query = "SELECT * FROM reservations WHERE 1=1"
    params = []

    if name:
        query += " AND customer_name = ?"
        params.append(name)

    if phone:
        query += " AND phone = ?"
        params.append(phone)

    if start_time:
        query += " AND start_time = ?"
        params.append(start_time)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "customer_name": row["customer_name"],
            "phone": row["phone"],
            "party_size": row["party_size"],
            "start_time": row["start_time"],  # keep as string
        })

    return results
def find_alternative_tables(conn,party_size,start_time,duration_hours=2):
    suggestions = []
    offsets = [30,-30,60,-60,90,-90]

    for minutes in offsets:
        new_start = start_time + timedelta(minutes=minutes)
        new_end = new_start + timedelta(hours=duration_hours)

        single = find_available_table(conn, party_size, new_start, new_end)
        
        if single:
            tables = [single]
        else:
            tables = find_available_tables_combined(conn, party_size, new_start, new_end)

        if tables:
            suggestions.append({
                "start_time": new_start.strftime("%Y-%m-%d %H:%M"),
                "end_time": new_end.strftime("%Y-%m-%d %H:%M")
            })
        
        if len(suggestions) > 3:
            break

    return suggestions
def find_available_tables_combined(conn,party_size,start_time,end_time):
    """
    if no single tables is fit, we will combine multiple tables
    can not combine in different zones
        1) get all the available tables
        2) sort by smallest
        3) combine until total seats >= party_size

    """
    
    cursor = conn.cursor()
    print("execute combine")
    # Get all free tables at this time
    cursor.execute("""
        SELECT 
            t.id,
            t.name,
            t.seats,
            t.zone
        FROM tables t
        WHERE NOT EXISTS (
            SELECT 1
            FROM reservation_tables rt
            JOIN reservations r ON r.id = rt.reservation_id
            WHERE rt.table_id = t.id
            AND r.start_time < ?
            AND r.end_time > ?
        )
    """, (start_time, end_time))
    print("fetching")
    free_tables = cursor.fetchall()
    print("grouping by zone")
    # Group by zone
    grouped = defaultdict(list)
    for table in free_tables:
        grouped[table[3]].append(table)  # table[3] = zone
    print("checking each zone")
    # Try each zone separately
    for zone, tables in grouped.items():
        print("sort small to large")
        # sort small to large (minimize wasted seats)
        tables.sort(key=lambda x: x[2])
        print("try combining up to 3 tables")
        # Try combinations up to 3 tables (safe limit)
        for r in range(1, min(4, len(tables)+1)):
            for combo in itertools.combinations(tables, r):
                total_seats = sum(t[2] for t in combo)
                if total_seats >= party_size:
                    return list(combo)
    print("ends")
    return None

# =========================
# CREATE RESERVATION
# =========================

def create_reservation(name, phone, party_size, start_time_value):
    conn = get_connection()
    cursor = conn.cursor()
    if name:
        name = name.lower()

    if isinstance(start_time_value, str):
        start_time = datetime.strptime(start_time_value, "%Y-%m-%d %H:%M")
    elif isinstance(start_time_value, datetime):
        start_time = start_time_value
    else:
        raise ValueError("Invalid start_time format")
    
    start_time = start_time.replace(second=0, microsecond=0)
    end_time = start_time + timedelta(hours=2)
    
    try:
        #It acquires write lock immediately
        #Prevents double booking
        #Safer than default deferred transaction
        conn.execute("BEGIN IMMEDIATE")
        print("single")
        single = find_available_table(conn, party_size, start_time, end_time)

        if single:
            tables = [single]
        else:
            print("miltiple tables")
            tables = find_available_tables_combined(conn, party_size, start_time, end_time)

        if not tables:
            print("alternatives")
            alternatives = find_alternative_tables(conn,party_size,start_time)

            conn.rollback()
            return {
                "status": "no_availability",
                "alternatives": alternatives
                }
        print("inserting reservation")
        cursor.execute("""
            INSERT INTO reservations
            (customer_name, phone, party_size, start_time, end_time)
            VALUES (?, ?, ?, ?, ?)
        """, (
            name,
            phone,
            party_size,
            start_time.strftime("%Y-%m-%d %H:%M"),
            end_time.strftime("%Y-%m-%d %H:%M")
        ))

        reservation_id = cursor.lastrowid
        print(f"reserve found  tables")
        print(f"{reservation_id}")

        for table in tables:
            cursor.execute("""
                INSERT INTO reservation_tables (reservation_id, table_id)
                VALUES (?, ?)
            """, (reservation_id, table[0]))
        print("commiting")
        conn.commit()
        print("success!")

        # Convert sqlite3.Row to plain dict
        clean_tables = []

        for t in tables:
            if isinstance(t, sqlite3.Row):
                clean_tables.append(dict(t))
            else:
                # If tuple
                clean_tables.append({
                    "id": t[0],
                    "name": t[1],
                    "seats": t[2],
                    "zone": t[3]
                })

        return { 
            "status": "success",
            "reservation_id": reservation_id,
            "tables": clean_tables,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M"),
        }
    
    except sqlite3.IntegrityError:
        conn.rollback()
        return {"status": "duplicate",}
    except Exception as e:
        conn.rollback()
        return {"status": f"internal error: {e}"}


# =========================
# CANCEL RESERVATION
# =========================

def cancel_reservation(name=None, phone=None, start_time=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM reservations WHERE 1=1"
    params = []

    if name:
        name = name.lower()
        query += " AND customer_name LIKE ?"
        params.append(f"%{name}%")

    if phone:
        query += " AND phone LIKE ?"
        params.append(f"%{phone}%")

    if start_time:
        query += " AND start_time = ?"
        params.append(start_time)

    cursor.execute(query, params)
    matches = cursor.fetchall()

    if len(matches) == 0:
        conn.close()
        return {"status": Status.NOT_FOUND.value}

    if len(matches) > 1:
        result = [
            {
                "reservation_id": r["id"],
                "customer_name": r["customer_name"],
                "start_time": r["start_time"]
            }
            for r in matches
        ]
        conn.close()
        return {
            "status": Status.MULTIPLE_MATCHES.value,
            "candidates": result
        }

    reservation_id = matches[0]["id"]

    cursor.execute("DELETE FROM reservations WHERE id = ?", (reservation_id,))
    conn.commit()
    conn.close()

    return {"status": Status.SUCCESS.value}

class CreateReservationInput(BaseModel):
    name: str
    phone: str
    party_size: int
    start_time: str

class CancelReservationInput(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    start_time: Optional[str] = None

class CheckAvailabilityInput(BaseModel):
    party_size: int
    start_time: str
# ==================================================
# Langgraph
# ==================================================
# -----------------------
# tools
# -----------------------

def create_reservation_tool(input: CreateReservationInput):
    return create_reservation(
        input.name,
        input.phone,
        input.party_size,
        input.start_time
    )

def cancel_reservation_tool(input: CancelReservationInput):
    return cancel_reservation(
        input.name,
        input.phone,
        input.start_time
    )

def check_availability_tool(input: CheckAvailabilityInput):
    return check_availability(
        input.party_size,
        input.start_time
    )
# -----------------------
# Configuration
# -----------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]
    action: Optional[str]
    name: Optional[str]
    phone: Optional[str]
    party_size: Optional[int]
    start_time: Optional[str]
    start_time_raw: Optional[str]
    missing_fields: List[str]
    tool_results: Optional[dict]
    
    loop_count: int
    candidates: List[str]
    target_reservation: str
    identity_failed: bool

# -----------------------
# Nodes
# -----------------------
intent_llm = ChatOllama(
    model="qwen2.5:7b-instruct-q4_K_M",
    temperature=0.0
)

extract_llm = ChatOllama(
    model="qwen2.5:7b-instruct-q4_K_M",
    temperature=0.0
)

conversation_llm = ChatOllama(
    model="qwen2.5:7b-instruct-q4_K_M",
    temperature=0.3
)

def get_last_user_message(state):
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return msg.content
    return ""

class ExtractSchema(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    party_size: Optional[int] = None
    start_time_raw: Optional[str] = None

def parse_time(raw_text):
    try:
        dt = parser.parse(raw_text, fuzzy=True)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return None 
    
def detect_intend(state: AgentState):
    user_input = get_last_user_message(state)
    prev_action = state.get("action", "UNKNOWN")
    detect_prompt = f"""
You are an intent classification engine for a restaurant system.

Your task:
Classify the user's latest message into ONE of the following categories:

RESERVE  → User wants to create a reservation
CANCEL   → User wants to cancel an existing reservation
CHECK    → User wants to check availability
UNKNOWN  → Anything else
Ground rule:
Consider the previous intent: {prev_action}
dont change the intent unless user has explicitly meant a different intention!
Rules:
- Output ONLY one word from the list.
- Do NOT explain.
- Do NOT add punctuation.
- Do NOT add extra text.
- If unsure, return UNKNOWN.

Examples:

User: I want a table for 4 tomorrow at 7
Output: RESERVE

User: Cancel my booking at 8
Output: CANCEL

User: Do you have space at 9pm?
Output: CHECK

User: Hello
Output: UNKNOWN                              
"""
    response = intent_llm.invoke([
        SystemMessage(content=detect_prompt),
        HumanMessage(content=user_input)
    ])

    print(f"\t\t[detect_intent] -> {response.content}")
    action = response.content.strip().upper()

    if action not in ["RESERVE", "CANCEL", "CHECK", "UNKNOWN"]:
        action = "UNKNOWN"
    
    if action != "UNKNOWN" and prev_action == "UNKNOWN":
        print("llm action")
        state["action"] = action
    else:
        print("prev_action")
        state["action"] = prev_action
    print(f"final action -> {state["action"]}")
    return state

def extract_fields(state: AgentState):
    recent_messages = state["messages"][-5:]
    extract_prompt = f"""
You are a strict information extraction engine.

Extract:

- name
- phone
- party_size
- start_time_raw (string, exact phrase from user like "today at 5pm")

Rules:
- Return ONLY valid JSON.
- Missing fields must be null.
- Never guess.
- Never default to current time.
- If the user did not explicitly specify a date/time, set start_time to null.

If conversion is uncertain → return null.
"""
    structured_llm  = extract_llm .with_structured_output(ExtractSchema)

    response = structured_llm.invoke(
        [SystemMessage(content=extract_prompt)] + recent_messages
    )

    extracted = response.model_dump()
    if extracted['start_time_raw'] is not None:
        start_time = parse_time(extracted['start_time_raw'])
        print(f"\t\t[extract_fields] -> {extracted}, start_time: {start_time}")
        state["start_time"] = start_time
    else:
        print(f"\t\t[extract_fields] -> {extracted}, start_time: None")
    for field, value in extracted.items():
        if value is not None:
            state[field] = value
    return state

def check_missing(state: AgentState):

    required = {
        "RESERVE": ["name", "phone", "party_size", "start_time"],
        "CHECK": ["name", "phone", "start_time"],
    }

    if state["action"] == "CANCEL":
        handle_cancel_identity_resolution(state)
        missing = state["missing_fields"]
    else:
        missing = []

        if state["action"] in required:
            for field in required[state["action"]]:
                if not state.get(field):
                    missing.append(field)
        state["missing_fields"] = missing

    print(f"\t\t[check_missing] -> {state['missing_fields']}")
    return state

def route_after_check(state: AgentState):
    if state["missing_fields"]:
        return "ask_missing"
    return "execute_tool"

def ask_missing(state: AgentState):
    state["loop_count"] += 1
    
    if state["loop_count"] > 6:
        state["messages"].append(
            AIMessage(content="I'm having trouble finding that reservation. Let’s start fresh — how can I help you today?")
        )
        return state
    
    missing = state["missing_fields"]
    action = state.get("action")
    candidates = state.get("candidates")
    identity_failed = state.get("identity_failed")

    context_blocks = []

    context_blocks.append(f"action: {action}")
    context_blocks.append(f"Missing fields: {missing}")

    if candidates:
        options = []
        for i, r in enumerate(candidates, 1):
            options.append(
                f"{i}. {r['customer_name']} - "
                f"{parse_time(r['start_time'])} - "
                f"Party of {r['party_size']}"
            )
        context_blocks.append("Multiple reservations found:")
        context_blocks.append("\n".join(options))

    if identity_failed:
        context_blocks.append("No reservation matched the provided details.")

    context_blocks.append(
        f"""Already known:
Name: {state.get("name")}
Phone: {state.get("phone")}
Party size: {state.get("party_size")}
Time: {state.get("start_time")}"""
    )

    dynamic_context = "\n\n".join(context_blocks)
    print(f"dynamic_context -> {dynamic_context}")

    ask_prompt = """
You are a professional and friendly restaurant host.

The guest is either:
- Making a reservation
- Cancelling a reservation

Your task:
Ask naturally and conversationally for ONLY the information needed next.

Rules:
- Be warm and human.
- Do not mention system logic.
- Do not repeat known details.
- If disambiguating between multiple reservations, clearly present options and ask the guest to identify one.
- If no reservation was found, gently suggest they may have used different details.
- Keep it concise.

If one field is missing → ask directly.
If multiple fields are missing → combine smoothly.
If identity failed → gently request clarification.
"""

    response = conversation_llm.invoke([
        SystemMessage(content=ask_prompt),
        HumanMessage(content=dynamic_context)
    ])

    print(f"\t\t[ask_missing] -> {missing}")
    print(f"\t\t response: {response.content}")

    state["messages"].append(AIMessage(content=response.content))
    return state

def execute_tool(state: AgentState):
    action = state["action"]
    print(f"\t\t[execute_tool] -> {action}")
    if action == "RESERVE":
        result = create_reservation(
            state.get("name"),
            state.get("phone"),
            state.get("party_size"),
            state.get("start_time")
        )

    elif action == "CANCEL":
        result = cancel_reservation(
            state.get("name"),
            state.get("phone"),
            state.get("start_time")
        )

    elif action == "CHECK":
        result = check_availability(
            state.get("party_size"),
            state.get("start_time")
        )

    elif action == "UNKNOWN":
        result = {
            "status": "unknown_intent",
            "message": "I didn't understand the request."
        }

    else:
        # Safety firewall
        result = {
            "status": "invalid_action",
            "message": f"Unsupported action: {action}"
        }
    state["tool_results"] = result
    return state

def intersept_results(state: AgentState):
    action = state["action"]    
    result = state["tool_results"]
    print(f"\t\t[intersept_results] -> {result}")
    intersept_prompt = f"""
You are a professional restaurant host assistant.

You are given a backend result JSON.
Generate a clear and polite response for the customer.
action taken: {action}
Rules:
- Be concise.
- Be professional.
- Do not expose internal system status codes.
- If reservation is successful, confirm details.
- If no table available, suggest alternatives if provided.
- If multiple matches found, ask for clarification.
- If not found, politely inform the user.
"""
    response = conversation_llm .invoke([
        SystemMessage(content=intersept_prompt),
        HumanMessage(content=json.dumps(result))
    ])

    print(f"\t[response content] ->{response.content}")
    state["messages"].append(AIMessage(content=response.content))

    return state

def handle_cancel_identity_resolution(state: AgentState):
    """
    Resolves reservation identity for cancellation.
    Determines whether we have:
        - exactly one match (ready to cancel)
        - multiple matches (need disambiguation)
        - no matches (need more identity or failed identity)
    Updates state with:
        - target_reservation
        - candidates
        - missing_fields
        - identity_failed
    """

    name = state.get("name")
    phone = state.get("phone")
    start_time = state.get("start_time")
    print(f"\t\t[cancel missing]({name,phone,start_time})",end="")
    # Query using available identity
    matches = find_reservations(
        name=name,
        phone=phone,
        start_time=start_time
    )

    state["candidates"] = []
    state["missing_fields"] = []
    state["identity_failed"] = False
    
    if not name and not phone and not start_time:
        print("no fields at all asking for name")    
        state["missing_fields"] = ["name"]
        return state
    # ------------------------------------------------
    # CASE 1 — Exactly one match → Ready
    # ------------------------------------------------
    if len(matches) == 1:
        print("-> 1 match")
        state["target_reservation"] = matches[0]
        return state

    # ------------------------------------------------
    # CASE 2 — Multiple matches → Need disambiguation
    # ------------------------------------------------
    if len(matches) > 1:
        print("-> multiple matches")
        state["candidates"] = matches

        # Ask for the most discriminating missing field
        if not start_time:
            print("no start")
            state["missing_fields"] = ["start_time"]
        elif not phone:
            print("no phone")
            state["missing_fields"] = ["phone"]
        else:
            # We already have all identity but still ambiguous
            print("confirmation choice")
            state["missing_fields"] = ["confirmation_choice"]

        return state

    # ------------------------------------------------
    # CASE 3 — No matches found
    # ------------------------------------------------
    if len(matches) == 0:
        print("-> no matches")

        # If no identity at all → ask for name first
        if not name and not phone and not start_time:
            state["missing_fields"] = ["name"]
            return state

        # If name given but no match → try narrowing with phone
        if name and not phone:
            state["missing_fields"] = ["phone"]
            return state

        # If phone given but no match → try name
        if phone and not name:
            state["missing_fields"] = ["name"]
            return state

        # If name + phone given but no match → maybe wrong date
        if name and phone and not start_time:
            state["missing_fields"] = ["start_time"]
            return state

        # If everything provided and still no match → identity failure
        state["identity_failed"] = True
        state["missing_fields"] = ["identity_retry"]

        return state

    return state


# -----------------------
# Graph
# -----------------------
workflow = StateGraph(AgentState)

workflow.add_node("detect_intend", detect_intend)
workflow.add_node("extract_fields", extract_fields)
workflow.add_node("check_missing", check_missing)
workflow.add_node("ask_missing", ask_missing)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("intersept_results", intersept_results)
workflow.set_entry_point("detect_intend")

workflow.add_edge("detect_intend","extract_fields")
workflow.add_edge("extract_fields","check_missing")
workflow.add_conditional_edges(
    "check_missing",
    route_after_check,
    {
        "ask_missing": "ask_missing",
        "execute_tool": "execute_tool"
    }
)
workflow.add_edge("ask_missing", END)#reaching out to the user, and redoing the whole steps if action was changed
workflow.add_edge("execute_tool", "intersept_results")
workflow.add_edge("intersept_results", END)


# ----------------------------------------------
# Tkinter App
# ----------------------------------------------


class RestaurantApp:

    def __init__(self, root):
        self.step = 1

        self.root = root
        self.root.title("AI Restaurant Manager")
        # -------- set Time --------
        self.time_var = tk.StringVar()
        self.time_var.set(datetime.now().strftime("%Y-%m-%d %H:%M"))

        self.time_frame = tk.Frame(root)
        self.time_frame.pack(padx=10)
        self.time_check = tk.Label(self.time_frame, text="Select Time:").pack(side=tk.LEFT)
        
        self.time_entry = tk.Entry(self.time_frame, textvariable=self.time_var, width=20)
        self.time_entry.pack(side=tk.LEFT, padx=5)

        self.refresh_button = tk.Button(
            self.time_frame, text="Refresh", command=self.update_layout
        )
        self.refresh_button.pack(side=tk.LEFT)

        # -------- Layout Frame --------
        self.layout_frame = tk.Frame(root)
        self.layout_frame.pack(padx=20)
        self.table_widgets = {}

        self.refresh_layout()
        # -------- Chat Frame --------
        self.agent_state = {
            "messages": [],
            "loop_count": 0
        }

        self.chat_frame = tk.Frame(root)
        self.chat_frame.pack(padx=10, pady=5)

        self.chat_history = scrolledtext.ScrolledText(
            self.chat_frame,
            height=10,
            width=60,
            state="disabled"
        )
        self.chat_history.pack()

        # -------- Input Frame --------
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(padx=10, pady=5)

        self.user_input = tk.Entry(self.input_frame, width=45)
        self.user_input.pack(side=tk.LEFT, padx=5)

        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack(side=tk.LEFT)

        # Initial layout load
        self.update_layout()

    # -----------------------
    # Tables Handling
    # -----------------------
    def highlight_reservation_tables(self, reservation_id,popup):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_id
            FROM reservation_tables
            WHERE reservation_id = ?
        """, (reservation_id,))

        table_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Highlight them
        for tid in table_ids:
            if tid in self.table_widgets:
                self.table_widgets[tid].config(bg="orange")

        # Auto restore after 3 seconds
        popup.protocol("WM_DELETE_WINDOW", lambda: self.close_popup(popup))
    
    def close_popup(self, popup):
        popup.destroy()
        self.update_layout()

    def cancel_reservation(self, reservation_id, popup_window):
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM reservations
            WHERE id = ?
        """, (reservation_id,))

        conn.commit()
        conn.close()

        popup_window.destroy()
        self.update_layout()

    def open_reservation_popup(self, table_id):
        check_time = self.time_var.get()

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT r.id, 
                r.customer_name,
                r.phone,
                r.party_size,
                r.start_time,
                r.end_time
            FROM reservations r
            JOIN reservation_tables rt
                ON r.id = rt.reservation_id
            WHERE rt.table_id = ?
            AND r.start_time < ?
            AND r.end_time > ?
        """, (table_id, check_time, check_time))

        reservation = cursor.fetchone()
        conn.close()

        if not reservation:
            return

        reservation_id = reservation[0]
        name = reservation[1]
        phone = reservation[2]
        party_size = reservation[3]
        start_time = reservation[4]
        end_time = reservation[5]


        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Reservation Details")
        popup.geometry("300x220")
        popup.resizable(False, False)

        self.highlight_reservation_tables(reservation_id,popup)

        tk.Label(popup, text="Reservation Details",
                font=("Arial", 12, "bold")).pack(pady=10)

        tk.Label(popup, text=f"Name: {name}").pack(pady=3)
        tk.Label(popup, text=f"Phone: {phone}").pack(pady=3)
        tk.Label(popup, text=f"Party Size: {party_size}").pack(pady=3)
        tk.Label(popup, text=f"Start: {start_time}").pack(pady=3)
        tk.Label(popup, text=f"End: {end_time}").pack(pady=3)

        tk.Button(
            popup,
            text="Cancel Reservation",
            fg="white",
            bg="red",
            command=lambda: self.cancel_reservation(reservation_id, popup)
        ).pack(pady=5)

    def refresh_layout(self):
        self.table_widgets = {}
        for widget in self.layout_frame.winfo_children():
            widget.destroy()

        check_time = self.time_var.get()

        try:
            tables = get_tables_status_at_time(check_time)
        except Exception as e:
            print("Time format error:", e)
            return

        # Sort tables by zone + seats
        zone_order = {
            "window": 0,
            "main": 1,
            "patio": 2,
            "private": 3
        }

        tables.sort(key=lambda x: (zone_order.get(x[3], 99), x[2]))

        # Group tables by zone
        from collections import defaultdict
        grouped = defaultdict(list)

        for table in tables:
            grouped[table[3]].append(table)  # table[3] = zone

        # Draw each zone as its own section
        for zone in sorted(grouped.keys(), key=lambda z: zone_order.get(z, 99)):

            zone_frame = tk.Frame(self.layout_frame, bd=2, relief="groove", padx=10, pady=10)
            zone_frame.pack(fill="x", pady=10)

            # -------- Zone Title --------
            tk.Label(
                zone_frame,
                text=f"{zone.upper()} AREA",
                font=("Arial", 13, "bold")
            ).pack(pady=(0, 10))

            # -------- Tables Container --------
            tables_frame = tk.Frame(zone_frame)
            tables_frame.pack()

            zone_tables = grouped[zone]
            total_tables = len(zone_tables)

            max_per_row = 4
            rows = (total_tables + max_per_row - 1) // max_per_row

            index = 0
            for r in range(rows):
                row_frame = tk.Frame(tables_frame)
                row_frame.pack()

                remaining = total_tables - index
                tables_in_this_row = min(max_per_row, remaining)

                # Center row by adding side padding
                empty_slots = max_per_row - tables_in_this_row
                left_padding = empty_slots // 2

                for _ in range(left_padding):
                    tk.Label(row_frame, width=12).pack(side="left", padx=10)

                for _ in range(tables_in_this_row):
                    table_id, name, seats, zone_name, reserved = zone_tables[index]
                    index += 1

                    color = "red" if reserved else "green"

                    lbl = tk.Label(
                        row_frame,
                        text=f"{name}\n{seats} seats",
                        bg=color,
                        fg="white",
                        width=12,
                        height=4,
                        relief="raised",
                        font=("Arial", 10, "bold"),
                        cursor="hand2" if reserved else "arrow"
                    )

                    lbl.pack(side="left", padx=10, pady=5)
                    
                    self.table_widgets[table_id] = lbl

                    # Make reserved table clickable
                    if reserved:
                        lbl.bind(
                            "<Button-1>",
                            lambda e, table_id=table_id: self.open_reservation_popup(table_id)
                        )


    # -----------------------
    # Chat Handling
    # -----------------------

    def send_message(self):
        user_text = self.user_input.get()
        if not user_text.strip():
            return

        self.append_chat("You", user_text)

        #response (LLM)
        self.agent_state["messages"].append(HumanMessage(content=user_text))
        
        response = chat_app.invoke(self.agent_state, config)
        
        self.agent_state = response
        self.step = 1 
        for msg in response["messages"]:
            
            # Check if the AI decided to call a tool
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"Step {self.step}: [ACTION] Calling Tool '{tc['name']}' with {tc['args']}")
            
            # Check if this message is the actual output from a tool
            if msg.type == "tool":
                print(f"Step {self.step}: [OBSERVATION] Tool returned: {msg.content[:100]}")

            if msg.type == "human":
                print(f"Step {self.step}: [USER INPUT] {msg.content[:100]}")

            # The final answer
            if msg.type == "ai" and not msg.tool_calls:
                print(f"Step {self.step}: [FINAL RESPONSE] {msg.content[:100]}")
            self.step += 1

        messages = response["messages"]
        last_ai_message = None
        for msg in reversed(messages):
            if msg.type == "ai":
                last_ai_message = msg.content
                break

        if last_ai_message:
            self.append_chat("AI", last_ai_message)

        self.user_input.delete(0, tk.END)

        # Refresh layout after each interaction
        self.update_layout()

    def append_chat(self, sender, message):
        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.yview(tk.END)

    # -----------------------
    # Layout Refresh
    # -----------------------

    def update_layout(self):
        cleanup_expired_reservations()
        self.refresh_layout()

#init_basic_restaurant()
checkpointer = MemorySaver()
chat_app = workflow.compile(checkpointer=checkpointer)

config = {"configurable" : {"thread_id" : "test"}}
#seed_random_reservations(20)
print(chat_app.get_graph().draw_ascii())


root = tk.Tk()
app = RestaurantApp(root)
root.mainloop()
