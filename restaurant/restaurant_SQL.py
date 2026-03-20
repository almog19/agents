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
import dateparser
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
BUFFER_MINUTES = 15

# ==================================================
# Restaurant facts — single source of truth
# The manager uses these to validate every AI output
# ==================================================
RESTAURANT_FACTS = {
    "name": "La Bella Cucina",
    "open":  f"{OPEN_HOUR}:00",
    "close": f"{CLOSE_HOUR}:00",
    "hours": f"{OPEN_HOUR}:00 – {CLOSE_HOUR}:00",
    "location": "123 Olive Street, Tel Aviv",
    "phone": "+972-3-555-1234",
}


# =========================
# STATUS ENUMS
# =========================

class Status(Enum):
    SUCCESS_RESERVE = "RESERVED"
    SUCCESS_CANCEL = "CANCELLED"
    INVALID_INPUT = "INVALID_INPUT"
    NO_TABLE_AVAILABLE = "NO_TABLE_AVAILABLE"
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
    ALREADY_EXISTS = "ALREADY_EXISTS"

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

def normalize_time(dt):
    return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)


# =========================
# CHECK TABLE AVAILABILITY
# =========================

def find_available_table(conn, party_size, start_time, end_time):
    cursor = conn.cursor()
    start_time = normalize_time(start_time)
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
def check_availability(party_size, start_time_value):
    conn = get_connection()
    
    if isinstance(start_time_value, str):
        start_time = datetime.strptime(start_time_value, "%Y-%m-%d %H:%M")
    elif isinstance(start_time_value, datetime):
        start_time = start_time_value
    else:
        raise ValueError("Invalid start_time format")
    start_time = normalize_time(start_time)
    end_time = start_time + timedelta(hours=2)
    #print(f"times {start_time} -> {end_time}")
    single = find_available_table(conn, party_size, start_time, end_time)

    if single:
        tables = [single]
    else:
        tables = find_available_tables_combined(conn, party_size, start_time, end_time)

    if not tables:
        alternatives = find_alternative_tables(conn,party_size,start_time)

        return {
            "status": "no_availability",
            "alternatives": alternatives
            }

    conn.close()

    if tables:
        tables_names = ""
        for table in tables:
            tables_names += f"{table["name"]}, "
        return {
            "status": "available",
            "tables": tables_names
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
def find_alternative_tables(conn, party_size, start_time, duration_hours=2):

    suggestions = []

    now = datetime.now()
    buffer_minutes = 15
    earliest_allowed = now + timedelta(minutes=buffer_minutes)

    search_window_before = 60
    search_window_after = 120
    step_minutes = 15

    start_range = start_time - timedelta(minutes=search_window_before)
    end_range = start_time + timedelta(minutes=search_window_after)

    t = start_range

    while t <= end_range:

        # Skip past times and closed time
        if t < earliest_allowed or t.hour < OPEN_HOUR:
            t += timedelta(minutes=step_minutes)
            continue

        new_start = t
        new_end = new_start + timedelta(hours=duration_hours)

        #no further search(close hour)
        if new_end.hour > CLOSE_HOUR and new_end.hour < OPEN_HOUR:
            break

        single = find_available_table(conn, party_size, new_start, new_end)

        if single:
            tables = [single]
        else:
            tables = find_available_tables_combined(conn, party_size, new_start, new_end)

        if tables:

            suggestions.append({
                "start_time": new_start.strftime("%H:%M"),
                "tables": tables
            })

        if len(suggestions) >= 3:
            break

        t += timedelta(minutes=step_minutes)

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
    #print("execute combine")
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
    """, (end_time.strftime("%Y-%m-%d %H:%M"),
        start_time.strftime("%Y-%m-%d %H:%M")
    ))
    #print("fetching")
    free_tables = cursor.fetchall()
    #print("grouping by zone")
    # Group by zone
    grouped = defaultdict(list)
    for table in free_tables:
        grouped[table[3]].append(table)  # table[3] = zone
    #print("checking each zone")
    # Try each zone separately
    for zone, tables in grouped.items():
        #print("sort small to large")
        # sort small to large (minimize wasted seats)
        tables.sort(key=lambda x: x[2])
        #print("try combining up to 3 tables")
        # Try combinations up to 3 tables (safe limit)
        for r in range(1, min(4, len(tables)+1)):
            for combo in itertools.combinations(tables, r):
                total_seats = sum(t[2] for t in combo)
                if total_seats >= party_size:
                    return list(combo)
    #print("ends")
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
    start_time = normalize_time(start_time)
    start_time = start_time.replace(second=0, microsecond=0)
    end_time = start_time + timedelta(hours=2)
    
    try:
        #It acquires write lock immediately
        #Prevents double booking
        #Safer than default deferred transaction
        conn.execute("BEGIN IMMEDIATE")
        single = find_available_table(conn, party_size, start_time, end_time)

        if single:
            print("single")
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
        print(f"reserve found tables {reservation_id}")
        print(f"table {tables[0]}, {tables[0]['id']}")
        for table in tables:
            cursor.execute("""
                INSERT INTO reservation_tables (reservation_id, table_id)
                VALUES (?, ?)
            """, (reservation_id, table['id']))
        print("commiting")
        conn.commit()
        print("success!")

        return { 
            "status": "RESERVED",
            "reservation_id": reservation_id,
            "tables": tables,
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

def find_reservation(name=None, phone=None, start_time=None):
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
    print(f"[MATCHES] {matches}")
    # zero matches
    if len(matches) == 0:
        conn.close()
        return {"status": Status.NOT_FOUND.value}

    #multiple matches
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

    #one match
    match = matches[0]
    result = {
        "reservation_id": match["id"],
        "customer_name": match["customer_name"],
        "phone_number": match["start_time"],
        "start_time": match["start_time"]
    }
    return {
        "status": Status.FOUND.value,
        "details": result
    }


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

    return {"status": Status.SUCCESS_CANCEL.value}
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
    action_pivot: Optional[str]
    _saved_action: Optional[str]
    confirmed: Optional[bool]
    _cache_key: Optional[str]

    _prev_action: Optional[str]
    name: Optional[str]
    phone: Optional[str]
    party_size: Optional[int]
    start_time: Optional[str]
    start_time_raw: Optional[str]

    missing_fields: List[str]
    _fields_changed: List[str]
    field_corrected: List[str]

    cancel_candidates: Optional[dict]

    tool_result: Optional[dict]
    output_reason: Optional[str]

    is_info_question: Optional[bool]

    loop_count: int
    candidates: List[str]


# -----------------------
# helpers
# -----------------------
def last_human(state):
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return msg.content
    return "no user message"

def last_ai(state):
    for msg in reversed(state["messages"]):
        if msg.type == "ai":
            return msg.content
    return "no ai message"

class ExtractSchema(BaseModel):
    # Field values
    name:           Optional[str]  = None
    phone:          Optional[str]  = None
    party_size:     Optional[int]  = None
    start_time_raw: Optional[str]  = None

    # Intent signals
    confirmed:        Optional[bool] = None  # True=yes/ok, False=no/cancel, None=neither
    action_pivot:     Optional[str]  = None  # "RESERVE"|"CANCEL"|"CHECK"|"INFO" if user changed goal
    field_corrected:  Optional[str]  = None  # name of field user just changed ("not 5, make it 3")
    is_info_question: bool           = False # "where are you", "what time do you close"


def parse_time(text):
    now = datetime.now()

    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now
        }
    )

    if not parsed:
        return None

    parsed = parsed.replace(second=0, microsecond=0)
    parsed = normalize_time(parsed)

    print(f"\t\t{parsed}")
    return parsed

def validate_input(state):

    errors = {}

    start_time = state.get("start_time")

    if start_time:
        if start_time.hour < OPEN_HOUR or start_time.hour >= CLOSE_HOUR:
            errors["start_time"] = "outside_opening_hours"

    party_size = state.get("party_size")

    if party_size is not None:
        if party_size <= 0:
            errors["party_size"] = "party_size is zero or negative"

    phone = state.get("phone")

    if phone:
        if not re.match(r"^\+?\d{8,15}$", phone):
            errors["phone"] = "invalid_phone phone number"

    if not errors:
        return {"status": "valid"}

    return {
        "status": "invalid",
        "fields": list(errors.keys()),
        "errors": errors
    }

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
    temperature=0.4
)

def manager(state: AgentState):
    """
    - decide the next step
        what the next node
        if current situation varaible enough to provide answer
        even not satisfied answer(missing information, fully booked)
            yes -> call output_praser
            no -> call necessery nodes

implement:
llm + hard constrains

criterion:

decide what to do next
decide whenever to end(return output)
guide:
1) reservation
    - (output)ask: time & party size
    - check availability
    available:
        - (output)ask: name & phone
        - make reservation
    not available:
        has alternatives?
            - yes: (output)suggest them as alternative time
            - no: (output)sorry we are fully booked
2) cancel reservation
    - identify reservation(name/phone/time)
    - (output)recheck it with the user
    - cancel
    - (output) result status

situations:
1) need information
user: can you make a reservation
manager -> determain_porpuse -> reservation
manager -> check_missing -> time & party_size
manager -> Output_praser -> what is time do you want, and how many guest will you be?
manager -> end
output: what is time do you want, and how many guest will you be?
user: today at 17, we will be 5
manager -> extract_fields -> start_time = 2026-03-18 17:00, party_size = 5
manager -> check availability -> {status: unavailable, alternatives [start_time 2026-03-18 17:30]}
manager -> Output_praser -> we are fully booked, we are available at 17:30 today, are you interesting?
manager -> end
output: we are fully booked, we are available at 17:30 today
user: yes
manager -> check_missing -> name & phone
manager -> Output_praser -> can I get your name and phone
manager -> end
output:can I get your name and phone
user: almog, 0501234567
manager -> extract_fields -> name = almog, phone = 0501234567
manager -> execute tool -> {status: RESERVED}
manager -> Output_praser -> reservation was complete, see you today at 17:30 almog
manager -> end
output:reservation was complete, see you today at 17:30 almog

2)
user: can you make a reservation for now?
manager -> extract_purpose -> make reservation
manager -> extract fields -> time = 2026-03-18 17:30
manager -> check missing -> party time
manager -> output_praser -> sure, how many guest will come?(answer the quest + continue the chat with necesseries information to complete the purpose)
manager -> end
output: sure, how many guest will come?
user: where is the restaurant?
manager -> general info -> 123 olive tel aviv(answer)
manager -> output_praser -> we are at 123 olive tel aviv, would you like to continue the reservation?
manager -> end
output: we are at 123 olive tel aviv, would you like to continue the reservation?
user: yes
manager -> check missing -> party_size
manager -> output_praser -> how many guest will you be?
manager -> end
output: how many guest will you be?
user: you know what lets cancel my friend reservation
manager -> extract_purpsoe -> cancel reservation
manager -> ask_missing -> time\\name\\phone
manager -> output -> do you have your friend name or phone or time of reservation?
manager -> end
output:do you have your friend name or phone or time of reservation?
user: yes his name is shiri, reservation is in two hours
manager -> extract_fields -> name = shiri, time = 2026-03-18 19:30
manager -> check_missing -> []
manager -> check_reservation -> {status : found multiple, shiri 0501234567 2026-03-18 17:30, shiri 0551234567 2026-03-18 17:30}
manager -> outpraser -> I have found 2 reservation both shiri at 2026-03-18 17:30, what his the shiri phone number? 0501234567 or 0551234567?
manager -> end
output: I have found 2 reservation both shiri at 2026-03-18 17:30, what his the shiri phone number? 0501234567 or 0551234567?
user: 0501234567
manager -> extract_fields -> phone = 0501234567
manager -> check_reservation -> {found: shiri 0501234567 2026-03-18 17:30}
manager -> cancel reservation -> {status: success}
manager -> output_praser -> successfuly canceled a reservation for shiri!
manager -> end
output: successfuly canceled a reservation for shiri!
    """
    print("[MANAGER]")
    state["loop_count"] = state.get("loop_count", 0) + 1

    if state["loop_count"] > 10:
        print(f"\tMAX LOOP REACHED")
        state["output_reason"] = "loop_limit"
        return state

    action = state.get("action")
    pivot = state.get("action_pivot")
    fields_changed = state.get("_fields_changed", [])
    is_info = state.get("is_info_question", False)
    tr = state.get("tool_result")

    # ── 1. INFO interrupt (mid-flow question about the restaurant) ────────────
    # Highest priority — always answer, then resume whatever was happening
    if is_info:
        print(f"[INFO]")
        state["_saved_action"] = action  # remember where we were
        state["output_reason"] = "info"
        state["is_info_question"] = False  # consume the signal
        return state

    # ── 2. Resume after INFO interrupt ───────────────────────────────────────
    # User was just asking a question, now continuing the flow
    if state.get("_saved_action") and not pivot:
        print(f"\t[CONTINUE FLOW] {action}")
        state["action"] = state["_saved_action"]
        state["_saved_action"] = None
        # Don't return — fall through so manager re-evaluates the restored action

    # ── 3. Action pivot — user changed their goal entirely ───────────────────
    if pivot and pivot != action:
        print(f"\t[PIVOTE] {action} → {pivot}")
        #state = reset_transaction(state)
        state["action"] = pivot
        action = pivot
        # Re-extract whatever fields the pivot message contained
        # (extract_fields already did this, values are in state)

    # ── 4. No intent yet — need to classify ──────────────────────────────────
    if not action:
        print(f"\t[NO INTENTION]")
        state["output_reason"] = None  # signals route_from_manager to call detect_intent
        return state

    # ── 5. Field correction — invalidate cached tool_result ──────────────────
    # e.g. "actually make it 3" — party_size changed, old availability check is stale
    cache_busting_fields = {"party_size", "start_time", "name", "phone"}
    if fields_changed and set(fields_changed) & cache_busting_fields:
        print(f"\t[CORRECTIONS] {fields_changed} — invalidating tool_result")
        state["tool_result"] = None
        state["confirmed"] = None

    # ── 6. UNKNOWN ────────────────────────────────────────────────────────────
    if action == "UNKNOWN":
        state["output_reason"] = "unknown"
        return state

    # ── 7. INFO (direct intent, not mid-flow interrupt) ──────────────────────
    if action == "INFO":
        state["output_reason"] = "info"
        state["action"] = state.get("_saved_action") or None  # reset after answering
        return state

    # =========================================================================
    # RESERVE
    # =========================================================================
    if action == "RESERVE":
        print(f"\t[RESERVATION]")

        # Phase 1 — need party_size + time to check availability
        missing = []
        if not state.get("party_size"): missing.append("party_size")
        if not state.get("start_time"): missing.append("start_time")
        if missing:
            state["missing_fields"] = missing
            state["output_reason"]  = "ask_missing"
            return state

        # Validate before hitting DB
        v = validate_input(state)
        if v["status"] == "invalid":
            print(f"\t\t[INVALID]")
            state["missing_fields"] = list(v["errors"].keys())
            state["_validation_errors"] = v["errors"]
            state["output_reason"] = "invalid_fields"
            return state

        # Phase 2 — run availability check (skip if cached for same party+time)
        cache_key = f"avail_{state['party_size']}_{state['start_time']}"
        if not tr or tr.get("_key") != cache_key:
            print(f"\t\t[AVAILABILITY CHECK] -> ({cache_key})")
            state["output_reason"] = "run_availability"
            state["_cache_key"] = cache_key
            return state

        # Phase 3 — fully booked
        if tr["status"] == Status.NO_AVAILABILITY.value:
            print(f"\t\t\t[INVALIDE]")
            alts = tr.get("alternatives", [])

            # User confirmed they want an alternative slot
            if state.get("confirmed") is True and alts:
                state["start_time"]  = f"{alts[0]['date']} {alts[0]['start_time']}"
                state["tool_result"] = None   # force re-check with new time
                state["confirmed"]   = None
                # Fall through — will re-run availability on next manager pass
                state["output_reason"] = "run_availability"
                state["_cache_key"]    = f"avail_{state['party_size']}_{state['start_time']}"
                return state

            # User declined alternatives or no alternatives exist
            if state.get("confirmed") is False:
                state["output_reason"] = "user_aborted"
                return state

            state["output_reason"] = "no_availability"
            return state

        # Phase 4 — available, need name + phone
        print(f"\t\t\t[AVAILABLE]")
        missing = []
        if not state.get("name"):  missing.append("name")
        if not state.get("phone"): missing.append("phone")
        if missing:
            state["missing_fields"] = missing
            state["output_reason"]  = "ask_missing"
            return state

        # Phase 5 — confirm details before writing to DB
        if state.get("confirmed") is None:
            state["output_reason"] = "confirm_reserve"
            return state
        if state.get("confirmed") is False:
            state["output_reason"] = "user_aborted"
            return state

        # Phase 6 — execute reservation (skip if already done)
        if not tr or tr.get("status") != Status.RESERVED.value:
            state["output_reason"] = "run_reserve"
            return state

        state["output_reason"] = "success_reserve"
        return state

    # =========================================================================
    # CANCEL
    # =========================================================================
    if action == "CANCEL":
        print(f"\t[CANCEL RESERVATION]")

        # Phase 1 — need at least one identifier
        has_any = state.get("name") or state.get("phone") or state.get("start_time")
        if not has_any:
            state["missing_fields"] = ["name_or_phone"]
            state["output_reason"]  = "ask_missing"
            return state

        # Phase 2 — search for reservation (re-run if fields changed or no result yet)
        if not tr or tr.get("_type") != "cancel_search":
            state["output_reason"] = "run_cancel_search"
            return state

        status = tr.get("status")

        # Not found — try asking for another identifier
        if status == Status.NOT_FOUND.value:
            asked  = state.get("_cancel_asked", [])
            for field in ["name", "phone", "start_time"]:
                if not state.get(field) and field not in asked:
                    state.setdefault("_cancel_asked", []).append(field)
                    state["missing_fields"] = [field]
                    state["output_reason"]  = "ask_missing"
                    state["tool_result"]    = None  # re-search next turn
                    return state
            # Exhausted all fields — give up
            state["output_reason"] = "not_found"
            return state

        # Multiple matches — disambiguate
        if status == Status.MULTIPLE_MATCHES.value:
            candidates = tr.get("candidates", [])
            state["cancel_candidates"] = candidates
            missing = []
            if not state.get("start_time"): missing.append("start_time")
            elif not state.get("phone"):    missing.append("phone")
            else:
                state["output_reason"] = "multiple_matches"
                return state
            state["missing_fields"] = missing
            state["output_reason"]  = "ask_missing"
            state["tool_result"]    = None  # re-search with new info
            return state

        # Exactly one match found
        if status == "FOUND":
            state["cancel_reservation"] = tr.get("reservation")

            # Confirm before deleting
            if state.get("confirmed") is None:
                state["output_reason"] = "confirm_cancel"
                return state
            if state.get("confirmed") is False:
                state["output_reason"] = "user_aborted"
                return state

            # Execute cancellation
            if tr.get("status") != Status.CANCELLED.value:
                state["output_reason"] = "run_cancel"
                return state

            state["output_reason"] = "success_cancel"
            return state

    # =========================================================================
    # CHECK
    # =========================================================================
    if action == "CHECK":
        print(f"\t[CHECK]")
        missing = []
        if not state.get("party_size"): missing.append("party_size")
        if not state.get("start_time"): missing.append("start_time")
        if missing:
            state["missing_fields"] = missing
            state["output_reason"]  = "ask_missing"
            return state

        cache_key = f"check_{state['party_size']}_{state['start_time']}"
        if not tr or tr.get("_key") != cache_key:
            state["output_reason"] = "run_availability"
            state["_cache_key"]    = cache_key
            return state

        state["output_reason"] = "check_result"
        return state

    # Fallback
    state["output_reason"] = "unknown"
    return state

def route_from_manager(state: AgentState) -> str:
    action = state.get("action")
    reason = state.get("output_reason")

    if not action:
        return "detect_intent"

    if reason in ("run_availability", "run_reserve", 
                  "run_cancel_search", "run_cancel"):
        return "execute_tool"

    return "output_phraser"  # everything else — manager decided we can respond
"""

So a single user turn might internally do:
```
extract_fields → manager → execute_tool → manager → output_parser → END
```

Or even two DB calls:
```
extract_fields → manager → execute_tool (availability)
                         → manager → execute_tool (reserve)  
                         → manager → output_parser → END
"""

# ==================================================
# NODE — output_parser
# The only node that speaks to the user.
# Reads output_reason + state variables → builds a
# grounded instruction → single constrained LLM call.
# ==================================================

# Maps field names to natural language phrases
FIELD_LABELS = {
    "name":          "their name",
    "phone":         "their phone number",
    "party_size":    "how many guests will be coming",
    "start_time":    "what time they'd like",
    "name_or_phone": "their name or phone number",
}

# Maps validation error keys to natural correction requests  
VALIDATION_MESSAGES = {
    "start_time": f"the time is outside our opening hours ({OPEN_HOUR}:00–{CLOSE_HOUR}:00)",
    "party_size": "the guest count must be a positive number",
    "phone":      "the phone number format looks wrong",
}

def _build_instruction(state: AgentState) -> str:
    """
    Pure logic — reads state, returns a grounded instruction string.
    The LLM only handles phrasing, never facts.
    """
    reason    = state.get("output_reason", "unknown")
    tr        = state.get("tool_result") or {}
    miss      = state.get("missing_fields", [])

    # ── Collected fields (used across multiple cases) ─────────────────────
    name       = state.get("name")
    phone      = state.get("phone")
    party_size = state.get("party_size")
    start_time = state.get("start_time")

    # ─────────────────────────────────────────────────────────────────────
    # ASK MISSING — ask for one field at a time
    # ─────────────────────────────────────────────────────────────────────
    if reason == "ask_missing":
        first = miss[0] if miss else "details"
        label = FIELD_LABELS.get(first, first)

        # If we already know some fields, acknowledge them naturally
        known_parts = []
        if party_size and first != "party_size": 
            known_parts.append(f"{party_size} guests")
        if start_time and first != "start_time":  
            known_parts.append(f"at {start_time}")
        if name and first not in ("name", "name_or_phone"):       
            known_parts.append(f"for {name}")

        if known_parts:
            context = f"You have: {', '.join(known_parts)}."
        else:
            context = ""

        return (
            f"{context} "
            f"Ask the guest for {label}. "
            f"One question only. Maximum 12 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # INVALID FIELDS — tell user what's wrong, ask to fix
    # ─────────────────────────────────────────────────────────────────────
    if reason == "invalid_fields":
        errors = state.get("_validation_errors", {})
        # Translate error keys into human messages
        problems = [
            VALIDATION_MESSAGES.get(field, f"{field} is invalid")
            for field in errors
        ]
        problems_str = " and ".join(problems)
        return (
            f"Tell the guest that {problems_str}. "
            f"Ask them to provide a corrected value. "
            f"Maximum 15 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # CONFIRM RESERVE — read back all details, ask yes/no
    # ─────────────────────────────────────────────────────────────────────
    if reason == "confirm_reserve":
        return (
            f"Read back the reservation details and ask the guest to confirm: "
            f"name={name}, phone={phone}, "
            f"{party_size} guests, at {start_time}. "
            f"Warm and natural. Maximum 25 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # CONFIRM CANCEL — read back what we found, ask to confirm deletion
    # ─────────────────────────────────────────────────────────────────────
    if reason == "confirm_cancel":
        res = state.get("cancel_reservation", {})
        return (
            f"Tell the guest you found this reservation and ask if they want to cancel it: "
            f"name={res.get('customer_name')}, "
            f"phone={res.get('phone')}, "
            f"{res.get('party_size')} guests, "
            f"at {res.get('start_time')}. "
            f"Maximum 25 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # NO AVAILABILITY — fully booked, offer alternatives if any
    # ─────────────────────────────────────────────────────────────────────
    if reason == "no_availability":
        alts = tr.get("alternatives", [])
        if alts:
            # Build exact time strings — LLM must not invent alternatives
            alt_times = ", ".join(
                f"{a['start_time']} on {a.get('date', 'today')}"
                for a in alts[:3]
            )
            return (
                f"Tell the guest we're fully booked at {start_time} for {party_size} people. "
                f"Suggest ONLY these exact available slots: {alt_times}. "
                f"Ask if any of those work. "
                f"Do NOT mention any other times. Maximum 30 words."
            )
        else:
            return (
                f"Tell the guest we're fully booked at {start_time} "
                f"and unfortunately have no alternative slots available. "
                f"Apologise and suggest they try a different day. "
                f"Maximum 20 words."
            )

    # ─────────────────────────────────────────────────────────────────────
    # CHECK RESULT — availability enquiry (not a booking)
    # ─────────────────────────────────────────────────────────────────────
    if reason == "check_result":
        if tr.get("status") == "available":
            tables = tr.get("tables", "a table")
            return (
                f"Tell the guest that {tables} is available for "
                f"{party_size} guests at {start_time}. "
                f"Ask if they'd like to make a reservation. "
                f"Maximum 20 words."
            )
        else:
            alts = tr.get("alternatives", [])
            if alts:
                alt_times = ", ".join(a["start_time"] for a in alts[:3])
                return (
                    f"Tell the guest no table is available at {start_time} "
                    f"for {party_size} people. "
                    f"Mention these nearby available slots: {alt_times}. "
                    f"Maximum 25 words."
                )
            return (
                f"Tell the guest no table is available at {start_time} "
                f"for {party_size} people, and no nearby alternatives either. "
                f"Maximum 15 words."
            )

    # ─────────────────────────────────────────────────────────────────────
    # SUCCESS RESERVE — booking confirmed
    # ─────────────────────────────────────────────────────────────────────
    if reason == "success_reserve":
        tables = tr.get("tables", "a table")
        return (
            f"Warmly confirm the reservation is booked: "
            f"{name}, {party_size} guests, at {start_time} ({tables}). "
            f"Maximum 20 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # SUCCESS CANCEL — cancellation confirmed
    # ─────────────────────────────────────────────────────────────────────
    if reason == "success_cancel":
        res = state.get("cancel_reservation", {})
        res_name = res.get("customer_name", name or "the reservation")
        return (
            f"Confirm that the reservation for {res_name} "
            f"at {res.get('start_time', start_time)} has been cancelled. "
            f"Warm and brief. Maximum 15 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # NOT FOUND — no reservation matched
    # ─────────────────────────────────────────────────────────────────────
    if reason == "not_found":
        tried = [f for f in ["name", "phone", "start_time"] if state.get(f)]
        tried_str = ", ".join(tried) if tried else "the provided details"
        return (
            f"Tell the guest you couldn't find a reservation using {tried_str}. "
            f"Ask if they'd like to try different details or if there's anything else you can help with. "
            f"Maximum 20 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # MULTIPLE MATCHES — ambiguous cancellation
    # ─────────────────────────────────────────────────────────────────────
    if reason == "multiple_matches":
        candidates = state.get("cancel_candidates", [])
        options = " | ".join(
            f"{c['customer_name']} — {c['start_time']} — party of {c['party_size']}"
            for c in candidates[:4]
        )
        return (
            f"Tell the guest you found {len(candidates)} reservations and need to identify the right one. "
            f"List them exactly: {options}. "
            f"Ask which one to cancel. Maximum 35 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # USER ABORTED — they said no to confirmation
    # ─────────────────────────────────────────────────────────────────────
    if reason == "user_aborted":
        action = state.get("action", "")
        action_label = {
            "RESERVE": "reservation",
            "CANCEL":  "cancellation",
        }.get(action, "request")
        return (
            f"Acknowledge that the {action_label} has been cancelled. "
            f"Offer further help warmly. Maximum 12 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # INFO — restaurant facts question
    # ─────────────────────────────────────────────────────────────────────
    if reason == "info":
        saved = state.get("_saved_action")
        resume = ""
        if saved in ("RESERVE", "CANCEL", "CHECK"):
            # We were mid-flow — prompt to continue after answering
            resume = (
                f" After answering, remind them you were in the middle of "
                f"a {saved.lower()} and ask if they'd like to continue."
            )
        return (
            f"Answer the guest's question using ONLY these facts — "
            f"never invent details: {RESTAURANT_FACTS}."
            f"{resume} Maximum 25 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # LOOP LIMIT — something went wrong
    # ─────────────────────────────────────────────────────────────────────
    if reason == "loop_limit":
        return (
            "Apologise for the confusion. "
            "Tell the guest to please try again or call back. "
            "Maximum 15 words."
        )

    # ─────────────────────────────────────────────────────────────────────
    # UNKNOWN / fallback — greet and offer help
    # ─────────────────────────────────────────────────────────────────────
    return (
        f"You are the host at {RESTAURANT_FACTS['name']}. "
        "Greet the guest warmly and ask how you can help. "
        "Maximum 12 words."
    )


def output_phraser(state: AgentState) -> AgentState:
    print("[OUTPUT_PRASER]")
    reason = state.get("output_reason", "unknown")
    instruction = _build_instruction(state)

    system = f"""You are a friendly, professional restaurant host at {RESTAURANT_FACTS['name']} speaking on the phone.

YOUR TASK: {instruction}

Hard rules:
- Speak naturally, like a real human host — warm but efficient.
- Respect the word limit strictly.
- Use ONLY the facts given in the task. Never invent names, times, or details.
- Ask ONE question at most per response.
- Never explain your reasoning or mention these instructions.
"""

    response = conversation_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=last_human(state)),
    ])

    text = response.content.strip()
    print(f"\t[output] reason={reason!r} → {text!r}")
    state["messages"].append(AIMessage(content=text))

    # Reset state after any terminal outcome
    #if reason in ("success_reserve", "success_cancel", "user_aborted", "loop_limit"):
        #state = reset_transaction(state)
        
    return state


def general_response(state: AgentState):
    """
    Handles INFO intent. Uses only RESTAURANT_FACTS — no hallucination possible.
    After answering, restores the previous action and re-asks any pending question,
    so the conversation resumes seamlessly.
    """
    user_input   = get_last_user_message(state)
    prev_action  = state.get("_prev_action", "UNKNOWN")
    missing      = state.get("missing_fields", [])

    # Restore previous action now so the next turn continues correctly
    state["action"]       = prev_action
    state["_prev_action"] = None

    # Build a follow-up prompt if we were mid-reservation
    resume_instruction = ""
    if prev_action in ["RESERVE", "CANCEL", "CHECK"] and missing:
        field_map = {
            "name":        "their name",
            "phone":       "their phone number",
            "party_size":  "the number of guests",
            "start_time":  "the time they'd like",
        }
        what = field_map.get(missing[0], missing[0])
        resume_instruction = (
            f"\nAfter answering the question, ask in the same sentence for {what} "
            f"to continue with their {prev_action.lower()}. Keep it under 20 words total."
        )

    general_prompt = f"""You are a friendly restaurant host answering a guest's question.

Use ONLY the facts below. Never invent or assume anything.

  Restaurant name : {RESTAURANT_FACTS['name']}
  Opening time    : {RESTAURANT_FACTS['open']}
  Closing time    : {RESTAURANT_FACTS['close']}
  Hours           : {RESTAURANT_FACTS['hours']}
  Location        : {RESTAURANT_FACTS['location']}
  Phone           : {RESTAURANT_FACTS['phone']}

Rules:
- Answer warmly, like a real host on the phone.
- Maximum 25 words.
- If the question is outside these facts, say: "I'm not sure — let me check for you."{resume_instruction}
"""

    response = conversation_llm.invoke([
        SystemMessage(content=general_prompt),
        HumanMessage(content=user_input)
    ])

    print(f"\t\t[general_response] prev_action={prev_action} missing={missing} -> {response.content}")
    state["messages"].append(AIMessage(content=response.content))
    return state


def detect_intent(state: AgentState):
    """
    Called only when manager routes here (fresh start / NEEDS_DETECT).
    Classifies the user's intent into a transactional action.
    INFO is never expected here — manager already handles it upstream.
    """
    user_input = get_last_user_message(state)
    detect_prompt = """
You are an intent classification engine for a restaurant reservation system.

Classify the user's message into ONE of:

RESERVE  → User wants to create a reservation
CANCEL   → User wants to cancel a reservation
CHECK    → User wants to check table availability
UNKNOWN  → Anything else (greetings, unclear, off-topic)

Rules:
- Output ONLY one word from the list above.
- No explanation, no punctuation, no extra text.
- If unsure, return UNKNOWN.

Examples:
User: I want a table for 4 tomorrow at 7  → RESERVE
User: Cancel my booking at 8              → CANCEL
User: Do you have space at 9pm?           → CHECK
User: Hello                               → UNKNOWN
"""
    response = intent_llm.invoke([
        SystemMessage(content=detect_prompt),
        HumanMessage(content=user_input)
    ])

    action = response.content.strip().upper()
    print(f"\t\t[detect_intent] -> {action}")

    if action not in ["RESERVE", "CANCEL", "CHECK", "UNKNOWN"]:
        action = "UNKNOWN"

    state["action"] = action
    print(f"\t\tfinal action -> {action}")
    return state


def extract_fields(state: AgentState) -> AgentState:
    print(f"[EXTRACT_FIELDS]")
    prompt = f"""You are an information extraction engine for a restaurant phone system.

The assistant's last message:
{last_ai(state)}

The user's latest message:
{last_human(state)}

Extract the following fields:

FIELD VALUES (extract if present):
- name           : person's name
- phone          : phone number (digits, may have + or dashes)
- party_size     : number of guests (integer)
- start_time_raw : any time reference ("7pm", "tomorrow 8", "in two hours", "now")

INTENT SIGNALS (classify the message):
- confirmed       : true  = user agreed / said yes / ok / sure
                    false = user declined / said no / changed mind
                    null  = neither (they gave info or asked something)

- action_pivot    : if the user is CHANGING what they want entirely, output one of:
                    "RESERVE", "CANCEL", "CHECK", "INFO"
                    null if they are continuing the same task
                    Examples:
                      "actually forget it, I want to cancel instead" → "CANCEL"
                      "where is the restaurant?"                      → null (that's is_info_question)
                      "I want to make a reservation"                  → "RESERVE" only if no action exists yet

- field_corrected : if the user is CORRECTING a previously given value, output the field name.
                    Examples:
                      "actually make it 3 not 5"  → "party_size"
                      "sorry wrong number, it's 0541234567" → "phone"
                      "change the time to 8pm"    → "start_time"
                    null if they are giving new info, not correcting

- is_info_question: true if the user is asking about the restaurant itself:
                    location, address, hours, phone number, name, parking, etc.
                    false otherwise

Rules:
- Return ONLY valid JSON matching the schema.
- null for any field you are not confident about.
- Never guess field values.
"""

    structured = extract_llm.with_structured_output(ExtractSchema)
    extracted  = structured.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=last_human(state))
    ]).model_dump()

    # Track which fields actually changed (for cache invalidation)
    changed_fields = []

    for field, value in extracted.items():
        if field == "start_time_raw":
            if value:
                parsed = parse_time(value)
                if parsed:
                    new_time = parsed
                    if state.get("start_time") != new_time:
                        changed_fields.append("start_time")
                    state["start_time"] = new_time
        elif field in ("confirmed", "action_pivot", "field_corrected", "is_info_question"):
            state[field] = value  # always overwrite signals
        elif value is not None:
            # Check if this is a correction of an existing value
            if state.get(field) is not None and state.get(field) != value:
                changed_fields.append(field)
            state[field] = value

    state["_fields_changed"] = changed_fields  # manager reads this

    print(f"\t[extract] name={state.get('name')} phone={state.get('phone')} "
          f"party={state.get('party_size')} time={state.get('start_time')} "
          f"confirmed={state.get('confirmed')} pivot={state.get('action_pivot')} "
          f"corrected={state.get('field_corrected')} info={state.get('is_info_question')} "
          f"changed={changed_fields}")
    return state

"""
def extract_fields(state: AgentState):
    recent_messages = state["messages"][-4:]
    missing_fields = state.get("missing_fields", [])
    last_question = get_last_ai_message(state)
    last_user = get_last_user_message(state)
    extract_prompt = f"w""
You are a strict information extraction engine for a restaurant reservation system.
important: Extact the field base on the latest QUESTION and the user answer!!
The assistant previously asked for:
{missing_fields}

latest question message:
{last_question}

User's answer message:
{last_user}

Extract these fields if present:

- name
- phone
- party_size
- start_time_raw

Interpretation rules:

If assistant asked for guests and user gives a number → party_size.

If assistant asked for phone and user gives digits → phone.

If assistant asked for name and user gives a word → name.

If assistant asked for time and user gives something relative to time
"now", "in an hour", "7", "7pm", "tonight", "tomorrow 8" → start_time_raw.

Rules:
- Return ONLY JSON.
- Missing fields must be null.
- Never guess unknown values.
"w""
    structured_llm  = extract_llm .with_structured_output(ExtractSchema)
    response = structured_llm.invoke(
        [SystemMessage(content=extract_prompt)] + recent_messages
    )

    extracted = response.model_dump()
    if extracted['start_time_raw'] is not None:
        start_time = parse_time(extracted['start_time_raw'])
        print(f"\t\t[extract_fields] -> {extracted}, start_time: {start_time}")
        print(f"\t\tlast_question: {last_question}")

        state["start_time"] = start_time
    else:
        print(f"\t\t[extract_fields] -> {extracted}, start_time: None")
        print(f"\t\tlast_question: {last_question}")
    for field, value in extracted.items():
        if value is not None:
            state[field] = value
    print(f"\t\t[variables] -> {state.get('name',"-")}, {state.get('phone',"-")}, {state.get('party_size',"-")}, {state.get('start_time_raw',"-")}({state.get('start_time',"-")})")
    return state
"""
def check_missing(state: AgentState):
    missing = []
    print(f"\t\tmissing {missing}")

    required = {
        "CHECK": ["name", "phone", "start_time"],
    }
    if state["action"] == "RESERVE":
        print(f"\t\t\t[reserve_1]")
        if not state.get("party_size"):
            missing.append("party_size")
        
        if not state.get("start_time"):
            missing.append("start_time")

        if not missing:
            result = check_availability(state["party_size"],state["start_time"])
            print(f"\t\t\t[reserve_2] -> {result}")
            state["tool_results"] = result
            
            if state["tool_results"]['status'] != "available":
                print(f"\t\tunavailable")
                state["missing_fields"] = missing
                return state
            
            if not state.get("name"):
                missing.append("name")

            if not state.get("phone"):
                missing.append("phone")

    elif state["action"] == "CANCEL":
        handle_cancel_identity_resolution(state)
        missing = state["missing_fields"]
    else:
        if state["action"] in required:
            for field in required[state["action"]]:
                if not state.get(field):
                    missing.append(field)
    state["missing_fields"] = missing

    print(f"\t\t[check_missing] -> {missing}")
    return state

def route_after_check(state: AgentState):
    if state["missing_fields"]:
        return "ask_missing"
    
    if state.get("tool_results") and state["tool_results"]['status'] != "available":
        return "intersept_results"
    
    return "execute_tool"

def ask_missing(state: AgentState):

    state["loop_count"] += 1

    if state["loop_count"] > 6:
        state["messages"].append(
            AIMessage(content="I'm having trouble. Let's start over.")
        )
        return state


    # =========================
    # VALIDATION CHECK
    # =========================

    missing = state["missing_fields"]
    
    invalid = validate_input(state)
    if invalid["status"] == "invalid":
        missing = "None"

    action = state.get("action")
    candidates = state.get("candidates")
    identity_failed = state.get("identity_failed")


    # =========================
    # BUILD CONTEXT
    # =========================

    context_blocks = []

    context_blocks.append(f"Action: {action}")
    context_blocks.append(f"Missing fields: {missing}")

    if invalid["status"] == "invalid":
        context_blocks.append(f"Invalid fields: {invalid['fields']}")
        context_blocks.append(f"Error reasons: {invalid['errors']}")

    if candidates:
        options = []
        for i, r in enumerate(candidates, 1):

            options.append(
                f"{i}. {r['customer_name']} - "
                f"{r['start_time']} - "
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


    # =========================
    # HOST PROMPT
    # =========================

    ask_prompt = f"""
You are a restaurant host speaking on the phone.

Rules:
- Maximum 10 words, start with the critical part of the sentence.
- Ask only ONE question.
- Speak naturally like a human host.
- Do NOT explain.

Priority rules:
1. If a field is INVALID → ask to fix it.
2. Otherwise ask for MISSING fields.

Invalid examples:

Invalid phone:
"That phone number looks wrong. Could you repeat it?"

Invalid party size:
"How many guests will be dining?"

Invalid time (open hour{OPEN_HOUR}, close hour{CLOSE_HOUR}):
"We're closed then. What time works instead?"

Missing examples:

Missing name:
"May I have your name?"

Missing phone:
"Phone number please?"

Missing party size:
"How many guests?"

Missing time:
"What time would you like?"

Multiple reservations:
"Which reservation? Tell me the time."

Identity failed:
"I couldn't find it. Name again please."
"""


    response = conversation_llm.invoke([
        SystemMessage(content=ask_prompt),
        HumanMessage(content=dynamic_context)
    ])


    print(f"\t\t[ask_missing] -> {missing}")
    print(f"\t\tcontext -> {dynamic_context}")
    print(f"\t\tresponse -> {response.content}")


    state["messages"].append(AIMessage(content=response.content))

    return state

def execute_tool(state: AgentState):
    reason = state.get("output_reason")
    print(f"[execute_tool] -> {reason}")
    if reason == "run_availability":
        print(f"\t[run availability]")
        result = check_availability(state["party_size"], state["start_time"])
        result["_key"] = state.get("_cache_key", "")

    elif reason == "run_reserve":
        print(f"\t[run reserve]")
        result = create_reservation(
            state["name"], state["phone"], state["party_size"], state["start_time"]
        )

    elif reason == "run_cancel_search":
        matches = find_reservation(
            name=state.get("name"),
            phone=state.get("phone"),
            start_time=state.get("start_time"),
        )
        if len(matches) == 0:
            result = {"status": Status.NOT_FOUND.value, "_type": "cancel_search"}
        elif len(matches) == 1:
            result = {"status": "FOUND", "_type": "cancel_search", "reservation": matches[0]}
        else:
            result = {"status": Status.MULTIPLE_MATCHES.value, "_type": "cancel_search",
                      "candidates": matches}

    elif reason == "run_cancel":
        res_id = state["cancel_reservation"]["id"]
        result = cancel_reservation(res_id)

    else:
        result = {"status": "error", "message": f"execute_tool called with unknown reason: {reason}"}

    print(f"\t[execute_tool] result → {result}")
    state["tool_result"]   = result
    state["output_reason"] = None  # let manager re-evaluate with fresh eyes
    return state

def intersept_results(state: AgentState):
    result = state["tool_results"]
    print(f"\t\t[intersept_results] -> {result}")

    # ── Build a factual, grounded context string ──────────────────────────────
    # We construct what the LLM is allowed to say BEFORE it speaks.
    # This prevents hallucinating alternative times.
    status = result.get("status", "")

    if status == "no_availability":
        alternatives = result.get("alternatives", [])
        if alternatives:
            alt_times = ", ".join(a["start_time"] for a in alternatives[:3])
            factual_context = (
                f"Fully booked at the requested time. "
                f"These exact alternative slots are available: {alt_times}. "
                f"Suggest only these times — do NOT invent others."
            )
        else:
            factual_context = (
                "Fully booked at the requested time. "
                "There are NO alternative slots available right now. "
                "Do NOT suggest any times. Ask if they'd like to try a different time themselves."
            )

    elif status == "RESERVED":
        factual_context = (
            f"Reservation confirmed. "
            f"Name: {result.get('reservation_name') or state.get('name', '?')}. "
            f"Time: {result.get('start_time', '?')}. "
            f"Guests: {result.get('party_size') or state.get('party_size', '?')}."
        )

    elif status == "CANCELLED":
        factual_context = "Reservation successfully cancelled."

    elif status == "MULTIPLE_MATCHES":
        factual_context = "Multiple reservations match. Ask guest to clarify."

    elif status == "NOT_FOUND":
        factual_context = "No reservation found with the provided details."

    elif status in ("duplicate", "ALREADY_EXISTS"):
        factual_context = "A reservation with these details already exists."

    else:
        factual_context = f"System returned: {status}. Ask the guest to try again."

    intersept_prompt = f"""You are a friendly restaurant host relaying a booking system result.

FACTUAL RESULT (report only what is stated here — nothing more):
{factual_context}

Rules:
- Maximum 15 words.
- Do NOT add times, names, or details that are not in the FACTUAL RESULT above.
- Speak naturally, like a host on the phone.
"""

    response = conversation_llm.invoke([SystemMessage(content=intersept_prompt)])

    print(f"\t[response content] -> {result}\n\t\t{response.content}")
    state["messages"].append(AIMessage(content=response.content))
    state["tool_results"] = None
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
#START → extract_fields → manager ──┬── detect_intent ──┐
#                          ▲        ├── execute_tool  ──┤
#                          └────────┘                   │
#                                   └── output_parser → END
# -----------------------
workflow = StateGraph(AgentState)

workflow.add_node("extract_fields", extract_fields)
workflow.add_node("manager", manager)
workflow.add_node("output_phraser",output_phraser)
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("execute_tool", execute_tool)

workflow.set_entry_point("extract_fields")
workflow.add_edge("extract_fields", "manager")

workflow.add_conditional_edges("manager", route_from_manager, {
    "detect_intent" : "detect_intent",
    "execute_tool" : "execute_tool",
    "output_phraser" : "output_phraser"
})

workflow.add_edge("detect_intent", "manager")
workflow.add_edge("execute_tool", "manager")
workflow.add_edge("output_phraser", END)

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
            "loop_count": 0,
            "_prev_action": None,
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
seed_random_reservations(20)
print(chat_app.get_graph().draw_ascii())


root = tk.Tk()
app = RestaurantApp(root)
root.mainloop()
