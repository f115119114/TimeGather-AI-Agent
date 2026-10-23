"""
A customer service agent for 拾光桌遊所 (Gradio app).

Every question goes to the agent first. The agent decides which tool to use:
shop information and game statuses come from a SQLite database, the code calculates the fee,
and anything the tools cannot answer is handed to a human.
The panel on the right shows which tools were called and what they returned.

Run it with:  python app-gradio.py
"""

import csv
import json
import os
import re
from pathlib import Path
import sqlite3

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. API setup
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env" if "__file__" in locals() else ".env")
CHAT_MODEL = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("LLM_API_URL"),
    timeout=60.0
)

# ---------------------------------------------------------------------------
# 2. The shop database: two tables, shop_info and games (SQLite)
# ---------------------------------------------------------------------------

connection = sqlite3.connect("shop_app.db", check_same_thread=False)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS shop_info")
cursor.execute("CREATE TABLE shop_info (topic TEXT PRIMARY KEY, content TEXT NOT NULL)")

cursor.execute("DROP TABLE IF EXISTS games")
cursor.execute("""CREATE TABLE games (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    category TEXT,
    players TEXT,
    hourly_rate INTEGER,
    available TEXT
)""")

shop_info_file = Path(__file__).parent / "shop_info.csv" if "__file__" in locals() else Path("shop_info.csv")
with open(shop_info_file, newline="", encoding="utf-8") as f:
    cursor.executemany(
        "INSERT INTO shop_info (topic, content) VALUES (?, ?)",
        [(r["topic"], r["content"]) for r in csv.DictReader(f)]
    )

games_file = Path(__file__).parent / "games.csv" if "__file__" in locals() else Path("games.csv")
with open(games_file, newline="", encoding="utf-8") as f:
    cursor.executemany(
        "INSERT INTO games (name, category, players, hourly_rate, available) VALUES (?, ?, ?, ?, ?)",
        [(r["name"], r["category"], r["players"], int(r["hourly_rate"]), r["available"]) for r in csv.DictReader(f)]
    )
connection.commit()

SHOP_TOPICS = [r["topic"] for r in cursor.execute("SELECT topic FROM shop_info")]
print(f"Ready: {len(SHOP_TOPICS)} shop topics loaded.")

# ---------------------------------------------------------------------------
# 3. The tools, as plain functions, and what the model sees
# ---------------------------------------------------------------------------

def get_shop_info(topic):
    """Return one piece of shop information, such as opening hours, address, or rules."""
    row = cursor.execute("SELECT content FROM shop_info WHERE topic = ?", (topic,)).fetchone()
    if row is None:
        return {"found": False, "topics_available": SHOP_TOPICS}
    return {"found": True, "topic": topic, "content": row["content"]}


def check_game_status(game_name):
    """Find board games whose name matches, returning player count, rate and availability."""
    like = f"%{game_name.strip()}%"
    rows = cursor.execute(
        "SELECT name, category, players, hourly_rate, available FROM games WHERE name LIKE ?",
        (like,)
    ).fetchall()
    if not rows:
        return {"found": False, "note": f"'{game_name}' 目前不在店內的遊戲清單中。"}
    return {
        "found": True,
        "games": [{
            "name": r["name"],
            "category": r["category"],
            "players": r["players"],
            "hourly_rate": r["hourly_rate"],
            "available": (r["available"] == "yes")
        } for r in rows]
    }


def calculate_play_fee(hours: int, people: int, is_student: bool = False):
    """Calculate table fee based on hours and people, applying special discount rules."""
    if hours <= 0 or people <= 0:
        return {"error": "時數與人數必須大於 0。"}

    base_rate = 60
    raw_total = hours * people * base_rate
    discount_note = []

    if is_student:
        discount_rate = 0.8
        discount_note.append("學生優惠 8 折 (20% off)")
    elif hours >= 3:
        discount_rate = 0.9
        discount_note.append("遊玩滿 3 小時享 9 折 (10% off)")
    else:
        discount_rate = 1.0
        discount_note.append("原價無折扣")

    final_total = int(round(raw_total * discount_rate))
    return {
        "hours": hours,
        "people": people,
        "is_student": is_student,
        "base_rate_per_hour_person": base_rate,
        "raw_total": raw_total,
        "final_total": final_total,
        "discount_applied": ", ".join(discount_note)
    }


def transfer_to_human():
    """Hand the conversation to human staff for complaints or unresolved inquiries."""
    return {
        "message": "請至拾光桌遊所櫃檯洽詢現場店長，或透過 FB/IG 官方粉絲專頁私訊，我們將由專人為您處理。",
        "note": (
            "The hand-over is done. First politely apologize or explain why in one sentence, "
            "then provide this contact message. Do not say that you will directly transfer/connect the call, "
            "because you cannot. Do not call this tool again."
        )
    }


TOOL_FUNCTIONS = {
    "get_shop_info": get_shop_info,
    "check_game_status": check_game_status,
    "calculate_play_fee": calculate_play_fee,
    "transfer_to_human": transfer_to_human,
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_shop_info",
            "description": f"Get store policies and general information about Time Gather Board Game Cafe. Topic must be one of: {', '.join(SHOP_TOPICS)}. Use 'outside_food' for food, meals, drinks, and snacks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "enum": SHOP_TOPICS}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_game_status",
            "description": "Check if a board game is in the shop library, player count, hourly rate, and availability. Supports game names like '卡坦島', '阿瓦隆', '卡卡頌', 'Catan'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "game_name": {"type": "string", "description": "The name or keyword of the board game"}
                },
                "required": ["game_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_play_fee",
            "description": "Calculate total fee for playing at the shop. Always use this whenever calculating cost or billing. Accepts hours, number of people, and whether student discount applies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Number of hours to play"},
                    "people": {"type": "integer", "description": "Number of players"},
                    "is_student": {"type": "boolean", "description": "True if customer mentioned being a student"}
                },
                "required": ["hours", "people"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Use this for complaints, wanting to talk to or complain to a manager, refunds, or shop questions that no other tool can answer.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

SYSTEM_PROMPT = """You are the online customer service agent of Time Gather Board Game Cafe (拾光桌遊所) located in Kaohsiung, Taiwan. All prices are in New Taiwan dollars, written like NT$60.

RULES
1. Answer only from tool results. Never answer from memory, and never invent anything.
2. If you cannot find the answer from tools, say so plainly. Do not guess.
3. Only talk about this board game cafe. Politely refuse unrelated topics (such as weather, coding, or homework).
4. Answer politely, in Traditional Chinese (繁體中文), in three sentences or less.
5. Store policy and general info questions (opening hours, address, fee rules, student discount, parking, wifi, booking, outside food, pet policy, teaching, private room) -> get_shop_info.
6. Board game questions (hourly rate, game availability, player count) -> check_game_status.
7. Table fees, discounts, price calculations -> calculate_play_fee. Never calculate prices yourself.
8. Customer complaints, manager requests, or store questions that no other tool can answer -> transfer_to_human."""

MAX_STEPS = 6

def run_agent(chat_history):
    """Answer the last question in chat_history. Returns (answer, trace)."""
    last_msg = chat_history[-1]
    raw_content = last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg[0]

    # 防呆：確保 question 是純文字字串（排除 list 或 dict 結構）
    if isinstance(raw_content, list):
        question = " ".join([item if isinstance(item, str) else item.get("text", str(item)) for item in raw_content])
    elif isinstance(raw_content, dict):
        question = raw_content.get("text", str(raw_content))
    else:
        question = str(raw_content)

    language = "Traditional Chinese" if re.search(r"[\u4e00-\u9fff]", question) else "English"

    # 清理標準化傳入 API 的歷史紀錄，確保每個 message 的 content 都是純字串
    clean_history = []
    for msg in chat_history[:-1]:
        if isinstance(msg, dict):
            c = msg.get("content", "")
            if isinstance(c, list):
                c = " ".join([i if isinstance(i, str) else i.get("text", str(i)) for i in c])
            clean_history.append({"role": msg.get("role", "user"), "content": str(c)})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            u, b = msg
            if u:
                clean_history.append({"role": "user", "content": str(u)})
            if b:
                clean_history.append({"role": "assistant", "content": str(b)})

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + clean_history
        + [{"role": "system", "content": f"Reply in {language}."}]
        + [{"role": "user", "content": question}]
    )
    trace = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SPECS,
            temperature=0.2,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return (message.content or "").strip(), trace

        messages.append(message)

        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            result = TOOL_FUNCTIONS[call.function.name](**arguments)
            trace.append({
                "step": step + 1,
                "tool": call.function.name,
                "arguments": arguments,
                "result": result
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

    return "抱歉，目前系統忙碌中無法處理您的請求，請至現場櫃檯洽詢店長或私訊官方粉絲專頁。", trace

# ---------------------------------------------------------------------------
# 4. The Gradio page
# ---------------------------------------------------------------------------

GREETING = (
    "Hello! Welcome to Time Gather Board Game Cafe (拾光桌遊所). "
    "Ask me about opening hours, games, room bookings, or fees!\n\n"
    "您好！歡迎來到「拾光桌遊所」線上客服。請問今天想了解營業時間、包廂預約、遊戲清單還是費用計算呢？"
)

EMPTY_TRACE = "### Tools used\n\n(nothing yet)"

SAMPLE_GROUPS_ZH = {
    "店家資訊（get_shop_info）": [
        "請問週末的營業時間是幾點？",
        "可以帶外食或飲料進去嗎？",
        "有包廂可以容納很多人嗎？需要低消嗎？",
    ],
    "遊戲查詢（check_game_status）": [
        "店裡有《卡坦島》可以玩嗎？",
        "請問《閃靈快手》現在在庫嗎？",
    ],
    "費用計算（calculate_play_fee）": [
        "我們 4 個人打算玩 3 小時，總共多少錢？",
        "我們是 5 位學生，預計玩 2 小時，有學生優惠嗎？算下來是多少？",
    ],
    "客訴、資料沒有、非客服範圍": [
        "我昨天把雨傘忘在店裡了，可以幫我找嗎？",
        "你們有賣 PS5 遊戲光碟嗎？",
        "明天高雄天氣好嗎？",
        "可以幫我寫期中報告嗎？",
    ],
    "多輪連續對話（依序點選）": [
        "請問店裡有《阿瓦隆》嗎？",
        "那我們 6 個人想玩 3 小時，費用怎麼算？",
        "太棒了，那我們可以帶外面的手搖飲進去喝嗎？",
    ],
}

SAMPLE_GROUPS = {
    "Shop information (get_shop_info)": [
        "What are your opening hours on weekends?",
        "Can I bring outside food or drinks?",
        "Do you have private rooms for large groups?",
    ],
    "Games & Availability (check_game_status)": [
        "Do you have Catan available?",
        "Is Geistesblitz (閃靈快手) available right now?",
    ],
    "Fee Calculation (calculate_play_fee)": [
        "How much does it cost for 4 people to play for 3 hours?",
        "We are 5 students playing for 2 hours, how much is the fee?",
    ],
    "Complaint, not in data, or out of scope": [
        "I left my umbrella at your shop yesterday, can you help?",
        "Do you sell Nintendo Switch games?",
        "What is the weather in Kaohsiung tomorrow?",
        "Can you write a Python script for my homework?",
    ],
    "Multi-turn conversation": [
        "Do you have Avalon?",
        "Great, we have 6 people and want to play for 3 hours, how much is that?",
        "Can we bring our own bubble tea?",
    ],
}

def format_trace(trace):
    """Show the tool calls of the last answer, so the choice is visible."""
    if not trace:
        return "### Tools used\n\n(no tool was needed for this answer)"
    lines = ["### Tools used", ""]
    for entry in trace:
        lines += [
            f"**{entry['step']}. {entry['tool']}**",
            "",
            f"arguments: `{json.dumps(entry['arguments'], ensure_ascii=False)[:160]}`",
            "",
            f"result: `{json.dumps(entry['result'], ensure_ascii=False)[:260]}`",
            "",
        ]
    return "\n".join(lines)

def start_chat():
    return [{"role": "assistant", "content": GREETING}]

def add_user_message(user_text, chat_history):
    user_text = user_text.strip()
    if not user_text:
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": user_text}]

def answer_question(chat_history):
    if not chat_history:
        return chat_history, EMPTY_TRACE
    last_msg = chat_history[-1]
    is_user = (isinstance(last_msg, dict) and last_msg.get("role") == "user") or (isinstance(last_msg, (list, tuple)) and last_msg[0] and not last_msg[1])
    if not is_user:
        return chat_history, EMPTY_TRACE
    try:
        answer, trace = run_agent(chat_history)
        panel = format_trace(trace)
    except Exception as error:
        answer = f"(system message) API error: {error}"
        panel = "### Tools used\n\n(the call failed)"
    return chat_history + [{"role": "assistant", "content": answer}], panel

with gr.Blocks(title="拾光桌遊所 - 客服 Agent") as demo:
    gr.Markdown(
        "## 拾光桌遊所 (Time Gather Board Game Cafe) 線上客服\n"
        "由 AI Agent 協助回答店面規則、桌遊在庫狀態與計費。所有資料均來自 SQLite 資料庫與專屬工具，右側即時顯示工具軌跡。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(value=start_chat(), height=430, label="Agent")
            with gr.Row():
                user_box = gr.Textbox(
                    placeholder="請輸入您的問題（例如：營業時間、卡坦島在庫嗎、費用計算）...",
                    show_label=False,
                    scale=8,
                )
                send_button = gr.Button("發送", variant="primary", scale=1)
                clear_button = gr.Button("清除", scale=1)

            gr.Markdown("**常見範例問題（點選後按發送）**")
            with gr.Tabs():
                with gr.Tab("中文"):
                    for group, questions in SAMPLE_GROUPS_ZH.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)
                with gr.Tab("English"):
                    for group, questions in SAMPLE_GROUPS.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)

        with gr.Column(scale=2):
            trace_panel = gr.Markdown(EMPTY_TRACE)

    send_button.click(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
        answer_question, chatbot, [chatbot, trace_panel]
    )
    user_box.submit(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
        answer_question, chatbot, [chatbot, trace_panel]
    )

    clear_button.click(start_chat, outputs=chatbot).then(
        lambda: ("", EMPTY_TRACE), outputs=[user_box, trace_panel]
    )

# ---------------------------------------------------------------------------
# 啟動 Gradio 服務
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("正在啟動 Gradio 服務...")
    demo.launch(inbrowser=True)