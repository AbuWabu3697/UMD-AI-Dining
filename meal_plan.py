import json
import os
import re
from typing import Dict, List, Any, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from openai import RateLimitError, OpenAIError, AuthenticationError


MODEL = "gpt-4o-mini"
MENU_PATH = "menu.csv"  # change to menu.json if needed
MAX_MENU_ROWS = 400     # keep prompts reasonable

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ----------------------------
# General Allergen Vocabulary
# ----------------------------
ALLERGEN_KEYWORDS: Dict[str, List[str]] = {
    "egg": ["egg", "eggs", "mayonnaise", "aioli", "meringue"],
    "milk": ["milk", "dairy", "cheese", "butter", "cream", "yogurt", "whey", "casein", "lactose"],
    "peanut": ["peanut", "groundnut"],
    "tree_nut": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "brazil nut"],
    "soy": ["soy", "soya", "tofu", "edamame", "soybean", "lecithin"],
    "wheat_gluten": ["wheat", "gluten", "flour", "bread", "pasta", "barley", "rye"],
    "sesame": ["sesame", "tahini"],
    "fish": ["fish", "salmon", "tuna", "cod", "anchovy"],
    "shellfish": ["shrimp", "crab", "lobster", "clam", "mussel", "oyster", "scallop"],
}

# User-friendly aliases -> canonical keys above
ALLERGEN_ALIASES: Dict[str, str] = {
    "dairy": "milk",
    "gluten": "wheat_gluten",
    "wheat": "wheat_gluten",
    "nuts": "tree_nut",
    "tree nuts": "tree_nut",
    "shell fish": "shellfish",
}

def normalize_allergen(token: str) -> Optional[str]:
    t = token.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = ALLERGEN_ALIASES.get(t, t)
    t = t.replace(" ", "_")
    return t if t in ALLERGEN_KEYWORDS else None

def extract_allergens_from_text(text: str) -> List[str]:
    """
    Heuristic extractor:
    - catches common patterns like "allergic to milk and eggs", "no dairy", "gluten-free"
    - also catches explicit allergen words anywhere in text
    """
    s = (text or "").lower()

    found: set[str] = set()

    # direct keyword spotting
    for canonical, kws in ALLERGEN_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", s):
                found.add(canonical)
                break

    # alias spotting
    for alias, canonical in ALLERGEN_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", s):
            found.add(canonical)

    return sorted(found)

def row_contains_allergen(row: pd.Series, allergens: List[str]) -> bool:
    """
    Checks multiple possible fields:
    - food_name / name / item / food
    - allergens column (if it exists)
    """
    if not allergens:
        return False

    # combine probable "name" columns into one string
    name_fields = []
    for col in ["food_name", "name", "item", "food"]:
        if col in row.index and pd.notna(row[col]):
            name_fields.append(str(row[col]))
    name_text = " ".join(name_fields).lower()

    allergens_text = ""
    if "allergens" in row.index and pd.notna(row["allergens"]):
        allergens_text = str(row["allergens"]).lower()

    haystack = (name_text + " " + allergens_text).strip()

    for a in allergens:
        for kw in ALLERGEN_KEYWORDS.get(a, []):
            if re.search(rf"\b{re.escape(kw)}\b", haystack):
                return True
    return False

# ---- Load menu once at startup ----
def load_menu(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else data.get("items", []))
    else:
        raise ValueError("menu file must be .csv or .json")

    df.columns = [c.strip() for c in df.columns]
    return df

def df_to_menu_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Keep only useful columns if present (helps token usage)
    desired = [
        # common menu fields
        "food_name", "name", "item", "food",
        # nutrition
        "calories", "protein", "carbs", "fat",
        "total_fat", "saturated_fat",
        # metadata
        "station", "dining_hall", "hall", "meal", "date",
        "diet_tags", "allergens",
        # link (your UMD data has this)
        "label_url",
        "serving_size"
    ]
    keep = [c for c in desired if c in df.columns]
    if keep:
        df = df[keep]

    if len(df) > MAX_MENU_ROWS:
        df = df.head(MAX_MENU_ROWS)

    return df.fillna("").to_dict(orient="records")

MENU_DF = load_menu(MENU_PATH)

# ---- In-memory sessions ----
SESSIONS: Dict[str, List[Dict[str, str]]] = {}
SESSION_STATE: Dict[str, Dict[str, Any]] = {}  # store extracted constraints like allergies

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(payload: ChatIn):
    session_id = payload.session_id.strip()
    user_msg = payload.message.strip()

    if not session_id:
        return {"error": "session_id required"}
    if not user_msg:
        return {"error": "message required"}

    history = SESSIONS.get(session_id, [])
    state = SESSION_STATE.get(session_id, {"allergies": []})

    # ----------------------------
    # Update session allergies
    # ----------------------------
    newly_found = extract_allergens_from_text(user_msg)
    # merge + de-dup
    merged = sorted(set(state.get("allergies", [])) | set(newly_found))
    state["allergies"] = merged
    SESSION_STATE[session_id] = state

    # ----------------------------
    # Filter menu for safety
    # ----------------------------
    allergies = state["allergies"]

    if allergies:
        safe_df = MENU_DF[~MENU_DF.apply(lambda r: row_contains_allergen(r, allergies), axis=1)].copy()
    else:
        safe_df = MENU_DF.copy()

    SAFE_MENU_ITEMS = df_to_menu_records(safe_df)

    # ----------------------------
    # System prompt
    # ----------------------------
    system = f"""
You are a meal-planning chatbot for a university dining hall.

You have access to SAFE_MENU_ITEMS (a list of menu objects). You must ONLY suggest items that appear in SAFE_MENU_ITEMS.
You must ask brief follow-up questions if user constraints are missing.

IMPORTANT:
- The user allergies are: {json.dumps(allergies)}
- Because SAFE_MENU_ITEMS is already filtered, NEVER suggest items that could contain those allergens.
- If allergies list is empty, proceed normally.

Your job:
1) Extract user constraints from conversation (goal, calorie target, macro preferences, allergies, dislikes, dietary pattern, number of meals, dining hall, time window).
2) When the user asks for a meal plan (or enough info is gathered), produce a plan using only SAFE_MENU_ITEMS.
3) Return plans as STRICT JSON only (no extra text) using this schema:

{{
  "type": "meal_plan",
  "constraints": {{
    "goal": "...",
    "calories_target": 0,
    "allergies": ["..."],
    "diet_style": "...",
    "meals_per_day": 4
  }},
  "day": "Day 1",
  "meals": [
    {{
      "name": "Breakfast",
      "items": [{{"name": "...", "servings": 1}}],
      "macros": {{"calories": 0, "protein": 0, "carbs": 0, "fat": 0}}
    }}
  ],
  "totals": {{"calories": 0, "protein": 0, "carbs": 0, "fat": 0}},
  "notes": ["..."]
}}

If you still need info, respond with JSON:
{{
  "type": "question",
  "question": "..."
}}

SAFE_MENU_ITEMS:
{json.dumps(SAFE_MENU_ITEMS, ensure_ascii=False)}
""".strip()

    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}]

    try:
        resp = client.responses.create(
            model=MODEL,
            input=messages
        )
        text = resp.output_text.strip()

    except RateLimitError:
        return {
            "type": "error",
            "message": "OpenAI quota/billing issue (429 insufficient_quota). Enable billing/credits on your OpenAI project."
        }

    except AuthenticationError:
        return {
            "type": "error",
            "message": "API key rejected. Check OPENAI_API_KEY."
        }

    except OpenAIError as e:
        return {
            "type": "error",
            "message": f"OpenAI error: {str(e)}"
        }


    # Parse JSON robustly
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            data = {"type": "error", "message": "Model returned non-JSON output.", "raw": text}

    # Save conversation history
    new_history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": json.dumps(data)}
    ]
    SESSIONS[session_id] = new_history[-20:]

    # Optional: return current allergy state so UI can show it
    if isinstance(data, dict) and data.get("type") in ("meal_plan", "question"):
        data.setdefault("session_state", {})
        data["session_state"]["allergies"] = allergies

    return data
