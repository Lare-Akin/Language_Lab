import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import random
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import plotly.graph_objects as go
import plotly.express as px
import time
from typing import List, Dict, Tuple, Optional

# =========================================
# CONFIG & PATHS
# =========================================

DATA_DIR = Path(__file__).parents[2] / "data"

LEARNING_DB_PATH = DATA_DIR / "Learning_Resource_Database.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.json"
PROGRESS_DATA_PATH = DATA_DIR / "progress_data.json"

DATA_DIR.mkdir(exist_ok=True)

# =========================================
# SESSION STATE INITIALISATION (TEACHER)
# =========================================

def init_session_state():
    defaults = {
        "user_id": "",
        "user_role": None,
        "logged_in": False,
        "current_user": None,
        "current_username": "",
        "learning_profile": {},
        "current_section": "practice_words",
        "sidebar_collapsed": False,
        "selected_student": None,
        "category_indices": {},
        "practice_mode": False,
        "practice_items": [],
        "practice_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# =========================================
# PROGRESS DATA MANAGEMENT
# =========================================

def init_progress_data():
    if not PROGRESS_DATA_PATH.exists():
        with open(PROGRESS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"users": {}, "analytics": {}}, f, ensure_ascii=False)

def load_progress_data():
    try:
        with open(PROGRESS_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}, "analytics": {}}

def save_progress_data(data: dict) -> bool:
    try:
        with open(PROGRESS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def record_quiz_result(username: str, quiz_type: str, score: int, total: int,
                       content_type: str = None, time_spent: int = 0):
    progress_data = load_progress_data()

    if username not in progress_data["users"]:
        progress_data["users"][username] = {
            "quiz_history": [],
            "category_performance": {},
            "weekly_progress": [],
            "streak_data": {
                "current_streak": 0,
                "longest_streak": 0,
                "last_activity": None
            }
        }

    timestamp = datetime.now().isoformat()
    quiz_record = {
        "timestamp": timestamp,
        "quiz_type": quiz_type,
        "score": score,
        "total": total,
        "accuracy": (score / total * 100) if total > 0 else 0,
        "content_type": content_type,
        "time_spent": time_spent
    }

    progress_data["users"][username]["quiz_history"].append(quiz_record)

    if content_type:
        if content_type not in progress_data["users"][username]["category_performance"]:
            progress_data["users"][username]["category_performance"][content_type] = {
                "attempts": 0,
                "correct": 0,
                "total_questions": 0
            }
        cat_perf = progress_data["users"][username]["category_performance"][content_type]
        cat_perf["attempts"] += 1
        cat_perf["correct"] += score
        cat_perf["total_questions"] += total

    week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
    weekly_progress = progress_data["users"][username]["weekly_progress"]
    week_found = False
    for week in weekly_progress:
        if week["week_start"] == week_start:
            week["quizzes_taken"] += 1
            week["total_score"] += score
            week["total_questions"] += total
            week_found = True
            break
    if not week_found:
        weekly_progress.append({
            "week_start": week_start,
            "quizzes_taken": 1,
            "total_score": score,
            "total_questions": total
        })

    streak_data = progress_data["users"][username]["streak_data"]
    today = datetime.now().date().isoformat()
    last_activity = streak_data.get("last_activity")

    if last_activity:
        last_date = datetime.fromisoformat(last_activity).date()
        if (datetime.now().date() - last_date).days == 1:
            streak_data["current_streak"] += 1
        elif (datetime.now().date() - last_date).days > 1:
            streak_data["current_streak"] = 1
    else:
        streak_data["current_streak"] = 1

    streak_data["last_activity"] = today
    if streak_data["current_streak"] > streak_data.get("longest_streak", 0):
        streak_data["longest_streak"] = streak_data["current_streak"]

    update_user_profile(username, {
        "learning_profile": {
            "total_quizzes": len(progress_data["users"][username]["quiz_history"]),
            "total_correct": sum(q["score"] for q in progress_data["users"][username]["quiz_history"]),
            "total_attempts": sum(q["total"] for q in progress_data["users"][username]["quiz_history"]),
            "streak_days": streak_data["current_streak"],
            "last_activity": today
        }
    })

    return save_progress_data(progress_data)

def get_student_progress(username: str):
    progress_data = load_progress_data()
    user_data = progress_data["users"].get(username, {})

    quiz_history = user_data.get("quiz_history", [])
    category_performance = user_data.get("category_performance", {})
    weekly_progress = user_data.get("weekly_progress", [])
    streak_data = user_data.get("streak_data", {})

    if not quiz_history:
        return {
            "username": username,
            "total_quizzes": 0,
            "total_correct": 0,
            "total_questions": 0,
            "overall_accuracy": 0,
            "category_stats": {},
            "weekly_stats": [],
            "streak_data": streak_data,
            "improvement_rate": 0,
            "quiz_history": []
        }

    total_quizzes = len(quiz_history)
    total_correct = sum(q["score"] for q in quiz_history)
    total_questions = sum(q["total"] for q in quiz_history)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    category_stats = {}
    for cat, perf in category_performance.items():
        if perf["total_questions"] > 0:
            category_stats[cat] = {
                "accuracy": (perf["correct"] / perf["total_questions"] * 100),
                "attempts": perf["attempts"],
                "total_questions": perf["total_questions"]
            }

    weekly_stats = []
    for week in weekly_progress:
        weekly_accuracy = (week["total_score"] / week["total_questions"] * 100) if week["total_questions"] > 0 else 0
        weekly_stats.append({
            "week": week["week_start"],
            "quizzes": week["quizzes_taken"],
            "accuracy": weekly_accuracy,
            "total_questions": week["total_questions"]
        })

    improvement_rate = 0
    if len(quiz_history) >= 10:
        first_5 = quiz_history[:5]
        last_5 = quiz_history[-5:]
        first_avg = sum(q["score"] / q["total"] for q in first_5 if q["total"] > 0) / 5 * 100
        last_avg = sum(q["score"] / q["total"] for q in last_5 if q["total"] > 0) / 5 * 100
        improvement_rate = last_avg - first_avg

    return {
        "username": username,
        "total_quizzes": total_quizzes,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "overall_accuracy": overall_accuracy,
        "category_stats": category_stats,
        "weekly_stats": weekly_stats,
        "streak_data": streak_data,
        "improvement_rate": improvement_rate,
        "quiz_history": quiz_history[-20:]
    }

def get_all_students_progress():
    progress_data = load_progress_data()
    user_profiles = load_user_profiles()

    students_progress = []
    for username, user_data in user_profiles.items():
        if user_data.get("role") == "student":
            progress = get_student_progress(username)
            progress["name"] = user_data.get("name", username)
            progress["last_login"] = user_data.get("last_login")
            progress["created_at"] = user_data.get("created_at")
            students_progress.append(progress)

    return sorted(students_progress, key=lambda x: x.get("overall_accuracy", 0), reverse=True)

# =========================================
# USER PROFILES / AUTH
# =========================================

def init_user_profiles():
    if not USER_PROFILES_PATH.exists():
        with open(USER_PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False)

def load_user_profiles():
    try:
        with open(USER_PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_user_profiles(profiles: dict) -> bool:
    try:
        with open(USER_PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def ensure_default_admin():
    profiles = load_user_profiles()
    if "admin" not in profiles:
        profiles["admin"] = {
            "password": hash_password("admin123"),
            "name": "Administrator",
            "role": "administrator",
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "learning_profile": {
                "total_quizzes": 0,
                "total_correct": 0,
                "total_attempts": 0,
                "streak_days": 0,
                "last_activity": None
            }
        }
        save_user_profiles(profiles)

def register_user(username: str, password: str, name: str, role: str = "administrator"):
    profiles = load_user_profiles()
    if username in profiles:
        return False, "Username already exists"

    profiles[username] = {
        "password": hash_password(password),
        "name": name,
        "role": role,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "learning_profile": {
            "total_quizzes": 0,
            "total_correct": 0,
            "total_attempts": 0,
            "streak_days": 0,
            "last_activity": None,
        },
    }

    ok = save_user_profiles(profiles)
    if not ok:
        return False, "Error saving user profile"

    progress_data = load_progress_data()
    if username not in progress_data["users"]:
        progress_data["users"][username] = {
            "quiz_history": [],
            "category_performance": {},
            "weekly_progress": [],
            "streak_data": {
                "current_streak": 0,
                "longest_streak": 0,
                "last_activity": None
            }
        }
        save_progress_data(progress_data)

    return True, "User registered successfully"

def login_user(username: str, password: str):
    profiles = load_user_profiles()
    if username not in profiles:
        return False, "Invalid username or password"

    user = profiles[username]
    if user["password"] != hash_password(password):
        return False, "Invalid username or password"

    if user.get("role") != "administrator":
        return False, "This app is for teachers/administrators only"

    user["last_login"] = datetime.now().isoformat()
    save_user_profiles(profiles)

    return True, user

def update_user_profile(username: str, updates: dict) -> bool:
    profiles = load_user_profiles()
    if username not in profiles:
        return False

    user = profiles[username]
    lp = user.get("learning_profile", {})

    for key, value in updates.items():
        if key == "learning_profile":
            lp.update(value)
            user["learning_profile"] = lp
        elif key.startswith("learning_profile."):
            lp_key = key.split(".")[1]
            lp[lp_key] = value
            user["learning_profile"] = lp
        else:
            user[key] = value

    profiles[username] = user
    return save_user_profiles(profiles)

def delete_user(username: str) -> bool:
    profiles = load_user_profiles()
    if username not in profiles:
        return False
    if username == "admin":
        return False
    del profiles[username]
    return save_user_profiles(profiles)

# =========================================
# AUTH UI
# =========================================

def login_section():
    st.sidebar.title("🔐 Teacher Account")
    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            success, result = login_user(username, password)
            if success:
                user = result
                st.session_state.logged_in = True
                st.session_state.user_role = user.get("role", "administrator")
                st.session_state.current_user = user.get("name", username)
                st.session_state.current_username = username
                st.session_state.user_id = username
                st.session_state.learning_profile = user.get("learning_profile", {})
                st.session_state.sidebar_collapsed = False
                st.success(f"✅ Welcome back, {st.session_state.current_user}!")
                st.rerun()
            else:
                st.error(f"❌ {result}")

    with tab_register:
        with st.form("register_form"):
            st.write("Create a new teacher/administrator account")
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name")

            submit_register = st.form_submit_button("Register as Teacher")

        if submit_register:
            if not new_username or not new_password or not full_name:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = register_user(new_username, new_password, full_name, "administrator")
                if success:
                    st.success("✅ Account created! Please login.")
                else:
                    st.error(f"❌ {message}")

def logout_button():
    if st.session_state.logged_in:
        if st.sidebar.button("🚪 Logout"):
            current_section = st.session_state.get("current_section", "practice_words")
            st.session_state.clear()
            init_session_state()
            st.session_state.current_section = current_section
            st.rerun()

def user_info_display():
    if not st.session_state.logged_in:
        return

    role_display = "👨‍💼 Teacher/Administrator"
    st.sidebar.success(f"**{role_display}**\nLogged in as: {st.session_state.current_user}")

    progress = get_student_progress(st.session_state.current_username)
    st.sidebar.info(
        f"📊 **Your Stats:**\n"
        f"• Quizzes: {progress['total_quizzes']}\n"
        f"• Accuracy: {progress['overall_accuracy']:.1f}%\n"
        f"• Streak: {progress['streak_data'].get('current_streak', 0)} days"
    )

    students = load_user_profiles()
    student_count = sum(1 for u in students.values() if u.get("role") == "student")
    st.sidebar.info(f"👥 **Students:** {student_count}")

# =========================================
# DATA LOADING
# =========================================

@st.cache_data
def load_learning_database():
    if not LEARNING_DB_PATH.exists():
        sample_data = {
            "Classification": [
                "Bulgarian_reference",
                "Bulgarian_Phrases",
                "Learning_From_Human_Conversation",
                "Dialog",
            ],
            "Category": ["Greetings", "Greetings", "General convo", "Sports"],
            "Bulgarian": [
                "Здравей",
                "Добро утро",
                "Здравей, как си?",
                "Обичаш ли да спортуваш?",
            ],
            "Pronunciation": ["Zdravey", "Dobro utro", "Zdravey, kak si?", "Obichash li da sportuvash?"],
            "English": ["Hello", "Good morning", "Hi, how are you?", "Do you like doing sports?"],
            "Grammar_Notes": [
                "Informal greeting",
                "Morning greeting",
                "Casual conversation starter",
                "Sports question",
            ],
        }
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(LEARNING_DB_PATH, index=False, encoding="utf-8-sig")
        return {
            "words": df_sample.iloc[0:1],
            "phrases": df_sample.iloc[1:2],
            "conversations": df_sample.iloc[2:3],
            "dialogues": df_sample.iloc[3:4],
        }, df_sample

    df = None
    for enc in ["utf-8-sig", "utf-8", "cp1251"]:
        try:
            df = pd.read_csv(LEARNING_DB_PATH, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            df = None
    if df is None:
        return {}, pd.DataFrame()

    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")

    mapping = {}
    for col in df.columns:
        cl = col.lower()
        if "classification" in cl:
            mapping[col] = "Classification"
        elif "category" in cl:
            mapping[col] = "Category"
        elif "bulgarian" in cl or cl == "bg":
            mapping[col] = "Bulgarian"
        elif "english" in cl or cl == "en" or "translation" in cl:
            mapping[col] = "English"
        elif "pronunciation" in cl or "transliteration" in cl:
            mapping[col] = "Pronunciation"
        elif "grammar" in cl or "notes" in cl:
            mapping[col] = "Grammar_Notes"

    df = df.rename(columns=mapping)

    for required in ["Classification", "Category", "Bulgarian", "English"]:
        if required not in df.columns:
            df[required] = ""

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    df = df[df["Bulgarian"].notna() & (df["Bulgarian"].str.strip() != "")]
    df = df.reset_index(drop=True)

    classification_map = {
        "Bulgarian_reference": "words",
        "Bulgarian_Phrases": "phrases",
        "Learning_From_Human_Conversation": "conversations",
        "Dialog": "dialogues",
    }

    out = {
        "words": pd.DataFrame(),
        "phrases": pd.DataFrame(),
        "conversations": pd.DataFrame(),
        "dialogues": pd.DataFrame(),
    }

    for original, key in classification_map.items():
        mask = df["Classification"].astype(str).str.strip().str.lower() == original.lower()
        out[key] = df[mask].reset_index(drop=True)

    return out, df

classifications, full_df = load_learning_database()
df_words = classifications.get("words", pd.DataFrame())
df_phrases = classifications.get("phrases", pd.DataFrame())
df_convo = classifications.get("conversations", pd.DataFrame())
df_dialogues = classifications.get("dialogues", pd.DataFrame())

# =========================================
# AUDIO & TEXT HELPERS
# =========================================

@st.cache_resource
def tts_audio(text: str, lang: str = "bg"):
    if not text or str(text).strip().lower() in ["", "nan"]:
        return None
    try:
        tts = gTTS(text=str(text), lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception:
        return None

def clean_dialogue_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"^\s*[\wА-Яа-я]+:\s*", "", text)
    return cleaned.strip() or text

# =========================================
# PAGINATED VIEW HELPER
# =========================================

def paginated_item_view(df: pd.DataFrame, key_prefix: str, title: str):
    st.header(title)

    if df.empty:
        st.info("No data available. Add some content first!")
        return

    categories = sorted(df["Category"].dropna().unique().tolist())
    selected_category = st.selectbox("Filter by Category:", ["All"] + categories, key=f"{key_prefix}_cat")

    display_df = df.copy()
    if selected_category != "All":
        display_df = display_df[display_df["Category"] == selected_category]

    if display_df.empty:
        st.info("No items in selected category.")
        return

    items = display_df.to_dict("records")
    cat_key = f"{key_prefix}_{selected_category}"

    if cat_key not in st.session_state.category_indices:
        st.session_state.category_indices[cat_key] = 0

    idx = st.session_state.category_indices[cat_key]
    idx = max(0, min(idx, len(items) - 1))
    st.session_state.category_indices[cat_key] = idx

    item = items[idx]

    st.markdown("---")
    st.subheader(f"Item {idx + 1} of {len(items)}")

    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"**🇧🇬 Bulgarian:** {item.get('Bulgarian', '')}")
        pron = item.get("Pronunciation")
        if pron and str(pron).strip().lower() != "nan":
            st.markdown(f"**🔊 Pronunciation:** {pron}")
        audio_path = tts_audio(item.get("Bulgarian", ""))
        if audio_path:
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

    with colB:
        st.markdown(f"**🌍 English:** {item.get('English', '')}")
        notes = item.get("Grammar_Notes")
        if notes and str(notes).strip().lower() != "nan":
            st.info(f"📘 Notes: {notes}")
        st.caption(f"Category: {item.get('Category', 'N/A')}")

    col1, col2, col3 = st.columns([1, 2, 1])
    if col1.button("◀ Previous", disabled=idx == 0, key=f"{cat_key}_prev"):
        st.session_state.category_indices[cat_key] = max(0, idx - 1)
        st.rerun()
    col2.markdown(f"**{idx + 1} / {len(items)}**")
    if col3.button("Next ▶", disabled=idx >= len(items) - 1, key=f"{cat_key}_next"):
        st.session_state.category_indices[cat_key] = min(len(items) - 1, idx + 1)
        st.rerun()

# =========================================
# PRACTICE SECTIONS
# =========================================

def practice_words_section():
    paginated_item_view(df_words, "words", "📝 Word Practice")

def practice_phrases_section():
    paginated_item_view(df_phrases, "phrases", "💬 Phrase Practice")

def practice_conversations_section():
    paginated_item_view(df_convo, "conversations", "🗣️ Conversation Practice")

def practice_dialogues_section():
    paginated_item_view(df_dialogues, "dialogues", "💭 Dialogue Practice")

# =========================================
# STUDENT PROGRESS DASHBOARD (HIGHLIGHTS)
# =========================================

def admin_student_progress_dashboard():
    st.header("📊 Student Progress Highlights")

    students = get_all_students_progress()
    if not students:
        st.info("No student data available yet.")
        return

    df_student = pd.DataFrame(students)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", len(df_student))
        st.metric("Avg Accuracy", f"{df_student['overall_accuracy'].mean():.1f}%")
    with col2:
        st.metric("Total Quizzes", int(df_student["total_quizzes"].sum()))
        st.metric("Active Streaks", int(sum(s.get("streak_data", {}).get("current_streak", 0) > 0 for s in students)))

    st.subheader("Accuracy vs Total Questions")
    fig = px.scatter(
        df_student,
        x="total_questions",
        y="overall_accuracy",
        hover_name="name",
        size="total_quizzes",
        color="overall_accuracy",
        color_continuous_scale="Blues",
        labels={
            "total_questions": "Total Questions Answered",
            "overall_accuracy": "Overall Accuracy (%)"
        },
        trendline="lowess"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Students by Accuracy")
    st.dataframe(
        df_student[["name", "username", "total_quizzes", "overall_accuracy"]]
        .sort_values("overall_accuracy", ascending=False)
        .reset_index(drop=True)
    )

# =========================================
# MAIN APP
# =========================================

def main():
    st.set_page_config(page_title="Teacher Dashboard", layout="wide")
    init_user_profiles()
    init_progress_data()
    ensure_default_admin()

    login_section()
    logout_button()
    user_info_display()

    if not st.session_state.logged_in:
        st.info("Please log in as a teacher/administrator to access the dashboard.")
        return

    st.title("Teacher Dashboard → Administrator")

    with st.sidebar.expander("📂 Navigation", expanded=True):
        st.write("**Learning**")
        if st.button("Words", key="nav_words"):
            st.session_state.current_section = "practice_words"
        if st.button("Phrases", key="nav_phrases"):
            st.session_state.current_section = "practice_phrases"
        if st.button("Conversations", key="nav_convo"):
            st.session_state.current_section = "practice_conversations"
        if st.button("Dialogues", key="nav_dialogues"):
            st.session_state.current_section = "practice_dialogues"

        st.write("---")
        st.write("**Analytics**")
        if st.button("Student Progress Highlights", key="nav_progress"):
            st.session_state.current_section = "progress_highlights"

    section = st.session_state.current_section

    if section == "practice_words":
        practice_words_section()
    elif section == "practice_phrases":
        practice_phrases_section()
    elif section == "practice_conversations":
        practice_conversations_section()
    elif section == "practice_dialogues":
        practice_dialogues_section()
    elif section == "progress_highlights":
        admin_student_progress_dashboard()
    else:
        practice_words_section()

if __name__ == "__main__":
    main()
