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

# =========================================
# CONFIG & PATHS (SHARED)
# =========================================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

LEARNING_DB_PATH = DATA_DIR / "Learning_Resource_Database.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.json"
QUIZ_HISTORY_PATH = DATA_DIR / "quiz_history.json"
PROGRESS_DATA_PATH = DATA_DIR / "progress_data.json"

DATA_DIR.mkdir(exist_ok=True)

# =========================================
# SESSION STATE INITIALISATION (STUDENT)
# =========================================

def init_session_state():
    """Initialize all session state variables for student app."""
    defaults = {
        # User/auth
        "user_id": "",
        "user_role": None,
        "logged_in": False,
        "current_user": None,
        "current_username": "",
        "learning_profile": {},
        
        # Navigation
        "current_section": "words",
        "sidebar_collapsed": False,
        
        # Content navigation indices (per category)
        "category_indices": {},
        
        # Progress tracking
        "progress_data": {},
        
        # Quiz state (namespaced)
        "dialogue_quiz": {
            "active": False,
            "finished": False,
            "questions": [],
            "current_idx": 0,
            "answers": {},   # q_i -> selected answer
            "options": {},   # opts_i -> frozen options list
        },
        "comprehensive_quiz": {
            "active": False,
            "finished": False,
            "questions": [],
            "current_idx": 0,
            "answers": {},   # q_i -> selected answer
            "options": {},   # opts_i -> frozen options list
            "source_df_class": "Mixed",  # "Mixed", "Words Only", etc.
        },
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =========================================
# PROGRESS DATA MANAGEMENT (SHARED)
# =========================================

def init_progress_data():
    """Initialize progress data storage."""
    if not PROGRESS_DATA_PATH.exists():
        with open(PROGRESS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"users": {}, "analytics": {}}, f, ensure_ascii=False)

def load_progress_data():
    """Load progress data for all users."""
    try:
        with open(PROGRESS_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}, "analytics": {}}

def save_progress_data(data: dict) -> bool:
    """Save progress data."""
    try:
        with open(PROGRESS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def record_quiz_result(username: str, quiz_type: str, score: int, total: int, 
                      content_type: str = None, time_spent: int = 0):
    """Record a quiz result for progress tracking."""
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
    
    # Update category performance
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
    
    # Update weekly progress
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
    
    # Update streak
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
    
    save_progress_data(progress_data)
    
    # Update user profile for backward compatibility
    update_user_profile(username, {
        "learning_profile": {
            "streak_days": streak_data["current_streak"],
            "last_activity": today
        }
    })

def get_student_progress(username: str):
    """Get detailed progress data for a student."""
    progress_data = load_progress_data()
    user_data = progress_data["users"].get(username, {})
    
    # Calculate overall stats
    quiz_history = user_data.get("quiz_history", [])
    category_performance = user_data.get("category_performance", {})
    weekly_progress = user_data.get("weekly_progress", [])
    streak_data = user_data.get("streak_data", {})
    
    if not quiz_history:
        return None
    
    total_quizzes = len(quiz_history)
    total_correct = sum(q["score"] for q in quiz_history)
    total_questions = sum(q["total"] for q in quiz_history)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    # Calculate performance by category
    category_stats = {}
    for cat, perf in category_performance.items():
        if perf["total_questions"] > 0:
            category_stats[cat] = {
                "accuracy": (perf["correct"] / perf["total_questions"] * 100),
                "attempts": perf["attempts"],
                "total_questions": perf["total_questions"]
            }
    
    # Calculate weekly progress
    weekly_stats = []
    for week in weekly_progress:
        weekly_accuracy = (week["total_score"] / week["total_questions"] * 100) if week["total_questions"] > 0 else 0
        weekly_stats.append({
            "week": week["week_start"],
            "quizzes": week["quizzes_taken"],
            "accuracy": weekly_accuracy,
            "total_questions": week["total_questions"]
        })
    
    # Calculate improvement rate (last 5 quizzes vs first 5 quizzes)
    improvement_rate = 0
    if len(quiz_history) >= 10:
        first_5 = quiz_history[:5]
        last_5 = quiz_history[-5:]
        first_avg = sum(q["score"] / q["total"] for q in first_5) / 5 * 100
        last_avg = sum(q["score"] / q["total"] for q in last_5) / 5 * 100
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
        "quiz_history": quiz_history[-20:]  # Last 20 quizzes
    }

# =========================================
# USER PROFILES: LOAD / SAVE / AUTH (SHARED)
# =========================================

def ensure_default_admin():
    """Ensure default admin exists in user profiles."""
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

def init_user_profiles():
    """Ensure user profiles JSON exists."""
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

def register_user(username: str, password: str, name: str, role: str = "student"):
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
    
    # Initialize progress data for new user
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
    
    # Update last login
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
        else:
            user[key] = value
    
    profiles[username] = user
    return save_user_profiles(profiles)

# =========================================
# AUTH UI (STUDENT)
# =========================================

def login_section():
    """Sidebar login/register for students."""
    st.sidebar.title("🔐 Student Account")
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
                # Only allow student login in student app
                if user.get("role") != "student":
                    st.error("❌ This app is for students only. Please use the Teacher App for administrator access.")
                    return
                
                st.session_state.logged_in = True
                st.session_state.user_role = user.get("role", "student")
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
            st.write("Create a new student account")
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name")
            
            submit_register = st.form_submit_button("Register as Student")
        
        if submit_register:
            if not new_username or not new_password or not full_name:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = register_user(new_username, new_password, full_name, "student")
                if success:
                    st.success("✅ Account created! Please login.")
                else:
                    st.error(f"❌ {message}")

def logout_button():
    if st.session_state.logged_in:
        if st.sidebar.button("🚪 Logout"):
            # Save learning profile
            if st.session_state.current_username:
                update_user_profile(
                    st.session_state.current_username,
                    {"learning_profile": st.session_state.learning_profile},
                )
            
            # Reset all state but keep current_section to avoid crash
            current_section = st.session_state.get("current_section", "words")
            st.session_state.clear()
            init_session_state()
            st.session_state.current_section = current_section
            st.rerun()

def user_info_display():
    if not st.session_state.logged_in:
        return
    
    role_display = "👨‍🎓 Student"
    profile = st.session_state.learning_profile or {}
    total_quizzes = profile.get("total_quizzes", 0)
    total_attempts = profile.get("total_attempts", 0)
    total_correct = profile.get("total_correct", 0)
    streak = profile.get("streak_days", 0)
    accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
    
    st.sidebar.success(f"**{role_display}**\nLogged in as: {st.session_state.current_user}")
    if total_quizzes > 0:
        st.sidebar.info(
            f"📊 {total_quizzes} quizzes | 🎯 {accuracy:.0f}% | 🔥 {streak} day streak"
        )

# =========================================
# DATA LOADING (SHARED)
# =========================================

@st.cache_data
def load_learning_database():
    """Load the single CSV and split by classification."""
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
    
    # Normalize columns
    mapping = {}
    for col in df.columns:
        cl = col.lower()
        if "classification" in cl:
            mapping[col] = "classification"
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
    
    for required in ["classification", "Category", "Bulgarian", "English"]:
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
        mask = df["classification"].astype(str).str.strip().str.lower() == original.lower()
        out[key] = df[mask].reset_index(drop=True)
    
    return out, df

classifications, full_df = load_learning_database()
df_words = classifications.get("words", pd.DataFrame())
df_phrases = classifications.get("phrases", pd.DataFrame())
df_convo = classifications.get("conversations", pd.DataFrame())
df_dialogues = classifications.get("dialogues", pd.DataFrame())

# =========================================
# AUDIO & TEXT HELPERS (SHARED)
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
    """Remove leading 'Speaker:' pattern from a line."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"^\s*[\wА-Яа-я]+:\s*", "", text)
    return cleaned.strip() or text

# =========================================
# UI COMPONENTS (SHARED)
# =========================================

def lesson_card(bulgarian, pronunciation, english, notes=None, show_audio=True):
    st.markdown("---")
    colA, colB, colC = st.columns(3)
    
    colA.markdown("**🇧🇬 Bulgarian:**")
    colA.markdown(
        f'<div style="font-family: Arial, Helvetica, sans-serif; font-size: 18px; font-weight: bold;">{bulgarian}</div>',
        unsafe_allow_html=True,
    )
    
    colB.markdown("**🔊 Pronunciation:**")
    if pronunciation and str(pronunciation).strip().lower() != "nan":
        colB.markdown(
            f'<div style="font-family: Arial, Helvetica, sans-serif; font-size: 16px;">{pronunciation}</div>',
            unsafe_allow_html=True,
        )
    else:
        colB.markdown("—")
    
    colC.markdown("**🌍 English:**")
    colC.markdown(
        f'<div style="font-family: Arial, Helvetica, sans-serif; font-size: 18px;">{english}</div>',
        unsafe_allow_html=True,
    )
    
    if show_audio:
        text_for_audio = clean_dialogue_text(bulgarian)
        audio_file = tts_audio(text_for_audio)
        if audio_file:
            st.audio(audio_file)
    
    if notes and str(notes).strip().lower() != "nan":
        st.info(f"📘 **Notes:** {notes}")

def category_navigator(df, title, no_data_msg, key_prefix):
    """Shared navigator for words/phrases/conversations/dialogues."""
    st.header(title)
    
    if df.empty:
        st.info(no_data_msg)
        return None, None
    
    categories = sorted(df["Category"].dropna().unique())
    if not categories:
        st.warning("No categories found.")
        return None, None
    
    selected_category = st.selectbox("Select Category:", categories, key=f"{key_prefix}_category")
    subset = df[df["Category"] == selected_category]
    items = subset["Bulgarian"].dropna().unique().tolist()
    
    if not items:
        st.info("No items in this category.")
        return None, None
    
    cat_key = f"{key_prefix}_{selected_category}"
    if cat_key not in st.session_state.category_indices:
        st.session_state.category_indices[cat_key] = 0
    
    idx = st.session_state.category_indices[cat_key]
    col1, col2, col3 = st.columns([1, 2, 1])
    
    if col1.button("◀ Previous", key=f"{cat_key}_prev"):
        idx = max(0, idx - 1)
        st.session_state.category_indices[cat_key] = idx
        st.rerun()
    
    col2.markdown(f"**{idx + 1} / {len(items)}**")
    
    if col3.button("Next ▶", key=f"{cat_key}_next"):
        idx = min(len(items) - 1, idx + 1)
        st.session_state.category_indices[cat_key] = idx
        st.rerun()
    
    current_text = items[idx]
    row = subset[subset["Bulgarian"] == current_text].iloc[0]
    
    return row, selected_category

# =========================================
# LEARNING SECTIONS (STUDENT)
# =========================================

def bulgarian_words_section():
    row, _ = category_navigator(
        df_words,
        "📚 Bulgarian Words",
        "ℹ️ No word data available. Add words with 'Bulgarian_reference' classification.",
        "words",
    )
    if row is not None:
        lesson_card(
            bulgarian=row["Bulgarian"],
            pronunciation=row.get("Pronunciation", ""),
            english=row["English"],
            notes=row.get("Grammar_Notes", ""),
        )

def bulgarian_phrases_section():
    row, _ = category_navigator(
        df_phrases,
        "💬 Bulgarian Phrases",
        "ℹ️ No phrase data available. Add phrases with 'Bulgarian_Phrases' classification.",
        "phrases",
    )
    if row is not None:
        lesson_card(
            bulgarian=row["Bulgarian"],
            pronunciation=row.get("Pronunciation", ""),
            english=row["English"],
        )

def human_conversations_section():
    row, _ = category_navigator(
        df_convo,
        "🗣️ Human Conversations",
        "ℹ️ No conversation data available. Add conversations with 'Learning_From_Human_Conversation' classification.",
        "convo",
    )
    if row is not None:
        lesson_card(
            bulgarian=row["Bulgarian"],
            pronunciation=row.get("Pronunciation", ""),
            english=row["English"],
        )

def dialogues_section():
    row, category = category_navigator(
        df_dialogues,
        "💭 Dialogues - Learn How to Respond",
        "ℹ️ No dialogue data available. Add dialogues with 'Dialog' classification.",
        "dialog",
    )
    if row is not None:
        st.markdown("#### 💬 Dialogue Scenario")
        st.write(f"**Situation:** {category}")
        lesson_card(
            bulgarian=row["Bulgarian"],
            pronunciation=row.get("Pronunciation", ""),
            english=row["English"],
            notes=row.get("Grammar_Notes", ""),
            show_audio=True,
        )

# =========================================
# QUIZ HELPERS (STUDENT)
# =========================================

def reset_quiz_state(name: str):
    """Reset quiz state for a given quiz namespace."""
    if name == "dialogue_quiz":
        st.session_state[name] = {
            "active": False,
            "finished": False,
            "questions": [],
            "current_idx": 0,
            "answers": {},
            "options": {},
        }
    elif name == "comprehensive_quiz":
        st.session_state[name] = {
            "active": False,
            "finished": False,
            "questions": [],
            "current_idx": 0,
            "answers": {},
            "options": {},
            "source_df_class": "Mixed",
        }

# =========================================
# DIALOGUE QUIZ (STUDENT)
# =========================================

def dialogue_quiz_section():
    st.header("🎭 Dialogue Response Quiz")
    
    quiz_state = st.session_state["dialogue_quiz"]
    
    if df_dialogues.empty:
        st.info("ℹ️ No dialogue data available for quizzes.")
        return
    
    if not quiz_state["active"]:
        col1, _ = st.columns(2)
        with col1:
            num_questions = st.slider("Number of questions:", 1, 10, 3, key="dialogue_quiz_num")
        
        if st.button("Start Dialogue Quiz 🎯", type="primary", key="start_dialogue_quiz"):
            sample_dialogues = df_dialogues.sample(
                min(num_questions, len(df_dialogues))
            ).to_dict("records")
            reset_quiz_state("dialogue_quiz")
            quiz_state = st.session_state["dialogue_quiz"]
            quiz_state["active"] = True
            quiz_state["questions"] = sample_dialogues
            st.rerun()
        return
    
    # Quiz in progress or finished
    questions = quiz_state["questions"]
    idx = quiz_state["current_idx"]
    
    if not quiz_state["finished"] and idx < len(questions):
        dialog = questions[idx]
        st.markdown(f"### Question {idx + 1} of {len(questions)}")
        st.markdown("---")
        
        st.write("**Listen to the dialogue:**")
        audio_text = clean_dialogue_text(dialog["Bulgarian"])
        audio_file = tts_audio(audio_text)
        if audio_file:
            st.audio(audio_file)
        
        st.write(f"**Situation:** {dialog.get('Category', 'General conversation')}")
        st.write("**How would you appropriately respond?**")
        
        # --------- FROZEN OPTIONS PER QUESTION ----------
        opts_key = f"opts_{idx}"
        if opts_key not in quiz_state["options"]:
            others = df_dialogues[df_dialogues["Bulgarian"] != dialog["Bulgarian"]]
            if len(others) >= 3:
                distractors = others.sample(3)["Bulgarian"].tolist()
            else:
                distractors = others["Bulgarian"].tolist()
            
            options = [dialog["Bulgarian"]] + distractors[:3]
            random.shuffle(options)
            quiz_state["options"][opts_key] = options
        else:
            options = quiz_state["options"][opts_key]
        # ------------------------------------------------
        
        answer_key = f"q_{idx}"
        saved_answer = quiz_state["answers"].get(answer_key)
        
        selected = st.radio(
            "Choose your response:",
            options,
            key=f"dialogue_radio_{idx}",
            index=options.index(saved_answer) if saved_answer in options else None,
        )
        
        if selected:
            quiz_state["answers"][answer_key] = selected
        
        col_prev, col_next = st.columns(2)
        
        with col_prev:
            if idx > 0 and st.button("◀ Previous Question", key=f"dialogue_prev_{idx}"):
                quiz_state["current_idx"] = idx - 1
                st.rerun()
        
        with col_next:
            if st.button("Next Question ▶", key=f"dialogue_next_{idx}"):
                if idx < len(questions) - 1:
                    quiz_state["current_idx"] = idx + 1
                    st.rerun()
                else:
                    quiz_state["finished"] = True
                    st.rerun()
    
    # Results
    if quiz_state["finished"]:
        score = 0
        questions = quiz_state["questions"]
        st.markdown("---")
        for i, dialog in enumerate(questions):
            key = f"q_{i}"
            user_answer = quiz_state["answers"].get(key)
            correct = user_answer == dialog["Bulgarian"]
            if correct:
                score += 1
        
        st.markdown(f"### 🎯 Your Score: {score}/{len(questions)}")
        
        # Record quiz result
        if st.session_state.logged_in:
            record_quiz_result(
                username=st.session_state.current_username,
                quiz_type="dialogue_quiz",
                score=score,
                total=len(questions),
                content_type="Dialogues"
            )
        
        st.markdown("### 📝 Review Your Answers")
        for i, dialog in enumerate(questions):
            key = f"q_{i}"
            user_answer = quiz_state["answers"].get(key)
            correct = user_answer == dialog["Bulgarian"]
            with st.expander(f"Question {i + 1}: {'✅' if correct else '❌'}"):
                st.write(f"**Situation:** {dialog.get('Category', 'General conversation')}")
                st.write(f"**Correct response:** {dialog['Bulgarian']}")
                st.write(f"**Your response:** {user_answer or 'No answer'}")
                if dialog.get("English"):
                    st.write(f"**Translation:** {dialog['English']}")
        
        # Update profile
        if st.session_state.logged_in:
            lp = st.session_state.learning_profile or {}
            new_lp = {
                "total_quizzes": lp.get("total_quizzes", 0) + 1,
                "total_correct": lp.get("total_correct", 0) + score,
                "total_attempts": lp.get("total_attempts", 0) + len(questions),
            }
            lp.update(new_lp)
            st.session_state.learning_profile = lp
            update_user_profile(
                st.session_state.current_username, {"learning_profile": lp}
            )
            
            if score >= len(questions) * 0.8:
                streak = lp.get("streak_days", 0) + 1
                lp["streak_days"] = streak
                st.session_state.learning_profile = lp
                update_user_profile(
                    st.session_state.current_username, {"learning_profile": lp}
                )
                st.success("🔥 Great job! Your streak has increased!")
        
        if st.button("Start New Quiz", type="primary"):
            reset_quiz_state("dialogue_quiz")
            st.rerun()

# =========================================
# COMPREHENSIVE QUIZ (STUDENT)
# =========================================

def custom_quiz_section():
    st.header("🎯 Comprehensive Quiz")
    
    quiz_state = st.session_state["comprehensive_quiz"]
    
    if full_df.empty:
        st.warning("No learning data available.")
        return
    
    # Map quiz type to dataframe
    type_to_df = {
        "Mixed": full_df,
        "Words Only": df_words,
        "Phrases Only": df_phrases,
        "Conversations Only": df_convo,
        "Dialogues Only": df_dialogues,
    }
    
    if not quiz_state["active"]:
        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.slider("Number of questions:", 5, 30, 10, key="comp_num")
        with col2:
            quiz_type = st.selectbox(
                "Quiz type:",
                list(type_to_df.keys()),
                key="comp_type",
            )
        
        if st.button("Start Quiz", type="primary", key="start_comp_quiz"):
            quiz_df = type_to_df[quiz_type]
            if quiz_df.empty:
                st.warning(f"No data available for {quiz_type}")
                return
            sample_data = quiz_df.sample(
                min(num_questions, len(quiz_df))
            ).to_dict("records")
            reset_quiz_state("comprehensive_quiz")
            quiz_state = st.session_state["comprehensive_quiz"]
            quiz_state["active"] = True
            quiz_state["questions"] = sample_data
            quiz_state["source_df_class"] = quiz_type
            st.rerun()
        return
    
    # Quiz in progress or finished
    questions = quiz_state["questions"]
    idx = quiz_state["current_idx"]
    
    if not quiz_state["finished"] and idx < len(questions):
        item = questions[idx]
        st.markdown(f"### Question {idx + 1} of {len(questions)}")
        st.markdown("---")
        
        classification = item.get("classification", "").lower()
        if classification == "dialog":
            st.write("**Listen to this dialogue:**")
            audio_text = clean_dialogue_text(item["Bulgarian"])
            audio_file = tts_audio(audio_text)
            if audio_file:
                st.audio(audio_file)
        
        st.write("**How would you say this in Bulgarian?**")
        st.info(f"**{item['English']}**")
        
        # Choose source dataframe for distractors
        source_df = type_to_df.get(quiz_state["source_df_class"], full_df)
        
        # --------- FROZEN OPTIONS PER QUESTION ----------
        opts_key = f"opts_{idx}"
        if opts_key not in quiz_state["options"]:
            others = source_df[source_df["Bulgarian"] != item["Bulgarian"]]
            if len(others) >= 3:
                distractors = others.sample(3)["Bulgarian"].tolist()
            else:
                distractors = others["Bulgarian"].tolist()
            
            options = [item["Bulgarian"]] + distractors[:3]
            random.shuffle(options)
            quiz_state["options"][opts_key] = options
        else:
            options = quiz_state["options"][opts_key]
        # ------------------------------------------------
        
        answer_key = f"q_{idx}"
        saved_answer = quiz_state["answers"].get(answer_key)
        
        selected = st.radio(
            "Choose the correct translation:",
            options,
            key=f"comp_radio_{idx}",
            index=options.index(saved_answer) if saved_answer in options else None,
        )
        
        if selected:
            quiz_state["answers"][answer_key] = selected
        
        col_prev, col_next = st.columns(2)
        with col_prev:
            if idx > 0 and st.button("◀ Previous Question", key=f"comp_prev_{idx}"):
                quiz_state["current_idx"] = idx - 1
                st.rerun()
        with col_next:
            if st.button("Next Question ▶", key=f"comp_next_{idx}"):
                if idx < len(questions) - 1:
                    quiz_state["current_idx"] = idx + 1
                    st.rerun()
                else:
                    quiz_state["finished"] = True
                    st.rerun()
    
    # Results
    if quiz_state["finished"]:
        score = 0
        questions = quiz_state["questions"]
        for i, item in enumerate(questions):
            key = f"q_{i}"
            user_answer = quiz_state["answers"].get(key)
            if user_answer == item["Bulgarian"]:
                score += 1
        
        st.markdown(f"### 🎯 Your Score: {score}/{len(questions)}")
        
        # Record quiz result
        if st.session_state.logged_in:
            record_quiz_result(
                username=st.session_state.current_username,
                quiz_type="comprehensive_quiz",
                score=score,
                total=len(questions),
                content_type=quiz_state["source_df_class"]
            )
        
        st.markdown("### 📝 Review Your Answers")
        
        for i, item in enumerate(questions):
            key = f"q_{i}"
            user_answer = quiz_state["answers"].get(key)
            correct = user_answer == item["Bulgarian"]
            with st.expander(f"Question {i + 1}: {'✅' if correct else '❌'}"):
                st.write(f"**English:** {item['English']}")
                st.write(f"**Correct Bulgarian:** {item['Bulgarian']}")
                st.write(f"**Your answer:** {user_answer or 'No answer'}")
                if item.get("Pronunciation"):
                    st.write(f"**Pronunciation:** {item['Pronunciation']}")
                if item.get("Grammar_Notes"):
                    st.write(f"**Notes:** {item['Grammar_Notes']}")
        
        # Update profile
        if st.session_state.logged_in:
            lp = st.session_state.learning_profile or {}
            new_lp = {
                "total_quizzes": lp.get("total_quizzes", 0) + 1,
                "total_correct": lp.get("total_correct", 0) + score,
                "total_attempts": lp.get("total_attempts", 0) + len(questions),
            }
            lp.update(new_lp)
            st.session_state.learning_profile = lp
            update_user_profile(
                st.session_state.current_username, {"learning_profile": lp}
            )
            
            if score >= len(questions) * 0.8:
                streak = lp.get("streak_days", 0) + 1
                lp["streak_days"] = streak
                st.session_state.learning_profile = lp
                update_user_profile(
                    st.session_state.current_username, {"learning_profile": lp}
                )
                st.success("🔥 Great job! Your streak has increased!")
        
        if st.button("Start New Quiz", type="primary"):
            reset_quiz_state("comprehensive_quiz")
            st.rerun()

# =========================================
# STUDENT PROGRESS DASHBOARD (STUDENT)
# =========================================

def student_progress_section():
    """Student's personal progress dashboard with graphs."""
    st.header("📊 Your Learning Progress Dashboard")
    
    if not st.session_state.logged_in:
        st.warning("Please login to view your progress.")
        return
    
    progress = get_student_progress(st.session_state.current_username)
    
    if not progress:
        st.info("📝 Complete some quizzes to see your progress here!")
        st.markdown(
            """
            **Get started:**
            1. Take a **Dialogue Quiz** to practice listening and responding  
            2. Try the **Comprehensive Quiz** to test all content types  
            3. Review your performance here  
            """
        )
        return
    
    # Overall Stats
    st.subheader("🎯 Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Quizzes", progress["total_quizzes"])
    col2.metric("Total Questions", progress["total_questions"])
    col3.metric("Overall Accuracy", f"{progress['overall_accuracy']:.1f}%")
    col4.metric("Current Streak", f"{progress['streak_data'].get('current_streak', 0)} days")
    
    # Progress over time graph
    st.subheader("📈 Progress Over Time")
    
    if progress["quiz_history"]:
        # Prepare data for line chart
        quiz_data = []
        for i, quiz in enumerate(progress["quiz_history"]):
            quiz_date = datetime.fromisoformat(quiz["timestamp"]).strftime("%Y-%m-%d %H:%M")
            quiz_data.append({
                "Quiz": i + 1,
                "Date": quiz_date,
                "Accuracy": quiz["accuracy"],
                "Score": quiz["score"],
                "Total": quiz["total"],
                "Type": quiz["quiz_type"].replace("_", " ").title()
            })
        
        df_quiz = pd.DataFrame(quiz_data)
        
        # Create line chart
        fig = px.line(df_quiz, x="Quiz", y="Accuracy", 
                     title="Quiz Accuracy Over Time",
                     markers=True,
                     color="Type",
                     hover_data=["Date", "Score", "Total"])
        
        fig.update_layout(
            xaxis_title="Quiz Number",
            yaxis_title="Accuracy (%)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quiz history available yet.")
    
    # Category Performance
    if progress["category_stats"]:
        st.subheader("📊 Performance by Category")
        
        categories = list(progress["category_stats"].keys())
        accuracies = [progress["category_stats"][cat]["accuracy"] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=accuracies, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Accuracy by Content Category",
            xaxis_title="Category",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display category details
        with st.expander("View Category Details"):
            for category, stats in progress["category_stats"].items():
                st.write(f"**{category}**: {stats['accuracy']:.1f}% accuracy "
                        f"({stats['attempts']} attempts, {stats['total_questions']} questions)")
    
    # Weekly Progress
    if progress["weekly_stats"]:
        st.subheader("📅 Weekly Progress")
        
        weeks = [w["week"] for w in progress["weekly_stats"]]
        weekly_acc = [w["accuracy"] for w in progress["weekly_stats"]]
        
        fig = go.Figure(data=[
            go.Scatter(x=weeks, y=weekly_acc, mode='lines+markers', name='Accuracy',
                      line=dict(color='green', width=3))
        ])
        
        fig.update_layout(
            title="Weekly Accuracy Trend",
            xaxis_title="Week Starting",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Improvement Rate
    if progress["improvement_rate"] != 0:
        st.subheader("📈 Improvement Analysis")
        col1, col2 = st.columns(2)
        
        improvement_color = "green" if progress["improvement_rate"] > 0 else "red"
        col1.metric("Improvement Rate", f"{progress['improvement_rate']:+.1f}%")
        
        if progress["improvement_rate"] > 0:
            col2.success("🎉 You're improving! Keep up the good work!")
        else:
            col2.warning("📉 You might need to review previous material.")
    
    # Recent Quiz History
    if progress["quiz_history"]:
        st.subheader("📝 Recent Quiz History")
        
        recent_quizzes = []
        for quiz in progress["quiz_history"][-10:]:  # Last 10 quizzes
            quiz_date = datetime.fromisoformat(quiz["timestamp"]).strftime("%Y-%m-%d %H:%M")
            recent_quizzes.append({
                "Date": quiz_date,
                "Type": quiz["quiz_type"].replace("_", " ").title(),
                "Score": f"{quiz['score']}/{quiz['total']}",
                "Accuracy": f"{quiz['accuracy']:.1f}%",
                "Content": quiz.get("content_type", "N/A")
            })
        
        df_recent = pd.DataFrame(recent_quizzes)
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

# =========================================
# MAIN STUDENT APP
# =========================================

def main():
    st.set_page_config(
        page_title="Laré BG Language Lab - Student",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        * { font-family: 'Roboto', 'Arial', 'Helvetica', sans-serif; }
        .main-header { font-size: 32px; font-weight: 700; margin-bottom: 0.5rem; }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        @media (max-width: 768px) {
            .stButton > button { width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    if "current_section" not in st.session_state:
        st.session_state.current_section = "words"
    
    if not st.session_state.logged_in:
        st.markdown('<h1 class="main-header">Laré BG Language Lab - Student</h1>', unsafe_allow_html=True)
        st.markdown("### Interactive Bulgarian Language Learning for Students")
        login_section()
        with st.expander("ℹ️ About the Student App", expanded=True):
            st.markdown(
                """
                **Features for Students:**
                - 📚 Words & Phrases by category  
                - 🗣️ Real-life conversations  
                - 💭 Dialogues to learn responses  
                - 🎯 Quizzes (dialogue + comprehensive)  
                - 📊 Personal progress tracking with graphs  
                
                **Get Started:**
                1. Register a student account  
                2. Login to access all features  
                3. Start learning Bulgarian!  
                
                **Mobile Friendly:** Works on phones and tablets!
                """
            )
        st.stop()
    
    st.markdown(
        f'<h1 class="main-header">Welcome → Добре дошли, {st.session_state.current_user}!</h1>',
        unsafe_allow_html=True,
    )
    
    # Sidebar toggle
    col_toggle, _ = st.columns([1, 20])
    with col_toggle:
        icon = "📂" if st.session_state.sidebar_collapsed else "📁"
        if st.button(icon, key="sidebar_toggle"):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()
    
    if not st.session_state.sidebar_collapsed:
        with st.sidebar:
            user_info_display()
            logout_button()
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📱 Navigation")
            
            nav_options = [
                ("📚 Words", "words"),
                ("💬 Phrases", "phrases"),
                ("🗣️ Conversations", "conversations"),
                ("💭 Dialogues", "dialogues"),
                ("🎭 Dialogue Quiz", "dialogue_quiz"),
                ("🎯 Comprehensive Quiz", "quiz"),
                ("📊 My Progress", "progress"),
            ]
            
            for label, key in nav_options:
                if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
                    # Reset quiz states when leaving quiz sections
                    if key != "dialogue_quiz" and st.session_state["dialogue_quiz"]["active"]:
                        reset_quiz_state("dialogue_quiz")
                    if key != "quiz" and st.session_state["comprehensive_quiz"]["active"]:
                        reset_quiz_state("comprehensive_quiz")
                    
                    st.session_state.current_section = key
                    st.rerun()
    
    section = st.session_state.current_section
    
    if section == "words":
        bulgarian_words_section()
    elif section == "phrases":
        bulgarian_phrases_section()
    elif section == "conversations":
        human_conversations_section()
    elif section == "dialogues":
        dialogues_section()
    elif section == "dialogue_quiz":
        dialogue_quiz_section()
    elif section == "quiz":
        custom_quiz_section()
    elif section == "progress":
        student_progress_section()
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 14px; padding: 20px 0;">
            Laré BG Language Lab • Student App • Bulgarian Learning Support Tool • Mobile Friendly<br>
            Made with ❤️ by Laré Akin • Cyrillic Supported
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    init_user_profiles()
    init_progress_data()
    ensure_default_admin()
    main()