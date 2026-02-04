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
# SESSION STATE INITIALISATION (TEACHER)
# =========================================

def init_session_state():
    """Initialize all session state variables for teacher app."""
    defaults = {
        # User/auth
        "user_id": "",
        "user_role": None,
        "logged_in": False,
        "current_user": None,
        "current_username": "",
        "learning_profile": {},
        
        # Navigation
        "current_section": "my_progress",  # Changed default to my_progress
        "sidebar_collapsed": False,
        
        # Content editing
        "edit_mode": False,
        "edit_index": None,
        
        # Student progress view
        "selected_student": None,
        
        # Quiz states (removed since we're removing quizzes)
        "quiz_started": False,
        "quiz_score": 0,
        "quiz_total": 0,
        "quiz_questions": [],
        "quiz_current_index": 0,
        "quiz_answers": [],
        "quiz_content_type": None,
        "quiz_start_time": None,
        
        # Practice states (removed since we're removing practice)
        "practice_mode": False,
        "practice_index": 0,
        "practice_items": [],
        
        # Dialog practice (removed since we're removing practice)
        "dialog_mode": False,
        "dialog_lines": [],
        "dialog_current": 0,
        "dialog_showing_answer": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =========================================
# PROGRESS DATA MANAGEMENT (SHARED) - FIXED
# =========================================

def init_progress_data():
    """Initialize progress data storage."""
    if not PROGRESS_DATA_PATH.exists():
        with open(PROGRESS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump({"users": {}}, f, ensure_ascii=False)

def load_progress_data():
    """Load progress data for all users."""
    try:
        with open(PROGRESS_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading progress data: {e}")
        return {"users": {}}

def save_progress_data(data: dict) -> bool:
    """Save progress data."""
    try:
        with open(PROGRESS_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving progress data: {e}")
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
    
    # Update user profile learning data
    update_user_profile(username, {
        "learning_profile.total_quizzes": 1,
        "learning_profile.total_correct": score,
        "learning_profile.total_attempts": total,
        "learning_profile.last_activity": datetime.now().isoformat()
    })
    
    return save_progress_data(progress_data)

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
        first_avg = sum(q["score"] / q["total"] for q in first_5 if q["total"] > 0) / 5 * 100 if first_5 else 0
        last_avg = sum(q["score"] / q["total"] for q in last_5 if q["total"] > 0) / 5 * 100 if last_5 else 0
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

def get_all_students_progress():
    """Get progress data for all students."""
    progress_data = load_progress_data()
    user_profiles = load_user_profiles()
    
    students_progress = []
    for username, user_data in user_profiles.items():
        if user_data.get("role") == "student":
            progress = get_student_progress(username)
            # Add user info
            progress["name"] = user_data.get("name", username)
            progress["last_login"] = user_data.get("last_login")
            progress["created_at"] = user_data.get("created_at")
            students_progress.append(progress)
    
    return sorted(students_progress, key=lambda x: x.get("overall_accuracy", 0), reverse=True)

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

def register_user(username: str, password: str, name: str, role: str = "administrator"):
    """Register user - for teacher app, defaults to administrator."""
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
    
    # Only allow administrators in teacher app
    if user.get("role") != "administrator":
        return False, "This app is for teachers/administrators only"
    
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
        elif key.startswith("learning_profile."):
            lp_key = key.split(".")[1]
            lp[lp_key] = value
            user["learning_profile"] = lp
        else:
            user[key] = value
    
    profiles[username] = user
    return save_user_profiles(profiles)

def delete_user(username: str) -> bool:
    """Delete a user profile."""
    profiles = load_user_profiles()
    if username not in profiles:
        return False
    
    # Don't allow deleting the default admin
    if username == "admin":
        return False
    
    del profiles[username]
    return save_user_profiles(profiles)

# =========================================
# AUTH UI (TEACHER)
# =========================================

def login_section():
    """Sidebar login/register for teachers/admins."""
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
            # Reset all state but keep current_section to avoid crash
            current_section = st.session_state.get("current_section", "my_progress")
            st.session_state.clear()
            init_session_state()
            st.session_state.current_section = current_section
            st.rerun()

def user_info_display():
    if not st.session_state.logged_in:
        return
    
    role_display = "👨‍💼 Teacher/Administrator"
    
    st.sidebar.success(f"**{role_display}**\nLogged in as: {st.session_state.current_user}")
    
    # Show teacher stats
    progress = get_student_progress(st.session_state.current_username)
    st.sidebar.info(f"""
    📊 **Your Stats:**
    • Quizzes: {progress['total_quizzes']}
    • Accuracy: {progress['overall_accuracy']:.1f}%
    • Streak: {progress['streak_data'].get('current_streak', 0)} days
    """)
    
    # Show admin stats
    students = load_user_profiles()
    student_count = sum(1 for u in students.values() if u.get("role") == "student")
    
    st.sidebar.info(f"👥 **Students:** {student_count}")

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

# Load the data
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
# TEACHER PROGRESS DASHBOARD
# =========================================
def teacher_progress_dashboard():
    """Progress dashboard for teacher's own learning."""
    st.header("📈 My Learning Progress")
    
    if not st.session_state.logged_in:
        st.warning("Please login to view your progress.")
        return
    
    progress = get_student_progress(st.session_state.current_username)
    
    # Overall stats
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Quizzes", progress["total_quizzes"])
    col2.metric("Accuracy", f"{progress['overall_accuracy']:.1f}%")
    col3.metric("Current Streak", f"{progress['streak_data'].get('current_streak', 0)} days")
    col4.metric("Longest Streak", f"{progress['streak_data'].get('longest_streak', 0)} days")
    
    # Category performance
    if progress["category_stats"]:
        st.subheader("📈 Performance by Category")
        
        categories = list(progress["category_stats"].keys())
        accuracies = [progress["category_stats"][cat]["accuracy"] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=accuracies,
                  marker_color=['lightgreen' if acc >= 80 else 
                               'lightblue' if acc >= 70 else 
                               'salmon' for acc in accuracies],
                  text=[f"{acc:.1f}%" for acc in accuracies],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title="Accuracy by Content Type",
            xaxis_title="Category",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Progress timeline - FIXED DATE HANDLING
    if progress["quiz_history"]:
        st.subheader("📅 Progress Timeline")
        
        quiz_dates = []
        quiz_accuracies = []
        quiz_types = []
        
        for quiz in progress["quiz_history"]:
            try:
                # Parse the date string to datetime
                quiz_date = datetime.fromisoformat(quiz["timestamp"])
                quiz_dates.append(quiz_date)
                quiz_accuracies.append(quiz["accuracy"])
                quiz_types.append(quiz["quiz_type"].replace("_", " ").title())
            except (ValueError, KeyError):
                continue  # Skip invalid dates
        
        if quiz_dates:  # Only create chart if we have valid dates
            df_progress = pd.DataFrame({
                "Date": quiz_dates,
                "Accuracy": quiz_accuracies,
                "Type": quiz_types
            })
            
            # Sort by date
            df_progress = df_progress.sort_values("Date")
            
            fig = px.scatter(
                df_progress,
                x="Date",
                y="Accuracy",
                color="Type",
                trendline="lowess",
                title="Your Quiz Performance Over Time",
                size_max=10
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid quiz dates found for timeline chart.")
    
    # Recent activity
    if progress["quiz_history"]:
        st.subheader("📝 Recent Activity")
        
        recent_quizzes = []
        for quiz in progress["quiz_history"][-10:]:  # Last 10 quizzes
            try:
                quiz_date = datetime.fromisoformat(quiz["timestamp"]).strftime("%Y-%m-%d %H:%M")
                recent_quizzes.append({
                    "Date": quiz_date,
                    "Type": quiz["quiz_type"].replace("_", " ").title(),
                    "Score": f"{quiz['score']}/{quiz['total']}",
                    "Accuracy": f"{quiz['accuracy']:.1f}%",
                    "Content": quiz.get("content_type", "Mixed").title()
                })
            except (ValueError, KeyError):
                continue
        
        if recent_quizzes:
            df_recent = pd.DataFrame(recent_quizzes)
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activity data available.")
    
    # Weekly progress - FIXED DATE HANDLING
    if progress["weekly_stats"]:
        st.subheader("📅 Weekly Progress")
        
        weeks = []
        week_accuracies = []
        
        for week in progress["weekly_stats"]:
            try:
                # Parse week start date
                week_date = datetime.strptime(week["week"], "%Y-%m-%d")
                weeks.append(week_date)
                week_accuracies.append(week["accuracy"])
            except (ValueError, KeyError):
                continue
        
        if weeks:  # Only create chart if we have valid dates
            fig = go.Figure(data=[
                go.Scatter(x=weeks, y=week_accuracies,
                          mode='lines+markers',
                          line=dict(color='royalblue', width=3),
                          marker=dict(size=10))
            ])
            
            fig.update_layout(
                title="Weekly Accuracy Trend",
                xaxis_title="Week",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)

# =========================================
# CONTENT MANAGEMENT (TEACHER) - FIXED VERSION
# =========================================

def admin_content_management():
    st.header("👨‍💼 Content Management")
    
    if not st.session_state.logged_in or st.session_state.user_role != "administrator":
        st.warning("⛔ Admin access required")
        return
    
    tab_add, tab_edit, tab_bulk = st.tabs(["➕ Add New Content", "✏️ Edit Content", "📁 Bulk Operations"])
    
    with tab_add:
        st.subheader("➕ Add New Learning Content")
        with st.form("add_content_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                classification = st.selectbox(
                    "Content Type:",
                    [
                        "Bulgarian_reference",
                        "Bulgarian_Phrases",
                        "Learning_From_Human_Conversation",
                        "Dialog",
                    ],
                )
                category = st.text_input("Category*", help="e.g., Greetings, Food, Travel")
                bulgarian = st.text_area("Bulgarian Text*", height=100, 
                                        help="Enter text in Bulgarian (Cyrillic)")
            
            with col2:
                english = st.text_area("English Translation*", height=100)
                pronunciation = st.text_input("Pronunciation", 
                                            help="Latin transcription for pronunciation")
                grammar_notes = st.text_area("Grammar Notes", height=100,
                                           help="Optional grammar or usage notes")
            
            submitted = st.form_submit_button("Add Content", type="primary")
        
        if submitted:
            if not category or not bulgarian or not english:
                st.error("Please fill in all required fields (*)")
            else:
                try:
                    if LEARNING_DB_PATH.exists():
                        existing_df = pd.read_csv(LEARNING_DB_PATH, encoding="utf-8-sig")
                    else:
                        existing_df = pd.DataFrame(
                            columns=[
                                "Classification",
                                "Category",
                                "Bulgarian",
                                "English",
                                "Pronunciation",
                                "Grammar_Notes",
                            ]
                        )
                    
                    new_row = {
                        "Classification": classification,
                        "Category": category,
                        "Bulgarian": bulgarian,
                        "English": english,
                        "Pronunciation": pronunciation,
                        "Grammar_Notes": grammar_notes,
                    }
                    
                    new_df = pd.DataFrame([new_row])
                    df = pd.concat([existing_df, new_df], ignore_index=True)
                    df.to_csv(LEARNING_DB_PATH, index=False, encoding="utf-8-sig")
                    
                    # Clear cache to force reload
                    st.cache_data.clear()
                    
                    st.success("✅ Content added successfully!")
                    st.balloons()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error adding content: {e}")
    
    with tab_edit:
        st.subheader("✏️ Edit Existing Content")
        
        # Reload data each time to get fresh state
        classifications, full_df = load_learning_database()
        
        if full_df.empty:
            st.warning("No content to edit.")
        else:
            # Search and filter options
            col_search, col_filter = st.columns(2)
            
            with col_search:
                search_term = st.text_input("Search content (Bulgarian or English):", 
                                          placeholder="Type to search...")
            
            with col_filter:
                filter_type = st.selectbox("Filter by type:", 
                                         ["All", "Words", "Phrases", "Conversations", "Dialogues"])
            
            # Apply filters
            display_df = full_df.copy()
            
            if search_term:
                mask = (display_df["Bulgarian"].str.contains(search_term, case=False, na=False) |
                       display_df["English"].str.contains(search_term, case=False, na=False) |
                       display_df["Category"].str.contains(search_term, case=False, na=False))
                display_df = display_df[mask]
            
            if filter_type != "All":
                type_map = {
                    "Words": "Bulgarian_reference",
                    "Phrases": "Bulgarian_Phrases",
                    "Conversations": "Learning_From_Human_Conversation",
                    "Dialogues": "Dialog"
                }
                display_df = display_df[display_df["Classification"] == type_map[filter_type]]
            
            if display_df.empty:
                st.info("No content found matching your criteria.")
            else:
                st.info(f"Found {len(display_df)} items")
                
                # Display items in a table
                display_df_display = display_df[["Classification", "Category", "Bulgarian", "English"]].copy()
                display_df_display.columns = ["Type", "Category", "Bulgarian", "English"]
                
                # Convert classification to readable names
                type_map_display = {
                    "Bulgarian_reference": "Word",
                    "Bulgarian_Phrases": "Phrase",
                    "Learning_From_Human_Conversation": "Conversation",
                    "Dialog": "Dialogue"
                }
                display_df_display["Type"] = display_df_display["Type"].map(type_map_display)
                
                for idx, row in display_df.iterrows():
                    with st.expander(f"{row['Bulgarian'][:50]}... - {row['English'][:50]}...", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Current Data:**")
                            st.write(f"**Type:** {row['Classification']}")
                            st.write(f"**Category:** {row['Category']}")
                            st.write(f"**Bulgarian:** {row['Bulgarian']}")
                            st.write(f"**English:** {row['English']}")
                            if pd.notna(row.get('Pronunciation')) and str(row['Pronunciation']).strip():
                                st.write(f"**Pronunciation:** {row['Pronunciation']}")
                            if pd.notna(row.get('Grammar_Notes')) and str(row['Grammar_Notes']).strip():
                                st.write(f"**Notes:** {row['Grammar_Notes']}")
                        
                        with col2:
                            st.write("**Edit Form:**")
                            with st.form(f"edit_form_{idx}"):
                                edit_classification = st.selectbox(
                                    "Content Type:",
                                    [
                                        "Bulgarian_reference",
                                        "Bulgarian_Phrases",
                                        "Learning_From_Human_Conversation",
                                        "Dialog",
                                    ],
                                    index=[
                                        "Bulgarian_reference",
                                        "Bulgarian_Phrases",
                                        "Learning_From_Human_Conversation",
                                        "Dialog",
                                    ].index(row.get("Classification", "Bulgarian_reference"))
                                    if row.get("Classification") in [
                                        "Bulgarian_reference",
                                        "Bulgarian_Phrases",
                                        "Learning_From_Human_Conversation",
                                        "Dialog",
                                    ]
                                    else 0,
                                    key=f"edit_class_{idx}"
                                )
                                edit_category = st.text_input("Category", 
                                                            value=row.get("Category", ""),
                                                            key=f"edit_cat_{idx}")
                                edit_bulgarian = st.text_area("Bulgarian Text", 
                                                            value=row.get("Bulgarian", ""),
                                                            height=80,
                                                            key=f"edit_bg_{idx}")
                                edit_english = st.text_area("English Translation", 
                                                          value=row.get("English", ""),
                                                          height=80,
                                                          key=f"edit_en_{idx}")
                                edit_pronunciation = st.text_input("Pronunciation", 
                                                                 value=row.get("Pronunciation", ""),
                                                                 key=f"edit_pron_{idx}")
                                edit_notes = st.text_area("Grammar Notes", 
                                                        value=row.get("Grammar_Notes", ""),
                                                        height=80,
                                                        key=f"edit_notes_{idx}")
                                
                                col_save, col_delete = st.columns(2)
                                with col_save:
                                    save_changes = st.form_submit_button("💾 Save Changes", 
                                                                        type="primary")
                                with col_delete:
                                    delete_content = st.form_submit_button("🗑️ Delete", 
                                                                          type="secondary")
                            
                            if save_changes:
                                try:
                                    # Load fresh data for editing
                                    classifications, full_df = load_learning_database()
                                    
                                    # Update the specific row
                                    full_df.loc[idx, "Classification"] = edit_classification
                                    full_df.loc[idx, "Category"] = edit_category
                                    full_df.loc[idx, "Bulgarian"] = edit_bulgarian
                                    full_df.loc[idx, "English"] = edit_english
                                    full_df.loc[idx, "Pronunciation"] = edit_pronunciation
                                    full_df.loc[idx, "Grammar_Notes"] = edit_notes
                                    
                                    # Save to file
                                    full_df.to_csv(LEARNING_DB_PATH, index=False, encoding="utf-8-sig")
                                    
                                    # Clear cache and show success
                                    st.cache_data.clear()
                                    st.success("✅ Content updated successfully!")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Error updating content: {e}")
                            
                            if delete_content:
                                try:
                                    # Load fresh data
                                    classifications, full_df = load_learning_database()
                                    
                                    # Check if we need confirmation
                                    if not st.session_state.get(f"confirm_delete_{idx}", False):
                                        st.session_state[f"confirm_delete_{idx}"] = True
                                        st.warning("⚠️ Click Delete again to confirm")
                                    else:
                                        # Delete the row
                                        full_df = full_df.drop(idx).reset_index(drop=True)
                                        full_df.to_csv(LEARNING_DB_PATH, index=False, encoding="utf-8-sig")
                                        
                                        # Clear cache and show success
                                        st.cache_data.clear()
                                        st.success("✅ Content deleted successfully!")
                                        st.rerun()
                                        
                                except Exception as e:
                                    st.error(f"Error deleting content: {e}")
    
    with tab_bulk:
        st.subheader("📁 Bulk Operations")
        
        col_import, col_export = st.columns(2)
        
        with col_import:
            st.write("**Import Content**")
            st.info("Upload a CSV file with the same structure as the learning database.")
            
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                    
                    # Validate required columns
                    required_cols = ["Classification", "Category", "Bulgarian", "English"]
                    missing_cols = [col for col in required_cols if col not in new_data.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        st.write("**Preview of imported data:**")
                        st.dataframe(new_data.head())
                        
                        if st.button("Merge with Existing Data", type="primary"):
                            # Load existing data
                            if LEARNING_DB_PATH.exists():
                                existing_df = pd.read_csv(LEARNING_DB_PATH, encoding="utf-8-sig")
                                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                            else:
                                combined_df = new_data
                            
                            # Remove duplicates (based on Bulgarian text)
                            combined_df = combined_df.drop_duplicates(subset=["Bulgarian"], keep="last")
                            
                            combined_df.to_csv(LEARNING_DB_PATH, index=False, encoding="utf-8-sig")
                            
                            # Clear cache and reload
                            st.cache_data.clear()
                            st.success(f"✅ Successfully imported {len(new_data)} items!")
                            st.info(f"Total items in database: {len(combined_df)}")
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error importing file: {e}")
        
        with col_export:
            st.write("**Export Content**")
            
            # Reload fresh data for export
            classifications, full_df = load_learning_database()
            
            export_options = st.multiselect(
                "Select content types to export:",
                ["Words", "Phrases", "Conversations", "Dialogues"],
                default=["Words", "Phrases", "Conversations", "Dialogues"]
            )
            
            type_map_export = {
                "Words": "Bulgarian_reference",
                "Phrases": "Bulgarian_Phrases",
                "Conversations": "Learning_From_Human_Conversation",
                "Dialogues": "Dialog"
            }
            
            if export_options and st.button("Generate Export File", type="primary"):
                selected_types = [type_map_export[opt] for opt in export_options]
                export_df = full_df[full_df["Classification"].isin(selected_types)].copy()
                
                if export_df.empty:
                    st.warning("No data to export for selected types.")
                else:
                    csv = export_df.to_csv(index=False, encoding="utf-8-sig")
                    
                    filename = f"bulgarian_content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                        type="primary"
                    )
                    
                    st.info(f"Exported {len(export_df)} items")

# =========================================
# STUDENT MANAGEMENT (TEACHER)
# =========================================

def student_management_section():
    st.header("👥 Student Management")
    
    if not st.session_state.logged_in or st.session_state.user_role != "administrator":
        st.warning("⛔ Admin access required")
        return
    
    tab_students, tab_add = st.tabs(["📋 Student List", "➕ Add Student"])
    
    with tab_students:
        user_profiles = load_user_profiles()
        
        # Filter only students
        students = {username: data for username, data in user_profiles.items() 
                   if data.get("role") == "student"}
        
        if not students:
            st.info("No students registered yet.")
            return
        
        st.subheader(f"📋 Student List ({len(students)} students)")
        
        # Display student table
        student_data = []
        for username, data in students.items():
            progress = get_student_progress(username)
            student_data.append({
                "Username": username,
                "Name": data.get("name", "N/A"),
                "Joined": data.get("created_at", "N/A")[:10] if data.get("created_at") else "N/A",
                "Last Login": data.get("last_login", "N/A")[:16] if data.get("last_login") else "Never",
                "Quizzes": progress["total_quizzes"],
                "Accuracy": f"{progress['overall_accuracy']:.1f}%",
                "Streak": progress["streak_data"].get("current_streak", 0),
            })
        
        df_students = pd.DataFrame(student_data)
        st.dataframe(df_students, use_container_width=True, hide_index=True)
        
        # Student actions
        st.subheader("🎯 Student Actions")
        selected_username = st.selectbox("Select student for actions:", 
                                       list(students.keys()),
                                       format_func=lambda x: f"{students[x].get('name', x)} ({x})")
        
        if selected_username:
            col_view, col_reset, col_delete = st.columns(3)
            
            with col_view:
                if st.button("📊 View Progress", use_container_width=True):
                    st.session_state.selected_student = selected_username
                    st.session_state.current_section = "student_progress"
                    st.rerun()
            
            with col_reset:
                if st.button("🔄 Reset Progress", use_container_width=True):
                    st.warning(f"Reset progress for {selected_username}?")
                    if st.button(f"Confirm Reset for {selected_username}", type="primary"):
                        progress_data = load_progress_data()
                        if selected_username in progress_data["users"]:
                            progress_data["users"][selected_username] = {
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
                            st.success(f"✅ Progress reset for {selected_username}")
                            st.rerun()
            
            with col_delete:
                if st.button("🗑️ Delete Account", use_container_width=True, type="secondary"):
                    st.error(f"Delete account for {selected_username}?")
                    if st.button(f"Confirm Delete {selected_username}", type="primary"):
                        if delete_user(selected_username):
                            progress_data = load_progress_data()
                            if selected_username in progress_data["users"]:
                                del progress_data["users"][selected_username]
                                save_progress_data(progress_data)
                            st.success(f"✅ Account deleted for {selected_username}")
                            st.rerun()
                        else:
                            st.error("Failed to delete account")
    
    with tab_add:
        st.subheader("➕ Add New Student")
        
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username*")
                new_password = st.text_input("Password*", type="password")
            
            with col2:
                confirm_password = st.text_input("Confirm Password*", type="password")
                full_name = st.text_input("Full Name*")
                email = st.text_input("Email (optional)")
            
            submitted = st.form_submit_button("Create Student Account", type="primary")
        
        if submitted:
            if not new_username or not new_password or not full_name:
                st.error("Please fill in all required fields (*)")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = register_user(new_username, new_password, full_name, "student")
                if success:
                    st.success("✅ Student account created successfully!")
                    
                    # Add email to profile if provided
                    if email:
                        update_user_profile(new_username, {"email": email})
                    
                    st.info(f"**Login credentials:**\nUsername: {new_username}\nPassword: {new_password}")
                else:
                    st.error(f"❌ {message}")

# =========================================
# STUDENT PROGRESS DASHBOARD (TEACHER)
# =========================================

def admin_student_progress_dashboard():
    """Admin dashboard to view all student progress."""
    st.header("📈 Student Progress Dashboard")
    
    if not st.session_state.logged_in or st.session_state.user_role != "administrator":
        st.warning("⛔ Admin access required")
        return
    
    tab_overview, tab_details, tab_analytics = st.tabs(["📊 Overview", "👤 Student Details", "📈 Analytics"])
    
    with tab_overview:
        students_progress = get_all_students_progress()
        
        if not students_progress:
            st.info("No student progress data available yet.")
            return
        
        # Overall statistics
        st.subheader("📊 Overall Class Statistics")
        
        total_students = len(students_progress)
        avg_accuracy = sum(s["overall_accuracy"] for s in students_progress) / total_students if total_students > 0 else 0
        total_quizzes = sum(s["total_quizzes"] for s in students_progress)
        avg_quizzes = total_quizzes / total_students if total_students > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", total_students)
        col2.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        col3.metric("Total Quizzes", total_quizzes)
        col4.metric("Avg Quizzes/Student", f"{avg_quizzes:.1f}")
        
        # Performance distribution
        st.subheader("📈 Performance Distribution")
        
        performance_bins = {
            "Excellent (90-100%)": 0,
            "Good (80-89%)": 0,
            "Average (70-79%)": 0,
            "Needs Improvement (<70%)": 0
        }
        
        for student in students_progress:
            accuracy = student["overall_accuracy"]
            if accuracy >= 90:
                performance_bins["Excellent (90-100%)"] += 1
            elif accuracy >= 80:
                performance_bins["Good (80-89%)"] += 1
            elif accuracy >= 70:
                performance_bins["Average (70-79%)"] += 1
            else:
                performance_bins["Needs Improvement (<70%)"] += 1
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=list(performance_bins.keys()), 
                  y=list(performance_bins.values()),
                  marker_color=['#2E7D32', '#7CB342', '#FFB300', '#E65100'])
        ])
        
        fig.update_layout(
            title="Student Performance Distribution",
            xaxis_title="Performance Level",
            yaxis_title="Number of Students",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Student ranking table
        st.subheader("🏆 Student Rankings")
        
        ranking_data = []
        for i, student in enumerate(students_progress, 1):
            ranking_data.append({
                "Rank": i,
                "Student": student["name"],
                "Username": student["username"],
                "Accuracy": f"{student['overall_accuracy']:.1f}%",
                "Quizzes": student["total_quizzes"],
                "Streak": student["streak_data"].get("current_streak", 0),
                "Improvement": f"{student.get('improvement_rate', 0):+.1f}%"
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)
        
        # Identify top performers and struggling students
        st.subheader("🎯 Highlights")
        
        col_top, col_struggle = st.columns(2)
        
        with col_top:
            top_students = [s for s in students_progress if s["overall_accuracy"] >= 85][:3]
            if top_students:
                st.success("🏅 **Top Performers:**")
                for student in top_students:
                    st.write(f"• {student['name']}: {student['overall_accuracy']:.1f}%")
        
        with col_struggle:
            struggling = [s for s in students_progress if s["overall_accuracy"] < 65]
            if struggling:
                st.warning("⚠️ **Needs Attention:**")
                for student in struggling[:3]:
                    st.write(f"• {student['name']}: {student['overall_accuracy']:.1f}%")
    
    with tab_details:
        students_progress = get_all_students_progress()
        
        if not students_progress:
            st.info("No student progress data available yet.")
            return
        
        # Student selection
        student_options = [f"{s['name']} ({s['username']})" for s in students_progress]
        selected_student = st.selectbox("Choose a student:", student_options, key="student_select")
        
        # Get selected student data
        selected_username = selected_student.split("(")[-1].strip(")")
        selected_progress = next((s for s in students_progress if s["username"] == selected_username), None)
        
        if selected_progress:
            st.markdown("---")
            
            # Student overview
            st.subheader(f"👤 {selected_progress['name']}'s Progress")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Quizzes", selected_progress["total_quizzes"])
            col2.metric("Accuracy", f"{selected_progress['overall_accuracy']:.1f}%")
            col3.metric("Current Streak", f"{selected_progress['streak_data'].get('current_streak', 0)} days")
            col4.metric("Longest Streak", f"{selected_progress['streak_data'].get('longest_streak', 0)} days")
            
            # Category performance for this student
            if selected_progress["category_stats"]:
                st.subheader("📊 Performance by Category")
                
                categories = list(selected_progress["category_stats"].keys())
                accuracies = [selected_progress["category_stats"][cat]["accuracy"] for cat in categories]
                
                # Create radar chart for category comparison
                fig = go.Figure(data=[
                    go.Scatterpolar(
                        r=accuracies,
                        theta=categories,
                        fill='toself',
                        name='Accuracy',
                        fillcolor='rgba(135, 206, 250, 0.5)',
                        line=dict(color='rgb(135, 206, 250)')
                    )
                ])
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title=f"{selected_progress['name']}'s Performance by Category"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Identify weak areas
                weak_categories = [cat for cat, stats in selected_progress["category_stats"].items() 
                                  if stats["accuracy"] < 70]
                
                if weak_categories:
                    st.warning(f"⚠️ Areas needing improvement: {', '.join(weak_categories)}")
                
                # Display detailed category table
                with st.expander("View Detailed Category Stats"):
                    category_data = []
                    for category, stats in selected_progress["category_stats"].items():
                        category_data.append({
                            "Category": category,
                            "Accuracy": f"{stats['accuracy']:.1f}%",
                            "Attempts": stats["attempts"],
                            "Questions": stats["total_questions"]
                        })
                    
                    df_categories = pd.DataFrame(category_data)
                    st.dataframe(df_categories, use_container_width=True, hide_index=True)
            
            # Progress over time for this student - FIXED DATE HANDLING
            if selected_progress["quiz_history"]:
                st.subheader("📈 Progress Timeline")
                
                quiz_dates = []
                quiz_accuracies = []
                quiz_types = []
                
                for quiz in selected_progress["quiz_history"]:
                    try:
                        # Parse the date string to datetime
                        quiz_date = datetime.fromisoformat(quiz["timestamp"])
                        quiz_dates.append(quiz_date)
                        quiz_accuracies.append(quiz["accuracy"])
                        quiz_types.append(quiz["quiz_type"].replace("_", " ").title())
                    except (ValueError, KeyError):
                        continue  # Skip invalid dates
                
                if quiz_dates:  # Only create chart if we have valid dates
                    df_student = pd.DataFrame({
                        "Date": quiz_dates,
                        "Accuracy": quiz_accuracies,
                        "Type": quiz_types
                    })
                    
                    # Sort by date
                    df_student = df_student.sort_values("Date")
                    
                    fig = px.scatter(
                        df_student,
                        x="Date",
                        y="Accuracy",
                        color="Type",
                        trendline="lowess",
                        title=f"{selected_progress['name']}'s Quiz Performance",
                        size_max=10
                    )
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Accuracy (%)",
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid quiz dates found for timeline chart.")
            
            # Recent activity
            if selected_progress["quiz_history"]:
                st.subheader("📝 Recent Activity")
                
                recent_quizzes = []
                for quiz in selected_progress["quiz_history"][-5:]:  # Last 5 quizzes
                    try:
                        quiz_date = datetime.fromisoformat(quiz["timestamp"]).strftime("%Y-%m-%d %H:%M")
                        recent_quizzes.append({
                            "Date": quiz_date,
                            "Type": quiz["quiz_type"].replace("_", " ").title(),
                            "Score": f"{quiz['score']}/{quiz['total']}",
                            "Accuracy": f"{quiz['accuracy']:.1f}%",
                            "Content": quiz.get("content_type", "Mixed")
                        })
                    except (ValueError, KeyError):
                        continue
                
                if recent_quizzes:
                    df_recent = pd.DataFrame(recent_quizzes)
                    st.dataframe(df_recent, use_container_width=True, hide_index=True)
                else:
                    st.info("No recent activity data available.")
    
    with tab_analytics:
        st.subheader("📈 Class Analytics")
        
        students_progress = get_all_students_progress()
        
        if not students_progress or len(students_progress) < 2:
            st.info("Need at least 2 students for comparative analytics.")
            return
        
        # Class comparison chart
        st.subheader("🏫 Class Comparison")
        
        student_names = [s["name"] for s in students_progress]
        student_accuracies = [s["overall_accuracy"] for s in students_progress]
        
        fig = go.Figure(data=[
            go.Bar(x=student_names, y=student_accuracies,
                  marker_color=['lightgreen' if acc >= 80 else 
                               'lightblue' if acc >= 70 else 
                               'salmon' for acc in student_accuracies],
                  text=[f"{acc:.1f}%" for acc in student_accuracies],
                  textposition='auto')
        ])
        
        fig.update_layout(
            title="Student Accuracy Comparison",
            xaxis_title="Student",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement analysis
        st.subheader("📊 Engagement Analysis")
        
        quizzes_per_student = [s["total_quizzes"] for s in students_progress]
        avg_quizzes = sum(quizzes_per_student) / len(quizzes_per_student) if quizzes_per_student else 0
        
        fig2 = go.Figure(data=[
            go.Bar(x=student_names, y=quizzes_per_student,
                  marker_color='cornflowerblue',
                  text=quizzes_per_student,
                  textposition='auto')
        ])
        
        fig2.update_layout(
            title="Quizzes Completed per Student",
            xaxis_title="Student",
            yaxis_title="Number of Quizzes",
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info(f"📊 **Engagement Stats:**")
        st.write(f"- Average quizzes per student: {avg_quizzes:.1f}")
        st.write(f"- Most active: {max(quizzes_per_student) if quizzes_per_student else 0} quizzes")
        st.write(f"- Least active: {min(quizzes_per_student) if quizzes_per_student else 0} quizzes")
        
        # Export options
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Class Progress CSV", use_container_width=True):
                export_data = []
                for student in students_progress:
                    export_data.append({
                        "Student Name": student["name"],
                        "Username": student["username"],
                        "Total Quizzes": student["total_quizzes"],
                        "Overall Accuracy": student["overall_accuracy"],
                        "Current Streak": student["streak_data"].get("current_streak", 0),
                        "Longest Streak": student["streak_data"].get("longest_streak", 0),
                        "Improvement Rate": student.get("improvement_rate", 0)
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                filename = f"class_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📈 Generate Progress Report", use_container_width=True):
                st.success("📋 **Class Progress Report Generated**")
                
                # Generate summary statistics
                weak_students = [s for s in students_progress if s["overall_accuracy"] < 65]
                strong_students = [s for s in students_progress if s["overall_accuracy"] >= 85]
                avg_improvement = sum(s.get("improvement_rate", 0) for s in students_progress) / len(students_progress) if students_progress else 0
                
                with st.expander("View Report Summary", expanded=True):
                    st.write("**📊 Class Summary:**")
                    st.write(f"- Total students: {len(students_progress)}")
                    st.write(f"- Average accuracy: {avg_accuracy:.1f}%")
                    st.write(f"- Average improvement rate: {avg_improvement:+.1f}%")
                    st.write(f"- Students needing help (<65%): {len(weak_students)}")
                    st.write(f"- Top performers (≥85%): {len(strong_students)}")
                    
                    if weak_students:
                        st.write("**⚠️ Students Needing Attention:**")
                        for student in weak_students:
                            st.write(f"- {student['name']} ({student['overall_accuracy']:.1f}%)")
                    
                    if strong_students:
                        st.write("**🏅 Top Performers:**")
                        for student in strong_students[:3]:
                            st.write(f"- {student['name']} ({student['overall_accuracy']:.1f}%)")

# =========================================
# DATABASE STATISTICS (TEACHER)
# =========================================

def database_statistics_section():
    st.header("📊 Database Statistics")
    
    if not st.session_state.logged_in or st.session_state.user_role != "administrator":
        st.warning("⛔ Admin access required")
        return
    
    # Content statistics
    st.subheader("📚 Content Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", len(full_df))
    col2.metric("Words", len(df_words))
    col3.metric("Phrases", len(df_phrases))
    col4.metric("Dialogues", len(df_dialogues))
    
    # Content by category
    st.subheader("📈 Content Distribution")
    
    if not full_df.empty and "Category" in full_df.columns:
        category_counts = full_df["Category"].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(x=category_counts.index, y=category_counts.values,
                  marker_color='lightcoral')
        ])
        
        fig.update_layout(
            title="Top 10 Categories by Item Count",
            xaxis_title="Category",
            yaxis_title="Number of Items",
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # User statistics
    st.subheader("👥 User Statistics")
    
    user_profiles = load_user_profiles()
    students = sum(1 for u in user_profiles.values() if u.get("role") == "student")
    admins = sum(1 for u in user_profiles.values() if u.get("role") == "administrator")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Users", len(user_profiles))
    col2.metric("Students", students)
    
    # System health
    st.subheader("⚙️ System Health")
    
    col1, col2, col3 = st.columns(3)
    
    # Check file sizes
    learning_db_size = LEARNING_DB_PATH.stat().st_size / 1024 if LEARNING_DB_PATH.exists() else 0
    user_profiles_size = USER_PROFILES_PATH.stat().st_size / 1024 if USER_PROFILES_PATH.exists() else 0
    progress_data_size = PROGRESS_DATA_PATH.stat().st_size / 1024 if PROGRESS_DATA_PATH.exists() else 0
    
    col1.metric("Learning DB", f"{learning_db_size:.1f} KB")
    col2.metric("User Profiles", f"{user_profiles_size:.1f} KB")
    col3.metric("Progress Data", f"{progress_data_size:.1f} KB")
    
    # Backup options
    st.subheader("💾 Backup & Maintenance")
    
    col_backup, col_clear = st.columns(2)
    
    with col_backup:
        if st.button("Create Backup", use_container_width=True):
            backup_dir = DATA_DIR / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup learning database
            if LEARNING_DB_PATH.exists():
                backup_path = backup_dir / f"Learning_Resource_Database_{timestamp}.csv"
                import shutil
                shutil.copy2(LEARNING_DB_PATH, backup_path)
            
            # Backup JSON files
            for json_file in [USER_PROFILES_PATH, PROGRESS_DATA_PATH]:
                if json_file.exists():
                    backup_path = backup_dir / f"{json_file.stem}_{timestamp}.json"
                    shutil.copy2(json_file, backup_path)
            
            st.success(f"✅ Backup created at: {backup_dir}/")
    
    with col_clear:
        if st.button("Clear Cache", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")

# =========================================
# MAIN TEACHER APP
# =========================================

def main():
    st.set_page_config(
        page_title="Laré BG Language Lab - Teacher",
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
        .stButton > button { width: 100%; }
        @media (max-width: 768px) {
            .stButton > button { width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    if "current_section" not in st.session_state:
        st.session_state.current_section = "my_progress"
    
    if not st.session_state.logged_in:
        st.markdown('<h1 class="main-header">Laré BG Language Lab - Teacher</h1>', unsafe_allow_html=True)
        st.markdown("### Teacher/Administrator Dashboard")
        login_section()
        with st.expander("ℹ️ About the Teacher App", expanded=True):
            st.markdown(
                """
                **Features for Teachers/Administrators:**
                - 📈 **Track Progress**: Your own learning analytics and progress timeline
                - 👨‍💼 **Content Management**: Add, edit, and delete learning content
                - 👥 **Student Management**: View, add, and manage student accounts
                - 📊 **Progress Monitoring**: Track student performance with detailed analytics
                - 📈 **Class Analytics**: Compare student performance and identify trends
                - 💾 **Database Tools**: Backup, export, and maintain the learning system
                
                **Get Started:**
                1. Login with admin credentials (default: admin/admin123)
                2. Use the sidebar to navigate different sections
                3. Monitor progress and manage content
                
                **Note:** This app is specifically designed for teacher/administrator management tasks.
                """
            )
        st.stop()
    
    st.markdown(
        f'<h1 class="main-header">Teacher Dashboard → {st.session_state.current_user}</h1>',
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
            
            # Only show Progress & Admin sections
            st.sidebar.markdown("#### 📊 Admin & Progress")
            admin_nav = [
    
                ("👨‍💼 Manage Content", "manage_content"),
                ("👥 Student Management", "student_management"),
                ("📈 Student Progress", "student_progress"),
                ("📊 Database Stats", "database_stats"),
            ]
            
            for label, key in admin_nav:
                if st.sidebar.button(label, key=f"admin_{key}", use_container_width=True):
                    st.session_state.current_section = key
                    st.rerun()
    
    section = st.session_state.current_section
    
    # Route to appropriate section
    if section == "my_progress":
        teacher_progress_dashboard()
    elif section == "manage_content":
        admin_content_management()
    elif section == "student_management":
        student_management_section()
    elif section == "student_progress":
        admin_student_progress_dashboard()
    elif section == "database_stats":
        database_statistics_section()
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 14px; padding: 20px 0;">
            Laré BG Language Lab • Teacher App • Administration Dashboard<br>
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
