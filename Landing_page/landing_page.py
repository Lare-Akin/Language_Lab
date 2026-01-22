import streamlit as st

# Page config
st.set_page_config(
    page_title="Laré BG Language Lab",
    page_icon="🇧🇬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        text-align: center;
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .welcome-text {
        text-align: center;
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    .card h3 {
        font-size: 1.8rem;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card p {
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    .card-link {
        display: inline-block;
        background-color: white;
        color: #1E3A8A;
        padding: 12px 30px;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
        border: 2px solid transparent;
    }
    .card-link:hover {
        background-color: transparent;
        color: white;
        border-color: white;
        transform: scale(1.05);
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 25px;
        margin-top: 40px;
    }
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
        display: block;
    }
    .feature-card h4 {
        color: #1E3A8A;
        font-size: 1.3rem;
        margin-bottom: 10px;
    }
    .feature-card p {
        color: #6B7280;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .footer {
        text-align: center;
        margin-top: 60px;
        padding-top: 30px;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 0.9rem;
    }
    .language-badge {
        display: inline-block;
        background: linear-gradient(45deg, #FF6B6B, #FFD166);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 10px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Bulgarian Language Learning Lab</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Master Bulgarian with Interactive Learning</p>', unsafe_allow_html=True)
st.markdown('<p class="welcome-text">Добре дошли в Лабораторията за български език!</p>', unsafe_allow_html=True)

st.markdown("---")

# Two main cards for Student and Teacher
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>👨‍🎓 Student Portal</h3>
        <p>Access interactive learning modules, practice vocabulary, take quizzes, track your progress, and master Bulgarian at your own pace.</p>
        <a href="https://lare-akin-language-lab-appsstudent-appstudent-app-lkys47.streamlit.app" target="_blank" class="card-link">🎯 Enter as Student</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>👨‍🏫 Teacher/Admin Portal</h3>
        <p>Manage content, monitor student progress, create quizzes, generate reports, and administer the learning platform.</p>
        <a href="https://teacherapppy-4csiurnebbmxwy2hrbky7j.streamlit.app" target="_blank" class="card-link">📊 Enter as Teacher</a>
    </div>
    """, unsafe_allow_html=True)

# Features section
st.markdown("## 🚀 Key Features")

features_html = """
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <h4>Interactive Lessons</h4>
        <p>Learn with engaging multimedia content and real-life dialogues tailored to your level.</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🎤</span>
        <h4>Pronunciation Practice</h4>
        <p>Improve your accent with audio exercises and voice recognition technology.</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <h4>Progress Tracking</h4>
        <p>Monitor your improvement with detailed analytics, reports, and achievement badges.</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🏆</span>
        <h4>Gamified Learning</h4>
        <p>Earn badges, compete in challenges, and stay motivated with our reward system.</p>
    </div>
</div>
"""

st.markdown(features_html, unsafe_allow_html=True)

# Additional info
st.markdown("---")

st.markdown("""
### 📚 About This Platform
This is a **prototype learning management system** for Bulgarian language education. 
The platform features:

- **Student App**: Complete learning environment with vocabulary practice, quizzes, and progress tracking
- **Teacher App**: Full administrative control with student monitoring, content management, and analytics
- **Real-time Updates**: Both apps sync with the same database for seamless experience
- **Mobile-Friendly**: Accessible on all devices
- **Cyrillic Support**: Full support for Bulgarian alphabet and pronunciation
""")

# How to use section
with st.expander("🚀 Getting Started", expanded=False):
    st.markdown("""
    ### For Students:
    1. Click **"Enter as Student"** above
    2. Login with demo credentials or register new account
    3. Start with "Word Practice" to build vocabulary
    4. Take quizzes to test your knowledge
    5. Track your progress in the "My Progress" section
    
    ### For Teachers:
    1. Click **"Enter as Teacher"** above  
    2. Login with admin credentials (default: admin/admin123)
    3. Explore content management in "Manage Content"
    4. Add students in "Student Management"
    5. Monitor progress in "Student Progress Dashboard"
    
    ### Demo Accounts:
    - **Student**: username: `demo`, password: `demopass`
    - **Teacher**: username: `admin`, password: `admin123`
    """)

# Footer
st.markdown("""
<div class="footer">
    Laré BG Language Lab Prototype • Made with ❤️ by Laré Akin<br>
    <small>This is a demonstration prototype. All data is stored securely and reset periodically.</small>
</div>
""", unsafe_allow_html=True)

# Add some interactivity
with st.sidebar:
    st.markdown("### 🔗 Quick Links")
    st.page_link("https://lare-akin-language-lab-appsstudent-appstudent-app-lkys47.streamlit.app", 
                label="🎯 Student App", icon="👨‍🎓")
    st.page_link("https://teacherapppy-4csiurnebbmxwy2hrbky7j.streamlit.app", 
                label="📊 Teacher App", icon="👨‍🏫")
    
    st.markdown("---")
    st.markdown("### ℹ️ About This Prototype")
    st.info("""
    This is a **working prototype** for:
    - Language learning research
    - Educational technology testing
    - User experience evaluation
    
    All feedback is welcome!
    """)