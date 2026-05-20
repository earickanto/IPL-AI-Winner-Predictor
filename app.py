import streamlit as st
import base64
import pickle
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IPL AI Predictor",
    page_icon="🏏",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = pickle.load(open("models/ipl_win_predictor.pkl", "rb"))

# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64(
    "images/ChatGPT Image May 20, 2026, 09_37_53 PM.png"
)

# =========================================================
# CSS
# =========================================================

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* =========================================================
GLOBAL
========================================================= */

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{

    background:
        radial-gradient(circle at top left, rgba(0,255,255,0.08), transparent 25%),
        radial-gradient(circle at bottom right, rgba(180,0,255,0.10), transparent 25%),
        linear-gradient(135deg, #020617 0%, #071122 40%, #020617 100%);

    color: white;

    overflow-x: hidden;
}}

header {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* =========================================================
MAIN
========================================================= */

.main-container {{
    padding: 30px;
}}

/* =========================================================
HERO
========================================================= */

.hero {{

    position: relative;

    padding: 70px 40px;

    border-radius: 40px;

    overflow: hidden;

    margin-bottom: 40px;

    background:
        linear-gradient(
            rgba(2,6,23,0.75),
            rgba(2,6,23,0.82)
        ),
        url("data:image/png;base64,{bg_image}");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    border:
        1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(25px);

    box-shadow:
        0 30px 70px rgba(0,0,0,0.45),
        0 0 60px rgba(0,255,255,0.08);
}}

.hero::before {{

    content: "";

    position: absolute;

    inset: -20%;

    background:
        conic-gradient(
            from 180deg,
            rgba(0,229,255,0.20),
            rgba(138,46,255,0.20),
            rgba(0,255,183,0.20),
            rgba(255,255,255,0.10),
            rgba(0,229,255,0.20)
        );

    filter: blur(70px);

    animation: rotateGlow 12s linear infinite;
}}

@keyframes rotateGlow {{

    0% {{
        transform: rotate(0deg);
    }}

    100% {{
        transform: rotate(360deg);
    }}
}}

/* =========================================================
LIVE BADGE
========================================================= */

.live-badge {{

    position: absolute;

    top: 25px;
    right: 30px;

    display: flex;
    align-items: center;
    gap: 10px;

    padding: 10px 18px;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.05);

    backdrop-filter: blur(20px);

    border:
        1px solid rgba(255,255,255,0.08);

    z-index: 10;
}}

.live-dot {{

    width: 12px;
    height: 12px;

    border-radius: 50%;

    background: #00ff9d;

    box-shadow:
        0 0 10px #00ff9d,
        0 0 20px #00ff9d;

    animation: pulse 1.5s infinite;
}}

@keyframes pulse {{

    0% {{
        transform: scale(1);
    }}

    50% {{
        transform: scale(1.4);
    }}

    100% {{
        transform: scale(1);
    }}
}}

/* =========================================================
TITLE
========================================================= */

.hero-title {{

    position: relative;

    z-index: 2;

    font-family: 'Orbitron', sans-serif;

    font-size: 72px;

    font-weight: 800;

    text-align: center;

    letter-spacing: 5px;

    margin-bottom: 20px;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #ffffff,
            #00ffb7,
            #8a2eff
        );

    background-size: 300%;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: shimmer 8s linear infinite;
}}

@keyframes shimmer {{

    0% {{
        background-position: 0%;
    }}

    100% {{
        background-position: 300%;
    }}
}}

.hero-sub {{

    position: relative;

    z-index: 2;

    max-width: 900px;

    margin: auto;

    text-align: center;

    font-size: 18px;

    line-height: 1.8;

    color: rgba(255,255,255,0.75);
}}

/* =========================================================
CARD
========================================================= */

.card-3d {{

    padding: 35px;

    border-radius: 30px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.06),
            rgba(255,255,255,0.015)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(25px);

    margin-bottom: 40px;

    box-shadow:
        0 25px 50px rgba(0,0,0,0.35),
        0 0 50px rgba(0,255,255,0.08);
}}

.section-title {{

    font-family: 'Orbitron', sans-serif;

    font-size: 30px;

    margin-bottom: 25px;

    letter-spacing: 3px;
}}

/* =========================================================
BUTTON
========================================================= */

.stButton > button {{

    width: 100%;

    padding: 16px;

    border-radius: 18px;

    border: none;

    font-family: 'Orbitron', sans-serif;

    font-size: 18px;

    font-weight: 700;

    color: white;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #8a2eff,
            #00ffb7
        );

    box-shadow:
        0 20px 40px rgba(0,0,0,0.35),
        0 0 30px rgba(0,255,255,0.25);

    transition: 0.3s ease;
}}

.stButton > button:hover {{

    transform:
        translateY(-4px);

    box-shadow:
        0 30px 50px rgba(0,0,0,0.4),
        0 0 50px rgba(0,255,255,0.35);
}}

/* =========================================================
SLIDER
========================================================= */

.stSlider > div > div > div > div {{

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #8a2eff,
            #00ffb7,
            #ffffff
        ) !important;

    height: 10px;

    border-radius: 999px;
}}

/* =========================================================
RESULT CARD
========================================================= */

.result-card {{

    padding: 40px;

    border-radius: 35px;

    text-align: center;

    margin-top: 40px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.07),
            rgba(255,255,255,0.02)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(20px);

    box-shadow:
        0 30px 60px rgba(0,0,0,0.4),
        0 0 60px rgba(0,255,255,0.08);
}}

.winner {{

    font-family: 'Orbitron', sans-serif;

    font-size: 54px;

    font-weight: 800;

    margin-bottom: 20px;

    background:
        linear-gradient(
            90deg,
            #00e5ff,
            #ffffff,
            #00ffb7
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.percent {{

    font-size: 24px;

    margin-top: 10px;
}}

/* =========================================================
SHOWDOWN
========================================================= */

.showdown {{

    margin-top: 40px;

    padding: 40px;

    border-radius: 35px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.06),
            rgba(255,255,255,0.015)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(25px);
}}

.showdown-grid {{

    display: grid;

    grid-template-columns: 1fr auto 1fr;

    gap: 30px;

    align-items: center;
}}

.team-side {{

    padding: 30px;

    border-radius: 25px;

    text-align: center;

    background:
        rgba(255,255,255,0.04);

    border:
        1px solid rgba(255,255,255,0.08);
}}

.team-name {{

    font-family: 'Orbitron', sans-serif;

    font-size: 36px;

    margin-bottom: 15px;
}}

.team-chance {{

    font-family: 'Orbitron', sans-serif;

    font-size: 48px;

    font-weight: 800;

    color: #00e5ff;
}}

.vs-badge {{

    width: 110px;
    height: 110px;

    border-radius: 50%;

    display: flex;
    align-items: center;
    justify-content: center;

    font-family: 'Orbitron', sans-serif;

    font-size: 32px;

    font-weight: 800;

    background:
        linear-gradient(
            135deg,
            rgba(0,229,255,0.25),
            rgba(138,46,255,0.25)
        );

    border:
        1px solid rgba(255,255,255,0.08);
}}

/* =========================================================
FLOAT PANEL
========================================================= */

.float-panel {{

    margin-top: 40px;

    padding: 35px;

    border-radius: 35px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.07),
            rgba(255,255,255,0.015)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(20px);
}}

.float-title {{

    font-family: 'Orbitron', sans-serif;

    font-size: 28px;

    margin-bottom: 25px;
}}

.float-grid {{

    display: grid;

    grid-template-columns: repeat(4, 1fr);

    gap: 20px;
}}

.float-item {{

    padding: 25px;

    border-radius: 22px;

    background:
        rgba(255,255,255,0.04);

    text-align: center;
}}

.float-label {{

    font-size: 13px;

    letter-spacing: 2px;

    color: rgba(255,255,255,0.65);

    margin-bottom: 10px;
}}

.float-value {{

    font-family: 'Orbitron', sans-serif;

    font-size: 32px;

    font-weight: 700;

    color: #00e5ff;
}}

/* =========================================================
RESPONSIVE
========================================================= */

@media screen and (max-width: 992px) {{

    .showdown-grid {{
        grid-template-columns: 1fr;
    }}

    .float-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}

    .hero-title {{
        font-size: 48px;
    }}
}}

@media screen and (max-width: 768px) {{

    .hero-title {{
        font-size: 34px;
    }}

    .hero-sub {{
        font-size: 14px;
    }}

    .float-grid {{
        grid-template-columns: 1fr;
    }}

    .showdown-grid {{
        grid-template-columns: 1fr;
    }}

    .vs-badge {{
        margin: auto;
    }}

    .team-name {{
        font-size: 26px;
    }}

    .team-chance {{
        font-size: 36px;
    }}

    .winner {{
        font-size: 34px;
    }}

    .live-badge {{

        position: relative;

        top: auto;
        right: auto;

        margin: auto auto 20px auto;

        width: fit-content;
    }}
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
#HERO
# =========================================================

st.markdown("""
<div class="main-container">

<div class="hero">

<div class="live-badge">
<div class="live-dot"></div>
LIVE AI ENGINE
</div>

<div class="hero-title">
IPL AI PREDICTOR
</div>

<div class="hero-sub">
A futuristic AI-powered IPL analytics platform with cinematic holographic visuals,
live prediction intelligence and immersive 3D sports-tech interface design.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
#DATA
# =========================================================

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans"
]

venues = [
    "MA Chidambaram Stadium",
    "Wankhede Stadium",
    "Eden Gardens",
    "Narendra Modi Stadium",
    "M Chinnaswamy Stadium"
]

# =========================================================
#INPUT
# =========================================================

st.markdown("""
<div class="card-3d">
<div class="section-title">
⚡ LIVE MATCH INPUT
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("🏏 Team A", teams)

with col2:
    team_b = st.selectbox("🏏 Team B", teams)

col3, col4 = st.columns(2)

with col3:
    venue = st.selectbox("📍 Venue", venues)

with col4:
    toss_winner = st.selectbox(
        "🎯 Toss Winner",
        [team_a, team_b]
    )

toss_decision = st.selectbox(
    "🔥 Toss Decision",
    ["bat", "field"]
)

score = st.slider(
    "💥 Expected Score",
    120,
    240,
    180
)

if team_a == team_b:
    st.error("⚠️ Team A and Team B cannot be same")

predict = st.button("⚡ PREDICT WINNER")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
#RESULT
# =========================================================

if predict and team_a != team_b:

    input_df = pd.DataFrame({

        "Bat First": [team_a],
        "Bat Second": [team_b],
        "Venue": [venue],
        "toss_winner": [toss_winner],
        "toss_decision": [toss_decision],
        "innings1_runs": [score]

    })

    prediction = model.predict_proba(input_df)

    team_a_percent = round(prediction[0][0] * 100)
    team_b_percent = round(prediction[0][1] * 100)

    winner = (
        team_a
        if team_a_percent > team_b_percent
        else team_b
    )

    st.markdown(f"""
    <div class="result-card">

    <div class="winner">
    🏆 {winner}
    </div>

    <div class="percent">
    {team_a}: {team_a_percent}%
    </div>

    <div class="percent">
    {team_b}: {team_b_percent}%
    </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("## 📊 LIVE WIN PROBABILITY")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            f"{team_a} Win Chance",
            f"{team_a_percent}%"
        )
        st.progress(team_a_percent / 100)

    with c2:
        st.metric(
            f"{team_b} Win Chance",
            f"{team_b_percent}%"
        )
        st.progress(team_b_percent / 100)

    st.markdown("## 🚀 LIVE MATCH ANALYTICS")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.info(f"🏟️ Venue: {venue}")

    with a2:
        st.warning(f"🎯 Toss Winner: {toss_winner}")

    with a3:
        st.success(f"🔥 Expected Score: {score}")
        

st.markdown("</div>", unsafe_allow_html=True)