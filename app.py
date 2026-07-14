import os, sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express       as px
import streamlit            as st
sys.path.insert(0, os.path.dirname(__file__))

# ─── page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title = "EDU PROPHET AI",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9; }

[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 4px;
}
.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #e6edf3;
    margin: 0;
    line-height: 1.2;
}
.page-sub {
    color: #8b949e;
    font-size: 15px;
    margin-bottom: 32px;
}
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-accent-green  { border-left: 4px solid #3fb950; }
.card-accent-yellow { border-left: 4px solid #d29922; }
.card-accent-red    { border-left: 4px solid #f85149; }
.card-accent-blue   { border-left: 4px solid #58a6ff; }

.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 1px;
}
.badge-low    { background:#1a4731; color:#3fb950; border:1px solid #3fb950; }
.badge-medium { background:#3d2d00; color:#d29922; border:1px solid #d29922; }
.badge-high   { background:#3d0c0c; color:#f85149; border:1px solid #f85149; }
.badge-ready  { background:#1a4731; color:#3fb950; border:1px solid #3fb950; }
.badge-not    { background:#3d0c0c; color:#f85149; border:1px solid #f85149; }

.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 15px;
    width: 100%;
    transition: all .2s;
}
.stButton > button:hover { background: #2ea043; transform: translateY(-1px); }

/* Text input styling */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}
.stTextInput label {
    color: #8b949e !important;
    font-size: 13px !important;
}

.stProgress > div > div > div > div { background: linear-gradient(90deg,#238636,#3fb950); }

.stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:8px; }
.stTabs [data-baseweb="tab"]      { color:#8b949e; }
.stTabs [aria-selected="true"]    { color:#58a6ff; border-bottom-color:#58a6ff; }

hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

MODEL_DIR = "models"
DATA_DIR  = "data"

@st.cache_resource
def load_dropout_bundle():
    path = os.path.join(MODEL_DIR, "dropout_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_placement_bundle():
    path = os.path.join(MODEL_DIR, "placement_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def gauge_chart(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"color": "#c9d1d9", "size": 14}},
        number = {"suffix": "%", "font": {"color": "#e6edf3", "size": 28}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#30363d",
                      "tickfont": {"color": "#8b949e"}},
            "bar":   {"color": color, "thickness": 0.7},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  40], "color": "#1a1f27"},
                {"range": [40, 70], "color": "#1e2530"},
                {"range": [70,100], "color": "#21293a"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=20, r=20, t=40, b=20), height=220,
    )
    return fig


def radar_chart(categories: list, values: list, title: str) -> go.Figure:
    fig = go.Figure(go.Scatterpolar(
        r     = values + [values[0]],
        theta = categories + [categories[0]],
        fill  = "toself",
        fillcolor = "rgba(35,134,54,0.2)",
        line  = dict(color="#3fb950", width=2),
    ))
    fig.update_layout(
        polar = dict(
            bgcolor    = "#161b22",
            radialaxis = dict(visible=True, range=[0,100],
                              gridcolor="#30363d", tickfont={"color":"#8b949e"}),
            angularaxis= dict(gridcolor="#30363d",
                              tickfont={"color":"#c9d1d9"}),
        ),
        paper_bgcolor = "#0d1117",
        plot_bgcolor  = "#0d1117",
        title = {"text": title, "font": {"color": "#e6edf3", "size": 15}},
        margin = dict(l=40, r=40, t=50, b=40),
        height = 340,
    )
    return fig


def bar_chart(x, y, title, color="#58a6ff", horiz=False) -> go.Figure:
    if horiz:
        fig = go.Figure(go.Bar(x=y, y=x, orientation="h",
                               marker_color=color,
                               marker_line_width=0))
    else:
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color,
                               marker_line_width=0))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        title={"text": title, "font": {"color": "#e6edf3", "size": 14}},
        font={"color": "#8b949e"},
        xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        margin=dict(l=20, r=20, t=45, b=20),
        height=320,
    )
    return fig


def setup_required():
    st.error("⚠️  Models not found. Run `python setup.py` first to generate data and train models.")
    st.code("python setup.py", language="bash")


def input_error(field, hint):
    st.error(f"⚠️ Invalid value for **{field}**. {hint}")


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR NAV
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
      <div style='font-size:36px'>🎓</div>
      <div style='font-family:"Space Mono",monospace; font-size:13px;
                  color:#58a6ff; letter-spacing:2px; margin-top:6px;'>
        STUDENT SUCCESS
      </div>
      <div style='font-family:"Space Mono",monospace; font-size:10px;
                  color:#8b949e; letter-spacing:2px;'>
        PREDICTION SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard",
         "🚨  Dropout Risk",
         "💼  Placement Readiness"],
        label_visibility="collapsed",
    )
    # ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Student Success Prediction System</h1>', unsafe_allow_html=True)
    

    dropout_df   = load_csv("dropout_data.csv")
    placement_df = load_csv("placement_data.csv")

    if dropout_df is None or placement_df is None:
        st.warning("📂 No datasets found. Run `python setup.py` to generate data and train models.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Students", f"{len(dropout_df):,}")
        high_risk = (dropout_df["dropout_risk"] == "High").sum()
        c2.metric("High Dropout Risk", f"{high_risk}", delta=f"{high_risk/len(dropout_df)*100:.1f}%", delta_color="inverse")
        placed = placement_df["placed"].sum()
        c3.metric("Placement Ready", f"{placed}", delta=f"{placed/len(placement_df)*100:.1f}%")
        avg_pkg = placement_df[placement_df["placed"]==1]["expected_package_lpa"].mean()
        c4.metric("Avg Package (LPA)", f"₹{avg_pkg:.1f}")
        avg_cgpa = placement_df["cgpa"].mean()
        c5.metric("Avg CGPA", f"{avg_cgpa:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            risk_counts = dropout_df["dropout_risk"].value_counts()
            colors = {"Low":"#3fb950","Medium":"#d29922","High":"#f85149"}
            fig = go.Figure(go.Pie(
                labels = risk_counts.index,
                values = risk_counts.values,
                hole   = 0.55,
                marker = dict(colors=[colors[l] for l in risk_counts.index],
                              line=dict(color="#0d1117", width=3)),
                textfont=dict(color="#e6edf3"),
            ))
            fig.update_layout(
                title={"text":"Dropout Risk Distribution","font":{"color":"#e6edf3","size":15}},
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                legend=dict(font=dict(color="#c9d1d9")),
                height=320, margin=dict(l=10,r=10,t=50,b=10),
                annotations=[dict(text="Risk", x=0.5, y=0.5,
                                  font_size=16, font_color="#8b949e",
                                  showarrow=False)],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            branch_place = placement_df.groupby("branch")["placed"].mean().sort_values() * 100
            fig = bar_chart(branch_place.index.tolist(),
                            branch_place.values.round(1).tolist(),
                            "Placement Rate by Branch (%)", "#58a6ff")
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(dropout_df, x="gpa", nbins=25,
                               color="dropout_risk",
                               color_discrete_map={"Low":"#3fb950","Medium":"#d29922","High":"#f85149"},
                               title="GPA vs Dropout Risk")
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                              font=dict(color="#8b949e"),
                              title_font_color="#e6edf3",
                              legend=dict(font=dict(color="#c9d1d9")),
                              height=300, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            placed_pkg = placement_df[placement_df["placed"]==1]["expected_package_lpa"]
            fig = go.Figure(go.Histogram(x=placed_pkg, nbinsx=30,
                                         marker_color="#58a6ff", opacity=0.8))
            fig.update_layout(
                title={"text":"Expected Package Distribution (LPA)",
                       "font":{"color":"#e6edf3","size":15}},
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#8b949e"),
                height=300, margin=dict(l=10,r=10,t=50,b=10),
                xaxis=dict(gridcolor="#30363d"),
                yaxis=dict(gridcolor="#30363d"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("""
            <div class='card card-accent-red'>
              <div style='font-size:28px'>🚨</div>
              <h3 style='margin:8px 0 4px; color:#e6edf3;'>Module 1: Dropout Risk</h3>
              <p style='color:#8b949e; margin:0;'>Predicts student dropout likelihood using
              attendance, GPA, backlogs, mental health, and family factors.
              Output: Low / Medium / High risk.</p>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            st.markdown("""
            <div class='card card-accent-green'>
              <div style='font-size:28px'>💼</div>
              <h3 style='margin:8px 0 4px; color:#e6edf3;'>Module 2: Placement Readiness</h3>
              <p style='color:#8b949e; margin:0;'>Predicts placement chances using CGPA,
              technical skills, communication, internships, projects, and more.
              Also estimates expected CTC.</p>
            </div>
            """, unsafe_allow_html=True)
            # ═══════════════════════════════════════════════════════════════════
#  PAGE 2 — DROPOUT RISK
# ═══════════════════════════════════════════════════════════════════
elif "Dropout" in page:
    st.markdown('<p class="section-title"></p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">🚨 Dropout Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Enter student details to assess dropout probability</p>', unsafe_allow_html=True)

    bundle = load_dropout_bundle()
    if bundle is None:
        setup_required()
    else:
        left, right = st.columns([1.1, 0.9])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📋 Academic Information**")
            c1, c2 = st.columns(2)
            attendance  = c1.text_input("Attendance (%)",        value="75",  placeholder="20 – 100")
            gpa         = c2.text_input("GPA (out of 10)",        value="7.0", placeholder="3.0 – 10.0")
            backlogs    = c1.text_input("Active Backlogs",        value="0",   placeholder="0 – 10")
            study_hrs   = c2.text_input("Study Hours / Day",      value="4.0", placeholder="0 – 12")

            st.markdown("<br>**👨‍👩‍👧 Personal & Socio-Economic**")
            c3, c4 = st.columns(2)
            income        = c3.text_input("Family Income (LPA)",          value="4.0", placeholder="0 – 30")
            distance      = c4.text_input("Distance from College (km)",   value="10",  placeholder="0 – 100")
            participation = c3.text_input("Participation Score (0–100)",  value="60",  placeholder="0 – 100")
            scholarship   = c4.text_input("Has Scholarship? (Yes / No)",  value="No",  placeholder="Yes or No")
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔍  Predict Dropout Risk"):
                errors = []
                try:
                    att_v = float(attendance)
                    if not 20 <= att_v <= 100: errors.append("Attendance must be 20–100.")
                except ValueError:
                    errors.append("Attendance must be a number.")

                try:
                    gpa_v = float(gpa)
                    if not 3.0 <= gpa_v <= 10.0: errors.append("GPA must be 3.0–10.0.")
                except ValueError:
                    errors.append("GPA must be a number.")

                try:
                    bl_v = int(backlogs)
                    if not 0 <= bl_v <= 10: errors.append("Backlogs must be 0–10.")
                except ValueError:
                    errors.append("Active Backlogs must be a whole number.")

                try:
                    sh_v = float(study_hrs)
                    if not 0 <= sh_v <= 12: errors.append("Study hours must be 0–12.")
                except ValueError:
                    errors.append("Study hours must be a number.")

                try:
                    inc_v = float(income)
                    if not 0 <= inc_v <= 30: errors.append("Family income must be 0–30.")
                except ValueError:
                    errors.append("Family income must be a number.")

                try:
                    dist_v = float(distance)
                    if not 0 <= dist_v <= 100: errors.append("Distance must be 0–100.")
                except ValueError:
                    errors.append("Distance must be a number.")

                try:
                    part_v = float(participation)
                    if not 0 <= part_v <= 100: errors.append("Participation score must be 0–100.")
                except ValueError:
                    errors.append("Participation score must be a number.")

                sch_str = scholarship.strip().lower()
                if sch_str not in ("yes", "no"):
                    errors.append("Scholarship must be 'Yes' or 'No'.")

                if errors:
                    for e in errors:
                        st.error(f"⚠️ {e}")
                else:
                    inp = {
                        "attendance_pct":      att_v,
                        "gpa":                 gpa_v,
                        "backlogs":            bl_v,
                        "family_income_lpa":   inc_v,
                        "participation_score": part_v,
                        "study_hours_day":     sh_v,
                        "distance_km":         dist_v,
                        "has_scholarship":     1 if sch_str == "yes" else 0,
                    }
                    st.session_state["dropout_result"] = inp

        with right:
            if "dropout_result" in st.session_state:
                from src.dropout_model import predict as d_predict
                res   = d_predict(st.session_state["dropout_result"])
                label = res["risk_label"]
                probs = res["probabilities"]
                conf  = res["confidence"]

                badge_cls = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high"}[label]
                st.markdown(f"""
                <div class='card'>
                  <p style='color:#8b949e; margin:0 0 8px;'>Dropout Risk Level</p>
                  <span class='badge {badge_cls}'>{label.upper()} RISK</span>
                  <p style='color:#8b949e; font-size:13px; margin:12px 0 0;'>
                    Confidence: <strong style='color:#e6edf3;'>{conf}%</strong>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                g1, g2, g3 = st.columns(3)
                colors = {"Low":"#3fb950","Medium":"#d29922","High":"#f85149"}
                cols   = [g1, g2, g3]
                for i, (risk, prob) in enumerate(probs.items()):
                    cols[i].plotly_chart(
                        gauge_chart(prob, risk, colors[risk]),
                        use_container_width=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                if label == "High":
                    tips = [
                        "🔴 Immediate counsellor intervention recommended",
                        "📚 Enrol in backlog clearance program",
                        "💰 Apply for emergency financial aid / scholarship",
                        "📅 Create personalised attendance recovery plan",
                    ]
                elif label == "Medium":
                    tips = [
                        "🟡 Monitor attendance and GPA closely",
                        "📖 Peer mentoring / study group allocation",
                        "💬 Regular check-in with class advisor",
                        "🎯 Goal-setting session with career counsellor",
                    ]
                else:
                    tips = [
                        "🟢 Student is on a healthy academic track",
                        "🚀 Encourage participation in competitive events",
                        "📈 Channel energy into placement preparation",
                    ]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**💡 Recommended Interventions**")
                for t in tips:
                    st.markdown(f"<div style='padding:4px 0; color:#c9d1d9;'>{t}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='card' style='text-align:center; padding:60px 20px;'>
                  <div style='font-size:48px;'>🎯</div>
                  <p style='color:#8b949e; margin-top:12px;'>
                    Fill in the student details on the left<br>and click <strong>Predict</strong> to see results.
                  </p>
                </div>
                """, unsafe_allow_html=True)
                # ═══════════════════════════════════════════════════════════════════
#  PAGE 3 — PLACEMENT READINESS
# ═══════════════════════════════════════════════════════════════════
elif "Placement" in page:
    st.markdown('<p class="section-title"></p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">💼 Placement Readiness Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Assess campus placement probability and estimated CTC</p>', unsafe_allow_html=True)

    bundle = load_placement_bundle()
    if bundle is None:
        setup_required()
    else:
        left, right = st.columns([1.1, 0.9])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📋 Academic & Technical Profile**")
            c1, c2 = st.columns(2)
            cgpa            = c1.text_input("CGPA (out of 10)",            value="7.5",  placeholder="4.0 – 10.0")
            technical_score = c2.text_input("Technical Score (0–100)",     value="65",   placeholder="0 – 100")
            communication   = c1.text_input("Communication Score (0–100)", value="70",   placeholder="0 – 100")
            aptitude        = c2.text_input("Aptitude Score (0–100)",      value="65",   placeholder="0 – 100")

            st.markdown("<br>**🛠 Experience & Extras**")
            c3, c4 = st.columns(2)
            internships      = c3.text_input("No. of Internships",           value="1",  placeholder="0 – 5")
            projects         = c4.text_input("No. of Projects",              value="2",  placeholder="0 – 10")
            certifications   = c3.text_input("Certifications",               value="2",  placeholder="0 – 10")
            backlogs         = c4.text_input("Active Backlogs",              value="0",  placeholder="0 – 10")
            hackathons       = c3.text_input("Hackathons Participated",      value="1",  placeholder="0 – 10")
            github_repos     = c4.text_input("GitHub Repos",                 value="3",  placeholder="0 – 50")
            leadership_roles = c3.text_input("Leadership Roles",             value="0",  placeholder="0 – 5")
            mock_interview   = c4.text_input("Mock Interview Score (0–100)", value="60", placeholder="0 – 100")

            st.markdown("<br>**📌 Other Details**")
            branch = st.text_input("Branch (e.g. CSE, CSE(AI&ML), IT, ECE)", value="CSE", placeholder="CSE / CSE(AI&ML) / IT / MECH / ECE")
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("💼  Predict Placement Readiness"):
                errors = []

                try:
                    cgpa_v = float(cgpa)
                    if not 4.0 <= cgpa_v <= 10.0: errors.append("CGPA must be 4.0–10.0.")
                except ValueError:
                    errors.append("CGPA must be a number.")

                try:
                    tech_v = float(technical_score)
                    if not 0 <= tech_v <= 100: errors.append("Technical score must be 0–100.")
                except ValueError:
                    errors.append("Technical score must be a number.")

                try:
                    comm_v = float(communication)
                    if not 0 <= comm_v <= 100: errors.append("Communication score must be 0–100.")
                except ValueError:
                    errors.append("Communication score must be a number.")

                try:
                    apt_v = float(aptitude)
                    if not 0 <= apt_v <= 100: errors.append("Aptitude score must be 0–100.")
                except ValueError:
                    errors.append("Aptitude score must be a number.")

                try:
                    int_v = int(internships)
                    if not 0 <= int_v <= 5: errors.append("Internships must be 0–5.")
                except ValueError:
                    errors.append("Internships must be a whole number.")

                try:
                    proj_v = int(projects)
                    if not 0 <= proj_v <= 10: errors.append("Projects must be 0–10.")
                except ValueError:
                    errors.append("Projects must be a whole number.")

                try:
                    cert_v = int(certifications)
                    if not 0 <= cert_v <= 10: errors.append("Certifications must be 0–10.")
                except ValueError:
                    errors.append("Certifications must be a whole number.")

                try:
                    bl_v = int(backlogs)
                    if not 0 <= bl_v <= 10: errors.append("Backlogs must be 0–10.")
                except ValueError:
                    errors.append("Backlogs must be a whole number.")

                try:
                    hack_v = int(hackathons)
                    if not 0 <= hack_v <= 10: errors.append("Hackathons must be 0–10.")
                except ValueError:
                    errors.append("Hackathons must be a whole number.")

                try:
                    github_v = int(github_repos)
                    if not 0 <= github_v <= 50: errors.append("GitHub repos must be 0–50.")
                except ValueError:
                    errors.append("GitHub repos must be a whole number.")

                try:
                    lead_v = int(leadership_roles)
                    if not 0 <= lead_v <= 5: errors.append("Leadership roles must be 0–5.")
                except ValueError:
                    errors.append("Leadership roles must be a whole number.")

                try:
                    mock_v = float(mock_interview)
                    if not 0 <= mock_v <= 100: errors.append("Mock interview score must be 0–100.")
                except ValueError:
                    errors.append("Mock interview score must be a number.")

                branch_v = branch.strip().upper()

                if errors:
                    for e in errors:
                        st.error(f"⚠️ {e}")
                else:
                    inp = {
                        "cgpa":                cgpa_v,
                        "technical_score":     tech_v,
                        "communication_score": comm_v,
                        "aptitude_score":      apt_v,
                        "internships":         int_v,
                        "projects_count":      proj_v,
                        "certifications":      cert_v,
                        "backlogs":            bl_v,
                        "hackathons":          hack_v,
                        "github_repos":        github_v,
                        "leadership_roles":    lead_v,
                        "mock_interview_score": mock_v,
                        "branch":              branch_v,
                    }
                    st.session_state["placement_result"] = inp

        with right:
            if "placement_result" in st.session_state:
                from src.placement_model import predict as p_predict
                res      = p_predict(st.session_state["placement_result"])
                placed   = res["placed"]
                raw_prob = res["probability"]
                # model returns either a float or a dict of class probabilities
                if isinstance(raw_prob, dict):
                    prob = raw_prob.get("Placement Ready", raw_prob.get("placed", next(iter(raw_prob.values()))))
                else:
                    prob = float(raw_prob)
                pkg      = res.get("expected_package_lpa", None)
                conf     = res["confidence"]

                badge_cls = "badge-ready" if placed else "badge-not"
                badge_txt = "PLACEMENT READY" if placed else "NOT READY"
                st.markdown(f"""
                <div class='card'>
                  <p style='color:#8b949e; margin:0 0 8px;'>Placement Status</p>
                  <span class='badge {badge_cls}'>{badge_txt}</span>
                  <p style='color:#8b949e; font-size:13px; margin:12px 0 0;'>
                    Confidence: <strong style='color:#e6edf3;'>{conf}%</strong>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # Placement probability gauge
                st.plotly_chart(
                    gauge_chart(prob, "Placement Probability", "#3fb950" if placed else "#f85149"),
                    use_container_width=True,
                )

                # Expected package
                if pkg is not None:
                    st.markdown(f"""
                    <div class='card card-accent-blue'>
                      <p style='color:#8b949e; margin:0 0 4px; font-size:13px;'>Estimated CTC</p>
                      <p style='color:#e6edf3; font-size:28px; font-weight:700; margin:0;'>
                        ₹{pkg:.1f} LPA
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Radar chart of skills
                inp = st.session_state["placement_result"]
                radar_cats = ["Technical", "Communication", "Aptitude", "Projects", "Internships", "Certifications"]
                radar_vals = [
                    inp["technical_score"],
                    inp["communication_score"],
                    inp["aptitude_score"],
                    min(inp["projects_count"]    * 10, 100),
                    min(inp["internships"]        * 20, 100),
                    min(inp["certifications"]     * 10, 100),
                ]
                st.plotly_chart(
                    radar_chart(radar_cats, radar_vals, "Skill Profile"),
                    use_container_width=True,
                )

                # Tips
                st.markdown("<br>", unsafe_allow_html=True)
                if placed:
                    tips = [
                        "✅ Strong candidate — target dream companies",
                        "🏆 Prepare for aptitude & technical rounds",
                        "🤝 Network on LinkedIn with industry professionals",
                        "📄 Keep resume updated with latest projects",
                    ]
                else:
                    tips = [
                        "🔧 Improve technical skills via online courses",
                        "🗣 Work on communication through mock interviews",
                        "📁 Add at least 1–2 more projects to portfolio",
                        "📜 Earn relevant certifications (AWS, Google, etc.)",
                        "🏢 Apply for internships to gain industry exposure",
                    ]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**💡 Recommended Actions**")
                for t in tips:
                    st.markdown(f"<div style='padding:4px 0; color:#c9d1d9;'>{t}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class='card' style='text-align:center; padding:60px 20px;'>
                  <div style='font-size:48px;'>💼</div>
                  <p style='color:#8b949e; margin-top:12px;'>
                    Fill in the student profile on the left<br>and click <strong>Predict</strong> to see results.
                  </p>
                </div>
                """, unsafe_allow_html=True)
