"""
Student Performance Prediction System - Light Modern UI
Perfect readability with light colors and dark text
Run: streamlit run app_light.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EduPredict - Student Performance System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for Light Theme with Dark Text
st.markdown("""
<style>
    /* Main container - Light background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hero section - Light gradient */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .hero h1 {
        color: white !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .hero p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem;
    }
    
    /* Card styling - White background, dark text */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    .card h3, .card h4, .card p, .card li {
        color: #1a1a1a !important;
    }
    
    /* Metric cards - Light gradient */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-card h4 {
        color: white !important;
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0 0 0;
        color: white !important;
    }
    
    /* Result cards */
    .result-pass {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .result-pass h2 {
        color: #155724 !important;
        font-size: 2rem;
        margin: 0;
    }
    
    .result-pass p {
        color: #155724 !important;
    }
    
    .result-fail {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .result-fail h2 {
        color: #721c24 !important;
        font-size: 2rem;
        margin: 0;
    }
    
    .result-fail p {
        color: #721c24 !important;
    }
    
    /* Tab styling - Light theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #333333 !important;
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Quick buttons */
    .quick-btn {
        background: #f8f9fa !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
    }
    
    /* Sidebar - Light theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box p {
        color: #1a1a1a !important;
    }
    
    /* Warning box */
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Dataframe */
    .dataframe {
        background: white;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress bar */
    .progress-container {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        height: 25px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: white;
        border-radius: 10px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = {
        'student_id': list(range(1, 21)),
        'study_hours': [2, 3, 1, 8, 7, 6, 2, 9, 5, 3, 4, 1, 8, 7, 2, 6, 3, 9, 1, 5],
        'attendance': [50, 60, 45, 90, 85, 78, 55, 95, 70, 60, 65, 40, 92, 88, 58, 80, 62, 97, 42, 72],
        'previous_marks': [40, 50, 35, 85, 80, 75, 48, 92, 65, 55, 60, 30, 88, 85, 50, 78, 55, 95, 38, 70],
        'assignments': [45, 55, 30, 88, 82, 70, 50, 90, 68, 52, 58, 35, 90, 84, 48, 75, 60, 94, 35, 68],
        'internal_marks': [40, 50, 35, 90, 85, 72, 45, 93, 67, 50, 60, 32, 87, 86, 45, 77, 58, 96, 37, 69],
        'final_result': [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)

@st.cache_resource
def train_model():
    df = load_data()
    feature_cols = ['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks']
    X = df[feature_cols]
    y = df['final_result']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    accuracy = accuracy_score(y, model.predict(X_scaled))
    
    return model, scaler, accuracy, df

# Load everything
df = load_data()
model, scaler, accuracy, _ = train_model()

# Prediction function
def predict_student(study_hours, attendance, previous_marks, assignments, internal_marks):
    input_data = [[study_hours, attendance, previous_marks, assignments, internal_marks]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    return "Pass" if prediction == 1 else "Fail", confidence

# Hero Section
st.markdown("""
<div class="hero">
    <h1>📚 EduPredict</h1>
    <p>AI-Powered Student Performance Prediction System</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Real-time Predictions | Early Intervention | Data-Driven Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-male.png", width=80)
    st.markdown("### Student Analytics Hub")
    st.markdown("---")
    
    pass_count = df['final_result'].sum()
    pass_rate = (pass_count/len(df))*100
    
    st.markdown(f"""
    <div style="background: #f0f0f0; border-radius: 15px; padding: 1rem; margin: 1rem 0;">
        <p style="color: #333; font-size: 0.9rem;">📊 Overall Pass Rate</p>
        <p style="font-size: 2rem; font-weight: bold; color: #667eea;">{pass_rate:.0f}%</p>
        <div class="progress-container">
            <div class="progress-fill" style="width: {pass_rate}%;">{pass_rate:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.info(f"**Algorithm:** Random Forest\n**Accuracy:** {accuracy:.1%}\n**Students:** {len(df)}")
    st.markdown("---")
    st.markdown("### Features Used")
    st.markdown("• Study Hours\n• Attendance\n• Previous Marks\n• Assignments\n• Internal Marks")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 PREDICTOR", "📋 STUDENT HUB", "⚠️ RISK MONITOR", "📊 INSIGHTS"])

# TAB 1: Predictor
with tab1:
    st.markdown("### AI Performance Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("#### 📚 Academic Metrics")
            study = st.slider("Study Hours per Day", 0.0, 12.0, 5.0, 0.5)
            attendance = st.slider("Attendance Percentage", 0, 100, 75)
            previous = st.slider("Previous Exam Marks", 0, 100, 65)
    
    with col2:
        with st.container():
            st.markdown("#### 📝 Assessment Scores")
            assignments = st.slider("Assignment Scores", 0, 100, 70)
            internal = st.slider("Internal Assessment Marks", 0, 100, 68)
    
    # Quick test buttons
    st.markdown("#### Quick Test Profiles")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        if st.button("🏆 High Achiever", use_container_width=True):
            study, attendance, previous, assignments, internal = 9.0, 95, 92, 94, 93
            st.rerun()
    with col_b:
        if st.button("📚 Average Student", use_container_width=True):
            study, attendance, previous, assignments, internal = 5.0, 70, 65, 68, 66
            st.rerun()
    with col_c:
        if st.button("⚠️ At Risk", use_container_width=True):
            study, attendance, previous, assignments, internal = 1.5, 42, 35, 38, 36
            st.rerun()
    with col_d:
        if st.button("🎓 Borderline", use_container_width=True):
            study, attendance, previous, assignments, internal = 4.0, 55, 48, 50, 49
            st.rerun()
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        result, confidence = predict_student(study, attendance, previous, assignments, internal)
        
        avg_score = (previous + assignments + internal) / 3
        study_efficiency = (study / 10) * 100
        
        if result == "Pass":
            st.markdown(f"""
            <div class="result-pass">
                <h2>✅ PASS</h2>
                <p>Predicted to <strong>PASS</strong> with {confidence:.1%} confidence</p>
                <hr>
                <p>📊 Average Score: {avg_score:.0f} | 📈 Study Efficiency: {study_efficiency:.0f}%</p>
                <p>🎯 Risk Level: <strong style="color: #155724;">Low Risk</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="result-fail">
                <h2>⚠️ FAIL</h2>
                <p>Predicted to <strong>FAIL</strong> with {confidence:.1%} confidence</p>
                <hr>
                <p>📊 Average Score: {avg_score:.0f} | 📈 Study Efficiency: {study_efficiency:.0f}%</p>
                <p>🎯 Risk Level: <strong style="color: #721c24;">High Risk - Immediate Action Required</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.snow()
        
        # Feature analysis chart
        st.markdown("#### Performance Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Study Hours', 'Attendance', 'Previous Marks', 'Assignments', 'Internal Marks']
        values = [study, attendance, previous, assignments, internal]
        colors = ['#2ecc71' if v >= 50 else '#e74c3c' for v in values]
        
        bars = ax.bar(features, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Passing Threshold (50%)')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Performance Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.set_facecolor('#f8f9fa')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f'{val}', ha='center', fontweight='bold', fontsize=11)
        
        st.pyplot(fig)
        plt.close()

# TAB 2: Student Hub
with tab2:
    st.markdown("### Complete Student Database")
    
    # Add predictions
    df_pred = df.copy()
    predictions = []
    confidences = []
    
    for _, row in df.iterrows():
        pred, conf = predict_student(row['study_hours'], row['attendance'],
                                      row['previous_marks'], row['assignments'],
                                      row['internal_marks'])
        predictions.append(pred)
        confidences.append(f"{conf:.1%}")
    
    df_pred['prediction'] = predictions
    df_pred['confidence'] = confidences
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Students</h4>
            <p class="value">{len(df)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pass_pred = len(df_pred[df_pred['prediction'] == 'Pass'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Pass</h4>
            <p class="value">{pass_pred}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fail_pred = len(df_pred[df_pred['prediction'] == 'Fail'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Fail</h4>
            <p class="value">{fail_pred}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Accuracy</h4>
            <p class="value">{accuracy:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display table
    st.dataframe(df_pred, use_container_width=True)
    
    # Download button
    csv = df_pred.to_csv(index=False)
    st.download_button("📥 Download Student Data (CSV)", csv, "student_data.csv", "text/csv")

# TAB 3: Risk Monitor
with tab3:
    st.markdown("### At-Risk Student Identification")
    
    # Identify at-risk students
    at_risk = []
    for _, row in df.iterrows():
        pred, conf = predict_student(row['study_hours'], row['attendance'],
                                      row['previous_marks'], row['assignments'],
                                      row['internal_marks'])
        if pred == 'Fail':
            at_risk.append({
                'Student ID': row['student_id'],
                'Study Hours': row['study_hours'],
                'Attendance': f"{row['attendance']}%",
                'Previous Marks': row['previous_marks'],
                'Assignments': row['assignments'],
                'Internal Marks': row['internal_marks'],
                'Risk Probability': f"{conf:.1%}"
            })
    
    if at_risk:
        st.warning(f"🚨 {len(at_risk)} students identified as AT-RISK! Immediate attention required.")
        
        at_risk_df = pd.DataFrame(at_risk)
        st.dataframe(at_risk_df, use_container_width=True)
        
        # Download at-risk list
        csv = at_risk_df.to_csv(index=False)
        st.download_button("📥 Download At-Risk List", csv, "at_risk_students.csv", "text/csv")
        
        st.markdown("---")
        st.markdown("### Recommended Intervention Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>🔴 Immediate Actions</h4>
                <p>• One-on-one tutoring sessions<br>
                • Parent-teacher conference within 48 hours<br>
                • Daily progress monitoring<br>
                • Study skills assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🟡 Support Measures</h4>
                <p>• Weekly group study sessions<br>
                • Peer mentoring program<br>
                • Assignment deadline extensions<br>
                • Bi-weekly progress review</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No at-risk students detected. All students are on track!")

# TAB 4: Insights
with tab4:
    st.markdown("### Performance Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pass/Fail Pie Chart
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        pass_cnt = df['final_result'].sum()
        fail_cnt = len(df) - pass_cnt
        ax1.pie([pass_cnt, fail_cnt], labels=['Pass', 'Fail'], 
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                explode=(0.05, 0.05), shadow=True, textprops={'fontsize': 12})
        ax1.set_title('Student Performance Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        # Study Hours Distribution
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        pass_data = df[df['final_result'] == 1]['study_hours']
        fail_data = df[df['final_result'] == 0]['study_hours']
        
        ax2.hist(pass_data, bins=10, alpha=0.7, label='Pass', color='#2ecc71', edgecolor='black')
        ax2.hist(fail_data, bins=10, alpha=0.7, label='Fail', color='#e74c3c', edgecolor='black')
        ax2.set_xlabel('Study Hours', fontsize=12)
        ax2.set_ylabel('Number of Students', fontsize=12)
        ax2.set_title('Study Hours Distribution by Result', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_facecolor('#f8f9fa')
        st.pyplot(fig2)
        plt.close()
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Attendance Box Plot
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        pass_att = df[df['final_result'] == 1]['attendance']
        fail_att = df[df['final_result'] == 0]['attendance']
        
        bp = ax3.boxplot([pass_att, fail_att], labels=['Pass', 'Fail'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax3.set_ylabel('Attendance (%)', fontsize=12)
        ax3.set_title('Attendance Distribution by Result', fontsize=14, fontweight='bold')
        st.pyplot(fig3)
        plt.close()
    
    with col4:
        # Feature Comparison
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        features = ['Study Hours', 'Attendance', 'Previous Marks', 'Assignments', 'Internal']
        pass_means = df[df['final_result']==1][['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks']].mean()
        fail_means = df[df['final_result']==0][['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks']].mean()
        
        x = range(len(features))
        width = 0.35
        ax4.bar([i - width/2 for i in x], pass_means, width, label='Pass', color='#2ecc71', edgecolor='black')
        ax4.bar([i + width/2 for i in x], fail_means, width, label='Fail', color='#e74c3c', edgecolor='black')
        ax4.set_xlabel('Features', fontsize=12)
        ax4.set_ylabel('Average Score', fontsize=12)
        ax4.set_title('Pass vs Fail Feature Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(features, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    # Correlation Heatmap
    st.markdown("#### Feature Correlation Matrix")
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    corr = df[['study_hours', 'attendance', 'previous_marks', 'assignments', 'internal_marks', 'final_result']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax5, fmt='.2f', square=True)
    ax5.set_title('Correlation Between Features', fontsize=14, fontweight='bold')
    st.pyplot(fig5)
    plt.close()

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 EduPredict | AI-Powered Student Performance Prediction System</p>
    <p style="font-size: 0.8rem;">Helping educators make data-driven decisions</p>
</div>
""", unsafe_allow_html=True)