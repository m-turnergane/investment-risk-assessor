import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
import hashlib

# Global database connection
conn = sqlite3.connect('user_profiles.db')
c = conn.cursor()

# Database setup
def setup_database():
    global conn, c
    
    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name_hash TEXT,
                  age INTEGER,
                  income REAL,
                  dependents INTEGER,
                  investment_experience TEXT,
                  risk_score INTEGER,
                  risk_tolerance TEXT,
                  equity_allocation INTEGER,
                  income_allocation INTEGER)''')

    # Add the new column if it doesn't exist
    c.execute("PRAGMA table_info(user_profiles)")
    columns = [column[1] for column in c.fetchall()]
    if 'name_hash' not in columns:
        c.execute("ALTER TABLE user_profiles ADD COLUMN name_hash TEXT")

    setup_feedback_table()

    conn.commit()

def setup_feedback_table():
    global conn, c
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  feedback TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()

# Call setup_database at the start of your script
setup_database()

# Risk assessment questions
questions = [
    {"type": "text", "question": "What is your name?", "key": "name"},
    {"type": "initial", "key": "initial", "questions": [
        {"type": "number", "question": "Age", "key": "age"},
        {"type": "select", "question": "Marital status", "key": "marital_status",
         "options": ["Single", "Common law", "Married", "Separated", "Divorced", "Widowed"]},
        {"type": "number", "question": "How many dependents do you have?", "min_value": 0, "max_value": 10, "key": "dependents"}
    ]},
    {"type": "employment_income", "key": "employment_income", "questions": [
        {"type": "radio", "question": "Are you currently employed?", "key": "employed",
         "options": ["Yes", "No"]},
        {"type": "number", "question": "What is your annual household income?", "key": "income"},
        {"type": "buttons", "question": "Which statement best describes your home ownership status?", "key": "home_ownership",
         "options": ["I don't own a home", "I'm paying a mortgage", "My mortgage is paid off"]}
    ]},
    {"type": "assets_liabilities", "key": "assets_liabilities", "questions": [
        {"type": "number", "question": "What is the total value of all your assets?", "key": "total_assets"},
        {"type": "number", "question": "What is the value of your fixed assets (e.g., property, vehicles)?", "key": "fixed_assets"},
        {"type": "number", "question": "What is the total value of your liabilities?", "key": "liabilities"}
    ]},
    {"type": "multiselect", "question": "What are your primary financial goals?", 
     "options": ["Retirement", "Home purchase", "Education", "Emergency fund", "Wealth accumulation"], 
     "key": "financial_goals"},
    {"type": "select", "question": "Which life stage best describes you?", 
     "options": ["Starting out", "Career building", "Peak earning years", "Pre-retirement", "Retirement"], 
     "key": "life_stage"},
    {"type": "image_buttons", "question": "How would you describe your investment experience?", 
     "options": [
         {"text": "Mostly Cash Savings", "image": "💰", "key": "cash_savings"},
         {"text": "Bonds, Income funds, GICs", "image": "📊", "key": "bonds_income"},
         {"text": "Mutual Funds and Exchange Traded Funds (ETFs)", "image": "📈", "key": "mutual_etfs"},
         {"text": "Self-Directed Investor: Stocks, Equities, Cryptocurrencies", "image": "🚀", "key": "self_directed"}
     ], 
     "key": "investment_experience"},
    {"type": "radio", "question": "How would you react if your investment lost 20% in a year?", 
     "options": ["Sell all investments", "Sell some", "Hold steady", "Buy more", "Buy a lot more"], "key": "market_reaction"},
    {"type": "chart", "question": "What level of volatility would you be the most comfortable with?", 
     "options": ["Low Volatility", "Balanced", "High Volatility"], 
     "key": "volatility_preference"},
    {"type": "radio", "question": "How long do you plan to hold your investments?", 
     "options": ["0-3 years", "3-5 years", "5+ years"], "key": "investment_horizon"},
    {"type": "radio", "question": "What's your risk capacity (ability to take risks)?", 
     "options": ["Very low", "Low", "Medium", "High", "Very high"], "key": "risk_capacity"},
    {"type": "slider", "question": "How confident are you in your investment knowledge?", 
     "min_value": 0, "max_value": 10, "step": 1, "key": "investment_confidence"}
]

def create_investment_chart(volatility_level):
    np.random.seed(42)  # For reproducibility
    x = np.arange(100)
    
    if volatility_level == "Low Volatility":
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 1, 100)
        y = 100 + trend + np.cumsum(noise) * 0.3
    elif volatility_level == "Balanced":
        trend = np.linspace(0, 20, 100)
        noise = np.random.normal(0, 1, 100)
        y = 100 + trend + np.cumsum(noise)
    else:  # High Volatility
        trend = np.linspace(0, 40, 100)  # Steeper overall trend
        volatility = np.random.normal(0, 1, 100) * 3  # Increased volatility
        momentum = np.cumsum(np.random.normal(0, 0.1, 100))  # Add momentum
        y = 100 + trend + np.cumsum(volatility) + momentum * 10
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('x', axis=alt.Axis(title='Time')),
        y=alt.Y('y', axis=alt.Axis(title='Value'), scale=alt.Scale(domain=[df.y.min()-10, df.y.max()+10])),
        tooltip=['x', 'y']
    ).properties(
        width=200,
        height=150,
        title=f"{volatility_level}"
    )
    
    return chart

def prepare_data_for_ml(answers):
    data = {}
    for key, value in answers.items():
        question = next((q for q in questions if q.get('key') == key), None)
        if question:
            if key == 'name':
                # Skip the name to maintain anonymity
                continue
            elif question['type'] == 'image_buttons' and key == 'investment_experience':
                # Handle investment experience separately
                for option in question['options']:
                    data[f"{key}_{option['key']}"] = 1 if value == option['text'] else 0
            elif 'options' in question:
                if isinstance(value, list):  # For multiselect questions
                    for option in question['options']:
                        data[f"{key}_{option}"] = 1 if option in value else 0
                elif isinstance(value, str):  # For categorical questions
                    for option in question['options']:
                        data[f"{key}_{option}"] = 1 if option == value else 0
            else:  # For numerical or text questions without options
                data[key] = value
        else:
            # Handle nested questions
            for q in questions:
                if q['type'] in ['initial', 'employment_income', 'assets_liabilities']:
                    sub_question = next((sq for sq in q['questions'] if sq.get('key') == key), None)
                    if sub_question:
                        if 'options' in sub_question and isinstance(value, str):
                            for option in sub_question['options']:
                                data[f"{key}_{option}"] = 1 if option == value else 0
                        else:
                            data[key] = value
                        break
    return data

def train_ml_model():
    data = pd.read_csv('user_data.csv')
    X = data.drop('risk_score', axis=1)
    y = data['risk_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'risk_assessment_model.joblib')
    return model

def calculate_risk_score_ml(answers):
    MIN_SAMPLES = 30  # Minimum number of samples required for ML

    if os.path.exists('user_data.csv'):
        data = pd.read_csv('user_data.csv')
        if len(data) >= MIN_SAMPLES:
            try:
                model = joblib.load('risk_assessment_model.joblib')
            except:
                model = train_ml_model()
            
            data = prepare_data_for_ml(answers)
            features = pd.DataFrame([data])
            return int(model.predict(features)[0])
        else:
            print(f"Using rule-based model. Need {MIN_SAMPLES - len(data)} more samples for ML.")
    else:
        print("No data file found. Using rule-based model.")

    return calculate_risk_score_rule_based(answers)

def calculate_risk_score_rule_based(answers):
    score = 0
    weights = {
        "age": lambda x: max(0, min(10, (65 - x) / 4)),  # Increased impact, 0-10 range
        "marital_status": {"Single": 8, "Common law": 6, "Married": 4, "Separated": 3, "Divorced": 2, "Widowed": 1},
        "dependents": lambda x: max(0, 8 - x * 2),  # 0 dependents = 8, 1 = 6, 2 = 4, 3 = 2, 4+ = 0
        "employed": {"Yes": 5, "No": 0},
        "income": lambda x: min(8, x / 25000),  # 1 point per $25k, max 8 points
        "home_ownership": {"I don't own a home": 0, "I'm paying a mortgage": 4, "My mortgage is paid off": 8},
        "investment_experience": {
            "Mostly Cash Savings and GICs": 0,
            "Bonds, Income funds, GICs": 3,
            "Mutual Funds and Exchange Traded Funds (ETFs)": 6,
            "Self-Directed Investor: Stocks, Equities, Cryptocurrencies": 10
        },
        "market_reaction": {"Sell all investments": 0, "Sell some": 3, "Hold steady": 6, "Buy more": 8, "Buy a lot more": 10},
        "volatility_preference": {"Low Volatility": 0, "Balanced": 5, "High Volatility": 10},
        "investment_horizon": {"0-3 years": 0, "3-5 years": 5, "5+ years": 10},
        "risk_capacity": {"Very low": 0, "Low": 3, "Medium": 6, "High": 8, "Very high": 10}
    }
    
    for key, value in answers.items():
        if key in weights:
            if callable(weights[key]):
                score += weights[key](value)
            elif isinstance(weights[key], dict):
                score += weights[key].get(value, 0)
            elif isinstance(value, (int, float)):
                score += value * weights[key]
    
    # Calculate net worth and add to score
    total_assets = answers.get('total_assets', 0)
    liabilities = answers.get('liabilities', 0)
    net_worth = total_assets - liabilities
    
    # Add net worth factor to score (0-10 points)
    net_worth_score = min(10, max(0, net_worth / 100000))  # 1 point per $100k net worth, max 10 points
    score += net_worth_score
    
    # Add liquidity factor to score (0-5 points)
    liquid_assets = total_assets - answers.get('fixed_assets', 0)
    liquidity_ratio = liquid_assets / total_assets if total_assets > 0 else 0
    liquidity_score = liquidity_ratio * 5  # 0-5 points based on liquidity ratio
    score += liquidity_score
    
    return score  # Note: We're not converting to int here to allow for more granularity

def get_risk_tolerance(score):
    if score < 30:
        return "Conservative"
    elif score < 50:
        return "Moderately Conservative"
    elif score < 70:
        return "Balanced"
    elif score < 90:
        return "Moderately Aggressive"
    else:
        return "Aggressive"

def get_allocation(risk_tolerance):
    allocations = {
        "Conservative": (20, 80),
        "Moderately Conservative": (40, 60),
        "Balanced": (60, 40),
        "Moderately Aggressive": (80, 20),
        "Aggressive": (100, 0)
    }
    return allocations[risk_tolerance]

def anonymize_data(data):
    # Hash sensitive information
    if 'name' in data:
        data['name_hash'] = hashlib.sha256(data['name'].encode()).hexdigest()
        del data['name']
    return data

def save_user_profile(user_data):
    global conn, c
    anonymized_data = anonymize_data(user_data)
    c.execute('''INSERT INTO user_profiles 
                 (name_hash, age, income, dependents, investment_experience, risk_score, risk_tolerance, equity_allocation, income_allocation) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
              (anonymized_data.get('name_hash', 'Anonymous'), anonymized_data['age'], anonymized_data['income'], 
               anonymized_data.get('dependents', 'No'), anonymized_data['investment_experience'], 
               anonymized_data['risk_score'], anonymized_data['risk_tolerance'], 
               anonymized_data['equity_allocation'], anonymized_data['income_allocation']))
    conn.commit()
    
    # Save data for ML
    ml_data = prepare_data_for_ml(anonymized_data)
    ml_data['risk_score'] = anonymized_data['risk_score']
    df = pd.DataFrame([ml_data])
    df.to_csv('user_data.csv', mode='a', header=False, index=False)

def display_summary(answers, risk_tolerance):
    st.subheader("Assessment Summary")
    for question in questions:
        if question['key'] in answers:
            st.write(f"**{question['question']}** Your answer: {answers[question['key']]}")
    
    st.subheader("Risk Tolerance Explanation")
    st.write(f"Based on your answers, your risk tolerance is: **{risk_tolerance}**")
    st.write("This assessment considers factors such as your age, financial situation, investment experience, and attitude towards market fluctuations. A higher risk tolerance suggests you might be more comfortable with investments that have potential for higher returns but also higher volatility.")

def main():
    st.set_page_config(layout="wide")
    
    # Custom CSS for larger and evenly spaced investment experience buttons
    st.markdown("""
    <style>
    .investment-button {
        width: 100%;
        height: 120px;
        white-space: normal;
        word-wrap: break-word;
        padding: 10px;
        font-size: 14px;
        line-height: 1.2;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .investment-button .emoji {
        font-size: 24px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Investment Risk Tolerance Assessment")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Display progress
    st.progress(st.session_state.step / len(questions))

    # Back and Start Over buttons
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("← Back") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()
    with col3:
        if st.button("↻ Start over"):
            st.session_state.step = 0
            st.session_state.user_answers = {}
            st.session_state.assessment_complete = False
            st.session_state.summary = None
            st.session_state.results = None
            st.rerun()

    # Display current question(s) or results
    if not st.session_state.assessment_complete:
        if st.session_state.step < len(questions):
            q = questions[st.session_state.step]
            
            if q['type'] == 'initial':
                st.header("Tell us a little bit about yourself")
                for sub_q in q['questions']:
                    if sub_q['type'] == 'number':
                        st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0, key=sub_q['key'])
                    elif sub_q['type'] == 'select':
                        st.session_state.user_answers[sub_q['key']] = st.selectbox(sub_q['question'], sub_q['options'], key=sub_q['key'])
                    elif sub_q['type'] == 'radio':
                        st.session_state.user_answers[sub_q['key']] = st.radio(sub_q['question'], sub_q['options'], key=sub_q['key'])
            elif q['type'] == 'employment_income':
                st.header("Tell us a little bit about yourself")
                for sub_q in q['questions']:
                    if sub_q['type'] == 'radio':
                        st.session_state.user_answers[sub_q['key']] = st.radio(sub_q['question'], sub_q['options'], horizontal=True, key=sub_q['key'])
                    elif sub_q['type'] == 'number':
                        st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0, key=sub_q['key'])
                    elif sub_q['type'] == 'buttons':
                        st.write(sub_q['question'])
                        cols = st.columns(3)
                        for i, option in enumerate(sub_q['options']):
                            if cols[i].button(option, key=f"{sub_q['key']}_{i}"):
                                st.session_state.user_answers[sub_q['key']] = option
            elif q['type'] == 'assets_liabilities':
                st.header("Tell us about your assets and liabilities")
                for sub_q in q['questions']:
                    st.session_state.user_answers[sub_q['key']] = st.number_input(sub_q['question'], min_value=0.0, value=0.0, step=1000.0, key=sub_q['key'])
            elif q['type'] == 'radio':
                st.session_state.user_answers[q['key']] = st.radio(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'chart':
                st.write(q['question'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.altair_chart(create_investment_chart("Low Volatility"))
                with col2:
                    st.altair_chart(create_investment_chart("Balanced"))
                with col3:
                    st.altair_chart(create_investment_chart("High Volatility"))
                st.session_state.user_answers[q['key']] = st.radio("Select your preferred volatility level:", q['options'], key=q['key'])
            elif q['type'] == 'text':
                st.session_state.user_answers[q['key']] = st.text_input(q['question'], key=q['key'])
            elif q['type'] == 'multiselect':
                st.session_state.user_answers[q['key']] = st.multiselect(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'select':
                st.session_state.user_answers[q['key']] = st.selectbox(q['question'], q['options'], key=q['key'])
            elif q['type'] == 'slider':
                st.session_state.user_answers[q['key']] = st.slider(q['question'], q['min_value'], q['max_value'], q['step'], key=q['key'])
            elif q['type'] == 'image_buttons':
                cols = st.columns(len(q['options']))
                for i, option in enumerate(q['options']):
                    if cols[i].button(f"{option['image']} {option['text']}", key=f"{q['key']}_{i}"):
                        st.session_state.user_answers[q['key']] = option['text']
                st.write(f"Selected: {st.session_state.user_answers.get(q['key'], 'None')}")
            
            # Next button with validation
            if st.button("Next"):
                if q['type'] == 'initial' or q['type'] == 'employment_income' or q['type'] == 'assets_liabilities':
                    # Check if all sub-questions are answered
                    all_answered = all(sub_q['key'] in st.session_state.user_answers for sub_q in q['questions'])
                else:
                    # For other question types, check if the main question is answered
                    all_answered = q['key'] in st.session_state.user_answers and st.session_state.user_answers[q['key']]
                
                if all_answered:
                    st.session_state.step += 1
                    st.rerun()
                else:
                    st.error("Please answer all questions to continue.")

        # Final submission
        elif st.session_state.step == len(questions):
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Submit", use_container_width=True):
                    risk_score = calculate_risk_score_ml(st.session_state.user_answers)
                    risk_tolerance = get_risk_tolerance(risk_score)
                    equity_allocation, income_allocation = get_allocation(risk_tolerance)

                    st.session_state.results = {
                        'risk_tolerance': risk_tolerance,
                        'equity_allocation': equity_allocation,
                        'income_allocation': income_allocation
                    }

                    user_data = {
                        **st.session_state.user_answers,
                        'risk_score': risk_score,
                        'risk_tolerance': risk_tolerance,
                        'equity_allocation': equity_allocation,
                        'income_allocation': income_allocation
                    }
                    save_user_profile(user_data)
                    st.session_state.summary = generate_summary(st.session_state.user_answers, risk_tolerance)
                    st.session_state.assessment_complete = True
                    st.rerun()

    # Display results
    if st.session_state.assessment_complete:
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
        if st.session_state.results:
            st.write(f"Your risk tolerance is: {st.session_state.results['risk_tolerance']}")
            st.write(f"Recommended allocation: {st.session_state.results['equity_allocation']}% Equities, {st.session_state.results['income_allocation']}% Income")
        st.success("Your profile has been saved!")

        # Add disclaimer on the final page
        st.markdown("""
        ---
        **Disclaimer**: This risk assessment tool is for educational purposes only and does not constitute financial advice. 
        Please consult with a qualified financial advisor before making any investment decisions.
        """)

def generate_summary(answers, risk_tolerance):
    summary = "## Assessment Summary\n\n"
    for question in questions:
        if question['key'] in answers:
            summary += f"**{question['question']}** Your answer: {answers[question['key']]}\n\n"
    
    summary += f"## Risk Tolerance Explanation\n\n"
    summary += f"Based on your answers, your risk tolerance is: **{risk_tolerance}**\n\n"
    summary += "This assessment considers factors such as your age, financial situation, investment experience, and attitude towards market fluctuations. A higher risk tolerance suggests you might be more comfortable with investments that have potential for higher returns but also higher volatility."
    
    return summary

# Add this at the end of your script
def close_db_connection():
    global conn
    conn.close()

if __name__ == "__main__":
    main()
    close_db_connection()