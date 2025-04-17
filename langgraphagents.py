import streamlit as st
import pandas as pd
import os
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
# Set OpenAI API key in environment (if not done automatically by Streamlit Secrets)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ✅ No need to pass api_key explicitly here
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# --- Scoring Logic ---
def calculate_score(row):
    score = (
        (5 - int(row['Risk_Class'])) +
        int(row['Medical_Incidence']) +
        (4 - int(row['Tech_Limitations'])) +
        (2 if row['Predicate_US'].strip().lower() == 'yes' else 0)
    )
    return score

# --- Agent Nodes ---
def risk_evaluator(state):
    row = state["row"]
    prompt = f"""
    You are a senior Regulatory Affairs expert specializing in Software as a Medical Device (SaMD).
    
    Analyze the **regulatory** and **technical** risks for launching a SaMD in the following country.
    
    Provide a structured assessment across these dimensions:
    1. Risk Class (1-5)
    2. Medical Incidence (0-5)
    3. Technical Limitations (1-4)
    4. Predicate Device in US (Yes/No)
    5. Population
    
    Country: {row['Country']}
    Risk Class: {row['Risk_Class']}
    Medical Incidence: {row['Medical_Incidence']}
    Technical Limitations: {row['Tech_Limitations']}
    Predicate Device in US: {row['Predicate_US']}
    Population: {row['Population']}
    Score: {row['Score']}
    
    Respond with:
    - Risk Summary (Structured)
    - Key Risk Factors
    - Final Risk Evaluation (Low / Medium / High)
    """
    response = llm.invoke([
        SystemMessage(content="You are a Regulatory Risk Evaluator with expertise in global SaMD compliance."),
        HumanMessage(content=prompt)
    ])
    state["risk_summary"] = response.content
    return state

def market_analyst(state):
    row = state["row"]
    prompt = f"""
    You are a Market Viability Analyst for healthcare technology in global markets.
    
    Using the provided score and risk summary, assess **market readiness** for SaMD launch.
    
    Country: {row['Country']}
    Population: {row['Population']}
    Score: {row['Score']}
    
    Risk Assessment:
    {state['risk_summary']}
    
    Analyze:
    - Population readiness
    - Technical infrastructure
    - Market maturity
    - Impact of risk on go-to-market strategy
    
    Respond with:
    - Market Viability Summary
    - Strengths & Weaknesses
    - Final Recommendation (Viable / Cautious / Unfavorable)
    """
    response = llm.invoke([
        SystemMessage(content="You are a strategic analyst assessing healthcare digital product markets."),
        HumanMessage(content=prompt)
    ])
    state["market_summary"] = response.content
    return state


def decision_maker(state):
    row = state["row"]
    prompt = f"""
    You are a senior strategic decision-maker in a med-tech company.
    
    Make a data-backed launch decision for the following country based on all assessments.
    
    Country: {row['Country']}
    Population: {row['Population']}
    Score: {row['Score']}
    
    Risk Assessment:
    {state['risk_summary']}
    
    Market Analysis:
    {state['market_summary']}
    
    Weigh the risk and market factors equally, and justify your decision based on all dimensions.
    
    Respond strictly in this format:
    Decision: [Launch / Hold / Reject]
    Explanation:
    - Justification from Risk
    - Justification from Market
    - Final Consideration
    """
    response = llm.invoke([
        SystemMessage(content="You are responsible for high-stakes go/no-go launch decisions."),
        HumanMessage(content=prompt)
    ])
    state["final_decision"] = response.content
    return state


# --- LangGraph Definition ---
graph_builder = StateGraph(dict)
graph_builder.add_node("risk_eval", risk_evaluator)
graph_builder.add_node("market_analysis", market_analyst)
graph_builder.add_node("decision", decision_maker)

graph_builder.set_entry_point("risk_eval")
graph_builder.add_edge("risk_eval", "market_analysis")
graph_builder.add_edge("market_analysis", "decision")
graph_builder.add_edge("decision", END)

agent_graph = graph_builder.compile()

# --- Streamlit UI ---
st.title("🚀 SaMD Agentic AI Launch Assessor (with LangGraph)")

uploaded_file = st.file_uploader("Upload Survey Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Score'] = df.apply(calculate_score, axis=1)
    st.write("### Uploaded Survey Data", df)

    result_rows = []

    for idx, row in df.iterrows():
        st.markdown(f"---\n### Assessing: {row['Country']}")
        state = agent_graph.invoke({"row": row.to_dict()})

        st.markdown(f"**🧪 Risk Assessment**\n{state['risk_summary']}")
        st.markdown(f"**📊 Market Analysis**\n{state['market_summary']}")
        st.markdown(f"**🚦 Final Decision**\n{state['final_decision']}")

        result_rows.append({
            "Country": row['Country'],
            "Risk Assessment": state["risk_summary"],
            "Market Analysis": state["market_summary"],
            "Final Decision": state["final_decision"]
        })

    result_df = pd.DataFrame(result_rows)
    st.download_button("📥 Download Assessment Report", result_df.to_csv(index=False), "launch_report.csv")
