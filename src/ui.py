#ui.py

#Libraries

import streamlit as st
from retrieve import load_chunks, hybrid_retrieve
from prompt import answer_question
import threading
import contextlib
from judge_llm import evaluate_single_interaction

# Page Configuration 
st.set_page_config(
    page_title="CancerCareAssist",
    layout="wide",
)

#  Header 
st.markdown(
    """
    <h1 style="text-align:center; color:#1C2A73; font-weight:700;">CancerCareAssist 🩺</h1>
    <p style="text-align:center; color:#555; font-size:16px;">
    A hybrid-retrieval, GPT-powered assistant that provides trusted cancer care information.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr style='border: 1px solid #D9E1F2;'>", unsafe_allow_html=True)

# Load Data 
@st.cache_resource
def get_chunks():
    return load_chunks()

chunks = get_chunks()

# Disclaimer sidebar
st.sidebar.title(" About")
st.sidebar.markdown(
    """
    **CancerCareAssist** answers general questions about navigating cancer care 
    using verified materials from:
    - **The American Cancer Society (ACS)**
    - **The National Cancer Institute (NCI)**  
    """
)
st.sidebar.divider()
st.sidebar.info(
    "This chatbot is for informational purposes only. Please consult a medical professional for personal health questions."
)
st.sidebar.markdown("---")
st.sidebar.caption("Built using BM25, Pinecone, GPT-3.5-Turbo, and Streamlit.")

#  Main Content 
st.markdown("### Ready to answer questions about navigating the cancer care journey.")
query = st.text_input("Type your question below:")

if query:
    with st.spinner("Retrieving relevant information and generating your answer..."):
        retrieved = hybrid_retrieve(query, chunks, top_k=10, alpha=0.6)
        answer = answer_question(query, retrieved)

    #  Answer Card
    st.markdown(
        f"""
        <div style="
            background-color:#E8ECFF;
            border-left:6px solid #1C2A73;
            padding:18px;
            border-radius:6px;
            margin-top:15px;
            color:#1C2A73;
            font-size:17px;
            line-height:1.6;
        ">
        {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieved Context
    st.divider()
    st.markdown("## Retrieved Context & Citations")
    for i, c in enumerate(retrieved, start=1):
        with st.expander(f"[{i}] {c['doc']} (p.{c['page']})"):
            st.write(c["text"])


#Background Evaluation
def run_judge():
        try:
            evaluate_single_interaction(query, retrieved, answer)
        except Exception as e:
            print(f"⚠️ Judge evaluation failed: {e}")

threading.Thread(target=run_judge, daemon=True).start()