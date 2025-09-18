# app.py
import streamlit as st
from ollama import Client
from retrieval_function import RAGRetriever

# ---------------------------
# Load retriever + Ollama client
# ---------------------------
@st.cache_resource
def load_resources():
    retriever = RAGRetriever()
    client = Client()
    return retriever, client

retriever, client = load_resources()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fitness & Nutrition Chatbot", page_icon="💪", layout="centered")
st.title("💬 Fitness & Nutrition Chatbot")
st.write("Ask me anything about fitness, workouts, recovery, or nutrition!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# Handle user query
# ---------------------------
if prompt := st.chat_input("Type your question here..."):
    # Save + display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get context from retriever
    context = retriever.get_context(prompt, top_k=3)

    # Build prompt
    full_prompt = f"""
    Answer the question based on the following context only.
    If the context doesn't contain relevant information, just say you don't know.

    Context:
    {context}

    Question: {prompt}
    """

    # Generate response with Llama (via Ollama)
    response = client.generate(model="llama3", prompt=full_prompt)
    answer = response["response"]

    # Save + display assistant answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
