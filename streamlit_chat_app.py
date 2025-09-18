# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your fine-tuned Flan-T5 model
model_name = "fine-tuned-flan-t5-martina"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.set_page_config(page_title="Fitness & Nutrition Chatbot", page_icon="💪", layout="centered")

st.title("💬 Fitness & Nutrition Chatbot")
st.write("Ask me anything about fitness, workouts, recovery, or nutrition!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        early_stopping=True,
        max_new_tokens=150,
        no_repeat_ngram_size=3
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)
