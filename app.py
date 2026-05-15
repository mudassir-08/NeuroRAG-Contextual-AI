import streamlit as st

from utils.chatbot_chain import rag_chat


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("📘 RAG Chatbot using Groq + LangChain")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User Input
user_input = st.chat_input("Ask a question about your PDFs...")


if user_input:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.spinner("Thinking..."):

        result = rag_chat(user_input)

        answer = result["answer"]

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Show assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("📚 Source Documents"):

            for i, doc in enumerate(result["source_documents"]):

                st.markdown(f"### Source {i+1}")

                st.write(doc.page_content[:1000])