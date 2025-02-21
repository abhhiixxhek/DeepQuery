import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key securely from Streamlit Secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Set environment variable
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Create memory to store conversation context
memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")

# Set page config
st.set_page_config(page_title="DeepQuery", page_icon="🚀", layout="wide")

# Custom CSS for modern UI
def load_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #0e1117;
                color: #ffffff;
                font-family: 'Inter', sans-serif;
            }
            .stChatMessage.user {
                background-color: #1f2937;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .stChatMessage.assistant {
                background-color: #4b5563;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .stTextInput>div>div>input {
                border: none;
                background-color: #1f2937;
                color: white;
                padding: 10px;
                border-radius: 8px;
            }
            .stButton>button {
                background-color: #2563eb;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #1e40af;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

load_css()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'llm' not in st.session_state:
    st.session_state.llm = llm
if 'memory' not in st.session_state:
    st.session_state.memory = memory

# Define Socratic prompting
socratic_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""
    You are a teaching assistant helping a student understand data science, machine learning, and artificial intelligence through the Socratic method. 
    Begin by providing a clear, detailed, and accurate explanation of the given concept, ensuring depth and relevance. Then, engage the student with 
    a series of thought-provoking questions (around five) that encourage critical thinking and practical application of their understanding.

    The student just said: "{input}"
    Previous conversation context:
    {chat_history}

    Now, generate a well-structured response with a detailed explanation followed by five insightful questions that drive deeper understanding.
    """
)

if 'chain' not in st.session_state:
    st.session_state.chain = LLMChain(
        llm=st.session_state.llm,
        prompt=socratic_prompt,
        memory=st.session_state.memory,
        verbose=True
    )

# Function to handle Socratic questioning
def socratic_assistant(student_input: str):
    response = st.session_state.chain.predict(input=student_input)
    return response

# Streamlit App UI
st.title("🚀 DeepQuery: AI-Powered Learning")
st.markdown("""
    ### Think, Question & Master AI & Data Science 🤖🔥
    Dive into the world of AI, ML, and Data Science with an interactive Socratic assistant. 
    Instead of just answers, get thought-provoking questions to enhance your understanding! 💡
""")

# Display chat messages with a modern UI
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

# Chat input handling
prompt = st.chat_input("Ask a question about AI, ML, or Data Science...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = socratic_assistant(prompt)
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
