import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
import os 

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.header("Chat with Neigborgood help chatbot")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Share me anything to feel free"}
    ]

def upload_pdf_to_directory(uploaded_file):
    data_dir = "data"  # Specify the data directory
    os.makedirs(data_dir, exist_ok=True)  # Create the data directory if it doesn't exist

    # Save the uploaded file to the data directory
    file_path = os.path.join(data_dir, "uploaded_pdf.pdf")
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())

    return file_path

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="House recommendation bot is loading"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5),
                system_prompt="You are a chatbot that provides information from the data supplied to you, assume all the questions are related to that only"
                        )
     
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_path = upload_pdf_to_directory(uploaded_file)
    st.success("PDF uploaded successfully.")

    # Create index from the uploaded PDF file
    index = load_data()

    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    prompt=st.chat_input("Your question")

    if prompt: # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
