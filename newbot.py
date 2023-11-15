import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from typing_extensions import Protocol
import os 

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.header("Chat with Neigborgood help chatbot")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Share me anything to feel free"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="House recommendation bot is loading"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5),
                system_prompt="I am a house recommender who is here to help you find the perfect place to call home. I will ask you questions about your needs and preferences, and then provide you with a list of neighborhoods and homes by comparing the global trust rates and list 3 homes that a person should buy according to thier needs. I will also provide you with information about the local community, schools, and amenities."
                        )
     
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

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



