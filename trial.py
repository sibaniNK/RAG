import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from csv_script import csv_file
from pdf import pdf_file
from text import text_file
from streamlit_chat import message

import streamlit as st
import os
 #chat history
def determine_file_type(file_name):
    _, file_extension = os.path.splitext(file_name)
    if file_extension.lower() == '.pdf':
        return 'PDF'
    elif file_extension.lower() == '.csv':
        return 'CSV'
    elif file_extension.lower() in ['.txt', '.text']:
        return 'Text'
    else:
        return None
if "chat_history" not in st.session_state: # st.session_state is a object that keep all the variables
    st.session_state.chat_history = []



uploaded_file = st.file_uploader("Upload a file")
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message('Human'):
            st.markdown(message.content)
    else:
        with st.chat_message('AI'):
            st.markdown(message.content)
user_query= st.chat_input("your message ")
if user_query is not None and user_query != "" and uploaded_file is not None:# not empty
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"): # human talking
        st.markdown(user_query)# markdown the user query
    with st.chat_message("AI"):
        file_type = determine_file_type(uploaded_file.name)
        if file_type is not None:
            # Execute the corresponding function based on file type
            if file_type == 'Text':
                ai_response = text_file(uploaded_file, user_query)

                st.markdown(ai_response)
            elif file_type == 'CSV':
                ai_response = csv_file(uploaded_file, user_query)

                st.markdown(ai_response)
            elif file_type == 'PDF':
                ai_response = pdf_file(uploaded_file, user_query)
                st.markdown(ai_response)
            else:
                st.warning("Unsupported file type. Please upload a PDF, CSV, or text file.")


    #aapend ai message
    st.session_state.chat_history.append(AIMessage(ai_response))
