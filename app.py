from csv_script import csv_file
from pdf import pdf_file
from text import text_file
from streamlit_chat import message

import streamlit as st
import os
import gc

# def conversational_chat(query, chain):
#     result = chain.invoke({"question": query, "chat_history": st.session_state['history']})
#     return result

def conversational_chat(query,chain):
    result = chain({"question": query,
                    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))

    return result["answer"]
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




def main_1():
    st.title('File Type Selector')

    # File upload
    uploaded_file = st.file_uploader("Upload a file")

    # Question input
    question = st.text_input("Enter your question")

    if uploaded_file is not None:
        file_type = determine_file_type(uploaded_file.name)
        if file_type is not None:
            # Execute the corresponding function based on file type
            if file_type == 'PDF':
                result = pdf_file(uploaded_file, question)
            elif file_type == 'CSV':
                result = csv_file(uploaded_file, question)
            elif file_type == 'Text':
                result = text_file(uploaded_file, question)

            st.write("Answer:", result)
        else:
            st.warning("Unsupported file type. Please upload a PDF, CSV, or text file.")



if __name__ == "__main__":
    main_1()