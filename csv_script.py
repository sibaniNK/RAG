from langchain_community.document_loaders import WebBaseLoader
#from bs4 import BeautifulSoup

from langchain_community.vectorstores import Chroma

from langchain_community import embeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub


def csv_file(file,question):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        path = tmp_file.name
    loader = CSVLoader(path)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                              chunk_overlap=100)
    chunks = splitter.split_documents(data)



    #vectorstore = Chroma.from_documents(chunks, embeddings)
    vectorstore = Chroma.from_documents( documents = chunks  ,collection_name = "rag-chroma",embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'))

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            """
    prompt = ChatPromptTemplate.from_template(template)
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # llm = HuggingFaceHub(
    #     repo_id=repo_id,
    #     model_kwargs={"temperature": 0.8, "top_k": 50},
    #     huggingfacehub_api_token=('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')
    # )
    llm = Ollama(model="mistral")
    output_parser = StrOutputParser()
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )


    return chain.invoke(question)


def main():
    st.title("Your Chatbot Title")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Ask user a question
        question = st.text_input("Ask your question")

        conversation_history = []

        # Process CSV file and get answer
        if st.button("Get Answer"):
            # Process CSV file and get answer
            answer = csv_file(uploaded_file, question)

            # Add user question and bot answer to conversation history
            conversation_history.append({"user": question, "bot": answer})

            # Display conversation history
            for message in conversation_history:
                st.write(f"User: {message['user']}")
                st.write(f"Bot: {message['bot']}")


# Run the Streamlit app
if __name__ == "__main__":
    main()
# st.title("Document Query with csv")
# uploaded_file = st.sidebar.file_uploader("Upload File", type="csv")
# if uploaded_file:
#     csv_file(file,question)
#
#
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
#     st.write(bytes_data)


