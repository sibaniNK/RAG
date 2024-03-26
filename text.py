from langchain_community import embeddings

from langchain_core.runnables import RunnablePassthrough
from io import StringIO
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader



def text_file(file, question):
    stringio= StringIO(file.getvalue().decode("utf-8"))
    data = stringio.read()
    #loader = TextLoader(data)
    text_splitter = CharacterTextSplitter(chunk_size= 500, chunk_overlap=0)
    pages = text_splitter.split_text(data)

    #data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents(pages)
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="rag-chroma",
                                        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'))
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    template = """Answer the question based only on the following context:
                {context}
                Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
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
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["txt"])
    if uploaded_file is not None:
        # Ask user a question
        question = st.text_input("Ask your question")
    conversation_history = []
    if st.button("Get Answer"):
        # Process CSV file and get answer
        answer = text_file(uploaded_file, question)
        conversation_history.append({"user": question, "bot": answer})

        # Display conversation history
        for message in conversation_history:
            st.write(f"User: {message['user']}")
            st.write(f"Bot: {message['bot']}")

    # Run the Streamlit app


if __name__ == "__main__":
    main()
