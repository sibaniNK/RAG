

from langchain_community import embeddings


from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama


def pdf_file(file ,question):
    from langchain_community.document_loaders import PyPDFLoader
    
    bytes_data = file.read()
    f = open(f"{file.name}.pdf", "wb")
    f.write(bytes_data)
    f.close()
    loader= PyPDFLoader(f"{file.name}.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents( documents = chunks,collection_name = "rag-chroma",embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'))
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

    #File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["pdf"])
    if uploaded_file is not None:

        # Ask user a question
        question = st.text_input("Ask your question")
    conversation_history =[]
    if st.button("Get Answer"):
        # Process CSV file and get answer
        answer = pdf_file(uploaded_file,question)
        conversation_history.append({"user": question, "bot": answer})

        # Display conversation history
        for message in conversation_history:
            st.write(f"User: {message['user']}")
            st.write(f"Bot: {message['bot']}")

    # Run the Streamlit app
if __name__ == "__main__":

    main()



    


