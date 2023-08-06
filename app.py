import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.prompts.prompt import PromptTemplate
from core_functions import read_knowledge_document, get_conversation_chain, get_text_chunks, get_vectorstore

# this function processes the question and displays the chat history based on LLM response


def handle_userinput(user_question):
    # Get response from LLM
    response = st.session_state.conversation({'question': user_question})

    # Update the chat history according to the response
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # Odd responses belong to user and even to agent/chatbot
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)  # write the response in custom template for user
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)  # write the response in custom template for bot


def main():
    # Load the api keys for openai and huggingface
    load_dotenv()

    # Read knowledge document
    raw_text = read_knowledge_document()

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    st.set_page_config(page_title="Chat for information on PAN card",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:  # check if any conversation is active
        # create conversation chain if it is not activated
        st.session_state.conversation = get_conversation_chain(
            vectorstore)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with agent for information on PAN Card :books:")
    user_question = st.text_input(
        "Agent will answer your questions based on infromation available to it:")

    # When this button is pressed new chat is started/ history is deleted
    if st.button("Start new chat"):
        st.session_state.conversation = get_conversation_chain(
            vectorstore)

    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
