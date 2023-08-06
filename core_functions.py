from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate

# read the knowledge document and return raw text


def read_knowledge_document():
    fileObject = open("KnowledgeDocument(pan_card_services).txt", "r")
    data = fileObject.read()
    return data

# split the text into chunks


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# create vector store from chunks of text


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()  # using Openai ada-002 embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # Using gpt-3.5-turbo from openai as our LLM
    llm = ChatOpenAI(temperature=0.7)
    # Now defining the prompt for reducing hallucinations
    template1 = """
You are an AI assistant for answering questions about PAN card in English.
You are given the following extracted parts of a long document and a question.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about pan card, politely inform them that you are tuned to 
only answer questions about pan card.

CONTEXT:
{context}
=========
QUESTION: {question}

"""
    QA_PROMPT = (PromptTemplate(template=template1,
                 input_variables=["question", "context"]))

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversation_chain
