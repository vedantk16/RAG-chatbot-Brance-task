# RAG Chatbot App

The task was to build a RAG(Retrieval Augmented Generation) chatbot. For the user question, RAG module would retrieve context from knowledge document and generation phase LLM would personalize answer using retrieval knowledge. An issue experienced by RAG chatbots is hallucinations and provided task also aimed to reduce this. 

## Introduction
------------
We build an application which provides information about the knowledge document, which is on Pan Card. We build the application using streamlit and langchain.

## How It Works
------------
The high-level logic of application is as follows:
1. Knowledge Document loading: The app reads the knowledge document (information on PAN Card) and extracts its text content.
2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.
3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the knowledge document.

To ensure that the answers provided by application are helpful, multiple steps and models need to be optimized. LLMs also have tendency to Hallucinate, i.e., include irrelevant, nonsensical, or factually incorrect facts in their answers. To prevent hallucination, it is important to ensure that the generated answer is consistent with the knowledge document. This can be done by using a variety of techniques including the following:
1.	Prompt Engineering Techniques: This includes producing the prompt in ways to Request for Evidence, Set Boundaries, step-by-step reasoning.
2.	Improving your information retrieval (IR) system: If the retriever grabs irrelevant documents or if the documents are not split accordingly, the completion will “hallucinate" most of time.
3.	Prevent incorrect information in context: This includes detecting when the information retrieval returns zero documents and taking steps to ensure this is conversed to the user. 
4.	Teach the model: We can train the model to avoid hallucinations and adjust the temperature to make the model more conservative.

We explore and implement majority of discussed methods to reduce hallucinations. We also evaluate our models by changing the prompt and temperature parameter using Rouge metric to find out the best combination for our task.

## Dependencies and Installation
----------------------------
To install the  RAG Chatbot App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```
OPENAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the RAG Chatbot App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Ask questions in natural language about the knowledge base i.e. PAN Card using the chat interface.
