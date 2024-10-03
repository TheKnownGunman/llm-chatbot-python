import streamlit as st

# Create the LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)

# Create the Embedding model

# embeddings = OpenAIEmbeddings(
#     openai_api_key=st.secrets["OPENAI_API_KEY"]
# )

# import streamlit as st
# from sentence_transformers import SentenceTransformer

# # Replace the OpenAI Embeddings with sentence-transformers model
model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')

# # Function to create embeddings
def create_embedding(content):
    return model.encode(content)

# # Example usage
# content = "This is an example sentence."
# content_embedding = create_embedding(content)