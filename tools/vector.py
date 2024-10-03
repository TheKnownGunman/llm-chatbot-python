import streamlit as st
from llm import llm, model
from graph import graph

# Create the Neo4jVector
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

neo4jvector = Neo4jVector.from_existing_index(
    model,                              # (1)
    graph=graph,                             # (2)
    index_name="vector",                 # (3)
    node_label="Chunk",                      # (4)
    text_node_property="text",               # (5)
    embedding_node_property="embedding", # (6)
    retrieval_query="""
RETURN
    node.text AS text,
    score,
    {
    }
    AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

# Create a function to call the chain
def get_medic_docs(input):
    return plot_retriever.invoke({"input": input})