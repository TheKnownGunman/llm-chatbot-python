# index_creation.py
import streamlit as st
# Import existing variables from graph.py and llm.py
from graph import graph  # Import the graph object (Neo4j connection)
from llm import create_embedding # Import the embedding object
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# Function to create and store embeddings for the existing documents in Neo4j
# Function to create embeddings for the existing documents in Neo4j
def create_vector_index():
    try:
        # Step 1: Connect to Neo4j and retrieve documents
        with graph._driver.session() as session:
            result = session.run("MATCH (doc:Document) WHERE doc.content IS NOT NULL RETURN doc.content AS content, id(doc) AS doc_id")

            for record in result:
                content = record["content"]
                doc_id = record["doc_id"]

                # Skip if content is None or empty
                if not content:
                    print(f"Skipping document with ID {doc_id} due to missing content")
                    continue

                # Step 2: Generate embeddings for the document content
                content_embedding = create_embedding(content)

                # Step 3: Store the embeddings back into Neo4j
                session.run("""
                    MATCH (doc:Document)
                    WHERE id(doc) = $doc_id
                    SET doc.contentEmbedding = $content_embedding
                """, doc_id=doc_id, content_embedding=content_embedding)
    finally:
        # Ensure the driver is closed after completing the process
        graph._driver.close()


# Function to query the vector index for relevant answers based on user queries
def query_vector_index(user_query):
    medical_vector_index = Neo4jVector.from_existing_index(
        create_embedding,
        graph=graph,
        index_name="vector",
        node_label="Document",  # The node label is now "Document"
        text_node_property="content",  # The text property that holds the document content
        embedding_node_property="contentEmbedding"  # The property storing the vector embeddings
    )

    # Perform similarity search with the user's query
    results = medical_vector_index.similarity_search(user_query)

    # Display results
    for doc in results:
        print(f"Document ID: {doc.metadata['doc_id']}")
        print(f"Relevant Text: {doc.page_content}")

import openai

# Define OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to get embeddings for a given text
# Assuming the embeddings instance is already created in llm.py
from llm import embeddings

# Use the embeddings instance to create an embedding for the content


# def create_embedding2(content):
#     return embeddings.embed_query([content])[0]  # Generate embeddings using OpenAI Embeddings


# Example of using the function to create the index and search
if __name__ == "__main__":
    create_vector_index()  # Run this once to store embeddings in Neo4j
    user_query = "How often do I need to backup data? Please detail answer Regulatory Requirements"
    query_vector_index(user_query)  # Perform search based on user's query