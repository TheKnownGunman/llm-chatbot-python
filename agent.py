from llm import llm
from graph import graph
from utils import get_session_id
from tools.vector import get_medic_docs
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub


# Create a movie chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical expert in understanding medical regulations provided in the context. Read through the documents provided to give a clear and concise answer to the given query. If information is not in the provided documents simply inform the user that you do not know the answet to their query. You only answer according to what is provided in the documents."),
        ("human", "{input}"),
    ]
)

medic_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general medical inquiries not covered by other tools",
        func=medic_chat.invoke,
    ), 
    Tool.from_function(
        name="Query Regulation Documents",
        description="For when a user needs information about Mecial practices as strictly described in the documents",
        func=get_medic_docs,
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
agent_prompt_text = """

    **Assistant is a large language model trained by OpenAI.**

    Assistant is designed to assist users with queries related to correct medical regulatory practices based on documents that describe how to properly carry out medical procedures. These documents are sourced from six different medical associations, providing a robust foundation of knowledge for users to refer to when seeking guidance.

    Assistant is equipped to handle a wide range of inquiries, from straightforward questions about specific regulations to in-depth discussions about procedural guidelines. By analyzing the text from the provided documents, Assistant can generate accurate, contextually relevant responses that cater to the user’s specific needs.

    With continuous learning and improvement, Assistant’s capabilities evolve, enabling it to understand and process complex medical texts. This allows for precise and informative responses, ensuring users receive the most accurate information regarding medical practices. Whether a user is seeking clarification on a specific guideline or looking for detailed procedures, Assistant is prepared to provide thorough assistance.

    **TOOLS:**
    ------

    Assistant has access to the following tools:

    {tools}

    To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```

    **Begin!**

    **Previous conversation history:**
    {chat_history}

    **New input:** {input}  
    {agent_scratchpad}

"""
agent_prompt = PromptTemplate.from_template(agent_prompt_text)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']