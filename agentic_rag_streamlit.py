# import basics
import os
from dotenv import load_dotenv

from pathlib import Path
import requests
import pandas as pd
from xml.etree import ElementTree as ET

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# import supabase db
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiating embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
# initiating llm
llm = ChatOpenAI(model="gpt-4o",temperature=0)

# pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Set up vocab search capability
VOCAB_XML_URL = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
VOCAB_CACHE = "cf-standard-name-table.xml"
VOCAB_DF = None

def load_vocab_xml(url=VOCAB_XML_URL, cache=VOCAB_CACHE):
    """Load XML document ready for searching."""
    if not Path(cache).exists():
        content = requests.get(url).text
        Path(cache).write_text(content)

    return ET.parse(cache)
    
    
def get_vocab_df():
    global VOCAB_DF
    if VOCAB_DF is None:
        # create a DataFrame of all "entry" elements in xml
        df = pd.DataFrame([], columns=["name", "canonical_units", "description"])
        
        xml = load_vocab_xml()

        for entry in xml.findall(".//entry"):
            name = entry.attrib["id"]
            canonical_units = entry[0].text
            description = entry[1].text
            df.loc[len(df)] = [name, canonical_units, description]
            
        VOCAB_DF = df
        
    return VOCAB_DF


# Create the vocab search tool
@tool
def search_standard_names(query: str, search_field: str="name"):
    """Search the CF Standard Names vocabulary based on a query term and search field"""
    df = get_vocab_df()
    _df = df.dropna(subset=[search_field])
    return _df[_df[search_field].str.contains(query, case=False)]

# combining all tools
tools = [retrieve, search_standard_names]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

st.subheader("What is this chatbot?")
st.caption("""
- Uses Retrieval Augmented Generation (RAG) to focus its responses on a knowledge base
- The knowledge base consists of:
  - The latest CF Conventions document
- An agentic approach that uses tools (functions) for deteministic queries:
  - Searching CF-standard names
""")

st.subheader("TODOs")
st.caption("""
- Add in more agents:
  - CF-Checker
- Add in login and persistent chat history
- Add in a much larger corpus of documents:
  - Other documentation on the CF site: https://cfconventions.org
  - The Unidata NetCDF documentation: https://docs.unidata.ucar.edu/netcdf-c/current/
  - The CF GitHub Discussions: https://github.com/orgs/cf-convention/discussions
""")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("Ask me about CF...")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

