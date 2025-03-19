# Agentic-RAG CF Chatbot

## Running in Docker Desktop

1. Start Docker Desktop

2. Start the `agstephens/ai-play:v2` container.

3. Login and set up:

```
# As root:
su --shell /bin/bash - ag

# As ag:
cd /home/ag/Agentic-RAG-with-LangChain
source ./setup-env.sh
```

4. Run:

```
streamlit run agentic_rag_streamlit.py --server.port 8000
```

5. View in browser (on laptop):

http://localhost:8182/
