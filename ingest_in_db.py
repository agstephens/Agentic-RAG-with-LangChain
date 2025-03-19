# import basics
import os
from dotenv import load_dotenv

# import langchain
 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load pdf docs from folder 'documents'
loader = PyPDFDirectoryLoader("documents")

urls = [
    ("https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12", "https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html"),
    ("https://docs.unidata.ucar.edu/netcdf-c", "https://docs.unidata.ucar.edu/netcdf-c/current/"),
    ("https://github.com/orgs/cf-convention/discussions", "https://github.com/orgs/cf-convention/discussions/")
]

documents = []

print("Just reading from 1 PDF of Conventions docs for now.")
urls = []

for base_url, web_path in urls:
    loader = RecursiveUrlLoader(web_path, prevent_outside=True, base_url=base_url)
    loaded_docs = loader.load()
    print(f"Found {len(loaded_docs)} from: {web_path}")
    documents += loaded_docs[:]


# split the documents in multiple chunks
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)
print(f"Split to {len(docs)} chunks.")

# store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)
print("Loading to vector store")
