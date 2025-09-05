# app.py

from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Defines the schema for the graph's shared memory.
class LogAnalysisState(TypedDict):
    """Represents the state of our graph, passing data between nodes."""
    log_file: str  # The path to the log file.
    document_chunks: List[Document]  # Log lines converted to documents.
    retrieved_snippets: List[Document]  # Snippets found by the RAG retriever.
    error_codes: List[int]  # The final list of unique error codes.
    vectorstore: FAISS  # The FAISS vector store for retrieval.

# ==============================================================================
# Step 2: Define the Nodes (the building blocks of the workflow)
# Each function below represents a "node" in the LangGraph.
# ==============================================================================
# Place this section immediately after Step 1

def ingest_and_embed(state: LogAnalysisState) -> LogAnalysisState:
    """Reads the log file, chunks it, and creates a vector store."""
    print("---INGESTING & EMBEDDING LOGS---")
    log_file_path = state["log_file"]
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()

    docs = [Document(page_content=line) for line in log_content.splitlines() if line.strip()]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    state["vectorstore"] = vectorstore
    state["document_chunks"] = docs
    
    return state

# ---

def retrieve_errors(state: LogAnalysisState) -> LogAnalysisState:
    """Retrieves log snippets likely containing error codes using a retriever."""
    print("---RETRIEVING ERROR LOGS---")
    vectorstore = state["vectorstore"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    query = "find all lines with error codes, error messages, or failure alerts"
    retrieved_docs = retriever.invoke(query)
    
    state["retrieved_snippets"] = retrieved_docs
    
    return state

# ---

class ErrorCodes(BaseModel):
    """A Pydantic model to ensure the LLM returns a list of integers."""
    unique_error_codes: List[int] = Field(..., description="A list of unique numeric error codes.")

def extract_codes(state: LogAnalysisState) -> LogAnalysisState:
    """Uses an LLM with structured output to extract numeric error codes."""
    print("---EXTRACTING ERROR CODES WITH LLM---")
    snippets = state["retrieved_snippets"]
    
    if not snippets:
        print("No snippets to extract from. Returning empty list.")
        state["error_codes"] = []
        return state

    context = "\n".join([doc.page_content for doc in snippets])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ErrorCodes)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting numeric error codes from log files. Extract all unique numeric error codes from the provided text. The codes can be in formats like ERR-1234, 0x1F4, 450, etc. Convert all of them to a single list of integers."),
        ("human", "Log Snippets:\n\n{context}")
    ])
    
    chain = prompt_template | structured_llm
    result = chain.invoke({"context": context})

    state["error_codes"] = sorted(list(set(result.unique_error_codes)))
    return state


# ==============================================================================
# Step 3: Build and Compile the LangGraph
# This section assembles the nodes and defines the workflow's structure.
# ==============================================================================
# Place this section immediately after the node definitions

def has_snippets(state: LogAnalysisState) -> str:
    """Checks if the retrieval step found any snippets."""
    if state["retrieved_snippets"]:
        return "continue"
    else:
        return "end"

builder = StateGraph(LogAnalysisState) #Create an instance of StateGraph

builder.add_node("ingest", ingest_and_embed)
builder.add_node("retrieve", retrieve_errors)
builder.add_node("extract", extract_codes)

builder.set_entry_point("ingest")
builder.add_edge("ingest", "retrieve")

builder.add_conditional_edges(
    "retrieve",
    has_snippets,
    {
        "continue": "extract",
        "end": END
    }
)
builder.add_edge("extract", END)

graph = builder.compile()

# ==============================================================================
# Step 4: Run the Application
# This is the main execution block of the script.
# ==============================================================================
# Place this section at the very end of the file

def create_sample_log(filename="sample_log.txt"):
    """Creates a dummy log file with various error formats."""
    with open(filename, "w") as f:
        f.write("2025-09-05 10:00:01 INFO: Starting application.\n")
        f.write("2025-09-05 10:00:05 ERROR: Database connection failed. Code: 500.\n")
        f.write("2025-09-05 10:01:10 WARNING: Disk usage is at 90%.\n")
        f.write("2025-09-05 10:02:20 CRITICAL: Fatal error, process terminated. (ERR-404)\n")
        f.write("2025-09-05 10:03:05 INFO: User 'Alice' logged in.\n")
        f.write("2025-09-05 10:04:15 ERROR: Unable to parse configuration file. Failure code: 0x1A.\n")
        f.write("2025-09-05 10:05:30 ERROR: Timeout occurred. HTTP status 408.\n")
    print(f"Sample log file '{filename}' created.")

if __name__ == "__main__":
    log_file_name = "sample_log.txt"
    create_sample_log(log_file_name)
    
    initial_state = {
        "log_file": log_file_name,
        "document_chunks": [],
        "retrieved_snippets": [],
        "error_codes": []
    }

    print("\n---RUNNING THE LANGGRAPH APPLICATION---")
    final_state = graph.invoke(initial_state)

    print("\n---FINAL OUTPUT---")
    output_json = {"unique_error_codes": final_state["error_codes"]}
    print(json.dumps(output_json, indent=4))
    
    os.remove(log_file_name)