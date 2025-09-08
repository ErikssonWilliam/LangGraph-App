# LangGraph-App
Small application using LangGraph for interview with Ericsson.

Here is a visual representation of the LangGraph workflow:

![Log Analysis Graph](graph_visualization.png)

This graph shows the flow of data through the various nodes:
1.  **Ingest**: Reads the log file and creates document chunks.
2.  **Retrieve**: Searches for relevant log snippets.
3.  **Extract**: Uses an LLM to extract error codes.
