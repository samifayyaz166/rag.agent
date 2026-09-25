from pathlib import Path
import sys
import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CHAT HISTORY
# ============================================================

from chat_history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# SRC FOLDER
# ============================================================

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))


# ============================================================
# LOADERS + VECTORSTORE
# ============================================================

from loaders import load_file, load_webpage
from vectorstore import create_vectorstore


# ============================================================
# LANGCHAIN
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)


# ============================================================
# AGENT + TOOLS
# ============================================================

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import create_retriever_tool

# Live Internet Search (NEW, non-deprecated Tavily package)
from langchain_tavily import TavilySearch


# ============================================================
# 1. DATA FOLDER
# ============================================================

data_folder = Path(__file__).parent / "data"


supported_extensions = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls",
}


# Find all supported files
files = [
    file
    for file in data_folder.rglob("*")
    if file.is_file()
    and file.suffix.lower() in supported_extensions
]


print("\n========================================")
print("FILES FOUND")
print("========================================")


if files:

    for file in files:
        print(f"  - {file.name}")

else:

    print("  No local files found.")


# ============================================================
# 2. WEBPAGES
# ============================================================

web_urls = [
    "https://example.com",
]


# ============================================================
# 3. LOAD ALL DOCUMENTS
# ============================================================

all_documents = []


for web_url in web_urls:

    try:

        print(f"\nLoading webpage: {web_url}")

        web_documents = load_webpage(web_url)

        all_documents.extend(web_documents)

        print(
            f"Loaded webpage: {web_url} "
            f"({len(web_documents)} documents)"
        )

    except Exception as e:

        print(
            f"ERROR loading webpage {web_url}: {e}"
        )


for file in files:

    try:

        print(
            f"\nLoading file: {file.name}"
        )

        documents = load_file(file)

        all_documents.extend(documents)

        print(
            f"Loaded: {file.name} "
            f"({len(documents)} documents)"
        )

    except Exception as e:

        print(
            f"ERROR loading {file.name}: {e}"
        )


# ============================================================
# 4. CHECK DOCUMENTS
# ============================================================

if not all_documents:

    raise ValueError(
        "No documents could be loaded."
    )


print(
    f"\nTotal documents loaded: "
    f"{len(all_documents)}"
)


# ============================================================
# 5. SPLIT DOCUMENTS
# ============================================================

print("\nSplitting documents...")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


split_documents = text_splitter.split_documents(
    all_documents
)


print(
    f"Total chunks created: "
    f"{len(split_documents)}"
)


# ============================================================
# 6. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Embeddings created successfully.")


# ============================================================
# 7. CREATE CHROMA VECTOR STORE
# ============================================================

print("\nCreating vector store...")


vectorstore = create_vectorstore(
    documents=split_documents,
    embeddings=embeddings
)


print(
    "Vector store created successfully."
)


# ============================================================
# 8. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# ============================================================
# 9. GROQ LLM
# ============================================================

print("\nInitializing Groq LLM...")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


print("Groq LLM ready.")


# ============================================================
# 10. RAG DOCUMENT SEARCH TOOL
# ============================================================

retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description=(
        "Search the user's local knowledge base. "
        "The knowledge base contains PDF, DOCX, TXT, CSV, "
        "XLSX, XLS files and loaded webpages. "
        "Use this tool when the question may be answered "
        "from the user's documents or loaded webpages."
    ),
)


# ============================================================
# 11. LIVE INTERNET SEARCH TOOL
# ============================================================

print("\nInitializing live internet search...")


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not TAVILY_API_KEY:

    raise ValueError(
        "\nTAVILY_API_KEY not found.\n"
        "Add your Tavily API key to the .env file.\n\n"
        "Example:\n"
        "TAVILY_API_KEY=your_api_key_here"
    )


web_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_raw_content=False,
    name="web_search",
    description=(
        "Search the live internet for current, recent, "
        "real-time, or up-to-date information. "
        "Use this for news, latest technology updates, "
        "current events, current prices, recent releases, "
        "or information not available in the local "
        "knowledge base."
    ),
)


print("Live internet search enabled.")


# ============================================================
# 12. TOOLS
# ============================================================

tools = [
    retriever_tool,
    web_search_tool
]


# ============================================================
# 13. SYSTEM PROMPT
# ============================================================

system_prompt = """
You are an intelligent RAG + Internet Search AI Agent.

You have TWO information sources.

============================================================
1. DOCUMENT SEARCH
============================================================

Tool name:
document_search

This searches the user's local knowledge base.

The knowledge base can contain:

- PDF
- DOCX
- TXT
- CSV
- XLSX
- XLS
- Loaded webpages


============================================================
2. LIVE INTERNET SEARCH
============================================================

Tool name:
web_search

This searches the live internet.

Use it when:

- The user asks for current information.
- The user asks for latest information.
- The user asks about recent news.
- The user asks about today's information.
- The user asks about current technology.
- The user asks about recent AI developments.
- The information is not available in the documents.
- The user explicitly asks you to search the internet.


============================================================
SEARCH STRATEGY
============================================================

For normal knowledge-base questions:

1. First use document_search.

2. If the local documents contain a useful answer,
   answer using the documents.

3. If the answer is not available in the documents,
   use web_search.

4. If the user explicitly asks for current/latest/recent
   information, use web_search.

5. If both sources are useful, you may use both.

6. Never invent information.


============================================================
CURRENT INFORMATION
============================================================

For questions containing words such as:

- latest
- current
- today
- recent
- newest
- now
- this week
- this month
- 2026

prefer web_search because the information may have changed.


============================================================
ANSWER STYLE
============================================================

Give clear and useful answers.

When using the local knowledge base, mention:

Source: Local Knowledge Base

When using internet search, mention:

Source: Live Internet Search

When you use web_search, you MUST include the actual URLs of the
pages you used as sources. List them clearly at the end of your
answer under a "Links:" heading, one URL per line, using the exact
URLs returned by the web_search tool. Never invent or guess a URL —
only use URLs that were actually returned by the tool. Do not use
vague citation markers like [0] or 【source】; always show the real
link instead.

When both are used, mention:

Sources: Local Knowledge Base + Live Internet Search


============================================================
IMPORTANT
============================================================

Do not pretend that old documents contain current information.

Do not make up sources.

Do not make up facts.

Use the appropriate tool before answering when external
information is required.

Use conversation history to understand follow-up questions.
"""


# ============================================================
# 14. AGENT PROMPT + CHAT HISTORY
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{input}"
        ),

        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
    ]
)


# ============================================================
# 15. CREATE TOOL-CALLING AGENT
# ============================================================

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)


# ============================================================
# 16. AGENT EXECUTOR
# ============================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
)


# ============================================================
# 17. CHAT HISTORY
# ============================================================

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ============================================================
# 18. AGENT READY
# ============================================================

print("\n========================================")
print("RAG + INTERNET SEARCH AGENT")
print("========================================")

print("\nSupported files:")
print("PDF, DOCX, TXT, CSV, XLSX, XLS")

print(
    f"\nLoaded webpage(s): "
    f"{', '.join(web_urls)}"
)

print("\nRAG Document Search: ENABLED")

print("Live Internet Search: ENABLED (Tavily)")

print("Chat History: ENABLED")

print("\nAgent is ready!")

print("\nType 'exit' to stop.")

print("========================================")


# ============================================================
# ============================================================
# 19. CLI CHAT LOOP
# ============================================================

def run_cli():

    while True:

        query = input(
            "\nEnter your query: "
        )

        if query.lower().strip() == "exit":

            print("\nGoodbye!")

            break

        if not query.strip():

            continue

        try:

            response = agent_with_history.invoke(
                {
                    "input": query
                },

                config={
                    "configurable": {
                        "session_id": "user_1"
                    }
                }
            )

            print(
                "\n========================================"
            )

            print("ANSWER")

            print(
                "========================================"
            )

            print(
                response["output"]
            )

            print(
                "========================================"
            )

        except Exception as e:

            print(
                "\n========================================"
            )

            print("ERROR")

            print(
                "========================================"
            )

            print(
                str(e)
            )

            print(
                "========================================"
            )


# ============================================================
# RUN CLI ONLY WHEN rag1.py IS DIRECTLY EXECUTED
# ============================================================

if __name__ == "__main__":

    run_cli()