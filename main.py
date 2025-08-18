import os
import json
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langchain.tools import Tool
from langgraph.graph import StateGraph, END


# ENV + MODELS
load_dotenv()
chat_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# COMPANIES & YEARS List
COMPANIES = {"GOOGL": "Google", "MSFT": "Microsoft", "NVDA": "NVIDIA"}
YEARS = ["2022","2023","2024"]


# LOAD 10-K FILINGS
def load_filings(base_folder: str = "filings") -> List:
    all_docs = []
    for ticker in COMPANIES:
        for year in YEARS:
            file_path = os.path.join(base_folder, f"{ticker}-{year}.pdf")
            if not os.path.exists(file_path):
                print(f"⚠️ Missing file: {file_path}")
                continue
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["company"] = ticker
                d.metadata["year"] = year
            all_docs.extend(docs)
            print(f"[+] Loaded {ticker} {year} ({len(docs)} pages)")
    return all_docs


# BUILD FAISS VECTORSTORE 
def build_vectorstore(docs):
    chunks = text_splitter.split_documents(docs)
    print(f"[DEBUG] Split into {len(chunks)} chunks")
    try:
        vectorstore = FAISS.from_documents(chunks, embedding=embedding)
        print("[+] FAISS Vector DB built successfully")
        return vectorstore
    except Exception as e:
        print(f"[ERROR] Failed to build FAISS vectorstore: {e}")
        return None


# RETRIEVER TOOL

def build_tools(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def retriever_tool_func(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found."
        results = []
        for d in docs:
            results.append({
                "company": d.metadata.get("company","Unknown"),
                "year": d.metadata.get("year","Unknown"),
                "page": d.metadata.get("page","?"),
                "excerpt": d.page_content[:300]
            })
        return json.dumps(results, indent=2)

    return [
        Tool.from_function(
            name="retriever_tool",
            func=retriever_tool_func,
            description="Retrieve top relevant filing excerpts for a financial query."
        )
    ]


# AGENT STATE

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state["messages"][-1]
    tool_calls = getattr(result, "tool_calls", None)
    return tool_calls is not None and len(tool_calls) > 0


# BUILD AGENT WITH DECOMPOSITION + SYNTHESIS

def build_agent(tools):
    llm = chat_llm.bind_tools(tools)
    tools_dict = {t.name: t for t in tools}

    # SYSTEM PROMPT
    system_prompt = """
        You are a financial Q&A expert & intelligent assistant helping users in answer their query. 
        You have access to 10-K filings for Google, Microsoft, NVIDIA (2022-2024).
        Give only answer in the answer key and do not give any reasoning to what you are about to do give only extracted information from the fillings in the answer.
        Search for the query asked to you in the fillings and return it in your answer.
        You can search the document thoroughly and completely without leave any part of the document to answer user query.
        The retriever tool supports anything you want to do to obtain the answer for the request.
        You have the ability to Cross-Company Ananlyis and to compare and determine the answer.
        You can perform Segment Analysis, the available tools allow you to calculate values for the companies and thier data.
        Always answer the query asked to you and do not give excuses in the answer you have no limitations to you.
        You are a excellent Strategist able to compare all the three companies and thier data and provide relevant answer.
        Be smart and check if the query should be Decomposed into sub-queries. Check for comparative or multi-company queries to decomposed.
        Synthesize the retrieved results into a JSON output:
        {
        "query": <original question>,
        "answer": <final synthesized answer>,
        "reasoning": <how you combined results>,
        "sub_queries": [list of sub-queries],
        "sources": [retrieved excerpts with company/year/page]
        }
        Always return valid JSON.
"""

    # Calling LLM
    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        message = llm.invoke(messages)
        return {"messages": [message]}

    # Calling Action Tool
    def take_action(state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        results = []
        if tool_calls:
            for t in tool_calls:
                print(f"[Tool Call] {t['name']} -> {t['args'].get('query','')}")
                if t["name"] not in tools_dict:
                    result = "Invalid tool"
                else:
                    result = tools_dict[t["name"]].invoke(t["args"].get("query",""))
                results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=result))
        return {"messages": results}

    # STATE GRAPH
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


# CLI RUNNER
def main():
    print("=== Financial RAG Agent ===")
    print("Loading filings...")
    docs = load_filings("filings")
    print(f"[DEBUG] Loaded {len(docs)} documents")
    if not docs:
        print("❌ No filings found. Place PDFs in 'filings/' folder.")
        return

    vectorstore = build_vectorstore(docs)
    if not vectorstore:
        print("❌ Vector store build failed. Exiting.")
        return

    tools = build_tools(vectorstore)
    print("[DEBUG] Tools ready")
    agent = build_agent(tools)
    print("[DEBUG] Agent ready")

    print("\n✅ Ready. Ask financial questions (type 'exit' to quit)\n")
    while True:
        try:
            query = input("You: ")
        except EOFError:
            print("\nExiting CLI.")
            break

        if query.lower() in ["exit","quit"]:
            break

        try:
            response = agent.invoke({"messages":[HumanMessage(content=query)]})
            if response.get("messages"):
                print("\nAI Response:\n", response["messages"][-1].content, "\n")
            else:
                print("[DEBUG] No messages returned.\n")
        except Exception as e:
            print(f"[ERROR] Agent failed: {e}\n")

if __name__ == "__main__":
    main()
