from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# CORRECT — this is the current import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import psycopg2



load_dotenv()

# ------------------ Vectorstore + LLM ------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Load product descriptions
file_path = "products_descriptions.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError("products_descriptions.txt NOT FOUND!")

with open(file_path, "r", encoding="utf8") as f:
    file_content = f.read()

documents = [Document(page_content=file_content)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs_split = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    persist_directory="./Ai-Agent-Products",
    collection_name="products"
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ------------------ Tool ------------------
@tool
def product_search_tool(query: str) -> str:
    """Searches the product descriptions and returns relevant matches."""
    docs = retriever.invoke(query)
    if not docs:
        return "No matching product found."
    return "\n\n".join([f"Match {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

tools = [product_search_tool]
tools_dict = {t.name: t for t in tools}


NEON_URL = os.getenv("NEON_URL")


@tool
def get_product_from_db(product_name: str) -> str:
    """Retrieve product details from Neon database."""
    try:
        conn = psycopg2.connect(NEON_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT name, description, skin_type, key_ingredients, benefits, usage, precautions, price, size
            FROM products
            WHERE name ILIKE %s
        """, (f"%{product_name}%",))
        
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            return f"""
Product Name: {row[0]}
Description: {row[1]}
Skin Type: {row[2]}
Key Ingredients: {row[3]}
Benefits: {row[4]}
Usage: {row[5]}
Precautions: {row[6]}
Price: {row[7]}
Size: {row[8]}
"""
        else:
            return "Product not found in database."
    except Exception as e:
        return f"Database error: {str(e)}"
    

# 3️⃣ Add the new tool to your agent

tools.append(get_product_from_db)
tools_dict[get_product_from_db.name] = get_product_from_db






# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
).bind_tools(tools)





# ------------------ Agent State ------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

conversation_state = {"messages": []}  # global memory

# System prompt for one-by-one questions
system_prompt = """
You are a skincare seller agent. Ask questions step by step:
1. Skin type
2. Main concern
3. Budget
4. Texture preference

Only ask ONE question at a time. After collecting all info, recommend the best product.
"""

def should_continue(state: AgentState):
    msg = state["messages"][-1]
    return hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    msgs = []

    for t in tool_calls:
        tool_name = t["name"]
        query = t["args"].get("query", "")

        # Step 1: Invoke the original tool
        result = tools_dict[tool_name].invoke(query)

        # Step 2: If product_search_tool found a match, automatically call DB tool
        if tool_name == "product_search_tool" and "Match 1:" in result:
            # crude extraction of first product name (assumes first line after 'Match 1:' is name)
            first_product_line = result.splitlines()[1]
            product_name = first_product_line.strip()

            # Step 3: Call Neon DB tool
            # db_result = get_product_from_db(product_name)
            db_result = get_product_from_db.invoke(product_name)

            html_card = generate_product_card(db_result)

            

            result += f"\n\n---\nOfficial Database Info:\n{db_result}"

        # Ensure result is string
        if isinstance(result, list):
            result = "\n\n".join([str(r) for r in result])
        else:
            result = str(result)

        msgs.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=result))

    return {"messages": msgs}


# Generate HTML card from DB result

def generate_product_card(db_result: str) -> str:
    """Convert the db_result string into an HTML card for display."""
    lines = db_result.strip().split("\n")
    data = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    
    html_card = f"""
    <div class="product-card" style="border:1px solid #ccc; padding:16px; border-radius:8px; width:300px; margin:10px;">
        <h3>{data.get('Product Name', '')}</h3>
        <p><strong>Skin Type:</strong> {data.get('Skin Type', '')}</p>
        <p><strong>Price:</strong> ${data.get('Price', '')} ({data.get('Size', '')})</p>
        <p><strong>Key Ingredients:</strong> {data.get('Key Ingredients', '')}</p>
        <p><strong>Benefits:</strong> {data.get('Benefits', '')}</p>
        <p><strong>Description:</strong> {data.get('Description', '')}</p>
        <button onclick="alert('Redirect to buy page for {data.get('Product Name', '')}')">Buy</button>
    </div>
    """
    return html_card



# Graph setup
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "tool_agent", False: END})
graph.add_edge("tool_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# ------------------ Flask ------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # your HTML page

@app.route("/ask", methods=["POST"])
def ask():
    global conversation_state
    user_input = request.json.get("message", "")

    # Add user's message
    new_state = {
        "messages": conversation_state["messages"] + [HumanMessage(content=user_input)]
    }

    # Invoke agent
    conversation_state = rag_agent.invoke(new_state)
    last_msg = conversation_state["messages"][-1]

    # Extract text content if it's structured
    ai_response = ""
    if hasattr(last_msg, "content"):
        content = last_msg.content
        if isinstance(content, list):
            # Extract text from each dict in the list
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            ai_response = "\n".join(texts)
        else:
            ai_response = str(content)
    else:
        ai_response = str(last_msg)

    return jsonify({"response": ai_response})




if __name__ == "__main__":
    app.run(debug=True)
