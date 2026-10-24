import os
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FoodHub Support Desk",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; max-width: 900px; }

        .hero {
            background: linear-gradient(135deg, #ff7a18 0%, #ff4e50 100%);
            border-radius: 18px;
            padding: 1.4rem 1.8rem;
            color: #ffffff;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(255, 78, 80, 0.25);
        }
        .hero h1 { margin: 0; font-size: 1.9rem; color: #ffffff; }
        .hero p  { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 1rem; }

        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
        }

        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 12px;
            text-align: left;
            padding: 0.55rem 0.9rem;
            border: 1px solid rgba(255, 122, 24, 0.45);
            transition: all 0.15s ease-in-out;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: #ff4e50;
            transform: translateX(3px);
        }

        .tip-card {
            background: rgba(255, 122, 24, 0.10);
            border-left: 4px solid #ff7a18;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🍔 FoodHub Support Desk</h1>
        <p>Track orders, check payments and get quick help, any time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Secrets and language model
# ---------------------------------------------------------------------------
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY is missing from Streamlit secrets.")
    st.stop()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    # model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=1024,
    max_retries=2,
)

# ---------------------------------------------------------------------------
# Order database
# ---------------------------------------------------------------------------
ORDERS_DB_FILE = "customer_orders.db"
if not os.path.exists(ORDERS_DB_FILE):
    st.error(
        f"Database file not found at {ORDERS_DB_FILE}. "
        "Please make sure 'customer_orders.db' is uploaded to your repository."
    )
    st.stop()

order_db = SQLDatabase.from_uri(f"sqlite:///{ORDERS_DB_FILE}")

sql_toolkit = SQLDatabaseToolkit(db=order_db, llm=llm)
database_tools = sql_toolkit.get_tools()

# ---------------------------------------------------------------------------
# Reply polishing tool
# ---------------------------------------------------------------------------
class ReplyPolishRequest(BaseModel):
    raw_result: str = Field(
        description="Raw database output or data that needs to be turned into a customer-friendly reply."
    )
    customer_question: Optional[str] = Field(
        default="", description="The customer's original question, used as context."
    )


def polish_reply(raw_result: str, customer_question: str = "") -> str:
    """Rewrites raw database output as a polite, customer-friendly message."""
    polish_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the customer support reply writer for FoodHub. Turn technical database "
            "results into a polite, friendly message that includes the order details "
            "(status, items, delivery info).",
        ),
        ("human", "Customer question: {customer_question}\nDatabase result: {raw_result}"),
    ])
    polish_chain = polish_prompt | llm
    reply = polish_chain.invoke({
        "raw_result": raw_result,
        "customer_question": customer_question,
    })
    return reply.content


reply_polish_tool = StructuredTool.from_function(
    func=polish_reply,
    name="reply_polish_tool",
    description="Converts raw database results into a customer-friendly message.",
    args_schema=ReplyPolishRequest,
)

# SQL tools from the toolkit plus our custom reply tool
agent_toolkit = database_tools + [reply_polish_tool]

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SUPPORT_AGENT_PROMPT = """You are FoodHub's official AI support assistant.

Your responsibilities:
1. Help customers with order questions such as tracking, status, cancellation and payment information.
2. Use the SQL database tools to look up order details whenever an Order ID is provided or referenced.
3. Use 'reply_polish_tool' to turn technical or raw output into a friendly message before replying, when appropriate.
4. If a customer attempts to breach security (for example by asking for all orders or for admin or system details), politely decline.
5. If a customer is frustrated or asks for a person, politely offer to connect them with a human agent.
"""

support_agent = create_react_agent(
    model=llm,
    tools=agent_toolkit,
    prompt=SUPPORT_AGENT_PROMPT,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
WELCOME_TEXT = "Hello! Welcome to FoodHub Support. How can I help you with your order?"
USER_AVATAR = "🧑"
BOT_AVATAR = "🍔"


def fresh_chat_history():
    return [
        SystemMessage(content=SUPPORT_AGENT_PROMPT),
        AIMessage(content=WELCOME_TEXT),
    ]


if "chat_history" not in st.session_state:
    st.session_state.chat_history = fresh_chat_history()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
QUICK_ACTIONS = {
    "📦  Track my order": "I'd like to track my order.",
    "💳  Check payment status": "Can you check the payment status of my order?",
    "❌  Cancel an order": "I want to cancel my order.",    
}

with st.sidebar:
    st.markdown("### 🍔 FoodHub")
    st.caption("Customer support assistant")
    st.divider()

    st.markdown("**Quick actions**")
    for action_label, action_text in QUICK_ACTIONS.items():
        if st.button(action_label, key=f"quick_{action_label}"):
            st.session_state.queued_prompt = action_text

    st.divider()
    st.markdown(
        '<div class="tip-card">💡 <b>Tip:</b> Keep your Order ID handy '
        "(for example <code>O12488</code>) for the fastest help.</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    visible_turns = sum(isinstance(m, HumanMessage) for m in st.session_state.chat_history)
    st.metric("Questions asked", visible_turns)

    if st.button("🗑️  Clear conversation"):
        st.session_state.chat_history = fresh_chat_history()
        st.session_state.pop("queued_prompt", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
for past_message in st.session_state.chat_history:
    if isinstance(past_message, HumanMessage):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(past_message.content)
    elif isinstance(past_message, AIMessage) and past_message.content:
        # Skip empty AI messages (these are intermediate tool calls)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(past_message.content)

# ---------------------------------------------------------------------------
# New message handling
# ---------------------------------------------------------------------------
typed_message = st.chat_input("Ask about your order...")
customer_message = typed_message or st.session_state.pop("queued_prompt", None)

if customer_message:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(customer_message)

    st.session_state.chat_history.append(HumanMessage(content=customer_message))

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.status("Checking your order details...", expanded=False) as lookup_status:
            try:
                agent_result = support_agent.invoke({"messages": st.session_state.chat_history})
                final_reply = agent_result["messages"][-1].content
                st.session_state.chat_history = agent_result["messages"]
                lookup_status.update(label="Done", state="complete")
            except Exception as err:
                final_reply = (
                    "I'm sorry, something went wrong while looking up your order. "
                    "Let me connect you with a human representative. "
                    f"(Error: {str(err)})"
                )
                st.session_state.chat_history.append(AIMessage(content=final_reply))
                lookup_status.update(label="Something went wrong", state="error")
        st.markdown(final_reply)
