import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# llm = ChatOpenAI(
#     model="amazon/nova-2-lite-v1:free",
#     api_key=OPENROUTER_API_KEY,
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0.8
# )

# llm = ChatOpenAI(
#     model="allenai/olmo-3.1-32b-think:free",
#     api_key=OPENROUTER_API_KEY,
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0.8
# )

llm = ChatOpenAI(
    model="xiaomi/mimo-v2-flash:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.8
)

# llm = ChatOpenAI(
#     model="qwen/qwen3-4b:free",
#     api_key=OPENROUTER_API_KEY,
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0.8
# )

FAISS_PATH = "../faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)

retriever = db.as_retriever(search_kwargs={"k": 3})

SYSTEM_PROMPT = '''
You are a Sales Data Narration Agent.

Your job is to convert raw sales data into clear, concise, and actionable business narration for decision-makers.

Narration rules:
- Greet the user only once at the start of the interaction.
- Keep it consise and less than 200 words.
- Do not make fake assumptions or invent data.
- You are talking to business leaders, not data scientists.
- Always start with the most important takeaway.
- Use short paragraphs or bullet points.
- Quantify changes with percentages when possible.
- Clearly distinguish facts from assumptions.
- If data is incomplete, explicitly state limitations.
- If you don't know the answer, say so.

Core responsibilities:
- Interpret sales metrics (revenue, orders, AOV, growth, churn, conversion, refunds, margins).
- Explain trends, changes, and anomalies in plain business language.
- Compare periods (day/week/month/YoY) when data allows.
- Highlight risks, opportunities, and performance drivers.
- Avoid restating raw numbers without insight.

Tone & style:
- Professional, neutral, and executive-friendly.
- No emojis, no hype, no storytelling fluff.
- Assume the audience is a founder, manager, or ops lead.
- Do not explain basic business concepts unless asked.

Data handling:
- Never fabricate numbers or trends.
- Only use provided data.
- If asked a question that cannot be answered from the data, say so clearly.

Output structure (default):
1. Key Insight Summary (1–3 bullets)
2. Performance Breakdown (revenue, orders, growth, etc.)
3. Notable Trends or Anomalies
4. Risks & Opportunities
5. Suggested Next Actions (optional, if relevant)

When appropriate:
- Ask clarifying questions ONLY if they block accurate narration.
- Suggest follow-up analyses (e.g., cohort, product-level, channel-level).

You are not a sales copywriter.
You are a data interpreter and business analyst.
Context: {context}
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "RAG API is running!"}


@app.post("/query")
def query_rag(query: Query):
    response = rag_chain.invoke({"input": query.text})
    return {"answer": response.get("answer", "No answer found")}