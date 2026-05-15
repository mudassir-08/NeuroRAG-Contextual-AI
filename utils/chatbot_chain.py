import os
import yaml

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.vector_store import load_vector_store
from utils.memory import memory

load_dotenv()


# Load Vector DB
db = load_vector_store()

# Retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# Load YAML Prompt
with open("prompt.yaml", "r", encoding="utf-8") as f:
    prompt_config = yaml.safe_load(f)

system_prompt = prompt_config["system_prompt"]

# LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
{system_prompt}

Conversation History:
{history}

Context:
{context}

User Question:
{question}
""")


def rag_chat(question):

    # Retrieve docs
    docs = retriever.invoke(question)

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Load memory
    history = memory.get_history()

    # Build final prompt
    final_prompt = prompt.format(
        system_prompt=system_prompt,
        history=history,
        context=context,
        question=question
    )

    # Generate response
    response = llm.invoke(final_prompt)

    answer = response.content

    # Save memory
    memory.save(question, answer)

    return {
        "answer": answer,
        "source_documents": docs
    }