#  Context-Aware RAG Chatbot using LangChain, FAISS, Groq, and Conversational Memory

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-red?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented--Generation-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b?style=for-the-badge&logo=streamlit)
![Transformers](https://img.shields.io/badge/Transformers-NLP-yellow?style=for-the-badge)
![Embeddings](https://img.shields.io/badge/Embeddings-Semantic_Search-teal?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-Llama3-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

</p>

---

#  Project Overview

This project implements a **production-style Context-Aware Conversational AI Chatbot** using:

- **Retrieval-Augmented Generation (RAG)**
- **LangChain**
- **FAISS Vector Database**
- **HuggingFace Embeddings**
- **Groq-hosted Llama 3**
- **Conversational Memory**
- **Streamlit Deployment**

The chatbot is capable of:

✅ Understanding user queries  
✅ Retrieving relevant knowledge from PDFs  
✅ Maintaining conversational memory  
✅ Generating context-aware responses  
✅ Performing semantic document retrieval  
✅ Supporting multi-turn dialogue  

---

#  Objective

The primary objective of this project is to build an intelligent AI assistant capable of:

- retrieving information from custom knowledge sources,
- remembering previous conversations,
- generating grounded responses,
- and maintaining conversational continuity.

Unlike traditional LLM systems, this chatbot combines:
- semantic retrieval,
- vector databases,
- conversational memory,
- and large language models.

This architecture reflects how modern enterprise AI systems are designed.

---

#  What is Retrieval-Augmented Generation (RAG)?

Traditional Large Language Models generate responses only from pretrained weights.

Mathematically:

\[
P(y|x)
\]

Where:
- \(x\) = user query
- \(y\) = generated response

This leads to:
- hallucinations,
- outdated knowledge,
- inability to access private documents,
- poor explainability.

---

#  RAG Solution

Retrieval-Augmented Generation modifies this process:

\[
P(y|x,D)
\]

Where:
- \(D\) = retrieved external documents

The model now generates responses using:
- user query,
- retrieved knowledge,
- conversational context.

This dramatically improves:
- factual grounding,
- reliability,
- domain adaptability,
- explainability.

---

#  Complete System Architecture

```text
PDF Documents
      ↓
Text Extraction
      ↓
Document Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Database
      ↓
Semantic Retriever
      ↓
Prompt Construction
      ↓
Conversational Memory
      ↓
Groq LLM (Llama 3)
      ↓
Generated Response
```

---

#  Core Technologies Used

| Technology | Purpose |
|---|---|
| LangChain | AI orchestration framework |
| FAISS | Vector similarity search |
| HuggingFace Embeddings | Semantic vector generation |
| Groq | High-speed LLM inference |
| Llama 3 | Response generation |
| Streamlit | Web deployment |
| PyPDF | PDF text extraction |
| YAML | Prompt engineering configuration |

---

#  Key AI Concepts Implemented

---

## 🔹 1. Embeddings

Text is transformed into semantic vectors:

\[
E(x)=v
\]

Where:
- \(x\) = text
- \(v \in \mathbb{R}^{384}\)

Embeddings capture semantic meaning instead of literal keywords.

---

## 🔹 2. Semantic Search

Similarity between vectors is computed using cosine similarity:

\[
\cos(\theta)=\frac{A \cdot B}{||A|| ||B||}
\]

Where:
- \(A\) = query embedding
- \(B\) = document embedding

This enables meaning-based retrieval.

---

## 🔹 3. Vector Databases

FAISS stores and indexes embeddings for:
- nearest-neighbor retrieval,
- semantic search,
- contextual document retrieval.

---

## 🔹 4. Conversational Memory

Conversation history is stored as:

\[
H = \{(u_1,a_1),(u_2,a_2),...,(u_n,a_n)\}
\]

Where:
- \(u_i\) = user query
- \(a_i\) = assistant response

The chatbot generates:

\[
P(y|q,D,H)
\]

Where:
- \(q\) = query
- \(D\) = retrieved documents
- \(H\) = dialogue history

This enables:
- contextual continuity,
- follow-up understanding,
- multi-turn reasoning.

---

## 🔹 5. Transformer Attention

The LLM uses self-attention:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

This allows:
- contextual reasoning,
- token interactions,
- long-range dependency modeling.

---

#  Features

##  Retrieval-Augmented Generation (RAG)

- Semantic document retrieval
- Context-aware generation
- Reduced hallucinations

---

##  Conversational Memory

- Multi-turn conversations
- Dialogue continuity
- Context preservation

---

##  FAISS Vector Database

- High-speed similarity search
- Efficient semantic retrieval
- Scalable vector indexing

---

##  Groq LLM Integration

- Ultra-fast inference
- Llama 3 integration
- Production-ready generation

---

##  Streamlit Deployment

- Interactive chatbot UI
- Real-time responses
- Source document visualization

---

#  Project Structure

```text
rag_chatbot/
│
├── app.py
├── prompt.yaml
├── .env
│
├── data/
│   ├── gpt4.pdf
│   └── agi.pdf
│
├── vectorstore/
│   └── db_faiss/
│
├── utils/
│   ├── chatbot_chain.py
│   ├── document_loader.py
│   ├── memory.py
│   └── vector_store.py
│
└── notebook.ipynb
```

---

#  Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/mudassir-08/NeuroRAG-Contextual-AI.git

cd rag-chatbot
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 3️⃣ Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -U \
langchain \
langchain-community \
langchain-core \
langchain-groq \
faiss-cpu \
sentence-transformers \
pypdf \
python-dotenv \
pyyaml \
streamlit
```

---

#  Environment Variables

Create:

```text
.env
```

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

---

#  Add PDF Files

Place PDFs inside:

```text
data/
```

Example:

```text
data/gpt4.pdf
data/agi.pdf
```

---

#  Create Vector Database

Run:

```python
from utils.document_loader import load_documents, split_documents
from utils.vector_store import create_vector_store

pdf_files = [
    "data/gpt4.pdf",
    "data/agi.pdf"
]

documents = load_documents(pdf_files)

chunks = split_documents(documents)

db = create_vector_store(chunks)

print("Vector DB created successfully")
```

---

#  Run Streamlit App

```bash
streamlit run app.py
```

---

#  Application Features

The deployed chatbot supports:

✅ Conversational querying  
✅ Semantic retrieval  
✅ Multi-turn memory  
✅ Source document inspection  
✅ Context-aware responses  

---

#  Example Questions

```text
What is GPT-4?
Explain AGI simply
What are limitations of large language models?
Who developed GPT-4?
```

---

#  Why This Architecture Matters

Modern enterprise AI systems rarely rely on pure LLM generation.

Instead, they combine:
- retrieval systems,
- vector search,
- memory architectures,
- and generative models.

This project mirrors architectures used in:
- Microsoft Copilot
- Perplexity AI
- OpenAI Retrieval Systems
- Enterprise Knowledge Assistants

---

#  Advantages of RAG Systems

| Traditional LLM | RAG System |
|---|---|
| Hallucinations | Grounded responses |
| Static knowledge | Dynamic retrieval |
| Poor explainability | Source transparency |
| No private knowledge | Custom document support |

---

#  Research-Level Insights

This project demonstrates how:
- semantic embeddings,
- vector similarity search,
- transformer-based reasoning,
- and conversational memory

can be integrated into a scalable AI system.

The architecture reflects modern trends in:
- enterprise AI,
- AI search engines,
- contextual assistants,
- knowledge-grounded generation.

---

#  Future Improvements

Potential future enhancements include:

## 🔹 Hybrid Retrieval
Combining:
- dense retrieval,
- sparse retrieval,
- reranking models.

---

## 🔹 Long-Term Memory
Persistent vectorized conversation memory.

---

## 🔹 Streaming Responses
Real-time token streaming.

---

## 🔹 Multi-User Sessions
Scalable session management.

---

## 🔹 Agentic AI
Tool-using autonomous workflows.

---

## 🔹 Multimodal Retrieval
Support for:
- images,
- audio,
- video,
- structured data.

---

#  Skills Demonstrated

This project demonstrates:

✅ Conversational AI Development  
✅ Retrieval-Augmented Generation  
✅ Vector Databases  
✅ Semantic Search  
✅ Embedding Models  
✅ Prompt Engineering  
✅ LLM Integration  
✅ Context Management  
✅ AI Deployment  

---

#  License

This project is licensed under the MIT License.

---

#  Author

Developed as a research-level Context-Aware Conversational AI project using:
- LangChain
- Groq
- FAISS
- HuggingFace Embeddings
- Streamlit

---

#  Final Conclusion

This project successfully demonstrates a modern production-style Retrieval-Augmented Generation chatbot system capable of:

- semantic retrieval,
- contextual reasoning,
- conversational memory,
- and grounded response generation.

The chatbot combines:
- retrieval,
- vector search,
- memory,
- and generation

to create a scalable and context-aware conversational AI assistant capable of answering questions from custom knowledge sources with factual grounding and conversational continuity.