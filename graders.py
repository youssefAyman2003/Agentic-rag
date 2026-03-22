from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field  
 
 
# ── Shared Groq LLM ──────────────────────────────────────────────────────────
def get_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0) -> ChatGroq:
    """Return a ChatGroq instance. API key is read from GROQ_API_KEY env var."""
    return ChatGroq(model=model, temperature=temperature)
 
 
# ── Pydantic schemas for structured output ───────────────────────────────────
 
class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""
    binary_score: str = Field(
        description="Document is relevant to the question. 'yes' or 'no'."
    )
 
 
class GradeHallucinations(BaseModel):
    """Binary score — is the generation grounded in the retrieved facts?"""
    binary_score: str = Field(
        description="Generation is grounded in the documents. 'yes' or 'no'."
    )
 
 
class GradeAnswer(BaseModel):
    """Binary score — does the generation resolve the user question?"""
    binary_score: str = Field(
        description="Generation addresses the question. 'yes' or 'no'."
    )
 
 
# ── 1. Retrieval Grader ──────────────────────────────────────────────────────
 
def build_retrieval_grader():
    """
    Returns a chain that scores whether a retrieved chunk is relevant.
    Input:  {"question": str, "document": str}
    Output: GradeDocuments(binary_score="yes"|"no")
    """
    llm = get_groq_llm()
    structured_llm = llm.with_structured_output(GradeDocuments)
 
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the question, "
        "grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' to indicate relevance."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ])
    return prompt | structured_llm
 
 
# ── 2. Hallucination Grader ──────────────────────────────────────────────────
 
def build_hallucination_grader():
    """
    Returns a chain that checks if a generation is grounded in retrieved docs.
    Input:  {"documents": str, "generation": str}
    Output: GradeHallucinations(binary_score="yes"|"no")
    """
    llm = get_groq_llm()
    structured_llm = llm.with_structured_output(GradeHallucinations)
 
    system = (
        "You are a grader assessing whether an LLM generation is grounded in "
        "and supported by a set of retrieved facts.\n"
        "Give a binary score 'yes' or 'no'.\n"
        "'yes' means the answer is grounded in the facts.\n"
        "'no' means the answer contains claims not supported by the facts."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved facts:\n\n{documents}\n\nLLM generation:\n{generation}"),
    ])
    return prompt | structured_llm
 
 
# ── 3. Answer Grader ─────────────────────────────────────────────────────────
 
def build_answer_grader():
    """
    Returns a chain that checks if a generation resolves the user question.
    Input:  {"question": str, "generation": str}
    Output: GradeAnswer(binary_score="yes"|"no")
    """
    llm = get_groq_llm()
    structured_llm = llm.with_structured_output(GradeAnswer)
 
    system = (
        "You are a grader assessing whether an answer fully addresses a question.\n"
        "Give a binary score 'yes' or 'no'.\n\n"
        "Score 'yes' if:\n"
        "  - The answer directly addresses the question using information from the context.\n"
        "  - The answer contains the actual CONTENT of the relevant entry, not just its title.\n"
        "  - For structured docs (licenses/permits): all available fields are present and filled.\n"
        "  - For CVs, books, articles: the answer is relevant and substantive.\n\n"
        "Score 'no' if:\n"
        "  - The answer is off-topic or does not address the question.\n"
        "  - The answer returns only a name or title with no supporting detail or field values.\n"
        "  - For structured docs: ANY labeled field from the source is missing.\n"
        "  - The answer says it could not find information but the question is clearly answerable."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question:\n\n{question}\n\nLLM generation:\n{generation}"),
    ])
    return prompt | structured_llm
 
 
# ── 4. Question Rewriter ─────────────────────────────────────────────────────
 
def build_question_rewriter():
    """
    Returns a chain that rewrites a question for better vectorstore retrieval.
    Input:  {"question": str}
    Output: str — improved question
    """
    llm = get_groq_llm()
 
    system = (
        "You are a question rewriter that converts an input question into a better version "
        "optimised for vectorstore retrieval.\n"
        "Analyse the semantic intent and rephrase to improve document matching.\n"
        "Return only the improved question — no explanation, no preamble."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Original question:\n\n{question}\n\nImproved question:"),
    ])
    return prompt | llm | StrOutputParser()
 
 
# ── 5. RAG Generation Chain ──────────────────────────────────────────────────
 
def build_rag_chain():
    """
    Returns a chain that generates an answer from context + question.
    Input:  {"context": str, "question": str}
    Output: str — the generated answer with full content + page citation
    """
    llm = get_groq_llm(model="llama-3.1-8b-instant", temperature=0)
 
    system = (
        "You are a friendly document assistant with two modes:\n\n"
 
        "MODE 1 — CHAT (empty context or greeting/smalltalk):\n"
        "Reply warmly and briefly. No citations. Examples:\n"
        "'hi'→'Hey! 👋 How can I help?', 'thanks'→'You're welcome! 😊',\n"
        "'your name'→'I'm your Self-RAG assistant!'\n\n"
 
        "MODE 2 — DOCUMENT QUERY (context has content):\n"
        "Use ONLY the context. Detect doc type then apply the matching rule:\n\n"
 
        "RULE A (license/permit/regulation):\n"
        "Reproduce EVERY field verbatim, one per line as 'Field: value'.\n"
        "Fields: Service, Sector, Authority, Description, Requirements,\n"
        "Documents, Steps, Fees, Processing Time, Validity, Renewal, Contact, Legal Ref.\n"
        "List ALL items in list-fields. Separate multiple matches with ────.\n"
        "Missing field → write 'Field: N/A'. NEVER return title only.\n\n"
 
        "RULE B (CV/resume): Answer with name, titles, companies, dates, skills.\n\n"
 
        "RULE C (article/report/book): Quote or paraphrase relevant passages directly.\n\n"
 
        "FALLBACK: If context lacks the answer, say so and state what's missing.\n\n"
 
        "CITATION (Mode 2 only): After full answer append exactly:\n"
        "📄 Source: <filename> — Page(s): <pages>\n"
        "Citation is a supplement — NEVER replace content with it."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    return prompt | llm | StrOutputParser()
 
 
# ── Bundle all graders for easy import ───────────────────────────────────────
 
def build_all_graders() -> dict:
    """
    Build and return all graders + the RAG chain as a dict.
    Called once at app startup and stored in st.session_state.
    """
    return {
        "retrieval_grader":     build_retrieval_grader(),
        "hallucination_grader": build_hallucination_grader(),
        "answer_grader":        build_answer_grader(),
        "question_rewriter":    build_question_rewriter(),
        "rag_chain":            build_rag_chain(),
    }