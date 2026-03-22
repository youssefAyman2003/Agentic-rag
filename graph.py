from typing import List
from typing_extensions import TypedDict
 
from langchain.schema import Document
try:
    from langgraph.graph import END, StateGraph          
except ImportError:
    from langgraph.graph import StateGraph               
    from langgraph.graph.graph import END                
 
 
# ── Graph State ──────────────────────────────────────────────────────────────
 
class GraphState(TypedDict):
    question:   str
    generation: str
    documents:  List[Document]
    steps:      List[str]       
 
 
# ── Helper ───────────────────────────────────────────────────────────────────
 
def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )
 
 
# ── Node factories (accept pre-built graders so graph is stateless) ──────────
 
def make_retrieve_node(retriever):
    def retrieve(state: GraphState) -> GraphState:
        question = state["question"]
        documents = retriever.get_relevant_documents(question)
        steps = state.get("steps", [])
        steps.append(f"🔍 Retrieved {len(documents)} chunks for: \"{question}\"")
        return {**state, "documents": documents, "steps": steps}
    return retrieve
 
 
def make_grade_documents_node(retrieval_grader):
    def grade_documents(state: GraphState) -> GraphState:
        question  = state["question"]
        documents = state["documents"]
        steps     = state.get("steps", [])
 
        filtered, relevant_count, irrelevant_count = [], 0, 0
        for doc in documents:
            score = retrieval_grader.invoke({
                "question": question,
                "document": doc.page_content,
            })
            if score.binary_score.lower() == "yes":
                filtered.append(doc)
                relevant_count += 1
            else:
                irrelevant_count += 1
 
        steps.append(
            f"📋 Document grading: {relevant_count} relevant, {irrelevant_count} filtered out"
        )
        return {**state, "documents": filtered, "steps": steps}
    return grade_documents
 
 
def make_generate_node(rag_chain):
    def generate(state: GraphState) -> GraphState:
        question  = state["question"]
        documents = state["documents"]
        steps     = state.get("steps", [])
 
        context    = _format_docs(documents)
        generation = rag_chain.invoke({"context": context, "question": question})
        steps.append("✍️ Generated answer from relevant documents")
        return {**state, "generation": generation, "steps": steps}
    return generate
 
 
def make_transform_query_node(question_rewriter):
    def transform_query(state: GraphState) -> GraphState:
        question = state["question"]
        steps    = state.get("steps", [])
 
        better_question = question_rewriter.invoke({"question": question})
        steps.append(f"🔄 Query rewritten: \"{better_question}\"")
        return {**state, "question": better_question, "steps": steps}
    return transform_query
 
 
# ── Conditional edge functions ───────────────────────────────────────────────
 
def decide_to_generate(state: GraphState) -> str:
    """After grading docs: generate if any relevant, else rewrite query."""
    if not state["documents"]:
        return "transform_query"
    return "generate"
 
 
def make_grade_generation_edge(hallucination_grader, answer_grader):
    def grade_generation(state: GraphState) -> str:
        """
        After generation:
          - Check hallucination  → if not grounded, retry generation
          - Check answer quality → if useful, END; else rewrite query
        """
        question   = state["question"]
        documents  = state["documents"]
        generation = state["generation"]
        steps      = state.get("steps", [])
 
        # 1. Hallucination check
        hall_score = hallucination_grader.invoke({
            "documents": _format_docs(documents),
            "generation": generation,
        })
        if hall_score.binary_score.lower() != "yes":
            steps.append("⚠️ Hallucination detected — retrying generation")
            state["steps"] = steps
            return "not supported"
 
        steps.append("✅ Generation is grounded in retrieved documents")
 
        # 2. Answer quality check
        ans_score = answer_grader.invoke({
            "question": question,
            "generation": generation,
        })
        if ans_score.binary_score.lower() == "yes":
            steps.append("✅ Answer resolves the question — done!")
            state["steps"] = steps
            return "useful"
 
        steps.append("❌ Answer does not address question — rewriting query")
        state["steps"] = steps
        return "not useful"
 
    return grade_generation
 
 
# ── Graph builder ─────────────────────────────────────────────────────────────
 
def build_graph(retriever, graders: dict):
    """
    Compile the full Self-RAG LangGraph.
 
    Args:
        retriever: Chroma retriever from vectorstore.py
        graders:   dict from graders.build_all_graders()
 
    Returns:
        app: compiled LangGraph application
    """
    # Unpack graders
    retrieval_grader     = graders["retrieval_grader"]
    hallucination_grader = graders["hallucination_grader"]
    answer_grader        = graders["answer_grader"]
    question_rewriter    = graders["question_rewriter"]
    rag_chain            = graders["rag_chain"]
 
    # Build nodes
    retrieve_node         = make_retrieve_node(retriever)
    grade_documents_node  = make_grade_documents_node(retrieval_grader)
    generate_node         = make_generate_node(rag_chain)
    transform_query_node  = make_transform_query_node(question_rewriter)
    grade_generation_edge = make_grade_generation_edge(hallucination_grader, answer_grader)
 
    # Assemble graph
    workflow = StateGraph(GraphState)
 
    workflow.add_node("retrieve",         retrieve_node)
    workflow.add_node("grade_documents",  grade_documents_node)
    workflow.add_node("generate",         generate_node)
    workflow.add_node("transform_query",  transform_query_node)
 
    workflow.set_entry_point("retrieve")
 
    workflow.add_edge("retrieve",        "grade_documents")
 
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate":        "generate",
        },
    )
 
    workflow.add_edge("transform_query", "retrieve")
 
    workflow.add_conditional_edges(
        "generate",
        grade_generation_edge,
        {
            "not supported": "generate",
            "useful":        END,
            "not useful":    "transform_query",
        },
    )
 
    app = workflow.compile()
    return app
 
 
# ── Run helper ────────────────────────────────────────────────────────────────
 
def run_graph(app, question: str) -> dict:
    """
    Run the compiled Self-RAG graph for a given question.
 
    Returns:
        result dict with keys: question, generation, documents, steps
    """
    inputs = {
        "question":   question,
        "generation": "",
        "documents":  [],
        "steps":      [],
    }
    result = None
    for output in app.stream(inputs):
        for _, value in output.items():
            result = value
 
    return result or inputs