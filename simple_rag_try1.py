# simple_rag.py

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
import os

# --------------------------------------------------
# Create Structured Output Models
# --------------------------------------------------


class ActionItem(BaseModel):
    task: str
    owner: str = "UNKNOWN"
    deadline: str = "UNKNOWN"


class MeetingAnalysis(BaseModel):
    summary: list[str]
    decisions: list[str]
    pending_questions: list[str]
    action_items: list[ActionItem]
    unclear_commitments: list[str]
    conflicting_statements: list[str]
    follow_up_email: str

# --------------------------------------------------
# 1. SOURCE DATA
# --------------------------------------------------

text = """
Project Review Meeting

Attendees:
Raj, Priya, John

Decision:
Cloud migration will proceed.

Action:
John will create migration plan by Nov 10.

Action:
Someone should validate security requirements.

Question:
Do we need additional budget approval?

Discussion:
Raj wants December launch.
Priya recommends January launch.
"""


def get_query_type(user_query: str) -> str:
    """Identify the requested meeting-analysis format."""
    query = user_query.casefold()
    if "summary" in query or "summarize" in query:
        return "summary"
    if "decision" in query or "question" in query:
        return "decisions"
    if "action" in query or "owner" in query or "deadline" in query or "table" in query:
        return "actions"
    if "conflict" in query or "unclear" in query or "ambigu" in query:
        return "conflicts"
    if "email" in query:
        return "email"
    return "complete"


def get_prompt(user_query: str, context: str) -> str:
    """Build a transcript-grounded prompt for the requested analysis."""
    query_type = get_query_type(user_query)
    instructions = {
        "summary": (
            "Return exactly five concise bullets in summary. Cover distinct "
            "meeting facts; do not add unsupported details."
        ),
        "decisions": (
            "Extract every explicit decision and every open question. Do not "
            "treat opinions or proposals as decisions."
        ),
        "actions": (
            "Extract every action item. Put the task, explicitly named owner, "
            "and explicitly stated deadline in separate fields. Use UNKNOWN "
            "when the owner or deadline is not stated."
        ),
        "conflicts": (
            "Identify differing positions as conflicting statements, even if "
            "the transcript does not resolve them. Identify vague assignments "
            "or missing owners as unclear commitments."
        ),
        "email": (
            "Draft a professional follow-up email for the participants. "
            "Include the decision, action items, differing launch views, and "
            "the open budget question. Do not invent details."
        ),
        "complete": "Provide a complete meeting analysis in all fields.",
    }[query_type]

    return f"""
You are an expert meeting-minutes assistant. Analyze the complete meeting
transcript and answer the user's request.

Rules:
- Use only facts in the transcript. Never invent owners, deadlines, or decisions.
- Use "UNKNOWN" when an action item's owner or deadline is not stated.
- Treat differing opinions as conflicting statements, not as a decision.
- Leave fields unrelated to the request empty, except summary, which should
  contain exactly five concise bullets.

Request-specific instructions: {instructions}

Meeting transcript:
{context}

User question:
{user_query}
"""


def validate_action_items(output: MeetingAnalysis, context: str) -> bool:
    """Check that each stated action-item owner appears in the source text."""
    source = context.casefold()
    return all(
        item.owner.casefold() == "unknown"
        or item.owner.casefold() in source
        for item in output.action_items
    )


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def format_answer(query: str, analysis: MeetingAnalysis) -> str:
    """Present the analysis in the format requested by the user."""
    query_type = get_query_type(query)

    if query_type == "summary":
        return "\n".join(f"- {item}" for item in analysis.summary)
    if query_type == "decisions":
        decisions = "\n".join(f"- {item}" for item in analysis.decisions)
        questions = "\n".join(
            f"- {item}" for item in analysis.pending_questions
        )
        return (
            f"Decisions:\n{decisions or '- None found.'}\n\n"
            f"Pending questions:\n{questions or '- None found.'}"
        )
    if query_type == "actions":
        rows = [
            "| Task | Owner | Deadline |",
            "| --- | --- | --- |",
        ]
        rows.extend(
            f"| {escape_table_cell(item.task)} "
            f"| {escape_table_cell(item.owner)} "
            f"| {escape_table_cell(item.deadline)} |"
            for item in analysis.action_items
        )
        if not analysis.action_items:
            rows.append("| None found. | - | - |")
        return "\n".join(rows)
    if query_type == "conflicts":
        conflicts = "\n".join(
            f"- {item}" for item in analysis.conflicting_statements
        )
        unclear = "\n".join(
            f"- {item}" for item in analysis.unclear_commitments
        )
        return (
            f"Conflicting statements:\n{conflicts or '- None found.'}\n\n"
            f"Unclear commitments:\n{unclear or '- None found.'}"
        )
    if query_type == "email":
        return analysis.follow_up_email or "No follow-up email was generated."

    return analysis.model_dump_json(indent=2)


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project "
            "directory or set it in your environment."
        )

    document = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
    )
    chunks = splitter.split_documents([document])
    print(f"\nCreated {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector store created.")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    structured_llm = llm.with_structured_output(MeetingAnalysis)

    while True:
        try:
            question = input("\nAsk a question (or type 'exit'): ").strip()
        except EOFError:
            print()
            break

        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue

        results = vectorstore.similarity_search(question, k=len(chunks))
        results.sort(key=lambda result: result.metadata["start_index"])
        context = "\n".join(result.page_content for result in results)
        response = structured_llm.invoke(get_prompt(question, context))

        if not validate_action_items(response, context):
            raise ValueError(
                "The model returned an action-item owner not found in the "
                "retrieved transcript context."
            )

        print("\nAnswer:")
        print(format_answer(question, response))


if __name__ == "__main__":
    main()