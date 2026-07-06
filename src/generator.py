import os
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def build_context(retrieved_docs: List[Dict]) -> str:
    parts = []

    for doc in retrieved_docs:
        meta = doc["metadata"]
        parts.append(
            f"Source: {meta.get('source_file')} | "
            f"Year: {meta.get('year')} | Month: {meta.get('month')} | "
            f"Chunk: {meta.get('chunk_id')}\n"
            f"{doc['text']}"
        )

    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, retrieved_docs: List[Dict]) -> str:
    context = build_context(retrieved_docs)

    # If no OpenAI key is set, return a retrieval-only placeholder.
    # This still lets you evaluate retriever metrics.
    if not os.getenv("OPENAI_API_KEY") or OpenAI is None:
        return (
            "OPENAI_API_KEY not set. Retrieval-only mode. "
            "Use the retrieved context to manually answer or set an API key."
        )

    client = OpenAI()

    prompt = f'''
You are a financial QA assistant.
Answer the question using only the provided Treasury Bulletin context.
If the answer is not in the context, say: "I could not find the answer in the provided context."

Question:
{question}

Context:
{context}

Final answer:
'''

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer financial questions using only provided source text.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content
