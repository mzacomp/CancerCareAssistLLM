# prompt.py

#Libraries
from openai import OpenAI
from dotenv import load_dotenv
import os

#Env Variables 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

#Build Context with Answering Questions 
def answer_question(query, retrieved_chunks):
    # Build context
    context = "\n\n".join([
        f"[{i+1}] ({c['doc']} p.{c['page']}) {c['text']}"
        for i, c in enumerate(retrieved_chunks)
    ])

    # LLM prompt with GPT-3.5-Turbo 
    prompt = f"""
You are a careful and grounded AI assistant answering questions only using the provided cancer care documents. 
You are not allowed to provide direct medical advice or diagnosis.

Your goal is to provide informative, well-structured answers using only the context provided.
You can use paragraph form or numbered lists as appropriate.

Cite each statement using [#] where # matches the context chunk number. 
If the answer is not directly supported by the context, reply only with:
"Apologies, the provided cancer care documents do not include that information."


Focus on clarity, completeness, and patient-centered readability. Please write an answer in 1-2 short paragraphs, about ~50-150 words.

CONTEXT:
{context}

QUESTION:
{query}
"""

    # Model Response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    return answer 
