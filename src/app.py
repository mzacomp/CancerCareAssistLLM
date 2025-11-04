# app.py

#Libraries 
from retrieve import load_chunks, hybrid_retrieve
from prompt  import answer_question

#Main Execution 
def clean_text(text):
    """Remove markdown bold/italics, asterisks, underscores."""
    for sym in ["**","*","_"]:
        text = text.replace(sym,"")
    return text.strip()

if __name__ == "__main__":
    chunks = load_chunks()
    print("\n Hi, I am CancerCareAssist. Ready to answer questions aboout navigating the cancer care journey.\n")

    while True:
        query = input("\nQuestion (or 'exit'): ")
        if query.lower() == "exit":
            break
        retrieved = hybrid_retrieve(query, chunks, top_k=10, alpha=0.6)
        answer = answer_question(query, retrieved)
        answer = clean_text(answer)
        #print model answer
        print("\n=== ANSWER ===\n")
        print(answer)
        #print retrieved context/citations after the answer 
        print("\n=== RETRIEVED CONTEXT & CITATIONS ===\n")
        for i, c in enumerate(retrieved, start=1):
            print(f"[{i}] {c['doc']} (p.{c['page']})")
        print("\n-------------------------")
