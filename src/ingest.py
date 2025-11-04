# ingest.py

#Libraries
import os, json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
from pinecone import Pinecone

# Load Env 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

#Data Directory
pdf_dir = "data/pdfs"

# Chunk PDFs 
def extract_chunks(pdf_dir, chunk_size=800, overlap=150, save_path="data/chunks.jsonl"):
    chunks = []
    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, fname)
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            words = text.split()
            for j in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[j:j + chunk_size])
                chunks.append({
                    "id": f"{fname}_{i}_{j}",
                    "doc": fname,
                    "page": i + 1,
                    "text": chunk
                })
    with open(save_path, "w") as f:
        for chunk in chunks:
            json.dump(chunk, f)
            f.write("\n")
    print(f" Extracted {len(chunks)} chunks and saved to {save_path}")
    return chunks

# Upserting into Pinecone Semantic indexc 
def build_pinecone_index(chunks):
    print(" Upserting embeddings to Pinecone...")
    for i, chunk in enumerate(chunks):
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["text"]
        )
        vector = emb.data[0].embedding
        index.upsert([
            (chunk["id"], vector, {"doc": chunk["doc"], "page": chunk["page"]})
        ])
        if (i + 1) % 15 == 0:
            print(f"  Uploaded {i+1}/{len(chunks)} chunks...")
    print("Pinecone upload complete")

if __name__ == "__main__":
    chunks = extract_chunks(pdf_dir)
    build_pinecone_index(chunks)
