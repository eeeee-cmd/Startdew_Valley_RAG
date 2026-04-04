import json
import pandas as pd

with open("/Users/harshpinge/Downloads/Rotman MMA/RSM8430/Startdew_Valley_RAG/data/processed/stardew_wiki_sections.jsonl") as f:
    lines = [json.loads(line) for line in f]

# basic stats
print(f"Total records: {len(lines)}")
print(f"Fields: {list(lines[0].keys())}")
print()

# dataframe overview
df = pd.DataFrame(lines)
print(df[["page_title", "heading", "text_length"]].describe())
print()

# null check
print("Nulls:")
print(df.isnull().sum())
print()

# heading distribution
print("Top headings:")
print(df["heading"].value_counts().head(20))
print()

# shortest and longest chunks
print("Shortest chunks:")
print(df.nsmallest(5, "text_length")[["page_title", "heading", "text"]])
print()

print("Longest chunks:")
print(df.nlargest(5, "text_length")[["page_title", "heading", "text"]])

# check how many modding/template pages exist
print('\nAfter applying chunker filters:')
print(f"Modding pages: {df[df['page_title'].str.startswith('Modding:')].shape[0]}")
print(f"Module pages: {df[df['page_title'].str.startswith('Module:')].shape[0]}")
print(f"PNG pages: {df[df['text'].str.startswith('PNG')].shape[0]}")

# check how many are under min_chars threshold
print(f"Under 50 chars: {(df['text_length'] < 50).sum()}")
print(f"Over 5000 chars: {(df['text_length'] > 5000).sum()}")

# simulate chunker filters
filtered = df[
    (df["text_length"] >= 50) &
    (~df["page_id"].str.startswith("Modding:")) &
    (~df["page_id"].str.startswith("Module:")) &
    (~df["text"].str.startswith("�PNG"))
]

print('\nAfter applying chunker filters:')
print(f"Original: {len(df)}")
print(f"After filters: {len(filtered)}")
print(f"Removed: {len(df) - len(filtered)}")

# inspecting chunking results
from chunker import load_jsonl_documents, chunk_documents

docs = load_jsonl_documents("/Users/harshpinge/Downloads/Rotman MMA/RSM8430/Startdew_Valley_RAG/data/processed/stardew_wiki_sections.jsonl")
chunks = chunk_documents(docs, strategy="section_recursive")

print(f"Docs loaded: {len(docs)}")
print(f"Chunks after splitting: {len(chunks)}")

# inspect a few chunks
for chunk in chunks[:3]:
    print("---")
    print("page_content:", chunk.page_content[:150])
    print("metadata:", chunk.metadata)