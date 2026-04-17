import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection("knicks")
print("Total documents:", col.count())
print()
results = col.peek(5)
for doc in results["documents"]:
    print("-", doc[:120])
