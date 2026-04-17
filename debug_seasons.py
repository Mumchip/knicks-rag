import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection("knicks")

# Get all docs and filter for season summaries
results = col.get(where=None, limit=2313)
season_docs = [(id_, doc) for id_, doc in zip(results["ids"], results["documents"]) if id_.startswith("season_")]
season_docs.sort()
for id_, doc in season_docs[:10]:
    print(f"{id_}: {doc}")
