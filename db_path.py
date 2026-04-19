import os
# Use absolute path so it works regardless of working directory
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
