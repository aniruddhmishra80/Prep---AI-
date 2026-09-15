"""
The RAG layer.

Interview line, memorise this sequence:
    Load -> Split -> Embed -> Store -> Retrieve -> Generate.

Load happens in parser.py, Generate happens in questions.py. This file owns the
middle four steps.

Why chunk overlap 50? So a sentence cut at a chunk boundary still survives
intact inside at least one chunk.

Why retrieve instead of pasting the whole resume into the prompt? For a 2-page
resume you honestly could paste it. Retrieval keeps the generator focused on the
relevant section instead of diluting attention across the whole document, and it
is the thing that still works when the JD is long or there are several documents.
"""

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import config
from backend.llm import get_embeddings


def _collection_path(session_id: str) -> str:
    """Each interview gets its own vector store, so sessions never mix."""
    return str(config.CHROMA_DIR / session_id)


def index_documents(session_id: str, resume_text: str, jd_text: str) -> int:
    """Split both documents, embed the chunks, persist them to Chroma."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    documents: List[Document] = []
    for source, text in (("resume", resume_text), ("jd", jd_text)):
        for chunk in splitter.split_text(text or ""):
            documents.append(Document(page_content=chunk, metadata={"source": source}))

    if not documents:
        return 0

    Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=_collection_path(session_id),
    )
    return len(documents)


def retrieve_context(session_id: str, query: str, k: int = None) -> str:
    """Return the k chunks most similar to the query, joined into one string."""
    k = k or config.RETRIEVE_K
    try:
        store = Chroma(
            persist_directory=_collection_path(session_id),
            embedding_function=get_embeddings(),
        )
        results = store.similarity_search(query, k=k)
        return "\n---\n".join(doc.page_content for doc in results)
    except Exception as exc:                       # noqa: BLE001
        print(f"[rag] retrieval failed ({exc})")
        return ""


def delete_store(session_id: str) -> bool:
    """
    Remove a session's vector store from disk.

    Interview line: "Every session writes its own Chroma directory, so without
    this the disk grows forever. Cleanup is explicit rather than automatic
    because the store has to outlive the interview - the report is generated
    from it."
    """
    import shutil

    path = _collection_path(session_id)
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:                       # noqa: BLE001
        print(f"[rag] could not delete store for {session_id} ({exc})")
        return False
