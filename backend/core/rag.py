"""
The RAG layer.

Interview line, memorise this sequence:
    Load -> Split -> Embed -> Store -> Retrieve -> Generate.

Load happens in parser.py, Generate happens in questions.py. This file owns the
middle four steps.

On Vercel / serverless: ChromaDB needs persistent disk storage between requests,
which serverless does not provide. In that environment we skip the vector store
and return the full resume + JD text directly from the SQLite session (Gemini
2.0 Flash has a 1M token context window, so no chunking is needed).
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import config

_ON_VERCEL = bool(os.getenv("VERCEL"))


def _collection_path(session_id: str) -> str:
    """Each interview gets its own vector store, so sessions never mix."""
    return str(config.CHROMA_DIR / session_id)


def index_documents(session_id: str, resume_text: str, jd_text: str) -> int:
    """Split both documents, embed the chunks, persist them to Chroma.

    On Vercel this is a no-op because /tmp is ephemeral between invocations.
    The text is already persisted in SQLite and retrieve_context reads it there.
    """
    if _ON_VERCEL:
        return 0  # no-op on serverless

    try:
        from langchain_chroma import Chroma
        from backend.llm import get_embeddings

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
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] indexing failed ({exc}); RAG disabled for this session")
        return 0


def retrieve_context(session_id: str, query: str, k: int = None) -> str:
    """Return relevant context for the query.

    On Vercel: reads resume_text + jd_text directly from SQLite.
    Locally: uses the ChromaDB vector store for semantic search.
    """
    if _ON_VERCEL:
        # Import here to avoid circular imports at module load time
        from backend import db
        session = db.get_session(session_id)
        if not session:
            return ""
        resume = session.get("resume_text") or ""
        jd = session.get("jd_text") or ""
        # Return a trimmed version — Gemini 2.0 Flash handles large context easily
        combined = f"=== RESUME ===\n{resume[:4000]}\n\n=== JOB DESCRIPTION ===\n{jd[:2000]}"
        return combined

    k = k or config.RETRIEVE_K
    try:
        from langchain_chroma import Chroma
        from backend.llm import get_embeddings

        store = Chroma(
            persist_directory=_collection_path(session_id),
            embedding_function=get_embeddings(),
        )
        results = store.similarity_search(query, k=k)
        return "\n---\n".join(doc.page_content for doc in results)
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] retrieval failed ({exc}); falling back to session text")
        # Fallback: read from SQLite
        try:
            from backend import db
            session = db.get_session(session_id)
            if session:
                return f"{session.get('resume_text', '')[:3000]}\n{session.get('jd_text', '')[:1500]}"
        except Exception:  # noqa: BLE001
            pass
        return ""


def delete_store(session_id: str) -> bool:
    """
    Remove a session's vector store from disk.

    Interview line: "Every session writes its own Chroma directory, so without
    this the disk grows forever. Cleanup is explicit rather than automatic
    because the store has to outlive the interview - the report is generated
    from it."
    """
    if _ON_VERCEL:
        return True  # nothing to delete

    import shutil

    path = _collection_path(session_id)
    try:
        shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] could not delete store for {session_id} ({exc})")
        return False
