"""
Persistent Vector Memory Store
================================
ChromaDB-backed persistent memory layer for agent continuity.
Stores: script history, character metadata, image references.

Supports:
  - Agent continuity across sessions
  - Recovery from failure
  - Future personalization
"""

import json
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

import config


class MemoryStore:
    """
    ChromaDB-backed vector memory store.
    Provides persistent storage for scripts, characters, and image references.
    """

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or config.CHROMA_COLLECTION

        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PROJECT MONTAGE Phase 1 memory store"}
        )

        print(f"  [Memory] Initialized ChromaDB at: {self.persist_dir}")
        print(f"  [Memory] Collection: {self.collection_name} ({self.collection.count()} documents)")

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the memory store."""
        # ChromaDB metadata values must be str, int, float, or bool
        safe_metadata = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_metadata[k] = v
                else:
                    safe_metadata[k] = json.dumps(v)

        self.collection.upsert(
            documents=[content],
            ids=[doc_id],
            metadatas=[safe_metadata] if safe_metadata else None
        )

    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query the memory store for relevant documents."""
        where_filter = None
        if filter_type:
            where_filter = {"type": filter_type}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, max(self.collection.count(), 1)),
                where=where_filter if filter_type else None
            )

            documents = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    entry = {
                        "content": doc,
                        "id": results['ids'][0][i] if results['ids'] else None,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results.get('distances') else None
                    }
                    documents.append(entry)

            return documents
        except Exception as e:
            print(f"  [Memory] Query error: {e}")
            return []

    def get_all(self, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all documents, optionally filtered by type."""
        try:
            where_filter = {"type": filter_type} if filter_type else None
            results = self.collection.get(where=where_filter)

            documents = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    entry = {
                        "content": doc,
                        "id": results['ids'][i],
                        "metadata": results['metadatas'][i] if results['metadatas'] else {}
                    }
                    documents.append(entry)
            return documents
        except Exception as e:
            print(f"  [Memory] Error retrieving documents: {e}")
            return []

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )


# ─── Global Memory Store Singleton ──────────────────────────────────────────
_global_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Get or create the global memory store."""
    global _global_store
    if _global_store is None:
        _global_store = MemoryStore()
    return _global_store


def reset_memory_store() -> None:
    """Reset the global memory store (for testing)."""
    global _global_store
    _global_store = None
