import hashlib
import json
import sqlite3

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import get_embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from datetime import datetime
import os


_SOURCE_INDEX_TABLE = "source_index"


def _ensure_source_index_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_SOURCE_INDEX_TABLE} (
            source_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            content_version TEXT NOT NULL,
            chunk_ids_json TEXT NOT NULL,
            stale_chunk_ids_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _open_source_index(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    _ensure_source_index_table(conn)
    return conn


def _decode_chunk_ids(value: str | None) -> list[str]:
    try:
        decoded = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [chunk_id for chunk_id in decoded if isinstance(chunk_id, str)]


def _get_file_documents(read_path: str) -> list[Document]:
    if read_path.endswith(".txt"):
        return txt_loader(read_path)
    if read_path.endswith(".pdf"):
        return pdf_loader(read_path)
    return []



class VectorStoreService:
    def __init__(self):
        persist_directory = get_abs_path(chroma_conf["persist_directory"])
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=get_embed_model(),
            persist_directory=persist_directory,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _reconcile_stale_chunks(
        self, conn: sqlite3.Connection, source_id: str, source_path: str, stale_chunk_ids: list[str]
    ) -> bool:
        if not stale_chunk_ids:
            return True
        try:
            self.vector_store.delete(ids=stale_chunk_ids)
        except Exception as exc:
            logger.error(
                "[加载知识库] source=%s stage=reconcile_stale error=%s",
                source_path,
                type(exc).__name__,
            )
            return False
        with conn:
            conn.execute(
                f"UPDATE {_SOURCE_INDEX_TABLE} SET stale_chunk_ids_json = ? WHERE source_id = ?",
                ("[]", source_id),
            )
        return True

    def load_document(self):
        """Index each supported source with deterministic IDs and a SQLite manifest."""
        data_path = get_abs_path(chroma_conf["data_path"])
        db_dir = get_abs_path(chroma_conf["md5_db_dir"])
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, chroma_conf["md5_hex_store"])
        allow_files_path = listdir_with_allowed_type(
            data_path, tuple(chroma_conf["allow_knowledge_file_type"])
        )

        conn = _open_source_index(db_path)
        try:
            for path in allow_files_path:
                source_path = os.path.relpath(path, data_path).replace("\\", "/")
                source_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
                content_version = get_file_md5_hex(path)
                if not content_version:
                    logger.error("[加载知识库] source=%s stage=hash error=unavailable", source_path)
                    continue

                row = conn.execute(
                    f"SELECT content_version, chunk_ids_json, stale_chunk_ids_json "
                    f"FROM {_SOURCE_INDEX_TABLE} WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
                previous_ids = _decode_chunk_ids(row[1]) if row else []
                stale_ids = _decode_chunk_ids(row[2]) if row else []
                if not self._reconcile_stale_chunks(conn, source_id, source_path, stale_ids):
                    continue
                if row and row[0] == content_version:
                    continue

                try:
                    documents = _get_file_documents(path)
                    if not documents:
                        logger.warning("[加载知识库] source=%s stage=load no_documents", source_path)
                        continue
                    split_documents = self.spliter.split_documents(documents)
                    if not split_documents:
                        logger.warning("[加载知识库] source=%s stage=split no_chunks", source_path)
                        continue
                except Exception as exc:
                    logger.error(
                        "[加载知识库] source=%s stage=load error=%s", source_path, type(exc).__name__
                    )
                    continue

                chunk_ids = [f"{source_id}:{content_version}:{index}" for index in range(len(split_documents))]
                indexed_documents = [
                    Document(
                        page_content=document.page_content,
                        metadata={
                            **document.metadata,
                            "source_id": source_id,
                            "source_path": source_path,
                            "content_version": content_version,
                            "chunk_index": index,
                        },
                    )
                    for index, document in enumerate(split_documents)
                ]
                try:
                    self.vector_store.add_documents(indexed_documents, ids=chunk_ids)
                except Exception as exc:
                    logger.error(
                        "[加载知识库] source=%s stage=add error=%s", source_path, type(exc).__name__
                    )
                    continue

                with conn:
                    conn.execute(
                        f"""
                        INSERT INTO {_SOURCE_INDEX_TABLE} (
                            source_id, source_path, content_version, chunk_ids_json,
                            stale_chunk_ids_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source_id) DO UPDATE SET
                            source_path = excluded.source_path,
                            content_version = excluded.content_version,
                            chunk_ids_json = excluded.chunk_ids_json,
                            stale_chunk_ids_json = excluded.stale_chunk_ids_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            source_id,
                            source_path,
                            content_version,
                            json.dumps(chunk_ids),
                            json.dumps(previous_ids),
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )

                if self._reconcile_stale_chunks(conn, source_id, source_path, previous_ids):
                    logger.info("[加载知识库] source=%s indexed", source_path)
        finally:
            conn.close()

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("="*20)
