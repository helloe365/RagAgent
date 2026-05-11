import sqlite3

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from datetime import datetime
import os



class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        """

        def _ensure_md5_table(conn: sqlite3.Connection):
            '''
            确保MD5表存在，并清理旧版本写入的空值脏数据
            '''
            conn.execute(
                f'''
                CREATE TABLE IF NOT EXISTS {chroma_conf["md5_hex_table"]} (
                    md5_hex TEXT PRIMARY KEY,
                    created_tm TEXT NOT NULL
                )
                '''
            )
            conn.execute(
                f'''DELETE FROM {chroma_conf["md5_hex_table"]}
                    WHERE md5_hex IS NULL OR md5_hex = ''
                '''
            )

        def _get_md5_db_connection(md5_db_path: str ):
            conn = sqlite3.connect(md5_db_path)
            _ensure_md5_table(conn)
            return conn

        def check_md5_hex(md5_for_check: str):
            """
            检查传入的md5字符串是否已经被处理过了
                return False(md5未处理过)  True(已经处理过，已有记录）
            """
            if not md5_for_check:
                return False

            md5_db_dir_path = get_abs_path(chroma_conf["md5_db_dir"])
            md5_db_path = os.path.join(md5_db_dir_path, chroma_conf["md5_hex_store"])
            os.makedirs(md5_db_dir_path,exist_ok=True)

            try:
                with _get_md5_db_connection(md5_db_path) as conn:
                    row = conn.execute(
                        f"SELECT 1 FROM {chroma_conf['md5_hex_table']} WHERE md5_hex = ? LIMIT 1",
                        (md5_for_check,),
                    ).fetchone()
                    return row is not None
            except sqlite3.Error:
                logger.error("Failed to check MD5 hex in database", exc_info=True)
                return False

        def save_md5_hex(md5_for_check: str):
            """将传入的md5字符串记录到SQLite数据库"""
            if not md5_for_check:
                logger.warning("[加载知识库]文件md5为空，跳过写入md5数据库")
                return

            md5_db_dir_path = get_abs_path(chroma_conf["md5_db_dir"])
            md5_db_path = os.path.join(md5_db_dir_path, chroma_conf["md5_hex_store"])

            try:
                with _get_md5_db_connection(md5_db_path) as conn:
                    conn.execute(
                        f"INSERT OR IGNORE INTO {chroma_conf['md5_hex_table']} (md5_hex, created_tm) VALUES (?, ?)",
                        (md5_for_check, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    )
                    conn.commit()
            except sqlite3.Error:
                logger.error("Failed to save MD5 hex to database", exc_info=True)

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allow_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf['data_path']),
            tuple(chroma_conf['allow_knowledge_file_type'])
        )

        for path in allow_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.error(f"[加载知识库]{path} md5计算失败，跳过")
                continue
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f'[加载知识库]{path}内没有有效文本内容，跳过')
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f'[加载知识库]{path}分片后没有有效文本内容，跳过')
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f'[加载知识库]{path} 内容加载成功')
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f'[加载知识库]{path}加载失败：{str(e)}', exc_info=True)
                continue

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("="*20)
