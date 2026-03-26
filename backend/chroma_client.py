import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class ChromaManager:
    """Chroma连接和集合管理"""

    def __init__(self):
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "embeddings_collection")
        # 使用用户配置的模型或默认的嵌入模型
        self.embedding_model = os.getenv("EMBEDDING_MODEL", os.getenv("HF_EMBEDDING_MODEL_PATH", "text-embedding-ada-002"))
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """初始化Chroma向量存储"""
        try:
            # 检查是否使用HuggingFace模型
            hf_model_path = os.getenv("HF_EMBEDDING_MODEL_PATH")
            if hf_model_path:
                # 使用HuggingFace嵌入模型
                model_kwargs = {"device": os.getenv("EMBEDDING_DEVICE", "cuda")}
                encode_kwargs = {"normalize_embeddings": True}
                embeddings = HuggingFaceEmbeddings(
                    model_name=hf_model_path,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
                print(f"使用HuggingFace嵌入模型: {hf_model_path}, 设备: {model_kwargs['device']}")
            else:
                # 使用OpenAI嵌入模型
                embeddings = OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_base=os.getenv("BASE_URL"),
                )
                print(f"使用OpenAI嵌入模型: {self.embedding_model}")
            
            # 创建或加载Chroma向量存储
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
            print(f"Chroma向量存储初始化成功: {self.persist_directory}")
        except Exception as e:
            print(f"Chroma向量存储初始化失败: {e}")
            self.vector_store = None

    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到Chroma"""
        if not self.vector_store:
            self._init_vector_store()
            if not self.vector_store:
                return {"error": "Chroma向量存储未初始化"}
        
        try:
            # 将字典转换为文本和元数据
            texts = []
            metadatas = []
            for doc in documents:
                texts.append(doc.get("text", ""))
                metadata = {
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                metadatas.append(metadata)
            
            # 添加文档到Chroma
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
           
            return {"success": True, "count": len(texts)}
        except Exception as e:
            return {"error": str(e)}

    def retrieve(self, query: str, top_k: int = 5, filter_expr: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """检索文档"""
        if not self.vector_store:
            self._init_vector_store()
            if not self.vector_store:
                return []
        
        try:
            # 如果有过滤条件，转换为Chroma的过滤格式
            where_filter = None
            if filter_expr:
                conditions = []
                if "chunk_level" in filter_expr:
                    conditions.append({"chunk_level": {"$eq": filter_expr["chunk_level"]}})
                if "filename" in filter_expr:
                    conditions.append({"filename": {"$eq": filter_expr["filename"]}})
                
                if conditions:
                    if len(conditions) == 1:
                        where_filter = conditions[0]
                    else:
                        where_filter = {"$and": conditions}
            

            results_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=where_filter,
            )
            
            # 格式化结果
            formatted_results = []
            for i, (doc, score) in enumerate(results_with_scores):
                formatted_results.append({
                    "id": i,
                    "text": doc.page_content,
                    "filename": doc.metadata.get("filename", ""),
                    "file_type": doc.metadata.get("file_type", ""),
                    "page_number": doc.metadata.get("page_number", 0),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "parent_chunk_id": doc.metadata.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.metadata.get("root_chunk_id", ""),
                    "chunk_level": doc.metadata.get("chunk_level", 0),
                    "chunk_idx": doc.metadata.get("chunk_idx", 0),
                    "score": float(score),  # 使用Chroma返回的真实相似度分数
                })
            
            return formatted_results
        except Exception as e:
            print(f"检索失败: {e}")
            return []

    def delete(self, filter_expr: Dict[str, Any]) -> bool:
        """删除文档"""
        if not self.vector_store:
            return False
        
        try:
            where_filter = {}
            if "filename" in filter_expr:
                where_filter["filename"] = filter_expr["filename"]
            
            self.vector_store.delete(where=where_filter)
            return True
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False

    def clear_collection(self):
        """清空集合"""
        if not self.vector_store:
            return False
        
        try:
            # 删除并重新创建集合
            self.vector_store.delete_collection()
            self._init_vector_store()
            return True
        except Exception:
            return False
