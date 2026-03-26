import re
import os
import json
import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from .schemas import (
    ChatRequest,
    ChatResponse,
    SessionListResponse,
    SessionInfo,
    SessionMessagesResponse,
    MessageInfo,
    SessionDeleteResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentUploadResponse,
    DocumentDeleteResponse,
)
from .agent import chat_with_agent, chat_with_agent_stream, storage
from .document_loader import DocumentLoader
from .parent_chunk_store import ParentChunkStore
# from .milvus_writer import MilvusWriter
# from .milvus_client import MilvusManager
from .chroma_client import ChromaManager
from .embedding import EmbeddingService

from fastapi.responses import Response
from .tts_service import TTSService

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

loader = DocumentLoader()
parent_chunk_store = ParentChunkStore()
# milvus_manager = MilvusManager()
chroma_manager = ChromaManager()
embedding_service = EmbeddingService()
# milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)  # 注释掉Milvus

router = APIRouter()

tts_service = TTSService()

# 添加新的API端点
@router.post("/chat/voice")
async def chat_with_voice(request: ChatRequest):
    """聊天并返回语音响应"""
    try:
        # 先获取文本响应
        resp = chat_with_agent(request.message, request.user_id, request.session_id)
        text_response = resp.get("response", "")
        
        # 转换为语音
        audio_data = tts_service.text_to_speech(text_response, request.speaker_id)
        
        if audio_data:
            return Response(
                content=audio_data,
                media_type="audio/mpeg"
            )
        else:
            return {"error": "语音转换失败"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{user_id}/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(user_id: str, session_id: str):
    """获取指定会话的所有消息"""
    try:
        data = storage._load()
        if user_id not in data or session_id not in data[user_id]:
            return SessionMessagesResponse(messages=[])
        
        session_data = data[user_id][session_id]
        messages = []
        for msg_data in session_data.get("messages", []):
            messages.append(MessageInfo(
                type=msg_data["type"],
                content=msg_data["content"],
                timestamp=msg_data["timestamp"],
                rag_trace=msg_data.get("rag_trace")
            ))
        
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{user_id}", response_model=SessionListResponse)
async def list_sessions(user_id: str):
    """获取用户的所有会话列表"""
    try:
        data = storage._load()
        if user_id not in data:
            return SessionListResponse(sessions=[])
        
        sessions = []
        for session_id, session_data in data[user_id].items():
            sessions.append(SessionInfo(
                session_id=session_id,
                updated_at=session_data.get("updated_at", ""),
                message_count=len(session_data.get("messages", []))
            ))
        
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{user_id}/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(user_id: str, session_id: str):
    """删除指定会话"""
    try:
        deleted = storage.delete_session(user_id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        resp = chat_with_agent(request.message, request.user_id, request.session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """跟 Agent 对话 (流式)"""
    async def event_generator():
        try:
            # chat_with_agent_stream 已经生成了 SSE 格式的字符串 (data: {...}\n\n)
            async for chunk in chat_with_agent_stream(
                request.message, 
                request.user_id, 
                request.session_id
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            # SSE 格式错误
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """获取已上传的文档列表"""
    try:
        # 创建文档元数据存储文件路径
        metadata_file = DATA_DIR / "documents_metadata.json"
        

        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                documents_metadata = json.load(f)
        else:
            documents_metadata = {}
        
        # 获取文档列表
        documents = []
        for filename, metadata in documents_metadata.items():
            documents.append(DocumentInfo(
                filename=filename,
                file_type=metadata.get("file_type", "Unknown"),
                chunk_count=metadata.get("chunk_count", 0)
            ))
        
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档并进行embedding"""
    try:
        filename = file.filename
        file_lower = filename.lower()
        if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))):
            raise HTTPException(status_code=400, detail="仅支持 PDF 和 Word 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # 删除旧的相关数据
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        # 使用Chroma存储文档
        result = chroma_manager.add_documents(leaf_docs)
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Chroma存储失败: {result['error']}")

        metadata_file = DATA_DIR / "documents_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                documents_metadata = json.load(f)
        else:
            documents_metadata = {}
        
        documents_metadata[filename] = {
            "file_type": "PDF" if filename.lower().endswith(".pdf") else "Word",
            "chunk_count": len(leaf_docs),
            "upload_time": datetime.datetime.now().isoformat()
        }
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(documents_metadata, f, ensure_ascii=False, indent=2)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入docstore）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str):
    """从RAG文档库中移除文档（保留本地文件）"""
    try:
# 删除本地文件
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            file_path.unlink()
        result = chroma_manager.delete({"filename": filename})
        parent_chunk_store.delete_by_filename(filename)

        metadata_file = DATA_DIR / "documents_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                documents_metadata = json.load(f)
            
            if filename in documents_metadata:
                del documents_metadata[filename]
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(documents_metadata, f, ensure_ascii=False, indent=2)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=0,
            message=f"成功从RAG文档库中移除文档 {filename}（本地文件已保留）",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
