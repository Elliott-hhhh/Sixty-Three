# Sixty-Three - 智能文档检索与对话系统

## 项目简介

Sixty-Three是一个基于RAG（检索增强生成）技术的智能文档检索与对话系统。该系统能够上传、处理和检索文档，支持自然语言查询，并提供基于文档内容的智能回答。

## 核心功能

- 📚 **文档管理**：支持上传PDF和Word文档
- 🔍 **智能检索**：基于向量数据库的相似度搜索
- 🤖 **AI对话**：支持基于文档内容的智能问答
- 🎤 **语音合成**：支持文本转语音功能
- 💬 **多轮对话**：支持上下文感知的多轮对话
- 📁 **文档删除**：支持从知识库中移除文档（保留本地文件）

## 技术栈

- **后端**：Python, FastAPI
- **前端**：HTML, CSS, JavaScript
- **向量数据库**：Chroma DB
- **嵌入模型**：支持HuggingFace和OpenAI嵌入模型
- **语言模型**：支持OpenAI兼容的LLM接口

## 项目结构

```
Sixty-Three/
├── backend/              # 后端代码
│   ├── agent.py          # 智能代理
│   ├── api.py            # API路由
│   ├── app.py            # FastAPI应用
│   ├── chroma_client.py  # Chroma向量数据库客户端
│   ├── document_loader.py # 文档加载器
│   ├── embedding.py      # 嵌入服务
│   ├── rag_pipeline.py   # RAG处理流程
│   ├── rag_utils.py      # RAG工具函数
│   ├── tools.py          # 工具函数
│   └── tts_service.py    # 语音合成服务
├── frontend/             # 前端代码
│   ├── index.html        # 主页面
│   ├── script.js         # JavaScript逻辑
│   └── style.css         # 样式文件
├── chroma_db/            # Chroma向量数据库存储
├── data/                 # 数据存储目录
│   ├── documents/        # 上传的文档
│   └── documents_metadata.json # 文档元数据
├── main.py               # 应用入口
├── pyproject.toml        # 项目配置
└── requirements.txt      # 依赖列表
```

## 快速开始

### 环境要求

- Python 3.12+
- uv（推荐）或pip

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd Sixty-Three
```

2. **安装依赖**
```bash
# 使用uv（推荐）
uv sync

# 或使用pip
pip install -r requirements.txt
```

3. **配置环境变量**
创建 `.env` 文件：
```env
# 基础配置
OPENAI_API_KEY=your-api-key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4.1

# 嵌入模型配置
EMBEDDING_MODEL=text-embedding-ada-002
HF_EMBEDDING_MODEL_PATH=moka-ai/m3e-base
EMBEDDING_DEVICE=cuda

# Chroma配置
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=embeddings_collection

# 语音合成配置
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your-amap-key
```

4. **启动服务**
```bash
# 使用uvicorn
uvicorn backend.app:app --reload

# 或使用python
python main.py
```

5. **访问应用**
打开浏览器访问：http://localhost:8000

## 使用指南

### 上传文档

1. 点击"上传文档"按钮
2. 选择PDF或Word文档
3. 等待文档处理完成

### 查询文档

- **普通查询**：直接输入问题，如"什么是人工智能？"
- **指定文档查询**：输入"ci.pdf讲了什么？"，系统会自动识别文件名并过滤检索

### 删除文档

点击文档列表中的删除按钮，系统会：
- 从向量数据库中删除文档的向量数据
- 从文档列表中移除文档
- **保留本地文件**（不会删除原始文件）

## RAG工作流程

1. **文档上传**：文档被分割成三级分块（L1-L3）
2. **向量存储**：文档分块被转换为向量并存储到Chroma数据库
3. **查询处理**：
   - 用户查询被转换为向量
   - 在向量数据库中搜索相似文档
   - 支持文件名过滤
4. **结果处理**：
   - 文档重排和合并
   - 生成最终回答

## API接口

### 文档管理
- `GET /documents` - 获取文档列表
- `POST /documents/upload` - 上传文档
- `DELETE /documents/{filename}` - 删除文档

### 对话接口
- `POST /chat` - 普通对话
- `POST /chat/stream` - 流式对话
- `POST /chat/voice` - 语音对话

### 会话管理
- `GET /sessions/{user_id}` - 获取会话列表
- `GET /sessions/{user_id}/{session_id}` - 获取会话消息
- `DELETE /sessions/{user_id}/{session_id}` - 删除会话

## 配置选项

### 分块配置
- `chunk_size`：分块大小（默认500字符）
- `chunk_overlap`：分块重叠（默认50字符）
- 三级分块策略：L1(1200字符)、L2(600字符)、L3(300字符)

### RAG配置
- `LEAF_RETRIEVE_LEVEL`：叶子节点检索级别（默认3）
- `AUTO_MERGE_ENABLED`：自动合并开关（默认开启）
- `AUTO_MERGE_THRESHOLD`：合并阈值（默认2）

## 故障排除

### 常见问题

1. **文档上传失败**
   - 检查文件格式是否支持（PDF/Word）
   - 检查文件大小是否过大

2. **检索不到文档**
   - 确认文档已成功上传
   - 检查查询关键词是否准确
   - 查看服务器日志获取详细错误信息

3. **向量存储初始化失败**
   - 检查嵌入模型配置
   - 确认CUDA环境是否正常（如果使用GPU）

### 日志查看

```bash
# 查看服务器日志
uvicorn backend.app:app --reload --log-level debug
```

## 部署

### Docker部署

```bash
docker-compose up -d
```


## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请联系项目维护者。
