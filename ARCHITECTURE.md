# Architecture Document

## Table of Contents
1. [Component Architecture](#component-architecture)
2. [Data Flow](#data-flow)
3. [Design Choices and Trade-offs](#design-choices-and-trade-offs)
4. [Anti-Hallucination Logic](#anti-hallucination-logic)
5. [Scaling to Production](#scaling-to-production)
6. [Safety and Validation Evolution](#safety-and-validation-evolution)



### Core Capabilities
- **Document Upload & Processing**: Accepts markdown documents, chunks them, and generates embeddings
- **Vector Storage**: Persists embeddings in ChromaDB for semantic search
- **RAG Query Pipeline**: Retrieves relevant context and generates answers using LLM with tool access
- **Anti-Hallucination Validation**: Validates LLM responses against retrieved context using similarity scoring

### Technology Stack
- **Web Framework**: FastAPI (0.121.1)
- **Vector Database**: ChromaDB (1.3.4)
- **Relational Database**: SQLite (via SQLModel 0.0.27)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **LLM Provider**: OpenRouter (via ChatOpenAI interface)
- **Agent Framework**: LangGraph (1.0.3) with LangChain (1.0.5)
- **Text Processing**: LangChain Text Splitters (1.0.0)

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                    │
│                         (main.py)                           │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼──────────┐              ┌──────────▼──────────┐
│  Documents Router│              │   Query Router      │
│  (upload)        │              │   (query)           │
└───────┬──────────┘              └──────────┬──────────┘
        │                                    │
        │                                    │
┌───────▼──────────────────┐      ┌─────────▼──────────────────┐
│ Document Pipeline        │      │ RAG Pipeline               │
│ (Chain of Responsibility)│      │ (Chain of Responsibility)  │
├──────────────────────────┤      ├────────────────────────────┤
│ 1. SaveToSQLHandler      │      │ 1. RetrieveHandler         │
│ 2. ChunkHandler          │      │ 2. ContextAugmentHandler   │
│ 3. EmbedHandler          │      │ 3. ToolHandler             │
│ 4. SaveToChromaHandler   │      │ 4. LLMHandler              │
└──────┬───────────────────┘      │ 5. ValidationHandler       │
       │                          │ 6. ResponseHandler         │
       │                          └─────────┬──────────────────┘
       │                                    │
   ┌───▼──────────┐                    ┌───▼──────────┐
   │ SQLite DB    │                    │ ChromaDB     │
   │ (Metadata)   │                    │ (Embeddings) │
   └──────────────┘                    └──────────────┘
```

### 1. Routers Layer (`routers/`)

**Purpose**: Organize API endpoints by domain

#### Documents Router (`routers/documents.py`)
- **Endpoint**: `POST /upload`
- **Responsibility**: Accept file uploads and orchestrate document processing pipeline
- **Error Handling**: Catches exceptions and returns structured error responses

#### Query Router (`routers/query.py`)
- **Endpoint**: `POST /query`
- **Responsibility**: Accept user queries and execute RAG pipeline
- **Input Schema**: `QueryRequest` (query string, optional top_k)
- **Output Schema**: `QueryResponse` (answer, context, refusal status)

### 2. Services Layer (`services/`)

#### Document Pipeline (`services/document_pipeline.py`)

**Pattern**: Chain of Responsibility

Processes uploaded documents through sequential handlers:

1. **SaveToSQLHandler**
   - Supposed to keep track of files uploaded in disk and chroma to give user the CRUD ability
   - Saves document metadata (filename, URL, timestamp) to SQLite
   - Assigns unique document ID
   - Updates context with `id`, `url`, and `file_path`

2. **ChunkHandler**
   - Reads document content from disk
   - Splits markdown using header-based strategy (H1, H2)
   - Stores chunks as LangChain `Document` objects
   - **Why markdown splitting**: Preserves semantic structure; headers indicate topic boundaries

3. **EmbedHandler**
   - Generates embeddings for each chunk using `all-MiniLM-L6-v2`
   - Yields embeddings lazily to optimize memory
   - Converts tensors to lists for ChromaDB compatibility

4. **SaveToChromaHandler**
   - Persists embeddings to ChromaDB `default` collection
   - Stores metadata: `db_id` (references SQL document ID)
   - Assigns unique IDs: `{filename}_chunk_{index}`

**Context Object**: `DocumentContext` dataclass carries state through pipeline (id, url, file_path, chunks, embeddings)

#### RAG Pipeline (`services/rag_pipeline.py`)

**Pattern**: Chain of Responsibility

Executes retrieval-augmented generation with validation:

1. **RetrieveHandler**
   - Encodes query using `all-MiniLM-L6-v2`
   - Queries ChromaDB for top-k similar chunks (default: 5)
   - Sets `related_data_found` flag based on results

2. **ContextAugmentHandler**
   - Constructs prompt with retrieved chunks as context
   - Adds strict instructions: "Use context strictly, say 'I don't know' if not in context"
   - Includes additional warning if `related_data_found=False`
   - **Purpose**: Reduce hallucination through explicit constraints

3. **ToolHandler**
   - Provides tools for LLM agent:
     - `time_tool_func`: Returns current UTC time
     - `user_info`: Returns hardcoded user profile (demo)
     - `retrieve_more_data`: Allows agent to retrieve additional documents
   - **Why tools**: Enables agent to fetch real-time data and expand context

4. **LLMHandler**
   - Creates LangChain agent with `ChatOpenRouter` model
   - System prompt: "You are here to help user get what ever data they need"
   - Invokes agent with augmented prompt
   - **LLM Provider**: OpenRouter API (flexible model selection)

5. **ValidationHandler** (Anti-Hallucination Node)
   - **Threshold**: 0.35 (cosine similarity) This might be better around 0.5 but for the test purposes i kept it at 0.35
   - **Process**:
     1. Splits LLM answer into sentences
     2. Generates embeddings for answer sentences and retrieved chunks
     3. Computes max cosine similarity for each sentence against chunks
     4. Sets `valid=True` if max similarity ≥ threshold
   - **Why 0.35**: Balances false positives (rejecting valid answers) vs false negatives (accepting hallucinations)
   - **Model**: Uses `all-MiniLM-L6-v2` (same as retrieval for consistency)

6. **ResponseHandler**
   - Formats final response based on validation result
   - If `valid=False`: Returns refusal message with retrieved context
   - If `valid=True`: Returns LLM answer with context

**Context Object**: Dictionary carries state through pipeline (query, top_k, retrieved_chunks, prompt, tools, raw_answer, valid, response)

#### File Service (`services/file_service.py`)

- **`upload_file`**: Saves uploaded file to `upload/` directory
- **`load_file_content`**: Reads file content with specified encoding
- **Design**: Static methods for stateless operations

#### Chroma Document Service (`services/chroma_document_service.py`)

- **`save_document`**: Adds documents with embeddings to collection
- **`delete_document`**: Removes documents by metadata filter
- **`retrieve_top_k`**: Queries similar documents by embedding
- **Design**: SQLModel schema with business logic methods

### 3. Models Layer (`models/`)

#### Database Configuration (`models/db.py`)

**SQL Engine**:
```python
engine = create_engine("sqlite:///./db.sqlite3")
```
- **Choice**: SQLite for simplicity in development and testing this project. If it was a real world project i would choose mongo or postgres based on whether how much data we will be dealing with.
- **Production Consideration**: Migrate to PostgreSQL for concurrent writes

**Chroma Client**:
```python
client = chromadb.PersistentClient(path="./chroma_db")
```
- **Choice**: Persistent storage (disk-based) for durability.
- **Why Not FIASS**: In this particular project for the simplicity.But if it was a real world project based on how much performance and simplicity trade offs we can accept as the team and company i would choose between FIASS and Chroma. 
- **Configuration**: Telemetry disabled for privacy

**Dependency Injection**:
- `get_db()`: Yields SQLModel session for request scope
- `get_chroma_client()`: Returns persistent ChromaDB client
- `get_or_create_chroma_collection()`: Ensures collection exists

#### Document Model (`models/document.py`)

```python
class DocumentModel(SQLModel, table=True):
    id: Optional[int]  # Primary key
    file_name: str
    url: str  # File path on disk
    uploaded_at: datetime  # Auto-generated timestamp
```
- **Purpose**: Store document metadata for tracking and retrieval
- **SQLModel**: Combines Pydantic validation with SQLAlchemy ORM

### 4. CRUD Layer (`crud/`)

#### Document CRUD (`crud/document.py`)

- **`create`**: Inserts document, flushes for ID generation, refreshes instance
- **`get`**: Retrieves document by ID
- **`list`**: Returns all documents
- **Design**: Static methods for stateless data access

### 5. Schemas Layer (`schemas/`)

#### Base Schemas (`schemas/base.py`)

- **`ResponseDTO[DataT]`**: Generic response wrapper (success, message, data)
- **`EmptyResponseDTO`**: Response without data payload
- **`MessageDTO`**: Structured messages (type: SUCCESS/ERROR/WARNING/INFO, text)
- **`PaginatedResponseDto[DataT]`**: Paginated responses (page, page_size, total)
- **`PaginatedQueryParams`**: Pagination parameters (page, page_size)

#### Query Schemas (`schemas/query.py`)

- **`QueryRequest`**: Query input (query: str, top_k: int = 5)
- **`QueryResponse`**: Query output (answer, context_used, refused, message)

### 6. Utils Layer (`utils/`)

#### Embedding (`utils/embedding.py`)

```python
def encode_to_embedding(document: Document | str) -> Tensor
```
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Input**: LangChain Document or string
- **Output**: PyTorch Tensor
- **Why all-MiniLM-L6-v2**:
  - 384-dimensional embeddings (compact)
  - Fast inference (22M parameters)
  - Good quality for general-domain retrieval

#### Markdown Splitter (`utils/markdown_splitter.py`)

```python
def split_markdown(markdown_content: str) -> List[Document]
```
- **Strategy**: Header-based splitting (H1, H2)
- **Why headers**: Semantic boundaries align with topic changes
- **Output**: LangChain Documents with preserved metadata

#### Tools (`utils/tools.py`)

```python
def get_current_time() -> str
```
- Returns ISO 8601 timestamp with 'Z' suffix (UTC indicator)

### 7. Configuration (`conf.py`)

```python
class Settings:
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "upload"
```
- **Bootstrap**: Auto-creates upload directory
- **PATH Management**: Adds BASE_DIR to sys.path for imports

### 8. Database Migrations (`alembic/`)

- **Tool**: Alembic (SQLAlchemy migration tool)
- **Current Revision**: `8d1356115b3d` (initial schema)
- **Metadata Source**: `SQLModel.metadata` for auto-generation
- **Modes**: Online (with DB connection) and offline (SQL generation)

---

## Data Flow

### Upload Flow

```
1. Client uploads file via POST /upload
        ↓
2. FileService saves to disk (upload/{filename})
        ↓
3. SaveToSQLHandler inserts metadata → db.sqlite3
        ↓
4. ChunkHandler reads file and splits by markdown headers
        ↓
5. EmbedHandler generates embeddings (all-MiniLM-L6-v2)
        ↓
6. SaveToChromaHandler persists to chroma_db/
        ↓
7. Returns EmptyResponseDTO with success message
```

### Query Flow

```
1. Client sends query via POST /query
        ↓
2. RetrieveHandler:
   - Encode query → embedding
   - ChromaDB similarity search → top-k chunks
        ↓
3. ContextAugmentHandler:
   - Build prompt with chunks
   - Add strict instructions
        ↓
4. ToolHandler:
   - Attach tools (time, user_info, retrieve_more_data)
        ↓
5. LLMHandler:
   - Create LangChain agent
   - Invoke with augmented prompt
   - Agent may use tools during reasoning
        ↓
6. ValidationHandler:
   - Split answer into sentences
   - Compute similarity with retrieved chunks
   - Check if max similarity ≥ 0.35
        ↓
7. ResponseHandler:
   - If valid: return answer + context
   - If invalid: return refusal message
        ↓
8. Returns QueryResponse to client
```

---

## Design Choices and Trade-offs

### 1. Why FastAPI?

**Chosen**: FastAPI 0.121.1

**Reasons**:
- **Performance**: ASGI-based async framework (high concurrency)
- **Type Safety**: Pydantic validation with automatic OpenAPI docs
- **Developer Experience**: Auto-generated interactive docs (`/docs`, `/redoc`)
- **Dependency Injection**: Clean pattern for database sessions and services

**Trade-offs**:
- **Learning Curve**: Async/await requires understanding of concurrency
- **Ecosystem Maturity**: Smaller ecosystem than Flask/Django (acceptable for modern projects)

### 2. Why ChromaDB?

**Chosen**: ChromaDB 1.3.4 (PersistentClient)

**Reasons**:
- **Simplicity**: Embedded database, no separate server required
- **Disk Persistence**: Data survives application restarts
- **Feature-Rich**: Supports metadata filtering, multiple collections
- **Local Development**: No infrastructure dependencies

**Trade-offs**:
- **Scalability**: Single-process, not suitable for high concurrency at scale
- **No Distributed Storage**: Cannot shard across nodes
- **Performance**: Slower than in-memory solutions (e.g., FAISS)

**Why Not FAISS**:
- FAISS (Facebook AI Similarity Search) is faster but:
  - **In-Memory Only**: Requires manual persistence layer
  - **No Metadata**: Must build separate metadata store
  - **Complexity**: Lower-level API, more boilerplate
  - **Trade-off**: ChromaDB chosen for ease of development; FAISS better for production scale

### 3. Embedding Model: all-MiniLM-L6-v2

**Reasons**:
- **Size**: 384 dimensions (vs 768 for base models) = 50% storage savings
- **Speed**: 22M parameters → fast inference on CPU
- **Quality**: SBERT-trained, good for semantic similarity tasks
- **General-Domain**: Works well without fine-tuning

**Trade-offs**:
- **Domain-Specific Tasks**: May underperform vs fine-tuned models (e.g., medical, legal)
- **Context Length**: 256 tokens max → requires chunking for long documents

### 4. Chain of Responsibility Pattern

**Applied**: Document Pipeline, RAG Pipeline

**Reasons**:
- **Modularity**: Each handler has single responsibility
- **Extensibility**: Easy to add/remove/reorder handlers
- **Testability**: Handlers can be tested in isolation
- **Context Passing**: Shared context object flows through chain

**Trade-offs**:
- **Debugging**: Errors in chain require tracing through multiple handlers
- **Performance**: Small overhead from handler dispatch (negligible in I/O-bound tasks)

### 5. SQLite for Metadata

**Reasons**:
- **Zero Config**: No separate database server
- **ACID Transactions**: Reliable for metadata consistency
- **File-Based**: Portable, easy to backup

**Trade-offs**:
- **Concurrency**: Write locks can cause contention
- **Production**: Should migrate to PostgreSQL for concurrent writes

### 6. LangGraph Agent Framework

**Reasons**:
- **Tool Use**: LLM can call functions (time, retrieval) during reasoning
- **Stateful**: Maintains conversation context across turns (future extension)
- **Observability**: LangSmith integration for debugging

**Trade-offs**:
- **Complexity**: Higher abstraction than direct LLM calls
- **Latency**: Tool calls add round-trips

### 7. Markdown Header-Based Chunking

**Reasons**:
- **Semantic Boundaries**: Headers indicate topic changes
- **Metadata Preservation**: Can track section hierarchy
- **Better Context**: Chunks are self-contained topics

**Trade-offs**:
- **Markdown-Only**: Requires markdown format (not PDF, DOCX, etc.)
- **Variable Chunk Size**: Headers may create very large/small chunks

---

## Anti-Hallucination Logic

### Validation Mechanism

The system implements a **post-generation validation layer** to detect and reject hallucinated answers.

#### Implementation: `ValidationHandler`

**Location**: `services/rag_pipeline.py:118-151`

#### Algorithm

1. **Input**:
   - `raw_answer`: LLM-generated response
   - `retrieved_chunks`: Top-k chunks from ChromaDB

2. **Sentence Extraction**:
   ```python
   sentences = [s.strip() for s in answer_text.replace("\n", ". ").split(".") if s.strip()]
   ```
   - Splits answer into sentences
   - Normalizes newlines to periods

3. **Embedding Generation**:
   ```python
   emb_sentences = st_model.encode(sentences, normalize_embeddings=True)
   emb_chunks = st_model.encode(chunks, normalize_embeddings=True)
   ```
   - Uses `all-MiniLM-L6-v2` (same as retrieval)
   - Normalizes embeddings for cosine similarity

4. **Similarity Computation**:
   ```python
   sims = [np.max([np.dot(sent_emb, chunk_emb) for chunk_emb in emb_chunks])
           for sent_emb in emb_sentences]
   ```
   - For each sentence, compute cosine similarity with all chunks
   - Take maximum similarity (best match)

5. **Thresholding**:
   ```python
   ctx["valid"] = max(sims) >= 0.35
   ```
   - Answer is valid if **any** sentence has similarity ≥ 0.35
   - **Threshold Explanation**:
     - **0.35**: Empirically chosen balance
     - **Lower** (e.g., 0.2): More permissive, risk of false positives (hallucinations pass)
     - **Higher** (e.g., 0.5): Stricter, risk of false negatives (valid answers rejected)

6. **Output**:
   - `valid=True`: Answer is grounded in retrieved context
   - `valid=False`: Answer likely hallucinated, return refusal

#### Refusal Behavior

```python
if not ctx["valid"]:
    ctx["response"] = {
        "answer": None,
        "context_used": ctx["retrieved_chunks"],
        "refused": True,
        "message": "I don't have enough information to answer that reliably."
    }
```

**User Experience**:
- Transparent: User knows answer was rejected
- Context Provided: Retrieved chunks shown for manual review
- Safe Default: No misinformation propagated

#### Limitations

1. **Paraphrasing**: LLM may rephrase context (correct but low similarity)
2. **Multi-Hop Reasoning**: Combining multiple chunks may reduce similarity
3. **Sentence Granularity**: Validates sentences independently (not logical consistency)
4. **Threshold Tuning**: 0.35 may not generalize across domains

---

## Scaling to Production

### Current Architecture Limitations

1. **Single-Process ChromaDB**: Cannot handle high query concurrency
2. **SQLite Write Locks**: Bottleneck for concurrent uploads
3. **No Caching**: Repeated queries re-compute embeddings and retrieve from DB
4. **Synchronous Embedding**: CPU-bound task blocks event loop
5. **No Observability**: Limited logging, metrics, and tracing

### Scaling Recommendations

#### 1. Vector Database Sharding

**Problem**: ChromaDB single-node cannot scale beyond ~100 QPS

**Solutions**:

- **Option A: Migrate to Qdrant/Weaviate/Milvus**
  - Distributed vector databases with horizontal scaling
  - **Qdrant**: Rust-based, high performance, RESTful API
  - **Weaviate**: Built-in semantic search, auto-sharding
  - **Milvus**: Mature, supports GPU indexing, S3 storage

- **Option B: Client-Side Sharding**
  - Partition documents across multiple ChromaDB instances
  - Use consistent hashing on `document_id`
  - **Trade-off**: Complex query routing, no global top-k

**Implementation**:
```python
# Example: Qdrant client with sharding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://qdrant-cluster:6333")
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    shard_number=4  # Distribute across 4 shards
)
```

#### 2. PostgreSQL Migration

**Problem**: SQLite write locks serialize uploads

**Solution**:
```python
# models/db.py
engine = create_engine(
    "postgresql://user:password@postgres:5432/vardast",
    pool_size=20,  # Connection pool
    max_overflow=10
)
```

**Benefits**:
- MVCC (Multi-Version Concurrency Control): No write locks
- Replication: Read replicas for query scaling
- Extensions: pg_vector for hybrid search (SQL + vector)

#### 3. Background Job Queue

**Problem**: Upload pipeline blocks HTTP response

**Solution**: Celery + Redis for async processing

```python
# workers/tasks.py
from celery import Celery

celery_app = Celery('vardast', broker='redis://localhost:6379/0')

@celery_app.task
def process_document_async(file_path: str, doc_id: int):
    # Run pipeline in background
    proceess_document_pipeline.handle(session, file_obj)

# routers/documents.py
@router.post("/upload")
async def upload_files(...):
    file_path = FileService.upload_file(file)
    doc = DocumentCRUD.create(session, file.filename, str(file_path))
    process_document_async.delay(str(file_path), doc.id)  # Async
    return {"status": "processing", "doc_id": doc.id}
```

**Benefits**:
- Non-blocking uploads (respond immediately)
- Retry logic for failed embeddings
- Rate limiting to avoid resource exhaustion

#### 4. Caching Layer

**Problem**: Repeated queries waste compute (embedding + retrieval)

**Solution**: Redis cache for query results

```python
# services/cache.py
import redis
import hashlib

cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_cached_query(query: str, top_k: int) -> dict | None:
    cache_key = hashlib.sha256(f"{query}:{top_k}".encode()).hexdigest()
    result = cache.get(cache_key)
    return json.loads(result) if result else None

def set_cached_query(query: str, top_k: int, result: dict, ttl=3600):
    cache_key = hashlib.sha256(f"{query}:{top_k}".encode()).hexdigest()
    cache.setex(cache_key, ttl, json.dumps(result))
```

**Cache Strategy**:
- **TTL**: 1 hour (documents don't change frequently)
- **Invalidation**: Clear cache on document upload
- **Hit Rate**: Monitor with Redis INFO stats

#### 5. Async Embedding Service

**Problem**: CPU-bound embedding blocks FastAPI event loop

**Solution**: Dedicated embedding microservice with GPU

```python
# embedding_service/main.py (Separate service)
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

@app.post("/embed")
async def embed(texts: List[str]):
    embeddings = model.encode(texts, batch_size=32)
    return {"embeddings": embeddings.tolist()}

# Main service calls embedding service
async def encode_to_embedding_remote(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://embedding-service:8001/embed",
            json={"texts": [text]}
        )
        return response.json()["embeddings"][0]
```

**Benefits**:
- GPU acceleration (10-100x faster)
- Horizontal scaling (multiple embedding workers)
- Non-blocking (async HTTP calls)

#### 6. Observability Stack

**Components**:

- **Logging**: Structured JSON logs (Loguru)
  ```python
  from loguru import logger
  logger.add("vardast.log", rotation="500 MB", format="{time} {level} {message}")
  ```

- **Metrics**: Prometheus + Grafana
  ```python
  from prometheus_fastapi_instrumentator import Instrumentator
  Instrumentator().instrument(app).expose(app)
  ```

- **Tracing**: OpenTelemetry + Jaeger
  ```python
  from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
  FastAPIInstrumentor.instrument_app(app)
  ```

**Key Metrics**:
- **Upload Pipeline**: Duration per handler, failure rate
- **RAG Pipeline**: Retrieval latency, validation rejection rate
- **ChromaDB**: Query latency, index size
- **LLM**: Token usage, response time

#### 7. Load Balancing & Auto-Scaling

**Architecture**:
```
[Client] → [Load Balancer (Nginx)] → [FastAPI Pod 1]
                                    → [FastAPI Pod 2]
                                    → [FastAPI Pod N]
                                         ↓
                            [PostgreSQL] + [Qdrant Cluster]
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vardast-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: vardast:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
        env:
        - name: DB_URL
          value: "postgresql://postgres:5432/vardast"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vardast-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vardast-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### 8. Data Partitioning Strategy

**Scenario**: 10M+ documents

**Solution**: Partition by collection

```python
# Partition by date or tenant
collection_name = f"docs_{tenant_id}_{year}_{month}"

# Query routing
def get_collection_for_query(tenant_id: str, date_range: tuple):
    collections = []
    for year, month in date_range:
        collections.append(f"docs_{tenant_id}_{year}_{month}")
    return collections

# Multi-collection search
results = []
for collection in collections:
    results.extend(retrieve_from_collection(collection, query, top_k))
results = sorted(results, key=lambda x: x.similarity)[:top_k]
```

---

## Safety and Validation Evolution

### Current Limitations

1. **Binary Decision**: Valid or invalid (no confidence score)
2. **Single Threshold**: 0.35 may not fit all query types
3. **Sentence-Level**: Ignores logical consistency across sentences
4. **No Uncertainty Quantification**: User doesn't know "how confident" the system is

### Evolution Roadmap

#### Phase 1: Confidence Scoring

**Implementation**:
```python
class ValidationHandler(BaseRagHandler):
    async def handle(self, ctx: dict):
        # Compute sentence-level similarities
        sims = [...]
        
        # Aggregate into confidence score
        ctx["confidence"] = {
            "score": np.mean(sims),  # Average similarity
            "min_similarity": min(sims),
            "max_similarity": max(sims),
            "sentence_count": len(sims)
        }
        
        # Dynamic threshold based on query complexity
        threshold = self.get_adaptive_threshold(ctx)
        ctx["valid"] = ctx["confidence"]["score"] >= threshold
        
        return await super().handle(ctx)
    
    def get_adaptive_threshold(self, ctx: dict):
        # Lower threshold for complex queries (more leniency)
        query_length = len(ctx["query"].split())
        if query_length > 20:
            return 0.25
        elif query_length > 10:
            return 0.30
        else:
            return 0.35
```

**User-Facing**:
```json
{
  "answer": "The capital of France is Paris.",
  "confidence": {
    "score": 0.87,
    "label": "HIGH"
  },
  "context_used": ["Paris is the capital...", ...]
}
```

#### Phase 2: Model-Critic Pipeline

**Architecture**: Two-model system (Generator + Critic)

```python
class CriticHandler(BaseRagHandler):
    def __init__(self, critic_model, next_handler=None):
        super().__init__(next_handler)
        self.critic = critic_model  # Separate LLM for evaluation
    
    async def handle(self, ctx: dict):
        answer = ctx["raw_answer"]
        chunks = ctx["retrieved_chunks"]
        
        # Critic prompt
        critic_prompt = f"""
        Evaluate if the following answer is supported by the context.
        
        Context:
        {chunks}
        
        Answer:
        {answer}
        
        Respond with:
        1. Supported: YES/NO
        2. Confidence: 0-100
        3. Reasoning: Brief explanation
        
        Format: {{"supported": "YES", "confidence": 95, "reasoning": "..."}}
        """
        
        critic_response = await self.critic.ainvoke(critic_prompt)
        critique = json.loads(critic_response.content)
        
        ctx["valid"] = critique["supported"] == "YES"
        ctx["critic_confidence"] = critique["confidence"]
        ctx["critic_reasoning"] = critique["reasoning"]
        
        return await super().handle(ctx)
```

**Benefits**:
- **Explainability**: Critic provides reasoning for rejection
- **Nuanced**: Not just yes/no, but confidence + explanation
- **Adversarial**: Harder to fool two models than one

**Trade-offs**:
- **Latency**: 2x LLM calls (consider async or smaller critic model)
- **Cost**: Double API usage

#### Phase 3: Entailment Classifier

**Model**: Fine-tuned NLI (Natural Language Inference) model

```python
from transformers import pipeline

class EntailmentHandler(BaseRagHandler):
    def __init__(self, next_handler=None):
        super().__init__(next_handler)
        self.nli = pipeline("text-classification", 
                           model="microsoft/deberta-v3-base-mnli")
    
    async def handle(self, ctx: dict):
        answer = ctx["raw_answer"]
        chunks = ctx["retrieved_chunks"]
        
        # Check each sentence against concatenated chunks
        context = " ".join(chunks)
        sentences = self.extract_sentences(answer)
        
        entailments = []
        for sent in sentences:
            result = self.nli(f"{context} [SEP] {sent}")
            # Result: {'label': 'ENTAILMENT'/'CONTRADICTION'/'NEUTRAL', 'score': 0.95}
            entailments.append(result)
        
        # Valid if all sentences are entailed or neutral
        ctx["valid"] = all(e["label"] != "CONTRADICTION" for e in entailments)
        ctx["entailment_scores"] = entailments
        
        return await super().handle(ctx)
```

**Benefits**:
- **Fast**: Smaller model than LLM (~400M params)
- **Accurate**: Trained specifically for entailment detection
- **Fine-Grained**: Detects contradictions (not just low similarity)

#### Phase 4: Uncertainty Quantification

**Method**: Ensemble predictions + calibration

```python
class UncertaintyHandler(BaseRagHandler):
    def __init__(self, llm_ensemble, next_handler=None):
        super().__init__(next_handler)
        self.models = llm_ensemble  # List of LLMs
    
    async def handle(self, ctx: dict):
        prompt = ctx["prompt"]
        
        # Generate multiple answers
        answers = []
        for model in self.models:
            response = await model.ainvoke(prompt)
            answers.append(response.content)
        
        # Compute agreement
        agreement = self.compute_agreement(answers)
        
        ctx["raw_answer"] = answers[0]  # Primary model
        ctx["uncertainty"] = 1 - agreement  # 0 = certain, 1 = uncertain
        ctx["valid"] = agreement >= 0.7  # Threshold on agreement
        
        return await super().handle(ctx)
    
    def compute_agreement(self, answers: List[str]) -> float:
        # Use sentence embeddings to measure semantic similarity
        embeddings = [self.embed(ans) for ans in answers]
        pairwise_sims = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                pairwise_sims.append(sim)
        return np.mean(pairwise_sims)
```

**User-Facing**:
```json
{
  "answer": "...",
  "uncertainty": 0.15,
  "interpretation": "LOW_UNCERTAINTY",
  "ensemble_size": 3
}
```

#### Phase 5: Human-in-the-Loop Feedback

**System**: Collect user feedback on rejected answers

```python
# schemas/feedback.py
class FeedbackRequest(SQLModel):
    query_id: str
    was_helpful: bool  # Did validation help or hurt?
    actual_hallucination: bool  # Was it really a hallucination?
    comment: Optional[str]

# routers/feedback.py
@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    # Store in database
    FeedbackCRUD.create(session, feedback)
    
    # Retrain threshold or fine-tune validation model
    if accumulated_feedback_count >= 100:
        retrain_validation_model()
```

**Benefits**:
- **Continuous Improvement**: Validation model learns from real usage
- **Threshold Tuning**: Adjust 0.35 based on precision/recall metrics
- **Error Analysis**: Identify systematic failures

#### Phase 6: Retrieval Quality Metrics

**Metrics**:

```python
class RetrievalMetricsHandler(BaseRagHandler):
    async def handle(self, ctx: dict):
        query = ctx["query"]
        chunks = ctx["retrieved_chunks"]
        
        # Compute retrieval quality
        ctx["retrieval_metrics"] = {
            "diversity": self.compute_diversity(chunks),
            "coverage": self.compute_coverage(query, chunks),
            "avg_similarity": np.mean(ctx.get("retrieval_scores", []))
        }
        
        # Warn if low quality
        if ctx["retrieval_metrics"]["avg_similarity"] < 0.3:
            ctx["warning"] = "Retrieved context may not be relevant"
        
        return await super().handle(ctx)
    
    def compute_diversity(self, chunks: List[str]) -> float:
        """Measures how different chunks are from each other"""
        embeddings = [self.embed(c) for c in chunks]
        sims = [cosine_similarity(embeddings[i], embeddings[j])
                for i in range(len(embeddings))
                for j in range(i+1, len(embeddings))]
        return 1 - np.mean(sims)  # High diversity = low similarity
    
    def compute_coverage(self, query: str, chunks: List[str]) -> float:
        """Measures how well chunks cover query terms"""
        query_terms = set(query.lower().split())
        chunk_terms = set(" ".join(chunks).lower().split())
        return len(query_terms & chunk_terms) / len(query_terms)
```

---

## Summary

**Vardast** is a RAG system designed for safety through validation. Key architectural principles:

1. **Modularity**: Chain of Responsibility enables flexible pipeline composition
2. **Validation First**: Anti-hallucination logic rejects ungrounded answers
3. **Developer-Friendly**: FastAPI + SQLModel + ChromaDB minimize infrastructure complexity
4. **Production-Ready Path**: Clear scaling strategy (PostgreSQL, Qdrant, caching, observability)

**Next Steps**:
- Migrate to PostgreSQL + Qdrant for production
- Implement confidence scoring and model-critic pipeline
- Add observability (Prometheus, Jaeger)
- Human-in-the-loop feedback for continuous improvement

