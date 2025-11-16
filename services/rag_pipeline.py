import numpy as np
from langchain.agents import create_agent
from langchain.tools import tool
from sentence_transformers import SentenceTransformer

from ai_model import ChatOpenRouter
from models import get_chroma_client, get_or_create_chroma_collection
from services.chroma_document_service import DocumentsChroma
from utils.embedding import encode_to_embedding
from utils.tools import get_current_time

st_model = SentenceTransformer("all-MiniLM-L6-v2") 
class BaseRagHandler:
    def __init__(self, next_handler=None):
        self.next = next_handler

    def set_next(self, handler):
        self.next = handler
        return handler

    async def handle(self, ctx: dict):
        if self.next:
            return await self.next.handle(ctx)
        return ctx


class RetrieveHandler(BaseRagHandler):
    async def handle(self, ctx: dict):
        query = ctx["query"]
        top_k = ctx["top_k"]

        db = get_chroma_client()
        col = get_or_create_chroma_collection(db, "default")

        query_emb = encode_to_embedding(query).tolist()

        results = col.query(query_embeddings=[query_emb], n_results=top_k)

        if results is not None:
            ctx["related_data_found"] = True
            ctx["retrieved_chunks"] = results.get("documents", [])[0]
            return await super().handle(ctx)

        ctx["related_data_found"] = False
        ctx["retrieved_chunks"] = []
        return super().handle(ctx)


class ContextAugmentHandler(BaseRagHandler):
    async def handle(self, ctx: dict):
        chunks = ctx["retrieved_chunks"]
        query = ctx["query"]

        more_strict_prompt = ""
        if not ctx["related_data_found"]:
            more_strict_prompt = "There were not sufficient documents related to user request, dont answer without needed data and assume your own data to be mostly outdated"

        ctx["prompt"] = (
            "Use the following context strictly. "
            'If any extra data needed tools to answer. If the answer is not in the retrieved data, say "I don\'t know."'
            "If the answer is not in the context and tools did not provide helpfull content, say 'I don't know.'\n\n"
            + more_strict_prompt
            + "Context:\n"
            + "\n".join(chunks)
            + "\n\nUser question:\n"
            + query
            # TODO: check about chunks being embeddings
        )
        return await super().handle(ctx)


class ToolHandler(BaseRagHandler):

    @tool(description="Returns the current UTC time in ISO format.")
    @staticmethod
    def time_tool_func():
        return get_current_time()

    @tool(description="Returns user info inside a dictionary")
    @staticmethod
    def user_info():
        return {"name": "alireza", "age": 25, "heigh": 178, "sex": "male"}

    @tool(description="Retrieves documents relevant to the query")
    @staticmethod
    def retrieve_more_data(query: str, top_k: int):
        return DocumentsChroma.retrieve_top_k(query, 1)

    async def handle(self, ctx: dict):
        ctx["tools"] = [
            self.time_tool_func,
            self.user_info,
            self.retrieve_more_data,
        ]
        return await super().handle(ctx)


class LLMHandler(BaseRagHandler):
    def __init__(self, next_handler=None):
        super().__init__(next_handler)
        self.llm = ChatOpenRouter()

    async def handle(self, ctx: dict):
        prompt = ctx["prompt"]

        agent = create_agent(
            model=self.llm,
            tools=ctx["tools"],
            system_prompt="You are here to help user get what ever data they need",
        )

        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        ctx["raw_answer"] = response
        return await super().handle(ctx)



class ValidationHandler(BaseRagHandler):
    #TODO: this needs some cleaning 
    def __init__(self, threshold=0.35, next_handler=None):
        super().__init__(next_handler)
        self.threshold = threshold

    async def handle(self, ctx: dict):
        raw_answer = ctx.get("raw_answer")
        chunks = ctx.get("retrieved_chunks")

        if not chunks or not raw_answer:
            ctx["valid"] = False
            return await super().handle(ctx)

        if isinstance(raw_answer, dict):
            answer_text = raw_answer.get("content") or raw_answer.get("answer") or str(raw_answer)
        else:
            answer_text = str(raw_answer)

        sentences = [s.strip() for s in answer_text.replace("\n", ". ").split(".") if s.strip()]
        if not sentences:
            ctx["valid"] = False
            return await super().handle(ctx)

        emb_sentences = st_model.encode(sentences, normalize_embeddings=True)
        emb_chunks = st_model.encode(chunks, normalize_embeddings=True)

        sims = [np.max([np.dot(sent_emb, chunk_emb) for chunk_emb in emb_chunks])
                for sent_emb in emb_sentences]

        ctx["valid"] = max(sims) >= self.threshold
        print("Sentence sims:", sims, "Max similarity:", max(sims))
        return await super().handle(ctx)

class ResponseHandler(BaseRagHandler):
    async def handle(self, ctx: dict):


        print(ctx["raw_answer"])
        if not ctx["valid"]:
            ctx["response"] = {
                "answer": None,
                "context_used": ctx["retrieved_chunks"],
                "refused": True,
                "message": "I don't have enough information to answer that reliably."
            }
        else:
            ctx["response"] = {
                "answer": ctx["raw_answer"],
                "context_used": ctx["retrieved_chunks"],
                "refused": False,
                "message": None
            }
        return await super().handle(ctx)


async def run_rag_pipeline(query: str, top_k: int = 5):
    ctx = {"query": query, "top_k": top_k}

    pipeline = RetrieveHandler()
    pipeline.set_next(ContextAugmentHandler()) \
            .set_next(ToolHandler()) \
            .set_next(LLMHandler()) \
            .set_next(ValidationHandler()) \
            .set_next(ResponseHandler())

    final_ctx = await pipeline.handle(ctx)
    return final_ctx["response"]
