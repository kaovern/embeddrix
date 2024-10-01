from time import monotonic_ns
from typing import Literal, Union, List
import os
import struct
from collections import OrderedDict

from aiomcache import Client as MemcacheClient
from blacksheep import Application, post, FromText
from blacksheep.server.openapi.v3 import OpenAPIHandler
from blacksheep.server.compression import use_gzip_compression
from openapidocs.v3 import Info
from loguru import logger
import orjson

from .embedding_worker import EmbeddingWorker

app = Application()
use_gzip_compression(app)

docs = OpenAPIHandler(info=Info(title="Embeddrix", version="0.0.1"))
docs.bind_app(app)

embedding_format = "1024d"  # 1024 doubles for jina-embeddings-v3 model

async def initialize_cache() -> MemcacheClient:
    memcached_host = os.getenv("MEMCACHED_HOST", "127.0.0.1")
    memcached_port = int(os.getenv("MEMCACHED_PORT", 11211))
    cache = MemcacheClient(memcached_host, memcached_port)
    logger.info("Memcache connected.")
    return cache

async def initialize_worker(cache: MemcacheClient) -> EmbeddingWorker:
    logger.info("Preparing worker...")
    worker = EmbeddingWorker()
    await make_embedding(worker, "Warmup text!", "retrieval.passage", cache)
    logger.info("Worker ready!")
    return worker

@app.on_start
async def on_start(app: Application):
    logger.info("Connecting to memcache...")
    cache = await initialize_cache()
    app.services.register(MemcacheClient, instance=cache)

    worker = await initialize_worker(cache)
    app.services.register(EmbeddingWorker, instance=worker)

@app.on_stop
async def on_stop(app: Application, worker: EmbeddingWorker):
    logger.info("Shutting down.")
    worker.close()


async def make_embedding(worker: EmbeddingWorker, text: Union[str, List[str]], task: Literal['retrieval.passage', 'retrieval.query'], cache: MemcacheClient) -> List[List[float]]:
    """
    Create embeddings for the provided text(s) using an embedding worker.
    Results are cached to avoid redundant computations.
    
    Args:
        worker (EmbeddingWorker): Worker object that provides the embeddings.
        text (Union[str, List[str]]): Single string or list of strings to be embedded.
        task (Literal): The type of embedding task, affects the embedding process.
        cache (MemcacheClient): Cache client to store and retrieve embeddings.
    
    Returns:
        List[List[float]]: A list of embeddings for the provided text(s).
    """
    
    # Normalize input text
    text_to_embed = OrderedDict()
    if isinstance(text, str):
        text = [text]  # Ensure text is a list for consistent processing
    
    # Initialize the text-to-embedding map
    for t in text:
        hash_key = str(hash(t))
        cached = await cache.get(f"embed:{task}:{hash_key}".encode("utf-8"))
        if cached:
            text_to_embed[t] = list(struct.unpack(embedding_format, cached))
        else:
            text_to_embed[t] = None

    # List of texts that need fresh embeddings
    texts_for_embed = [t for t in text_to_embed.keys() if text_to_embed[t] is None]

    # Fetch required embeddings
    if texts_for_embed:
        embeds = await worker.get_embeddings(texts_for_embed, task)
        for t, e in zip(texts_for_embed, embeds):
            e = e.tolist()
            text_to_embed[t] = e
            await cache.set(f"embed:{task}:{hash_key}".encode("utf-8"), struct.pack(embedding_format, *e))

    return [text_to_embed[t] for t in text]
    

@post("/passage")
async def passage(worker: EmbeddingWorker, text: FromText, cache: MemcacheClient) -> List[List[float]]:
    return await make_embedding(worker, text.value, "retrieval.passage", cache)

@post("/query")
async def query(worker: EmbeddingWorker, text: FromText, cache: MemcacheClient) -> List[List[float]]:
    return await make_embedding(worker, text.value, "retrieval.query", cache)
