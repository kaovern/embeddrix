from time import monotonic_ns
from typing import Union, List
import os
import struct
from collections import OrderedDict
from functools import wraps

from aiomcache import Client as MemcacheClient
from blacksheep import Application, post, FromJSON
from blacksheep.server.openapi.v3 import OpenAPIHandler
from blacksheep.server.compression import use_gzip_compression
from openapidocs.v3 import Info, License
from openapidocs.common import Format
from loguru import logger
from pydantic import BaseModel

from .embedding_worker import EmbeddingWorker, TaskType

app = Application()
use_gzip_compression(app)

app.use_cors(
    allow_methods="*",
    allow_origins="*",
    allow_headers="*",
    max_age=300,
)

docs = OpenAPIHandler(info=Info(
    title="Embeddrix",
    version="0.0.1",
    description="A stupid simple service to generate text embeddings.",
    license=License("MIT")
), preferred_format=Format.YAML)
docs.bind_app(app)

embedding_format = "1024d"  # 1024 doubles for jina-embeddings-v3 model

class InputData(BaseModel):
    texts: list[str]

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


def timeit(func):
    fn_name = func.__name__
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = monotonic_ns()
        res = await func(*args, **kwargs)
        end = monotonic_ns()
        logger.info(f"{fn_name} took {(end-start)/1_000_000} ms")
        return res
    return wrapper

@timeit
async def make_embedding(worker: EmbeddingWorker, text: Union[str, List[str]], task: TaskType, cache: MemcacheClient) -> List[List[float]]:
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
async def passage(worker: EmbeddingWorker, input: FromJSON[InputData], cache: MemcacheClient) -> List[List[float]]:
    """Generate Embeddings for Passage Texts
    
    <br><br>
    <b>Endpoint:</b><br>
        POST /passage<br>
    <br>
    <b>Description:</b><br>
        This endpoint generates 1024-dimensional embeddings for passage texts. The embeddings are intended for retrieval 
        tasks where the passage serves as the content to be searched.<br>

    <b>Request Body:</b><br>
        The request body should be a JSON object that contains a list of text passages.<br>
        <br>
        - `texts`: A list of strings, where each string is a passage text to be embedded.<br>
    <br>
    <b>JSON Example:</b><br>
        <pre>
        <code>
        {<br>
            "texts": [<br>
                "Sample passage text 1.",<br>
                "Sample passage text 2.",<br>
                "Sample passage text 3."<br>
            ]<br>
        }<br>
        </code>
        </pre>
    <br>
    <b>Response:</b><br>
        The response will be a JSON array, where each element is a list of 1024 floating-point numbers 
        representing the embedding of the corresponding input text.<br>

    <b>JSON Example:</b><br>
        <pre>
        <code>
        [<br>
            [0.123, 0.456, /* ... 1022 more numbers ... */, 0.789],<br>
            [0.234, 0.567, /* ... 1022 more numbers ... */, 0.890],<br>
            [0.345, 0.678, /* ... 1022 more numbers ... */, 0.901]<br>
        ]<br>
        </code>
        </pre>
    """
    return await make_embedding(worker, input.value.texts, "retrieval.passage", cache)

@post("/query")
async def query(worker: EmbeddingWorker, input: FromJSON[InputData], cache: MemcacheClient) -> List[List[float]]:
    """Generate Embeddings for Query Texts
    
    <br><br>
    <b>Endpoint:</b><br>
        POST /passage<br>
    <br>
    <b>Description:</b><br>
        This endpoint generates 1024-dimensional embeddings for query texts. The embeddings are intended for retrieval 
        tasks where the query serves as the question to be answered.<br>

    <b>Request Body:</b><br>
        The request body should be a JSON object that contains a list of text queries.<br>
        <br>
        - `texts`: A list of strings, where each string is a query text to be embedded.<br>
    <br>
    <b>JSON Example:</b><br>
        <pre>
        <code>
        {<br>
            "texts": [<br>
                "What is the capital of Russia?",<br>
                "When did Boston tea party happen?",<br>
                "Why don't we see stars anymore?"<br>
            ]<br>
        }<br>
        </code>
        </pre>
    <br>
    <b>Response:</b><br>
        The response will be a JSON array, where each element is a list of 1024 floating-point numbers 
        representing the embedding of the corresponding input text.<br>

    <b>JSON Example:</b><br>
        <pre>
        <code>
        [<br>
            [0.123, 0.456, /* ... 1022 more numbers ... */, 0.789],<br>
            [0.234, 0.567, /* ... 1022 more numbers ... */, 0.890],<br>
            [0.345, 0.678, /* ... 1022 more numbers ... */, 0.901]<br>
        ]<br>
        </code>
        </pre>
    """
    return await make_embedding(worker, input.value.texts, "retrieval.query", cache)
