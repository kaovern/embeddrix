import asyncio
import multiprocessing
from multiprocessing import Process, Queue
from threading import Thread
from typing import List, Literal, Dict
import uuid

from sentence_transformers import SentenceTransformer

# Define the type for the task parameter
TaskType = Literal['retrieval.query', 'retrieval.passage', 'text-matching', 'classification', 'separation']


def _worker_process(input_queue: Queue, output_queue: Queue):
    """
    The worker process that loads the model and processes tasks from the input queue.
    """
    # Load the model in the separate process
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True, device='cuda')
    while True:
        task = input_queue.get()
        if task is None:
            break  # Exit signal
        task_id, texts, task_type = task
        try:
            embeddings = model.encode(
                texts,
                task=task_type,
                prompt_name=task_type
            )
            # Put the result along with the task ID into the output queue
            output_queue.put((task_id, embeddings))
        except Exception as e:
            # Put the exception into the output queue
            output_queue.put((task_id, e))


class EmbeddingWorker:
    def __init__(self):
        """
        Initialize the EmbeddingWorker by creating queues, starting the worker process,
        and starting the result reader thread.
        """
        self.loop = asyncio.get_event_loop()
        self.input_queue: Queue = multiprocessing.Queue()
        self.output_queue: Queue = multiprocessing.Queue()
        self.pending_tasks: Dict[str, asyncio.Future] = {}
        self.process = Process(target=_worker_process, args=(self.input_queue, self.output_queue), daemon=True)
        self.process.daemon = True
        self.process.start()

        # Start the result reader thread
        self._result_thread = Thread(target=self._result_reader, daemon=True)
        self._result_thread.start()

    def _result_reader(self):
        """
        The result reader thread that reads from the output queue and sets results on the futures.
        """
        while True:
            task_id, result = self.output_queue.get()
            future = self.pending_tasks.pop(task_id, None)
            if future is not None:
                if isinstance(result, Exception):
                    self.loop.call_soon_threadsafe(future.set_exception, result)
                else:
                    self.loop.call_soon_threadsafe(future.set_result, result)

    async def get_embeddings(self, texts: List[str], task: TaskType) -> List[List[float]]:
        """
        Asynchronously generate embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to encode.
            task (TaskType): The task type, either 'retrieval.query' or 'retrieval.passage'.

        Returns:
            List[List[float]]: A list of embeddings for the input texts.
        """
        task_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self.pending_tasks[task_id] = future
        # Put the task into the input queue
        self.input_queue.put((task_id, texts, task))

        # Wait for the result asynchronously
        return await future

    def close(self):
        """
        Clean up the worker process and queues.
        """
        # Send exit signal to worker process
        self.input_queue.put(None)
        self.process.join()
        self._result_thread.join()
        self.input_queue.close()
        self.output_queue.close()
