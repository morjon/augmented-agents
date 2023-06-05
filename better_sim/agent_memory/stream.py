from typing import List
from lanchain.schema import Document


class MemoryStream:  # maybe
    def __init__(self):
        self.stream = []
        self.importance_weight = 0.5

    def store_memories(self, documents: List[Document]) -> List[str]:
        result = []
        for document in documents:
            self.stream.append(document)
            result.append(document.content)
        return result

    def retrieve_and_filter_memories(self, observation: str) -> List[str]:
        result = []
        # Filter memories based on observation
        for memory in self.stream:
            if observation in memory.content:
                result.append(memory.content)
        return result
