from pydantic import BaseModel
from typing import List
from langchain_community.llms import Cohere
from langchain_community.chat_models import ChatCohere
import cohere
import os
import asyncio
from sklearn.metrics.pairwise import cosine_similarity

# Get API Key
with open('Coherekey', 'r') as file:
    CKey = file.read().strip()

os.environ["COHERE_API_KEY"] = CKey

class MemoryAgent:
    def __init__(self, api_key, role, goal):
        self.llm = cohere.Client(api_key)
        self.role = role
        self.goal = goal
        self.vector_memory = []

    async def embed_text(self, text, is_query=False):
        response = self.llm.embed(
            texts=[text],
            model='embed-english-v3.0',
            input_type='search_query' if is_query else 'search_document'
        )
        return response.embeddings[0]

    async def store_memory(self, text):
        embedding = await self.embed_text(text, is_query=False)
        self.vector_memory.append({
            'text': text,
            'vector': embedding
        })

    async def retrieve_relevant_context(self, query, k=3):
        query_embedding = await self.embed_text(query, is_query=True)
        similarities = []
        
        for memory in self.vector_memory:
            similarity = cosine_similarity(
                [query_embedding], 
                [memory['vector']]
            )[0][0]
            similarities.append((similarity, memory['text']))
        
        similarities.sort(reverse=True)
        return [text for _, text in similarities[:k]]

async def test_similarity():
    memory_agent = MemoryAgent(CKey, "Test", "Testing similarities")
    
    # Store some test memories
    test_memories = [
        "John is a software engineer who loves gaming",
        "Mary enjoys painting and drawing in her free time",
        "The weather is sunny today in London",
        "John is a talented musician who plays the guitar",
        "There is a cat in singapore with elite programming skills"
    ]
    
    for memory in test_memories:
        await memory_agent.store_memory(memory)
    
    # Test query
    query = "What are John's programming skills?"
    relevant = await memory_agent.retrieve_relevant_context(query, k=6)
    
    print("\nQuery:", query)
    print("\nMemories sorted by relevance:")
    for i, memory in enumerate(relevant, 1):
        print(f"{i}. {memory} - Relevance Score: {cosine_similarity([await memory_agent.embed_text(query, is_query=True)], [await memory_agent.embed_text(memory, is_query=False)])[0][0]}")

if __name__ == "__main__":
    asyncio.run(test_similarity())