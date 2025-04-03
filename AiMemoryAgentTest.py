from pydantic import BaseModel
from typing import List
from langchain_community.llms import Cohere
from langchain_community.chat_models import ChatCohere
import cohere
import os
import asyncio
from sklearn.metrics.pairwise import cosine_similarity

with open('Coherekey', 'r') as file:
    CKey = file.read().strip()
    
os.environ["COHERE_API_KEY"] = CKey

class SimpleAgent:
    def __init__(self, api_key, role, goal):
        self.llm = cohere.Client(api_key)
        self.role = role
        self.goal = goal

    async def think(self, input_text):
        prompt = f"{self.role}\nGoal: {self.goal}\nInput: {input_text}\nResponse:"
        response = self.llm.generate(
            model='command',  # or any other Cohere model
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        return response.generations[0].text

class MemoryAgent:
    def __init__(self, api_key, role, goal):
        self.llm = cohere.Client(api_key)
        self.role = role
        self.goal = goal
        self.vector_memory = []
        self.SummarizedMemory = []
        self.recent_interactions = []  # Store last 3 interactions

    async def embed_text(self, text):
        response = self.llm.embed(
            texts=[text],
            model='embed-english-v3.0',
            input_type='search_document'  # Added input_type parameter
        )
        return response.embeddings

    async def store_memory(self, text):
        embedding = await self.embed_text(text)
        self.vector_memory.append({
            'text': text,
            'vector': embedding
        })

    async def retrieve_relevant_context(self, query, k=3):
        query_embedding = await self.embed_text(query)
        similarities = []
        
        for memory in self.vector_memory:
            similarity = cosine_similarity(
                [query_embedding], 
                [memory['vector']]
            )[0][0]
            similarities.append((similarity, memory['text']))
        
        similarities.sort(reverse=True)
        return [text for _, text in similarities[:k]]

    #Use Vector Memory to store and retrieve relevant memories/summaries
    async def ManageMemory(self, input_text, response_text):
        try:    
            # Store input and response as vectors
            interaction = f"User: {input_text}\nAssistant: {response_text}"
            await self.store_memory(interaction)
            
            # Add to recent interactions
            self.recent_interactions.append({
                'input': input_text,
                'response': response_text
            })
            
            # If we have more than 3 recent interactions
            if (len(self.recent_interactions)+1) % 3 == 0:
             
                # Combine interactions for context
                combined_context = ""
                for interaction in self.recent_interactions:
                    combined_context += f"User: {interaction['input']}\nAssistant: {interaction['response']}\n"
                
                # Create summary prompt for the group
                previous_summary = ""
                if self.SummarizedMemory:
                    previous_summary = f"Previous Summary:\n{self.SummarizedMemory[-1]}\n\n"

                prompt = f"""
                {self.role}
                Goal: Create a concise summary of these related interactions
                Previous Interactions:
                {previous_summary}
                {combined_context}
                Create a brief summary that captures key information from all interactions:"""     
                
                summary_for_memory = self.llm.generate(
                    model='command',
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.2
                )
                
                # Store group summary
                await self.store_memory( 'Summary:' + summary_for_memory.generations[0].text)
                self.SummarizedMemory.append(summary_for_memory.generations[0].text)
                self.recent_interactions = []  # Clear interactions for next group
                print("Memory Updated")

        except Exception as e:
            print(f"Memory management error: {str(e)}")

    async def think(self, input_text, context=None):
        # Combine recent interactions and summaries for context
        recent_context = "\nRecent Interactions:\n"
        for interaction in self.recent_interactions:
            recent_context += f"User: {interaction['input']}\nAssistant: {interaction['response']}\n"
        
        summary_context = "\nOlder Context:\n" + "\n".join(self.SummarizedMemory)
        memory_context = recent_context + summary_context

        prompt = f"{self.role}\nGoal: {self.goal}\nContext:{memory_context}\nInput: {input_text}\nResponse:"
        response = self.llm.generate(
            model='command',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        
        # Store new input and response in memory
        await self.ManageMemory(input_text, response.generations[0].text)
        return response.generations[0].text


async def main():
    #agent = SimpleAgent(CKey, "Planner", "Create engaging and creative story content")
    #result = await agent.think("Write a story about a magical forest")
    #print(result)

    memory_agent = MemoryAgent(CKey, "Game Master", "Run a quick game of Dungeons and Dragons")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = await memory_agent.think(user_input)
        print(f"\nAssistant: {response}\n")
            
    
# Run the async function
if __name__ == "__main__":
    asyncio.run(main())