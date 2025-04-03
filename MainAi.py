import os
from typing import List

import asyncio
import cohere
from cohere.types import tool
from langchain_community.chat_models import ChatCohere
from langchain_community.llms import Cohere
from pydantic import BaseModel

from AITools import  ToolManager
from AiMemory import AiMemoryManager

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
CKey = os.environ["COHERE_API_KEY"]

#with open('Coherekey', 'r') as file:
#    CKey = file.read().strip()
    
class ToolAgent:
    def __init__(self, api_key, role, goal):
        self.llm = cohere.Client(api_key)
        self.role = role
        self.goal = goal
        self.tool_manager = ToolManager()

    def get_tool_descriptions(self):
        tool_descriptions = self.tool_manager.get_tool_descriptions()
        return tool_descriptions 

    async def think(self, input_text):
        tool_descriptions = self.tool_manager.get_tool_descriptions()
        
        tool_selection_prompt = f"""
        {self.role}
        Goal: {self.goal}
        Available Tools:
        {tool_descriptions}
        
        User Input: {input_text}
        Which tool should I use? Respond with just the tool name or 'none':"""
        
        tool_choice = self.llm.generate(
            model='command',
            prompt=tool_selection_prompt,
            max_tokens=50,
            temperature=0.2
        ).generations[0].text.strip().lower()
        
        if tool_choice in self.tool_manager.tools:
            # Prepare tool-specific parameters
            tool_params = {}
            if tool_choice in ['wiki', 'google']:
                tool_params['query'] = input_text
            elif tool_choice == 'time':
                tool_params['timezone'] = None  # Or parse timezone from input_text if needed
            
            tool_result = await self.tool_manager.use_tool(tool_choice, **tool_params)
            
            # Generate final response using tool result
            response_prompt = f"""
            {self.role}
            Goal: {self.goal}
            Tool Used: {tool_choice}
            Tool Result: {tool_result}
            User Input: {input_text}
            Generate a helpful response:"""
            
            response = self.llm.generate(
                model='command',
                prompt=response_prompt,
                max_tokens=300,
                temperature=0.7
            )
            return response.generations[0].text
        else:
            return "I don't need any tools to answer this. " #+ await self.direct_response(input_text)

    async def direct_response(self, input_text):
        # Handle responses without tools
        prompt = f"{self.role}\nGoal: {self.goal}\nInput: {input_text}\nResponse:"
        response = self.llm.generate(
            model='command',
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        return response.generations[0].text

class SimpleAgent:
    def __init__(self, api_key, role, goal):
        self.llm = cohere.Client(api_key)
        self.role = role
        self.goal = goal
        self.memory_service = AiMemoryManager(api_key, "Memory Manager", "Store and retrieve relevant context")
        self.tool_agent = ToolAgent(api_key, "Tool Assistant", "Help the SimpleAgent AiAgent with specific tasks using tools")

    async def think(self, input_text):
        Context = await self.memory_service.retrieve_relevant_context(input_text, k=5)
        
        tool_decision_prompt = f"""
        {self.role}
        Goal: {self.goal}
        Context: {Context}
        Available Tools: {self.tool_agent.get_tool_descriptions()}
        Input: {input_text}
        Should we use tools for this task? Answer with just 'yes' or 'no':"""
        
        need_tools = self.llm.generate(
            model='command',
            prompt=tool_decision_prompt,
            max_tokens=50,
            temperature=0.2
        ).generations[0].text.strip().lower()

        if 'yes' in need_tools:
            tool_response = await self.tool_agent.think(input_text)
            
            prompt = f"""{self.role}
                    \nGoal: {self.goal}
                    \nContext: {Context}
                    \nInput: {input_text}
                    \nTool Result: {tool_response}
                    \nUsing the tool's result, answer the initial query:"""
            
            final_response = self.llm.generate(
                model='command',
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            ).generations[0].text
            
            TroubleshootingResponse = f"\nTool Response: {tool_response}\n\nFinal response: {final_response}"
            await self.memory_service.ManageMemory(input_text, tool_response, final_response)
            return TroubleshootingResponse #final_response
        else:
            response = self.llm.generate(
                model='command',
                prompt=f"""{self.role}
                        \nGoal: {self.goal}
                        \nContext: {Context}
                        \nInput: {input_text}
                        \nResponse:""",
                max_tokens=300,
                temperature=0.7
            )
            await self.memory_service.ManageMemory(input_text, "No tools used", response.generations[0].text)
            return response.generations[0].text
            

async def main():

    agent = SimpleAgent(CKey, "Ai Assistant", "Use any tools at your disposal to answer the user's question")

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = await agent.think(user_input)
        print(f"\nAssistant: {response}\n")
            
# Run the async function
if __name__ == "__main__":
    asyncio.run(main())