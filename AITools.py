import asyncio
import datetime
import os
from typing import Any, Dict

import cohere
from langchain_community.chat_models import ChatCohere
from langchain_community.llms import Cohere
from pydantic import BaseModel
import pyjokes
import pywhatkit
import wikipedia

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
CKey = os.environ["COHERE_API_KEY"]

#with open('Coherekey', 'r') as file:
#    CKey = file.read().strip()

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, str]
    
    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError("Tool must implement execute method")

class WikiSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="wiki_search",
            description="Search Wikipedia for information about a topic",
            parameters={"query": "The topic to search for"}
        )
    
    async def execute(self, query: str) -> str:
        try:
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

class JokeTool(Tool):
    def __init__(self):
        super().__init__(
            name="tell_joke",
            description="Tell a random programming joke",
            parameters={}
        )
    
    async def execute(self) -> str:
        return pyjokes.get_joke()

class GoogleSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="google_search",
            description="Search Google for a topic",
            parameters={"query": "The search query"}
        )
    
    async def execute(self, query: str) -> str:
        try:
            # Open browser for search
            pywhatkit.search(query)
            # Get search information
            search_info = pywhatkit.info(query, lines=2)
            return f"Searched Google for: {query}\nQuick summary: {search_info}"
        except Exception as e:
            return f"Error performing Google search: {str(e)}"

class TimeTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_time",
            description="Get the current time",
            parameters={"timezone": "Optional timezone"}
        )
    
    async def execute(self, timezone: str | None = None) -> str:
        return datetime.datetime.now().strftime("%I:%M %p")

class ToolManager:
    def __init__(self):
        self.tools = {
            "wiki": WikiSearchTool(),
            "joke": JokeTool(),
            "google": GoogleSearchTool(),
            "time": TimeTool()
        }
    
    def get_tool_descriptions(self) -> str:
        return "\n".join([
            f"{name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
    
    async def use_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
        
        tool = self.tools[tool_name]
        return await tool.execute(**kwargs)

'''
class ToolAgent:
    def __init__(self, api_key, role, goal):
        self.llm = cohere.Client(api_key)
        self.role = role
        self.goal = goal
        self.tool_manager = ToolManager()

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
            return "I don't need any tools to answer this. " + await self.direct_response(input_text)

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
'''

"""
async def main():
    agent = ToolAgent(CKey, "Assistant", "Help users with various tasks using tools")
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = await agent.think(user_input)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
"""