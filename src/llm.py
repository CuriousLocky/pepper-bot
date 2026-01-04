import json
import re
from datetime import datetime
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Tuple
from config import Config
from memory import MemoryManager
from websearch import web_search
from get_url_content import get_url_content

class LLMClient:
    def __init__(self, config: Config, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.client = AsyncOpenAI(
            api_key=config.api.key,
            base_url=config.api.url
        )
        self.maintenance_tools = self._define_tools()
        # Chat tools exclude 'add_long_term_memory'
        self.chat_tools = [t for t in self.maintenance_tools if t['function']['name'] != 'add_long_term_memory']

    def _clean_response(self, text: str) -> str:
        if not text:
            return ""
        # Remove redundant formatting info like "[msg 8] Pepper (reply to msg 7): "
        # Matches leading pattern of optional "[msg XX]" + optional Name/Nickname + optional "(reply to msg XX)" + optional ":"
        
        # Build list of names to filter: primary name + nicknames
        names = [self.config.bot.name] + self.config.bot.nicknames
        # Escape names for regex
        names_pattern = "|".join([re.escape(n) for n in names])
        
        pattern = rf'^(?:\s*\[msg \d+\]\s*)?(?:(?:{names_pattern})\s*)?(?:\(reply to msg \d+\)\s*)?:\s*'
        
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def _define_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "add_short_term_memory",
                    "description": "Add a significant event or fact to short-term memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content of the memory to save."
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_long_term_memory",
                    "description": "Add a permanent fact or summarized event to long-term memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to save permanently."
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_user_info",
                    "description": "Update or add information about a specific user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "integer",
                                "description": "The Telegram User ID."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name to call the user by."
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the user's personality, habits, etc."
                            }
                        },
                        "required": ["user_id", "name", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_url_content",
                    "description": "Fetch and read the text content of a specific webpage URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the webpage to read."
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_scheduled_task",
                    "description": "Schedule a task to be executed later.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_in_minute": {
                                "type": "integer",
                                "description": "Delay in minutes before execution. Max at 1440 (24 hours)."
                            },
                            "title": {
                                "type": "string",
                                "description": "Title of the task."
                            },
                            "content": {
                                "type": "string",
                                "description": "Content/Instruction for the task."
                            }
                        },
                        "required": ["time_in_minute", "title", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_scheduled_task_list",
                    "description": "Get the list of currently scheduled tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    async def _execute_tools(self, tool_calls: List[Any], tool_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tool_output = ""
            try:
                if function_name == "add_short_term_memory":
                    self.memory_manager.add_short_term_event(function_args["content"])
                    tool_output = "Short-term memory added successfully."
                elif function_name == "add_long_term_memory":
                    self.memory_manager.append_long_term(function_args["content"])
                    tool_output = "Long-term memory added successfully."
                elif function_name == "update_user_info":
                    self.memory_manager.update_user_info(
                        function_args["user_id"],
                        function_args["name"],
                        function_args["description"]
                    )
                    tool_output = "User info updated successfully."
                elif function_name == "web_search":
                    search_results = web_search(function_args["query"], self.config.search)
                    tool_output = json.dumps(search_results, ensure_ascii=False)
                elif function_name == "get_url_content":
                    tool_output = get_url_content(function_args["url"])
                elif function_name == "set_scheduled_task":
                    if tool_context and "schedule_func" in tool_context:
                        tool_output = await tool_context["schedule_func"](
                            function_args["time_in_minute"],
                            function_args["title"],
                            function_args["content"]
                        )
                    else:
                        tool_output = "Error: Scheduling context not available."
                elif function_name == "get_scheduled_task_list":
                    if tool_context and "list_func" in tool_context:
                        tool_output = await tool_context["list_func"]()
                    else:
                        tool_output = "Error: Scheduling context not available."
                else:
                    tool_output = f"Unknown tool: {function_name}"
            except Exception as e:
                tool_output = f"Error executing tool {function_name}: {str(e)}"
            
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_output,
            })
        return results

    async def get_response(self, 
                           messages: List[Dict[str, Any]], 
                           system_prompt_template: str,
                           tool_context: Dict[str, Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        
        system_prompt = system_prompt_template.replace(
            "{{date-time}}", datetime.now().strftime("%Y-%m-%d %H:%M")
        ).replace(
            "{{short-mem}}", self.memory_manager.get_short_term_str()
        ).replace(
            "{{long-mem}}", self.memory_manager.get_long_term_str()
        ).replace(
            "{{user-info}}", self.memory_manager.get_user_info_str()
        )

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        new_messages = []

        try:
            response = await self.client.chat.completions.create(
                model=self.config.api.model,
                messages=full_messages,
                tools=self.chat_tools,
                tool_choice="auto",
                temperature=self.config.model_params.temperature,
                max_tokens=self.config.context.max_ai_response_token,
                reasoning_effort=self.config.model_params.reasoning_effort
            )
            
            response_message = response.choices[0].message
            new_messages.append(response_message.model_dump())

            while response_message.tool_calls:
                full_messages.append(response_message.model_dump())
                
                tool_results = await self._execute_tools(response_message.tool_calls, tool_context)
                full_messages.extend(tool_results)
                new_messages.extend(tool_results)

                # Get next response from AI
                response = await self.client.chat.completions.create(
                    model=self.config.api.model,
                    messages=full_messages,
                    tools=self.chat_tools,
                    tool_choice="auto",
                    temperature=self.config.model_params.temperature,
                    max_tokens=self.config.context.max_ai_response_token,
                    reasoning_effort=self.config.model_params.reasoning_effort
                )
                response_message = response.choices[0].message
                new_messages.append(response_message.model_dump())
            
            return self._clean_response(response_message.content or ""), new_messages

        except Exception as e:
            return f"Error communicating with AI: {str(e)}", []

    async def consolidate_memory(self, expired_events: List[Any], system_prompt_template: str):
        if not expired_events:
            return

        events_str = "\n".join([f"- {e.content}" for e in expired_events])
        
        prompt = f"""
The following short-term memories are expiring:
{events_str}

Review these events.
- If an event contains important long-term information (e.g., user preferences, major life events), save it to long-term memory.
- If it reveals new long-term information about a user, update the user info.
- If it is trivial, do nothing (it will be forgotten).

Use the provided tools to take action.
"""
        
        sys_prompt_base = "You are a memory manager for a chatbot. Your job is to consolidate short-term memories into long-term storage or discard them."
        sys_prompt = sys_prompt_base + f"\nCurrent Long-term Memory:\n{self.memory_manager.get_long_term_str()}\n\nCurrent User Info:\n{self.memory_manager.get_user_info_str()}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.config.api.model,
                messages=messages,
                tools=self.maintenance_tools,
                tool_choice="auto",
                temperature=0.3, # Lower temperature for maintenance tasks
                reasoning_effort=self.config.model_params.reasoning_effort
            )
            
            response_message = response.choices[0].message
            if response_message.tool_calls:
                await self._execute_tools(response_message.tool_calls)
                
        except Exception as e:
            print(f"Error consolidating memory: {e}")