import json
import re
from datetime import datetime
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Tuple
from config import Config
from memory import MemoryManager
from websearch import web_search
from get_url_content import get_url_content
from image_gen import generate_image
from random import randint
import os

class LLMClient:
    def __init__(self, config: Config, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.client = AsyncOpenAI(
            api_key=config.api.key,
            base_url=config.api.url
        )
        
        # Tool model client for memory consolidation
        tool_api_key = config.tool_model.api_key or config.api.key
        tool_api_url = config.tool_model.api_url or config.api.url
        self.tool_client = AsyncOpenAI(
            api_key=tool_api_key,
            base_url=tool_api_url
        )
        
        all_tools = self._define_all_tools()
        
        # Chat tools: everything except memory management tools
        excluded_names = {'add_long_term_memory', 'add_knowledge', 'update_user_info'}
        self.chat_tools = [t for t in all_tools if t['function']['name'] not in excluded_names]
        
        # Maintenance tools: specific memory management tools
        maintenance_names = {'add_long_term_memory', 'add_knowledge', 'update_user_info'}
        self.maintenance_tools = [t for t in all_tools if t['function']['name'] in maintenance_names]

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

    def _define_all_tools(self) -> List[Dict[str, Any]]:
        tool_list = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image based on a description.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "The detailed description of the image to generate."
                            },
                            "msg_id": {
                                "type": "integer",
                                "description": "Optional. The message ID of an image in chat history to use as input reference."
                            }
                        },
                        "required": ["description"]
                    }
                }
            },
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
                                "description": "The content of the memory to save (timestamp automatically added)."
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
                    "description": "Add a permanent event or fact to long-term memory. Use this for specific occurrences, events, or time-dependent information about what happened.",
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
                    "name": "add_knowledge",
                    "description": "Add a permanent knowledge to the knowledge store. Use this for plain facts about the bot itself, rules the bot should follow, methodologies, or procedures that define how the bot operates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The knowledge to save permanently."
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
            },
            {
                "type": "function",
                "function": {
                    "name": "randint",
                    "description": "Return random integer in range [a, b], including both end points.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "integer",
                                "description": "Beginning of the range (inclusive)."
                            },
                            "b": {
                                "type": "integer",
                                "description": "End of the range (inclusive)."
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
        if self.config.black_list.enable:
            # If blacklist feature is enabled, add a tool for it
            tool_list.append({
                "type": "function",
                "function": {
                    "name": "block_user",
                    "description": "Block a user from interacting.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                    "type": "integer",
                                    "description": "The Telegram User ID."
                                },
                            "duration_minutes": {
                                    "type": "integer",
                                    "description": f"Duration in minutes of blocking (max: {self.config.black_list.max_minute})."
                                },                            
                        },
                        "required": ["user_id", "duration_minutes"]
                    }
                }
            })
        if self.config.skills.enabled:
            tool_list.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_skill",
                        "description": "Fetch and load a skill by name.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "skill_name": {
                                    "type": "string",
                                    "description": "The name of the skill"
                                }
                            },
                            "required": ["skill_name"]
                        }
                    }
                }
            ])
        return tool_list

    async def _execute_tools(self, tool_calls: List[Any], tool_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tool_output = ""
            try:
                if function_name == "generate_image":
                    msg_id = function_args.get("msg_id")
                    image_input = None
                    
                    if msg_id is not None:
                        if tool_context and "chat_history" in tool_context:
                            hist = tool_context["chat_history"]
                            # Find message by internal ID
                            target_msg = next((m for m in hist.messages if m.message_id == msg_id), None)
                            if target_msg:
                                if target_msg.image_url:
                                    image_input = target_msg.image_url
                                else:
                                    tool_output = f"Error: Message {msg_id} does not contain an image."
                                    # Skip generation if validation fails
                                    results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": tool_output,
                                    })
                                    continue
                            else:
                                tool_output = f"Error: Message {msg_id} not found."
                                results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_output,
                                })
                                continue
                        else:
                            tool_output = "Error: Chat history context not available."
                            results.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": tool_output,
                            })
                            continue

                    success, full_res_base64, resized_base64, text_content = await generate_image(
                        function_args["description"], 
                        self.config, 
                        image_base64=image_input
                    )
                    
                    if success and full_res_base64:
                        tool_output = "The image is successfully generated and will attach to the next message"
                        if text_content:
                            tool_output += f"\n\n{text_content}"
                            
                        if tool_context is not None:
                            if "generated_images" not in tool_context:
                                tool_context["generated_images"] = []
                            # Store full resolution base64 for delivery to chat
                            tool_context["generated_images"].append(full_res_base64)
                            
                            if "history_images" not in tool_context:
                                tool_context["history_images"] = {}
                            # Store resized base64 for history context
                            tool_context["history_images"][tool_call.id] = resized_base64
                    else:
                        tool_output = f"Failed to generate image. Model output:\n{text_content}"
                elif function_name == "add_short_term_memory":
                    await self.memory_manager.add_short_term_event(function_args["content"])
                    tool_output = "Short-term memory added successfully."
                elif function_name == "add_long_term_memory":
                    await self.memory_manager.append_long_term(function_args["content"])
                    tool_output = "Long-term memory added successfully."
                elif function_name == "add_knowledge":
                    await self.memory_manager.append_knowledge(function_args["content"])
                    tool_output = "Knowledge added successfully."
                elif function_name == "update_user_info":
                    await self.memory_manager.update_user_info(
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
                elif function_name == "block_user":
                    if tool_context and "block_user_func" in tool_context:
                        tool_output = await tool_context["block_user_func"](
                            function_args["user_id"],
                            function_args["duration_minutes"]
                        )
                    else:
                        tool_output = "Error: Blacklist context not available."
                elif function_name == "randint":
                    a = function_args["a"]
                    b = function_args["b"]
                    rand_value = randint(a, b)
                    tool_output = str(rand_value)
                elif function_name == "list_skills":
                    tool_output = await tool_context["list_skills_func"]()
                elif function_name == "fetch_skill":
                    skill_name = function_args["skill_name"]
                    tool_output = await tool_context["fetch_skill_func"](skill_name)
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
                           tool_context: Dict[str, Any] = None,
                           current_user_id: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        
        # Determine query for memory retrieval from the last few messages
        query_text = ""
        query_input = ""
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        
        if user_messages:
            last_content = user_messages[-1]
            
            # Helper to get text from content
            def get_text(c):
                if isinstance(c, str): return c
                if isinstance(c, list): return " ".join([p.get("text", "") for p in c if p.get("type") == "text"])
                return ""
            
            last_text = get_text(last_content)
            
            if len(user_messages) > 1:
                second_last_content = user_messages[-2]
                prev_text = get_text(second_last_content)
                
                query_text = prev_text + "\n" + last_text
                
                # Construct query_input for embedding
                if isinstance(last_content, list):
                    # Multimodal
                    query_input = [item.copy() for item in last_content]
                    if prev_text:
                         query_input.insert(0, {"type": "text", "text": prev_text + "\n"})
                elif isinstance(last_content, str):
                    query_input = prev_text + "\n" + last_content
            else:
                query_text = last_text
                query_input = last_content

        # Pre-generate query embedding to save API calls
        query_embeddings = None
        if query_input:
            query_embeddings = await self.memory_manager.get_embeddings([query_input])

        short_mem = await self.memory_manager.get_short_term_str(query_text, query_embeddings=query_embeddings)
        long_mem = await self.memory_manager.get_long_term_str(query_text, query_embeddings=query_embeddings)
        user_info = await self.memory_manager.get_user_info_str(query_text, current_user_id, query_embeddings=query_embeddings)
        knowledges = self.memory_manager.get_all_knowledges_str()
        skill_list = self.list_skills()

        system_prompt = system_prompt_template.replace(
            "{{date-time}}", datetime.now().strftime("%Y-%m-%d %H:%M")
        ).replace(
            "{{short-mem}}", short_mem
        ).replace(
            "{{long-mem}}", long_mem
        ).replace(
            "{{user-info}}", user_info
        ).replace(
            "{{knowledges}}", knowledges
        ).replace(
            "{{skill-list}}", skill_list
        )

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        new_messages = []

        try:
            # Prepare common arguments
            kwargs = {
                "model": self.config.api.model,
                "messages": full_messages,
                "tools": self.chat_tools,
                "tool_choice": "auto",
                "temperature": self.config.model_params.temperature,
                "max_tokens": self.config.context.max_ai_response_token,
            }
            if self.config.model_params.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.model_params.reasoning_effort

            response = await self.client.chat.completions.create(**kwargs)
            
            response_message = response.choices[0].message
            new_messages.append(response_message.model_dump())

            while response_message.tool_calls:
                full_messages.append(response_message.model_dump())
                
                tool_results = await self._execute_tools(response_message.tool_calls, tool_context)
                full_messages.extend(tool_results)
                new_messages.extend(tool_results)

                # Update messages for next call
                kwargs["messages"] = full_messages
                
                # Get next response from AI
                response = await self.client.chat.completions.create(**kwargs)
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
- If it is a plain fact about the bot itself, a rule the bot should follow, a methodology, or a procedure, save it to knowledge.
- If it is an event (something special happened on a particular day, personal interactions), save it to long-term memory.
- If it reveals new long-term information about a user, update the user info. Be very careful to not overwrite existing info unless it's a clear update.
- If it is trivial, do nothing (it will be forgotten).

Use the provided tools to take action.
"""
        
        sys_prompt_base = "You are a memory manager for a chatbot. Your job is to consolidate short-term memories into knowledge (facts, rules, methodologies) or long-term events."
        sys_prompt = sys_prompt_base + f"\nCurrent Knowledge:\n{self.memory_manager.get_all_knowledges_str()}\n\nCurrent Long-term Memory:\n{self.memory_manager.get_all_long_term_str()}\n\nCurrent User Info:\n{self.memory_manager.get_all_user_info_str()}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            kwargs = {
                "model": self.config.tool_model.model,
                "messages": messages,
                "tools": self.maintenance_tools,
                "tool_choice": "auto",
                "temperature": 0.3,
            }

            response = await self.tool_client.chat.completions.create(**kwargs)
            
            response_message = response.choices[0].message
            if response_message.tool_calls:
                await self._execute_tools(response_message.tool_calls)
                
        except Exception as e:
            print(f"Error consolidating memory: {e}")
            
    def list_skills(self) -> str:
        if not self.config.skills.enabled:
            return "Available Skills:\n" + "Skills feature is disabled."
        
        skills_root = self.config.skills.root_path
        
        # create skills root if not exists
        if not os.path.isdir(skills_root):
            os.makedirs(skills_root)
            return "Available Skills:\n" + "No skills available."
        
        try:
            skill_files = [f for f in os.listdir(skills_root) if f.endswith('.md')]
            if not skill_files:
                return "Available Skills:\n" + "No skills available."
            skill_names = [os.path.splitext(f)[0] for f in skill_files]
            return "Available Skills:\n" + "\n".join(f"- {name}" for name in skill_names)
        except Exception as e:
            return f"Error listing skills"