from typing import Any, Dict, List

from pepperbot.config import Config


class ToolRegistry:
    def __init__(self, config: Config):
        self.config = config

    def chat_tools(self) -> List[Dict[str, Any]]:
        image_properties: Dict[str, Any] = {
            "description": {
                "type": "string",
                "description": "A self-contained image prompt including relevant subjects, style and details from the conversation. The image backend does not receive chat history.",
            },
        }
        if self.config.image_generation.accepts_image_input():
            image_properties["msg_id"] = {
                "type": "string",
                "description": "Optional internal message ID of a reference image in thread history, such as m12.",
            }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_user_info",
                    "description": "Fetch available Telegram user/profile information, current chat membership, known-user notes and accessible profile photos. Some fields may be unavailable due to Telegram privacy/access limits. Profile images are included when the chat model supports vision.",
                    "parameters": {
                        "type": "object",
                        "properties": {"user_id": {"type": "integer"}},
                        "required": ["user_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image based on a description.",
                    "parameters": {
                        "type": "object",
                        "properties": image_properties,
                        "required": ["description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_short_term_memory",
                    "description": "Add a significant event or fact to short-term memory. Timestamp is automatically attached. Content should NOT contain timestamp data.",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_user_info",
                    "description": "Update or add information about a specific Telegram user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "integer", "description": "The Telegram user ID."},
                            "name": {"type": "string", "description": "The stable name to call the user by."},
                            "description": {"type": "string", "description": "Description of the user's personality, habits, preferences, or important context."},
                            "telegram_username": {
                                "type": "string",
                                "description": "Optional Telegram username. Must start with @, for example @alice.",
                                "pattern": "^@",
                            },
                        },
                        "required": ["user_id", "name", "description"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_url_content",
                    "description": "Fetch and read text content from a webpage URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "set_scheduled_task",
                    "description": "Schedule a task to be executed later.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_in_minute": {"type": "integer"},
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["time_in_minute", "title", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_scheduled_task_list",
                    "description": "Get the list of currently scheduled tasks.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "randint",
                    "description": "Return random integer in range [a, b], including both end points.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                },
            },
        ]
        if self.config.black_list.enable:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "block_user",
                        "description": "Temporarily block a user from interacting with the bot.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "integer"},
                                "duration_minutes": {"type": "integer"},
                            },
                            "required": ["user_id", "duration_minutes"],
                        },
                    },
                }
            )
        if self.config.skills.enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_skill",
                        "description": "Fetch and load a skill by name.",
                        "parameters": {
                            "type": "object",
                            "properties": {"skill_name": {"type": "string"}},
                            "required": ["skill_name"],
                        },
                    },
                }
            )
        return tools
