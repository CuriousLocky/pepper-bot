import re
from pathlib import Path
from typing import Dict, List, Optional

from pepperbot.config import TemplateConfig, TemplateMessage
from pepperbot.providers.base import ChatMessage


MACRO_RE = re.compile(r"{{\s*([a-zA-Z0-9_\-.]+)\s*}}")


class TemplateRenderer:
    def __init__(self, template: TemplateConfig):
        self.template = template

    def render_messages(self, variables: Dict[str, str]) -> List[ChatMessage]:
        rendered: List[ChatMessage] = []
        for message in self.template.messages:
            rendered_message = self.render_template_message(message, variables)
            if rendered_message is not None:
                rendered.append(rendered_message)
        return rendered

    def render_template_message(
        self,
        message: TemplateMessage,
        variables: Dict[str, str],
    ) -> Optional[ChatMessage]:
        if message.type != "text":
            return None
        if not message.role:
            raise ValueError(f"Template text message '{message.id}' must define role")
        content = self._load_content(message)
        content = self._replace_macros(content, variables).strip()
        if message.omit_if_empty and not content:
            return None
        return ChatMessage(role=message.role, content=content)

    def render_repair_message(self, variables: Dict[str, str]) -> ChatMessage:
        message = self.template.empty_response_repair_message
        rendered = self.render_template_message(message, variables)
        if rendered is None:
            raise ValueError("empty_response_repair_message must be a text message")
        return rendered

    def _load_content(self, message: TemplateMessage) -> str:
        if message.content_file:
            return Path(message.content_file).read_text(encoding="utf-8")
        return message.content

    def _replace_macros(self, content: str, variables: Dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            return variables.get(match.group(1), "")

        return MACRO_RE.sub(replace, content)
