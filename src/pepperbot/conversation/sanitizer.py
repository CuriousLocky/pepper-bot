import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class ParsedResponse:
    text: str
    retry: bool = False


class ResponseSanitizer:
    def __init__(self, bot_names: Iterable[str]):
        escaped = [re.escape(name) for name in bot_names if name]
        self.bot_names = escaped or ["Pepper"]

    def clean(self, text: str) -> str:
        return self.parse(text).text

    def parse(self, text: str) -> ParsedResponse:
        if not text:
            return ParsedResponse("", retry=True)
        cleaned = self._strip_code_fence(text.strip()).strip()
        xml_text = self._extract_xml_content(cleaned)
        if xml_text is not None:
            cleaned_xml = self._clean_plain_text(xml_text)
            return ParsedResponse(cleaned_xml, retry=not bool(cleaned_xml))
        if self._looks_xml_like(cleaned):
            return ParsedResponse("", retry=True)
        return ParsedResponse(self._clean_plain_text(cleaned), retry=False)

    def _clean_plain_text(self, text: str) -> str:
        cleaned = html.unescape(text or "").strip()
        for _ in range(3):
            next_value = self._strip_leading_prefix(cleaned).strip()
            if next_value == cleaned:
                break
            cleaned = next_value
        return cleaned.strip()

    def _strip_code_fence(self, text: str) -> str:
        match = re.fullmatch(r"```(?:xml|text|markdown)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else text

    def _extract_xml_content(self, text: str) -> Optional[str]:
        parsed = self._extract_with_elementtree(text)
        if parsed is not None:
            return parsed
        for tag in ("telegram_reply", "reply", "response", "content"):
            match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        message_matches = list(
            re.finditer(r"<message\b(?P<attrs>[^>]*)>(?P<body>.*?)</message>", text, re.DOTALL | re.IGNORECASE)
        )
        if message_matches:
            assistant = [m for m in message_matches if re.search(r'role\s*=\s*(["\'])assistant\1', m.group("attrs"), re.IGNORECASE)]
            if assistant:
                return assistant[-1].group("body").strip()
            if len(message_matches) == 1:
                return message_matches[0].group("body").strip()
        return None

    def _extract_with_elementtree(self, text: str) -> Optional[str]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None
        root_tag = self._local_name(root.tag)
        if root_tag in {"telegram_reply", "reply", "response", "content"}:
            return self._element_text(root)
        if root_tag == "message":
            return self._element_text(root)
        for wanted in ("telegram_reply", "reply", "response", "content"):
            found = root.find(f".//{wanted}")
            if found is not None:
                return self._element_text(found)
        assistant_messages = [
            element
            for element in root.findall(".//message")
            if (element.attrib.get("role") or "").lower() == "assistant"
        ]
        if assistant_messages:
            return self._element_text(assistant_messages[-1])
        messages = root.findall(".//message")
        if len(messages) == 1:
            return self._element_text(messages[0])
        return None

    def _element_text(self, element: ET.Element) -> str:
        return "".join(element.itertext()).strip()

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    def _looks_xml_like(self, text: str) -> bool:
        if not text.lstrip().startswith("<"):
            return False
        return bool(re.search(r"</?[a-zA-Z_][\w:.-]*(?:\s|>|/>)", text))

    def _strip_leading_prefix(self, text: str) -> str:
        name_pattern = "|".join(self.bot_names)
        generic = r"(?:assistant|bot|机器人|回复)"
        patterns = [
            rf"^\s*(?:\[?msg\s*#?\s*\d+\]?\s*)?(?:{name_pattern}|{generic})\s*(?:\([^)]*\)\s*)?[：:]\s*",
            rf"^\s*\[?msg\s*#?\s*\d+\]?\s*(?:{name_pattern})?\s*(?:\([^)]*\)\s*)?[：:]\s*",
            rf"^\s*(?:<[^>]+>\s*)?(?:{name_pattern})\s+[（(]reply\s+to\s+msg\s+\d+[）)]\s*[：:]\s*",
        ]
        for pattern in patterns:
            stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            if stripped != text:
                return stripped
        return text
