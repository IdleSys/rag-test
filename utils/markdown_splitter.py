from typing import List

from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter


def split_markdown(markdown_content: str) -> List[Document]:
    headers = [
        ("#", "h1"),
        ("##", "h2"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers)
    return splitter.split_text(markdown_content)
