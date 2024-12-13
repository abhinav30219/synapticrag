"""
LLM interface module for SynapticRAG.
"""
from .llm_interface import LLMInterface, LLMOutput
from .prompts import Prompts, PromptTemplate

__all__ = [
    'LLMInterface', 'LLMOutput',
    'Prompts', 'PromptTemplate'
]
