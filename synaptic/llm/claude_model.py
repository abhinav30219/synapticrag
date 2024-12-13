"""Claude LLM interface for SynapticRAG"""
import os
import json
import re
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from anthropic import Anthropic
from dotenv import load_dotenv

@dataclass
class ClaudeConfig:
    """Configuration for Claude model"""
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50

class ClaudeModel:
    """Interface for Anthropic's Claude model"""
    
    def __init__(
        self,
        config: Optional[ClaudeConfig] = None,
        api_key: Optional[str] = None
    ):
        self.config = config or ClaudeConfig()
        
        # Load API key from environment if not provided
        if not api_key:
            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No API key provided and none found in environment")
        
        # Initialize client
        self.client = Anthropic(api_key=api_key)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        keyword_extraction: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, List[str]]]:
        """Generate text using Claude"""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response
        response = self.client.messages.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            **kwargs
        )
        
        text = response.content[0].text
        
        # Handle keyword extraction specially
        if keyword_extraction:
            try:
                # Find JSON in response
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Parse JSON
                    keywords = json.loads(json_str)
                    # Ensure expected structure
                    if not isinstance(keywords, dict):
                        raise ValueError("Expected dict response")
                    if "high_level_keywords" not in keywords or "low_level_keywords" not in keywords:
                        raise ValueError("Missing required keyword fields")
                    return keywords
                else:
                    raise ValueError("No JSON found in response")
            except Exception as e:
                # Return empty keywords on error
                return {
                    "high_level_keywords": [],
                    "low_level_keywords": []
                }
        
        return text
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts"""
        responses = []
        for prompt in prompts:
            response = self.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            responses.append(response)
        return responses
