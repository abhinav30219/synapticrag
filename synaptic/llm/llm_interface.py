from typing import Dict, List, Optional, Union, Any
import torch
import anthropic
import requests
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMOutput:
    """Container for LLM generation output"""
    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class LLMInterface:
    """Interface for interacting with language models"""
    
    def __init__(
        self,
        model_name_or_path: str,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize API clients
        self.anthropic_client = None
        self.openrouter_headers = None
        
        if "claude" in model_name_or_path.lower():
            if not anthropic_api_key:
                raise ValueError("Anthropic API key required for Claude models")
            self.anthropic_client = anthropic.Client(api_key=anthropic_api_key)
        else:
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key required for non-Claude models")
            self.openrouter_headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "HTTP-Referer": "https://github.com/yourusername/synapticrag",
                "X-Title": "SynapticRAG"
            }

        # Store generation parameters
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", 50)
        
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Union[LLMOutput, List[LLMOutput]]:
        """Generate text based on prompt"""
        
        # Handle single prompt
        if isinstance(prompt, str):
            return self._generate_single(
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                **kwargs
            )
        
        # Handle multiple prompts
        return [
            self._generate_single(
                p,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                **kwargs
            )
            for p in prompt
        ]
    
    def _generate_single(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> LLMOutput:
        """Generate text for a single prompt"""
        
        if self.anthropic_client:
            return self._generate_with_anthropic(
                prompt,
                max_new_tokens,
                temperature,
                **kwargs
            )
        else:
            return self._generate_with_openrouter(
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                **kwargs
            )
    
    def _generate_with_anthropic(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMOutput:
        """Generate text using Anthropic's Claude API"""
        
        try:
            message = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=max_new_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response = message.content[0].text
            
            # Convert response to tokens (approximate)
            tokens = list(range(len(response.split())))  # Placeholder
            
            return LLMOutput(
                text=response,
                tokens=tokens,
                metadata={
                    "model": self.model_name,
                    "usage": message.usage
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            raise
    
    def _generate_with_openrouter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> LLMOutput:
        """Generate text using OpenRouter API"""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self.openrouter_headers,
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_new_tokens or self.max_tokens,
                    "temperature": temperature or self.temperature,
                    "top_p": top_p or self.top_p,
                    "top_k": top_k or self.top_k
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result["choices"][0]["message"]["content"]
            
            # Convert response to tokens (approximate)
            tokens = list(range(len(generated_text.split())))  # Placeholder
            
            return LLMOutput(
                text=generated_text,
                tokens=tokens,
                metadata={
                    "model": self.model_name,
                    "usage": result.get("usage", {})
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating with OpenRouter: {e}")
            raise
    
    def encode(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        """Encode text to embeddings using local models or APIs"""
        raise NotImplementedError("Encoding not implemented for API-based interface")
    
    def decode(
        self,
        tokens: Union[List[int], torch.Tensor],
        **kwargs
    ) -> str:
        """Decode tokens to text"""
        raise NotImplementedError("Decoding not implemented for API-based interface")
