"""Clue generator for memory-augmented retrieval"""
import json
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Clue:
    """Container for clues"""
    text: str
    score: float
    level: str  # 'high' or 'low'
    metadata: Optional[Dict] = None

class ClueGenerator:
    """Generates clues from text using LLM"""
    
    def __init__(
        self,
        llm: Any,
        max_clues: int = 5,
        min_score: float = 0.3,
        temperature: float = 0.7,
        max_tokens: int = 150
    ):
        self.llm = llm
        self.max_clues = max_clues
        self.min_score = min_score
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize prompt templates
        self.keyword_extraction_prompt = """Given the following text, extract two types of keywords:

1. High-level keywords: Abstract concepts, themes, and key topics that capture the broader meaning
2. Low-level keywords: Specific entities, terms, and details that are directly mentioned

Text: {text}

Please format your response as JSON with two lists:
{{
    "high_level_keywords": ["keyword1", "keyword2", ...],
    "low_level_keywords": ["keyword1", "keyword2", ...]
}}

Focus on the most relevant and important keywords. Each list should have 3-5 keywords.

Examples:

Text: "The new AI model demonstrated remarkable performance in natural language processing tasks, achieving state-of-the-art results on multiple benchmarks."
{{
    "high_level_keywords": ["artificial intelligence advancement", "technological innovation", "performance evaluation"],
    "low_level_keywords": ["AI model", "natural language processing", "benchmarks", "state-of-the-art results"]
}}

Text: "What are the environmental consequences of deforestation on biodiversity?"
{{
    "high_level_keywords": ["environmental impact", "ecosystem disruption", "conservation"],
    "low_level_keywords": ["deforestation", "biodiversity", "species loss", "habitat destruction"]
}}

Now extract keywords from the following text:
{text}
""".strip()
    
    def generate_clues(
        self,
        text: str,
        memory_state: Optional[Dict] = None,
        max_clues: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Clue]:
        """Generate clues from text using LLM"""
        if not text or not isinstance(text, str) or not text.strip():
            return []
        
        max_clues = max_clues or self.max_clues
        min_score = min_score or self.min_score
        
        # Generate keywords
        prompt = self.keyword_extraction_prompt.format(text=text)
        
        try:
            # Use generate() with keyword_extraction=True
            response = self.llm.generate(
                prompt,
                temperature=self.temperature,
                keyword_extraction=True
            )
            
            # Process high-level keywords
            high_level_clues = []
            for keyword in response.get("high_level_keywords", []):
                # Higher base score for abstract concepts
                score = 0.8
                if memory_state is not None:
                    score *= 1.2  # 20% boost for memory-influenced clues
                
                high_level_clues.append(Clue(
                    text=keyword.strip(),
                    score=score,
                    level="high",
                    metadata={
                        "memory_state": bool(memory_state),
                        "type": "concept"
                    }
                ))
            
            # Process low-level keywords
            low_level_clues = []
            for keyword in response.get("low_level_keywords", []):
                # Calculate score based on presence in text
                text_words = set(text.lower().split())
                keyword_words = set(keyword.lower().split())
                overlap = len(text_words & keyword_words) / len(keyword_words)
                score = 0.6 + (0.4 * overlap)  # Base score plus overlap influence
                
                if memory_state is not None:
                    score *= 1.1  # 10% boost for memory-influenced clues
                
                low_level_clues.append(Clue(
                    text=keyword.strip(),
                    score=score,
                    level="low",
                    metadata={
                        "memory_state": bool(memory_state),
                        "type": "entity",
                        "overlap": overlap
                    }
                ))
            
            # Combine and sort clues
            all_clues = high_level_clues + low_level_clues
            all_clues = [clue for clue in all_clues if clue.score >= min_score]
            all_clues.sort(key=lambda x: x.score, reverse=True)
            
            return all_clues[:max_clues]
            
        except Exception as e:
            print(f"Error generating clues: {str(e)}")
            return []
    
    def batch_generate_clues(
        self,
        texts: List[str],
        batch_size: int = 8,
        memory_states: Optional[List[Dict]] = None,
        max_clues: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[List[Clue]]:
        """Generate clues for multiple texts in batches"""
        if not texts:
            return []
            
        # Process texts in batches
        all_clues = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating clues"):
            batch_texts = texts[i:i + batch_size]
            batch_states = None
            if memory_states is not None:
                batch_states = memory_states[i:i + batch_size]
            
            # Generate clues for each text in batch
            batch_clues = []
            for j, text in enumerate(batch_texts):
                state = batch_states[j] if batch_states else None
                clues = self.generate_clues(
                    text,
                    memory_state=state,
                    max_clues=max_clues,
                    min_score=min_score
                )
                batch_clues.append(clues)
            
            all_clues.extend(batch_clues)
        
        return all_clues
