from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Template for system prompts"""
    template: str
    input_variables: List[str]

class Prompts:
    """Collection of system prompts for different operations"""
    
    # Memory-related prompts
    MEMORY_CONTEXT = PromptTemplate(
        template="""You are processing a document to form a memory of its content. 
Read and understand the following text carefully:

{context}

Acknowledge once you have processed the content.""",
        input_variables=["context"]
    )
    
    MEMORY_QUERY = PromptTemplate(
        template="""Based on your memory of the processed documents, help me understand:

{question}

Generate a response that demonstrates your understanding of the content.""",
        input_variables=["question"]
    )
    
    GENERATE_CLUES = PromptTemplate(
        template="""Based on the following query, generate specific clues that would help retrieve relevant information from the documents:

Query: {query}

Generate 3-5 specific clues that would help find relevant information. Each clue should be a specific phrase or statement that might appear in the documents.

Format your response as a list of clues, one per line.""",
        input_variables=["query"]
    )
    
    # Entity extraction prompts
    EXTRACT_ENTITIES = PromptTemplate(
        template="""Analyze the following text and extract key entities and their relationships:

{text}

For each entity, provide:
1. Entity name
2. Entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.)
3. Brief description

For each relationship between entities, provide:
1. Source entity
2. Target entity
3. Relationship type
4. Brief description
5. Keywords describing the relationship

Format your response as JSON with 'entities' and 'relationships' arrays.""",
        input_variables=["text"]
    )
    
    # Graph-based retrieval prompts
    GRAPH_QUERY = PromptTemplate(
        template="""Given the following query, identify key concepts and their potential relationships that would be relevant in a knowledge graph:

Query: {query}

Identify:
1. Key entities/concepts to look for
2. Important relationships between these entities
3. Any specific attributes or properties that might be relevant

Format your response as a structured list.""",
        input_variables=["query"]
    )
    
    # Response generation prompts
    GENERATE_RESPONSE = PromptTemplate(
        template="""Based on the following context and query, generate a comprehensive response:

Query: {query}

Relevant Context:
{context}

Memory Clues:
{clues}

Generate a response that:
1. Directly answers the query
2. Uses information from both the context and memory clues
3. Maintains factual accuracy
4. Provides a coherent and well-structured explanation

Response:""",
        input_variables=["query", "context", "clues"]
    )
    
    # Summarization prompts
    SUMMARIZE_DOCUMENT = PromptTemplate(
        template="""Create a comprehensive summary of the following document:

{text}

Focus on:
1. Main topics and themes
2. Key findings or arguments
3. Important relationships between concepts
4. Significant conclusions

Provide a well-structured summary that captures the essential information.""",
        input_variables=["text"]
    )
    
    # Hybrid retrieval prompts
    HYBRID_QUERY = PromptTemplate(
        template="""Process the following query for both memory-based and graph-based retrieval:

Query: {query}

1. Memory Perspective:
- What key concepts should we recall from the document memory?
- What contextual clues would help retrieve relevant information?

2. Graph Perspective:
- What entities and relationships are we looking for?
- What graph patterns might be relevant?

Provide structured guidance for both retrieval approaches.""",
        input_variables=["query"]
    )
    
    # Error handling prompts
    HANDLE_AMBIGUITY = PromptTemplate(
        template="""The following query contains ambiguity that needs to be resolved:

Query: {query}

Identify:
1. Sources of ambiguity
2. Possible interpretations
3. Clarifying questions needed
4. Suggested approaches for each interpretation

Provide guidance on how to proceed with this ambiguous query.""",
        input_variables=["query"]
    )
    
    # Quality control prompts
    VALIDATE_RESPONSE = PromptTemplate(
        template="""Validate the following response against the provided context:

Response: {response}

Context: {context}

Check for:
1. Factual accuracy
2. Completeness
3. Relevance to the query
4. Logical consistency

Provide an assessment of the response quality.""",
        input_variables=["response", "context"]
    )

    @classmethod
    def format(cls, template_name: str, **kwargs) -> str:
        """Format a prompt template with provided variables"""
        template = getattr(cls, template_name)
        
        # Validate input variables
        missing_vars = [
            var for var in template.input_variables
            if var not in kwargs
        ]
        if missing_vars:
            raise ValueError(
                f"Missing required variables for {template_name}: {missing_vars}"
            )
        
        return template.template.format(**kwargs)

    @classmethod
    def get_template(cls, template_name: str) -> PromptTemplate:
        """Get a prompt template by name"""
        return getattr(cls, template_name)

    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available prompt templates"""
        return [
            attr for attr in dir(cls)
            if isinstance(getattr(cls, attr), PromptTemplate)
        ]
