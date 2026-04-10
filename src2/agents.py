"""
agents.py — Specialized agent implementations for Stardew Valley chatbot.

Three agents:
  1. ItemFinder — Find items, resources, crafting, locations, tools
  2. FriendshipFinder — Village relationships, romance, heart events
  3. CropPlanner — Farming, crops, seasons, growth, profit
  4. DefaultAgent — Fallback for general Stardew questions

Each agent has:
  - A system prompt with specific instructions
  - Access to the retriever (for RAG-based answers)
  - An answer() method that returns a response
"""

from dataclasses import dataclass
from typing import Optional

from retriever import Retriever
from llm import LLMClient


@dataclass
class AgentResponse:
    """Response from an agent."""
    answer: str
    agent_type: str
    reasoning: Optional[str] = None
    sources: Optional[list[dict]] = None
    tokens_used: Optional[dict] = None


class Agent:
    """Base class for all agents."""
    
    def __init__(self, retriever: Retriever, llm: LLMClient, name: str):
        self.retriever = retriever
        self.llm = llm
        self.name = name
        self.system_prompt = ""  # override in subclasses
    
    def answer(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.25,
        include_reasoning: bool = False,
    ) -> AgentResponse:
        """
        Answer a question using RAG + LLM.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            min_score: Minimum relevance score threshold
            include_reasoning: Include chain-of-thought
        
        Returns:
            AgentResponse with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant context
        context = self.retriever.build_context(query, top_k=top_k, min_score=min_score)
        chunks = self.retriever.retrieve_with_threshold(
            query, top_k=top_k, min_score=min_score
        )
        
        if context == "No relevant wiki content found.":
            return AgentResponse(
                answer="I couldn't find relevant information to answer that question. Could you rephrase or ask something more specific about Stardew Valley?",
                agent_type=self.name,
                sources=[],
                tokens_used={},
            )
        
        # Step 2: Format message for LLM
        user_message = f"<wiki_context>\n{context}\n</wiki_context>\n\nQuestion: {query}"
        
        # Step 3: Call LLM with agent's system prompt
        try:
            llm_resp = self.llm.complete(
                messages=[{"role": "user", "content": user_message}],
                system=self.system_prompt,
                max_tokens=1024,
                temperature=0.6,
            )
        except Exception as e:
            return AgentResponse(
                answer=f"Error querying LLM: {str(e)}",
                agent_type=self.name,
                sources=[],
                tokens_used={},
            )
        
        # Step 4: Format response
        return AgentResponse(
            answer=llm_resp.answer,
            agent_type=self.name,
            reasoning=llm_resp.reasoning if include_reasoning else None,
            sources=[
                {
                    "page_title": c.page_title,
                    "heading": c.heading,
                    "url": c.url,
                    "score": round(c.score, 4),
                }
                for c in chunks
            ],
            tokens_used={
                "input": llm_resp.input_tokens,
                "output": llm_resp.output_tokens,
                "total": llm_resp.total_tokens,
            },
        )


class ItemFinder(Agent):
    """Agent specializing in items, resources, materials, and locations."""
    
    def __init__(self, retriever: Retriever, llm: LLMClient):
        super().__init__(retriever, llm, "ItemFinder")
        self.system_prompt = """\
You are an expert on Stardew Valley items, resources, tools, and locations.
Answer the player's question about items using ONLY the wiki context provided in the <wiki_context> block.

Rules:
- Be concise and practical. Use bullet points for lists.
- Include item prices, where to get items, and how to obtain/craft them when relevant.
- CRITICAL: Your answer MUST end cleanly. Do NOT add any text after your final bullet point or sentence.
- CRITICAL: NO citations, NO links, NO "For details, see...", NO "Source:", NO "Check the wiki".
- Your response ends where your content ends — nothing after that.
- FORBIDDEN ENDINGS: "For details, see [link]", "Source: [link]", "Check the wiki", "According to [page]"
- If the context doesn't answer the question, say so — do not invent facts.
- Never discuss topics unrelated to Stardew Valley items and resources.
"""


class FriendshipFinder(Agent):
    """Agent specializing in friendships, romance, and villager relationships."""
    
    def __init__(self, retriever: Retriever, llm: LLMClient):
        super().__init__(retriever, llm, "FriendshipFinder")
        self.system_prompt = """\
You are an expert on Stardew Valley villager relationships, friendships, and romance.
Answer the player's question about villagers using ONLY the wiki context provided in the <wiki_context> block.

Rules:
- Be concise and helpful. Use bullet points for multi-step instructions.
- Include favorite gifts, schedules, heart events, and dating/marriage info when relevant.
- CRITICAL: Your answer MUST end cleanly. Do NOT add any text after your final bullet point or sentence.
- CRITICAL: NO citations, NO links, NO "For details, see...", NO "Source:", NO "Check the wiki".
- Your response ends where your content ends — nothing after that.
- FORBIDDEN ENDINGS: "For details, see [link]", "Source: [link]", "Check the wiki", "According to [page]"
- If the context doesn't answer the question, say so — do not invent facts.
- Never discuss topics unrelated to Stardew Valley villagers and relationships.
"""


class CropPlanner(Agent):
    """Agent specializing in crops, farming, seasons, and agricultural planning."""
    
    def __init__(self, retriever: Retriever, llm: LLMClient):
        super().__init__(retriever, llm, "CropPlanner")
        self.system_prompt = """\
You are an expert on Stardew Valley farming, crops, and seasonal planning.
Answer the player's question about farming using ONLY the wiki context provided in the <wiki_context> block.

Rules:
- Be concise and practical. Use bullet points for multi-step instructions or crop lists.
- Include planting seasons, growth time, profit, and watering needs when relevant.
- CRITICAL: Your answer MUST end cleanly. Do NOT add any text after your final bullet point or sentence.
- CRITICAL: NO citations, NO links, NO "For details, see...", NO "Source:", NO "Check the wiki".
- Your response ends where your content ends — nothing after that.
- FORBIDDEN ENDINGS: "For details, see [link]", "Source: [link]", "Check the wiki", "According to [page]"
- If the context doesn't answer the question, say so — do not invent facts.
- Never discuss topics unrelated to Stardew Valley crops and farming.
"""


class DefaultAgent(Agent):
    """Fallback agent for general Stardew Valley questions."""
    
    def __init__(self, retriever: Retriever, llm: LLMClient):
        super().__init__(retriever, llm, "DefaultAgent")
        self.system_prompt = """\
You are a knowledgeable and friendly guide for the farming simulation game Stardew Valley.
Answer the player's question using ONLY the wiki context provided in the <wiki_context> block.

Rules:
- Be concise and helpful. Use bullet points for multi-step instructions or lists.
- CRITICAL: Your answer MUST end cleanly. Do NOT add any text after your final bullet point or sentence.
- CRITICAL: NO citations, NO links, NO "For details, see...", NO "Source:", NO "Check the wiki".
- Your response ends where your content ends — nothing after that.
- FORBIDDEN ENDINGS: "For details, see [link]", "Source: [link]", "Check the wiki", "According to [page]"
- If the context doesn't fully answer the question, say so — do not invent facts.
- If the question is ambiguous, ask one brief clarifying question.
- Never discuss topics unrelated to Stardew Valley.
"""


def get_agent(
    agent_type: str, retriever: Retriever, llm: LLMClient
) -> Agent:
    """
    Factory function to get the appropriate agent.
    
    Args:
        agent_type: "items", "friendship", "crops", or "default"
    
    Returns:
        Agent instance
    """
    if agent_type == "items":
        return ItemFinder(retriever, llm)
    elif agent_type == "friendship":
        return FriendshipFinder(retriever, llm)
    elif agent_type == "crops":
        return CropPlanner(retriever, llm)
    else:
        return DefaultAgent(retriever, llm)
