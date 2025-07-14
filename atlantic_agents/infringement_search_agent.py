#!/usr/bin/env python3
"""
Patent Infringement Evidence Search Agent v7.0 

CONTRACT:
- Host (Python/Ruby/etc) provides: search, storage, persistence, rate limiting
- Agent (LLM) provides: domain reasoning, source classification, term extraction, 
  quality judgement, search strategy, vocabulary building

PORTING GUIDE:
1. Implement 5 simple tools: search_web, store_evidence, store_knowledge, 
   get_knowledge, get_evidence_count
2. Add basic JSON storage and HTTP retry logic (Juli will manage that in Ruby backend)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import httpx

import requests
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, function_tool, Runner
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    sys.stderr.write("[ERROR] Set OPENAI_API_KEY and TAVILY_API_KEY environment variables\n")
    sys.exit(1)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('patent_search_agent.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("patent_search_agent")


KNOWLEDGE_BASE_PATH = Path("./patent_knowledge_base.json")
EVIDENCE_DB_PATH = Path("./patent_evidence_db.json")
CHECKPOINT_PATH = Path("./patent_analysis_checkpoint.json")


class SimpleStore:
    """
    Generic JSON storage - no logic, just save/load
    NOTE: Not concurrency-safe. Use one process per store or add locking in production.
    """
    
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}
    
    def save(self):
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.path.write_text(json.dumps(self.data, indent=2))
    
    def clear(self):
        self.data = {}
        self.save()


knowledge_base = SimpleStore(KNOWLEDGE_BASE_PATH)
evidence_store = SimpleStore(EVIDENCE_DB_PATH)
checkpoint_store = SimpleStore(CHECKPOINT_PATH)


MAX_CONTENT_CHARS = 1500  # Prevent token bloat
MAX_SEARCHES_PER_TURN = 8  # Prevent runaway recursion

# Evidence schema for validation 
EVIDENCE_SCHEMA = {
    "required_fields": ["url", "title", "quote", "source_quality", "relevance", "confidence"],
    "source_quality_values": ["Primary", "Secondary", "Tertiary"],
    "confidence_values": ["High", "Medium", "Low"],
    "min_quote_length": 50
}

# Tool implementations 
@function_tool
async def search_web(query: str, max_results: int = 10) -> str:
    """
    Simple web search. Returns raw results for agent to analyze.
    [{"url": "...", "title": "...", "content": "...", "raw_content": "..."}]
    Rate-limited to prevent runaway costs.
    """
    
    search_count = checkpoint_store.data.get("searches_this_turn", 0)
    if search_count >= MAX_SEARCHES_PER_TURN:
        log.warning(f"[SEARCH] Hit limit of {MAX_SEARCHES_PER_TURN} searches")
        return json.dumps({"error": f"Search limit reached ({MAX_SEARCHES_PER_TURN} per turn)", "results": []})
    
    log.info(f"[SEARCH] Query {search_count + 1}: {query}")
    
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def _search():
        headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_raw_content": True,
            "max_results": max_results
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers=headers,
                timeout = 30.0
            )
            response.raise_for_status()
            await asyncio.sleep(0.5)
            return response.json()
    
    try:
        results = await _search()
        
        
        checkpoint_store.data["searches_this_turn"] = search_count + 1
        checkpoint_store.save()
        
        # Format with size limits to prevent token bloat
        formatted = []
        for r in results.get("results", []):
            formatted.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", "")[:MAX_CONTENT_CHARS].encode('utf-8', errors='ignore').decode('utf-8'),
                "raw_content": r.get("raw_content", "")[:MAX_CONTENT_CHARS].encode('utf-8', errors='ignore').decode('utf-8') if r.get("raw_content") else ""
            })
        
        log.info(f"[SEARCH] Returned {len(formatted)} results")
        return json.dumps(formatted)
        
    except Exception as e:
        log.error(f"Search error: {e}")
        return json.dumps({"error": str(e), "results": []})

@function_tool
async def store_evidence(element_id: str, evidence_data: str) -> str:
    """
    Store evidence with basic validation to prevent downstream breaks.
    Evidence can be a single dict or array of dicts.
    Ensures required fields exist but doesn't judge quality (agent's job).
    """
    log.info(f"[STORE] Evidence for {element_id}")
    
    try:
        evidence = json.loads(evidence_data)
        
        
        if isinstance(evidence, dict):
            evidence = [evidence]
        elif not isinstance(evidence, list):
            return f"Error: Evidence must be a dict or list, got {type(evidence).__name__}"
        
        
        if element_id not in evidence_store.data:
            evidence_store.data[element_id] = []
        
        # Simple dedupp by URL
        existing_urls = {e.get("url") for e in evidence_store.data[element_id] if "url" in e}
        
        stored = 0
        validation_errors = []
        
        for item in evidence:
            # Basic validation
            missing_fields = [f for f in EVIDENCE_SCHEMA["required_fields"] if f not in item]
            if missing_fields:
                validation_errors.append(f"Missing fields: {missing_fields}")
                continue
            
            
            if len(item.get("quote", "")) < EVIDENCE_SCHEMA["min_quote_length"]:
                validation_errors.append(f"Quote too short (<{EVIDENCE_SCHEMA['min_quote_length']} chars)")
                continue
            
            
            if item.get("source_quality") not in EVIDENCE_SCHEMA["source_quality_values"]:
                validation_errors.append(f"Invalid source_quality: {item.get('source_quality')}")
                continue
            
            if item.get("confidence") not in EVIDENCE_SCHEMA["confidence_values"]:
                validation_errors.append(f"Invalid confidence: {item.get('confidence')}")
                continue
            
            
            if item["url"] in existing_urls:
                continue
            
            
            evidence_store.data[element_id].append(item)
            existing_urls.add(item["url"])
            stored += 1
        
        evidence_store.save()
        
        if validation_errors:
            return f"Stored {stored} items. Validation errors: {'; '.join(validation_errors[:3])}"
        return f"Stored {stored} new evidence items"
        
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        log.error(f"Store error: {e}")
        return f"Error: {str(e)}"

@function_tool
async def store_knowledge(key: str, value: str, max_items: int = 50) -> str:
    """
    Store knowledge with automatic size management.
    For list values, keeps only the most recent max_items entries.
    """
    try:
        parsed_value = json.loads(value) if value.startswith('[') or value.startswith('{') else value
        
        # If storing a list and key already exists with a list, append and trim
        if isinstance(parsed_value, list) and key in knowledge_base.data:
            existing = knowledge_base.data.get(key, [])
            if isinstance(existing, list):
                # Append new items and keep only recent ones
                combined = existing + parsed_value
                knowledge_base.data[key] = combined[-max_items:]
                log.info(f"[KNOWLEDGE] Updated {key}, kept last {max_items} items")
            else:
                knowledge_base.data[key] = parsed_value
        else:
            knowledge_base.data[key] = parsed_value
        
        knowledge_base.save()
        return f"Stored knowledge: {key}"
    except Exception as e:
        return f"Error: {str(e)}"

@function_tool
async def get_knowledge(key: str = None, limit: int = 20) -> str:
    """
    Retrieve stored knowledge with size limits to prevent token bloat.
    If no key provided, return recent entries up to limit.
    """
    if key:
        value = knowledge_base.data.get(key, None)
        if value is None:
            return json.dumps({"error": f"Key '{key}' not found"})
        
        
        if isinstance(value, list):
            return json.dumps({key: value[-limit:]})
        return json.dumps({key: value})
    else:
        
        all_items = list(knowledge_base.data.items())
        recent_items = all_items[-limit:]
        
        # For each list value, trim to reasonable size
        result = {}
        for k, v in recent_items:
            if isinstance(v, list):
                result[k] = v[-10:]  
            else:
                result[k] = v
        
        return json.dumps(result)

@function_tool
async def get_evidence_count(element_id: str) -> str:
    """
    Simple count of evidence stored for an element.
    """
    count = len(evidence_store.data.get(element_id, []))
    return json.dumps({"element_id": element_id, "evidence_count": count})

# Agent sys prompt 

AGENT_SYSTEM_PROMPT = """You are an intelligent patent evidence search agent. Your role is to find implementation evidence for patent claims across ANY domain.

CORE MISSION:
Analyze patent elements and find evidence of how companies implement the described functionality.

YOU ARE RESPONSIBLE FOR:
1. Understanding the patent domain from its content
2. Discovering how companies describe these concepts
3. Judging source quality
4. Extracting relevant terms for follow-up searches
5. Storing high-quality evidence

TOOLS AVAILABLE:
- search_web(query): Returns search results (limited to 8 per element)
- store_evidence(element_id, evidence_data): Store validated evidence
- store_knowledge(key, value): Remember patterns, terms, queries (lists auto-trim to 50 items)
- get_knowledge(key=None, limit=20): Retrieve what you've learned
- get_evidence_count(element_id): Check how much evidence exists

CONSTRAINTS:
- Maximum 8 searches per element (plan carefully)
- Evidence must include ALL required fields (see below)
- Raw content is limited to 1500 chars per result

WORKFLOW FOR EACH ELEMENT:

1. UNDERSTAND THE DOMAIN
Read the patent's field and element text. What industry is this? What terminology would practitioners use?

2. CHECK EXISTING KNOWLEDGE
Call get_knowledge() to see if you've learned relevant patterns, terms, or successful queries.

3. INITIAL SEARCH
search_web with broad terms: "[Company] [domain terms]"

4. ANALYZE RESULTS
For each result, YOU determine:
- Source quality: Is this official documentation? Marketing? Blog?
- Relevance: Does it show HOW they implement this?
- Key terms: What vocabulary should I search next?

5. STRATEGIC FOLLOW-UP
Based on discovered terms, search more specifically:
- Product names you found
- Technical terms from the results
- Related documentation
Remember: You have LIMITED searches, so make each count!

6. STORE EVIDENCE
For high-quality, relevant results, store evidence with ALL these fields:
{
  "url": "source URL",
  "title": "page title",
  "quote": "200-500 char excerpt showing implementation",
  "source_quality": "Primary/Secondary/Tertiary",
  "relevance": "how this maps to the patent element",
  "confidence": "High/Medium/Low"
}

## CRITICAL: Evidence Storage Format
When calling store_evidence, ALWAYS format as JSON array:
- CORRECT: [{"url": "...", "title": "...", ...}]
- WRONG: {"url": "...", "title": "...", ...}

Even for single evidence items, wrap in array brackets []!

7. BUILD KNOWLEDGE
Store successful patterns for future elements:
- store_knowledge("vocab_[domain]", ["term1", "term2"])
- store_knowledge("effective_queries", ["query that worked well"])
- store_knowledge("product_names", ["discovered products"])

SOURCE QUALITY GUIDELINES:
- Primary: Official docs, technical specifications, datasheets, regulatory filings
- Secondary: White papers, research, academic sources, conference papers
- Tertiary: Blogs, news, forums (only use if no better sources exist)

DOMAIN ADAPTATION:
- Software: Look for APIs, implementations, configurations
- Chemistry: Look for processes, formulations, methods
- Biotech: Look for protocols, procedures, assays
- Mechanical: Look for specifications, designs, operations
- Medical: Look for clinical data, FDA submissions, technical specs
- Adapt your terminology to what you discover

SEARCH EFFICIENCY:
- Start broad to discover vocabulary
- Use discovered terms for targeted follow-ups
- Don't waste searches on similar queries
- If you hit the search limit, prioritize quality over quantity

IMPORTANT:
- You do ALL the analysis - the tools just fetch and store
- Learn from each search to improve the next
- Store evidence that shows HOW something works, not just that it exists
- Build vocabulary dynamically from what you find
- Plan your searches carefully - you have a limit!"""

# Main Agent Class 
class PatentSearchAgent:
    """Idea is to have a minimal wrapper - agent should do all the work"""
    
    def __init__(self, company: str):
        self.company = company
        self.agent = None
        self.session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_store.data["searches_this_turn"] = 0
        checkpoint_store.save()
    
    async def initialize(self):
        """Create agent with tools"""
        self.agent = Agent(
            name="Patent Evidence Search Agent v7.0",
            instructions=AGENT_SYSTEM_PROMPT,
            model=MODEL_NAME,
            tools=[
                search_web,
                store_evidence,
                store_knowledge,
                get_knowledge,
                get_evidence_count
            ],
            model_settings=ModelSettings(
                temperature=0.3,
                max_tokens=3000
            )
        )
        log.info(f"Agent v7.0 initialized for {self.company}")
    
    async def analyze_element(self, element: Dict[str, Any], patent_context: Dict[str, Any]) -> Dict[str, Any]:
        """Let the agent analyze an element"""
        
        # Reset search counter for this element
        checkpoint_store.data["searches_this_turn"] = 0
        checkpoint_store.save()
        
        prompt = f"""
Analyze this patent element for {self.company}:

PATENT: {patent_context['patent_id']}
FIELD: {patent_context['field_of_invention']}

ELEMENT: {element['element_id']}
TEXT: {element['text']}

Steps:
1. Check existing knowledge with get_knowledge()
2. Search for evidence of how {self.company} implements this
3. Analyze results and extract key terms
4. Do follow-up searches with discovered terms
5. Store high-quality evidence
6. Save useful patterns for future elements

NOTE: You have a limit of {MAX_SEARCHES_PER_TURN} searches per element.

Return a summary of what you found.
"""
        
        try:
            log.info(f"Analyzing element {element['element_id']}")
            
            # Let the agent workworkwork
            result = await Runner.run(self.agent, prompt)
            
            
            evidence_count = len(evidence_store.data.get(element['element_id'], []))
            
           
            searches_used = checkpoint_store.data.get("searches_this_turn", 0)
            
            return {
                "element_id": element['element_id'],
                "evidence_count": evidence_count,
                "searches_executed": searches_used,
                "agent_summary": str(result.content) if hasattr(result, 'content') else str(result)
            }
            
        except Exception as e:
            log.error(f"Error analyzing element: {e}")
            return {
                "element_id": element['element_id'],
                "error": str(e),
                "evidence_count": 0,
                "searches_executed": checkpoint_store.data.get("searches_this_turn", 0)
            }
    
    async def analyze_patent(self, patent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all elements in the patent"""
        
        results = {
            "patent_id": patent_data["patent_id"],
            "company": self.company,
            "field": patent_data.get("field_of_invention", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "agent_version": "v7.0_minimal_agentic",
            "claims": []
        }
        
        patent_context = {
            "patent_id": patent_data["patent_id"],
            "field_of_invention": patent_data.get("field_of_invention", ""),
            "company": self.company
        }
        
        # we process each claim
        for claim in patent_data.get("independent_claims", []):
            claim_result = {
                "claim_number": claim["claim_number"],
                "elements": []
            }
            
            # ..and each element
            for element in claim["elements"]:
                element_result = await self.analyze_element(element, patent_context)
                claim_result["elements"].append(element_result)
                
                # checkpoint simnples
                checkpoint_store.data["current"] = {
                    "claim": claim["claim_number"],
                    "element": element["element_id"]
                }
                checkpoint_store.save()
                
                await asyncio.sleep(1)  # Rate limit
            
            results["claims"].append(claim_result)
        
        
        total_elements = sum(len(c["elements"]) for c in results["claims"])
        total_evidence = sum(e["evidence_count"] for c in results["claims"] for e in c["elements"])
        total_searches = sum(e.get("searches_executed", 0) for c in results["claims"] for e in c["elements"])
        
        results["summary"] = {
            "total_elements": total_elements,
            "total_evidence": total_evidence,
            "total_searches": total_searches,
            "average_evidence_per_element": round(total_evidence / max(total_elements, 1), 1),
            "average_searches_per_element": round(total_searches / max(total_elements, 1), 1)
        }
        
        return results

# Main execution
async def main():
    parser = argparse.ArgumentParser(
        description="Minimal Patent Search Agent v7.0 - Portable Agentic Design"
    )
    parser.add_argument("patent_file", type=Path, help="Structured patent JSON")
    parser.add_argument("--company", required=True, help="Target company")
    parser.add_argument("--output", type=Path, help="Output file")
    parser.add_argument("--clear", action="store_true", help="Clear previous data")
    
    args = parser.parse_args()
    
    
    if not args.patent_file.exists():
        log.error(f"Patent file not found: {args.patent_file}")
        sys.exit(1)
    
    patent_data = json.loads(args.patent_file.read_text())
    log.info(f"Loaded patent {patent_data['patent_id']}")
    
    
    if args.clear:
        knowledge_base.clear()
        evidence_store.clear()
        checkpoint_store.clear()
        log.info("Cleared all data")
    
   
    agent = PatentSearchAgent(args.company)
    await agent.initialize()
    
    try:
        log.info(f"Starting analysis for {args.company}")
        results = await agent.analyze_patent(patent_data)
        
        
        if args.output:
            args.output.write_text(json.dumps(results, indent=2))
            log.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        
        print(f"\n{'='*60}")
        print(f"Analysis Complete (v7.0): {args.company}")
        print(f"Patent: {patent_data['patent_id']}")
        print(f"Elements: {results['summary']['total_elements']}")
        print(f"Evidence: {results['summary']['total_evidence']}")
        print(f"Searches: {results['summary']['total_searches']}")
        print(f"Avg Evidence/Element: {results['summary']['average_evidence_per_element']}")
        print(f"Avg Searches/Element: {results['summary']['average_searches_per_element']}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        log.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())