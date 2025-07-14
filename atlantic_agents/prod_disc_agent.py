#!/usr/bin/env python3
"""
Patent Product Discovery Agent v2.0 - Enhanced with Meta-Cognitive Tools
Pure transaction layer - all intelligence lives in the LLM agent

PURPOSE:
Find products/services that relate to patent claims across ANY domain.
This agent discovers WHAT products exist, not HOW they work.

ENHANCEMENTS IN V2.0:
- Self-reflection loop for coverage analysis
- Functional role extraction for better understanding
- Search variation suggestions for broader discovery
- Multi-angle discovery strategy

CONTRACT:
- Host provides: search, storage, persistence, rate limiting, meta-tools
- Agent provides: domain reasoning, product identification, relevance scoring
- No domain logic in host code - completely portable to any language
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

import requests
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, function_tool, Runner
from tenacity import retry, stop_after_attempt, wait_exponential
import jsonschema

# CConfigss
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    sys.stderr.write("[ERROR] Set OPENAI_API_KEY and TAVILY_API_KEY environment variables\n")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('product_discovery_agent.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("product_discovery_agent")


PRODUCT_KNOWLEDGE_PATH = Path("./product_knowledge_base.json")
PRODUCT_DB_PATH = Path("./discovered_products_db.json")
CHECKPOINT_PATH = Path("./product_discovery_checkpoint.json")


class SimpleStore:
    """
    
    NOTE: NOT concurrency-safe for now. Use one process per store or add locking in production @Juli.
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


knowledge_base = SimpleStore(PRODUCT_KNOWLEDGE_PATH)
product_store = SimpleStore(PRODUCT_DB_PATH)
checkpoint_store = SimpleStore(CHECKPOINT_PATH)


MAX_CONTENT_CHARS = 1500  # Prevent token bloat
MAX_SEARCHES_PER_TURN = 8  # Prevent runaway recursion

# JSON Schema for product validation 
PRODUCT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["product_name", "company", "category", "url", "description", "relevance", "confidence"],
    "properties": {
        "product_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200
        },
        "company": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100
        },
        "category": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100
        },
        "url": {
            "type": "string",
            "minLength": 4,
            "pattern": "^https?://"
        },
        "description": {
            "type": "string",
            "minLength": 20,
            "maxLength": 500
        },
        "relevance": {
            "type": "string",
            "minLength": 20,
            "maxLength": 500
        },
        "confidence": {
            "type": "string",
            "enum": ["High", "Medium", "Low"]
        }
    },
    "additionalProperties": False
}

# Tooling implementations 
@function_tool
async def search_products(query: str, max_results: int = 10) -> str:
    """
    Search for products/services. Returns raw results for agent to analyze.
    Agent must determine what products are mentioned and their relevance.
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
        
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        await asyncio.sleep(0.5)  # Rate limit
        return response.json()
    
    try:
        results = await _search()
        
        
        checkpoint_store.data["searches_this_turn"] = search_count + 1
        checkpoint_store.save()
        
        
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
async def store_product(element_id: str, product_data: str) -> str:
    """
    Store discovered products with validation.
    Products must include name, company, category, url, description, relevance, and confidence.
    Enforces element_id matching to prevent data corruption.
    """
    log.info(f"[STORE] Product for {element_id}")
    
    try:
        products = json.loads(product_data)
        
        
        if isinstance(products, dict):
            products = [products]
        elif not isinstance(products, list):
            return f"Error: Products must be a dict or list, got {type(products).__name__}"
        
        
        if element_id not in product_store.data:
            product_store.data[element_id] = []
        
        
        existing_products = {(p.get("product_name"), p.get("company")) 
                           for p in product_store.data[element_id] 
                           if "product_name" in p and "company" in p}
        
        stored = 0
        validation_errors = []
        
        for item in products:
            # ELEMENT-ID GUARDRAIL: Prevent storing under wrong element
            if "element_id" in item and item["element_id"] != element_id:
                validation_errors.append(f"Element ID mismatch: expected {element_id}, got {item['element_id']}")
                log.warning(f"[GUARDRAIL] Blocked wrong element_id: {item.get('element_id')} != {element_id}")
                continue
            
            # JSON Schema validation
            try:
                jsonschema.validate(instance=item, schema=PRODUCT_SCHEMA)
            except jsonschema.exceptions.ValidationError as e:
                validation_errors.append(f"Schema validation: {e.message}")
                continue
            
            
            product_key = (item["product_name"], item["company"])
            if product_key in existing_products:
                continue
            
            # Store validated product
            product_store.data[element_id].append(item)
            existing_products.add(product_key)
            stored += 1
        
        product_store.save()
        
        if validation_errors:
            return f"Stored {stored} products. Validation errors: {'; '.join(validation_errors[:3])}"
        return f"Stored {stored} new products"
        
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        log.error(f"Store error: {e}")
        return f"Error: {str(e)}"

@function_tool
async def store_knowledge(key: str, value: str, max_items: int = 50) -> str:
    """
    Store discovered patterns, categories, or terminology.
    """
    try:
        parsed_value = json.loads(value) if value.startswith('[') or value.startswith('{') else value
        
        # If storing a list and key already exists with a list, append and trim
        if isinstance(parsed_value, list) and key in knowledge_base.data:
            existing = knowledge_base.data.get(key, [])
            if isinstance(existing, list):
                
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
    Retrieve stored patterns and discoveries.
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
        
        
        result = {}
        for k, v in recent_items:
            if isinstance(v, list):
                result[k] = v[-10:]  
            else:
                result[k] = v
        
        return json.dumps(result)

@function_tool
async def get_product_count(element_id: str) -> str:
    """
    Count of products found for an element.
    """
    count = len(product_store.data.get(element_id, []))
    return json.dumps({"element_id": element_id, "product_count": count})



@function_tool
async def reflect_on_coverage(element_id: str, element_text: str) -> str:
    """
    Agent reflects on whether discovered products cover all functional aspects of the patent element.
    Returns analysis and suggestions for gaps in coverage.
    """
    log.info(f"[REFLECT] Coverage analysis for {element_id}")
    
    # Get current products for this element
    current_products = product_store.data.get(element_id, [])
    
    # Prepare reflection context
    reflection_context = {
        "element_id": element_id,
        "element_text": element_text,
        "discovered_products": [
            {
                "name": p.get("product_name", ""),
                "category": p.get("category", ""),
                "description": p.get("description", "")
            }
            for p in current_products
        ],
        "product_count": len(current_products),
        "instruction": "Analyze if the discovered products cover all functional aspects of the patent element. Consider: 1) Are all functional roles addressed? 2) Are products from diverse categories? 3) What aspects might be missing? 4) Suggest new search angles."
    }
    
    return json.dumps(reflection_context)

@function_tool
async def extract_functional_roles(element_text: str) -> str:
    """
    Extract abstract functional roles from patent text.
    Returns structured representation of what the element does, not how.
    """
    log.info(f"[EXTRACT] Functional roles from element")
    
    extraction_context = {
        "element_text": element_text,
        "instruction": "Extract functional roles as abstract concepts. Identify: 1) What function is being performed? 2) What is being acted upon? 3) What is the purpose/outcome? Return as structured roles without technology-specific terms."
    }
    
    return json.dumps(extraction_context)

@function_tool
async def suggest_search_variations(current_terms: str, functional_context: str = "") -> str:
    """
    Suggest alternative search terms based on functional understanding.
    Helps broaden search beyond initial terminology.
    """
    log.info(f"[SUGGEST] Search variations")
    
    try:
        terms = json.loads(current_terms) if current_terms.startswith('[') else [current_terms]
    except:
        terms = [current_terms]
    
    suggestion_context = {
        "current_terms": terms,
        "functional_context": functional_context,
        "instruction": "Suggest alternative search terms that: 1) Express the same function differently 2) Target different layers (infrastructure vs application vs service) 3) Use industry variations 4) Consider adjacent technologies. Return diverse alternatives."
    }
    
    return json.dumps(suggestion_context)

#  Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are an intelligent product discovery agent with meta-cognitive capabilities. Your role is to find products and services that relate to patent claims across ANY domain.

CORE MISSION:
Discover WHAT products/services exist that relate to patent elements, not HOW they work.
Use meta-cognitive tools to ensure comprehensive discovery across functional layers.

YOU ARE RESPONSIBLE FOR:
1. Understanding patent functionality abstractly
2. Finding products across diverse categories
3. Reflecting on coverage gaps
4. Adapting search strategies dynamically
5. Building cross-element insights

TOOLS AVAILABLE:
- search_products(query): Search for products (limited to 8 per element)
- store_product(element_id, product_data): Store discovered products
- store_knowledge(key, value): Remember patterns and categories
- get_knowledge(key=None, limit=20): Retrieve what you've learned
- get_product_count(element_id): Check how many products found
- reflect_on_coverage(element_id, element_text): Analyze coverage gaps
- extract_functional_roles(element_text): Understand abstract functionality
- suggest_search_variations(current_terms, functional_context): Get search alternatives

## MULTI-ANGLE DISCOVERY STRATEGY

### 1. FUNCTIONAL ANALYSIS FIRST
- Call extract_functional_roles() to understand WHAT the patent element does abstractly
- Don't fixate on specific technical terms - think about the function
- Identify the core purpose, not the implementation

### 2. INITIAL BROAD SEARCH
- Use functional understanding to create initial queries
- Combine company name with functional concepts
- Cast a wide net to discover the landscape

### 3. SEARCH DIVERSIFICATION
- After 2-3 searches, call suggest_search_variations()
- Try different angles: infrastructure vs application vs service layer
- Look for products at different abstraction levels

### 4. COVERAGE REFLECTION
- After finding 3-5 products, call reflect_on_coverage()
- Ask: "What aspects of this element haven't I explored?"
- Are all products from the same family? Branch out!
- Use reflection insights to guide remaining searches

### 5. CROSS-ELEMENT PATTERN RECOGNITION
- If multiple elements mention similar concepts, they might use ONE product
- Look for the connecting tissue between elements
- Store cross-element insights in knowledge base

### 6. ITERATIVE REFINEMENT
- Use each search result to inform the next
- Don't waste searches on similar queries
- Adapt based on what you discover

## WORKFLOW FOR EACH ELEMENT:

1. EXTRACT FUNCTIONAL ROLES
Call extract_functional_roles(element_text) FIRST to understand abstract functionality.

2. CHECK EXISTING KNOWLEDGE
Call get_knowledge() to leverage patterns from previous elements.

3. INITIAL FUNCTIONAL SEARCH
Based on roles, search broadly for products that fulfill those functions.

4. ANALYZE AND DIVERSIFY
- Identify product families mentioned
- Call suggest_search_variations() for new angles
- Search different layers/categories

5. REFLECT ON COVERAGE
Call reflect_on_coverage() to identify gaps and guide remaining searches.

6. TARGETED GAP FILLING
Use reflection insights to find products in underserved functional areas.

7. STORE PRODUCTS
For each discovered product, store with ALL fields:
{
  "product_name": "Official product/service name",
  "company": "Company offering it",
  "category": "Type of product/service",
  "url": "Product page or reference",
  "description": "Brief description of what it is",
  "relevance": "How this relates to the patent element",
  "confidence": "High/Medium/Low"
}

## CRITICAL: Product Storage Format
When calling store_product, ALWAYS format as JSON array:
- CORRECT: [{"product_name": "...", "company": "...", ...}]
- WRONG: {"product_name": "...", "company": "...", ...}

DO NOT include "element_id" in the product data - it's handled automatically.

## BUILD KNOWLEDGE
Store insights for future elements:
- store_knowledge("functional_patterns", ["pattern1", "pattern2"])
- store_knowledge("product_layers", {"infrastructure": [...], "application": [...], "service": [...]})
- store_knowledge("cross_element_products", ["products that span multiple elements"])

## IMPORTANT PRINCIPLES:
- Think functionally, not literally
- Explore multiple abstraction layers
- Use meta-tools to guide strategy
- Build understanding across elements
- Adapt based on discoveries"""

# Main Agent Class
class ProductDiscoveryAgent:
    
    def __init__(self, company: str):
        self.company = company
        self.agent = None
        self.session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        # Reset search counter on startup
        checkpoint_store.data["searches_this_turn"] = 0
        checkpoint_store.save()
    
    async def initialize(self):
        """ereating agent with tools"""
        self.agent = Agent(
            name="Product Discovery Agent v2.0",
            instructions=AGENT_SYSTEM_PROMPT,
            model=MODEL_NAME,
            tools=[
                search_products,
                store_product,
                store_knowledge,
                get_knowledge,
                get_product_count,
                reflect_on_coverage,
                extract_functional_roles,
                suggest_search_variations
            ],
            model_settings=ModelSettings(
                temperature=0.3,
                max_tokens=3000
            )
        )
        log.info(f"Product Discovery Agent v2.0 initialized for {self.company}")
    
    async def analyze_element(self, element: Dict[str, Any], patent_context: Dict[str, Any]) -> Dict[str, Any]:
        """Let the agent discover products for an element"""
        
        # Reset search counter for this element
        checkpoint_store.data["searches_this_turn"] = 0
        checkpoint_store.save()
        
        prompt = f"""
Discover products/services from {self.company} that relate to this patent element:

PATENT: {patent_context['patent_id']}
FIELD: {patent_context['field_of_invention']}

ELEMENT: {element['element_id']}
TEXT: {element['text']}

ENHANCED DISCOVERY PROCESS:
1. Call extract_functional_roles() to understand abstract functionality
2. Check existing knowledge with get_knowledge()
3. Search broadly based on functional understanding
4. After 2-3 searches, call suggest_search_variations() for new angles
5. When you have 3-5 products, call reflect_on_coverage() to identify gaps
6. Use reflection insights to guide remaining searches
7. Store all discovered products with complete information

Remember: You have {MAX_SEARCHES_PER_TURN} searches. Use meta-cognitive tools to maximize discovery across functional layers.

Return a summary of products discovered and your coverage analysis.
"""
        
        try:
            log.info(f"Discovering products for element {element['element_id']}")
            
            
            result = await Runner.run(self.agent, prompt)
            
            
            product_count = len(product_store.data.get(element['element_id'], []))
            
       
            searches_used = checkpoint_store.data.get("searches_this_turn", 0)
            
            return {
                "element_id": element['element_id'],
                "product_count": product_count,
                "searches_executed": searches_used,
                "agent_summary": str(result.content) if hasattr(result, 'content') else str(result)
            }
            
        except Exception as e:
            log.error(f"Error discovering products: {e}")
            return {
                "element_id": element['element_id'],
                "error": str(e),
                "product_count": 0,
                "searches_executed": checkpoint_store.data.get("searches_this_turn", 0)
            }
    
    async def discover_products(self, patent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process all elements to discover related products"""
        
        results = {
            "patent_id": patent_data["patent_id"],
            "company": self.company,
            "field": patent_data.get("field_of_invention", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "agent_version": "v2.0_enhanced_discovery",
            "claims": []
        }
        
        patent_context = {
            "patent_id": patent_data["patent_id"],
            "field_of_invention": patent_data.get("field_of_invention", ""),
            "company": self.company
        }
        
        
        for claim in patent_data.get("independent_claims", []):
            claim_result = {
                "claim_number": claim["claim_number"],
                "elements": []
            }
            
            
            for element in claim["elements"]:
                element_result = await self.analyze_element(element, patent_context)
                claim_result["elements"].append(element_result)
                
                
                checkpoint_store.data["current"] = {
                    "claim": claim["claim_number"],
                    "element": element["element_id"]
                }
                checkpoint_store.save()
                
                await asyncio.sleep(1)  # Rate limit
            
            results["claims"].append(claim_result)
        
        
        all_products = []
        for claim in results["claims"]:
            for element in claim["elements"]:
                element_products = product_store.data.get(element["element_id"], [])
                for product in element_products:
                    
                    product["element_id"] = element["element_id"]
                    all_products.append(product)
        
        
        unique_products = {}
        for product in all_products:
            key = (product["product_name"], product["company"])
            if key not in unique_products:
                unique_products[key] = product
        
        
        category_distribution = {}
        for product in all_products:
            category = product.get("category", "Unknown")
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        
        total_elements = sum(len(c["elements"]) for c in results["claims"])
        total_products = len(all_products)
        unique_product_count = len(unique_products)
        total_searches = sum(e.get("searches_executed", 0) for c in results["claims"] for e in c["elements"])
        
        results["discovered_products"] = list(unique_products.values())
        results["summary"] = {
            "total_elements": total_elements,
            "total_products_found": total_products,
            "unique_products": unique_product_count,
            "total_searches": total_searches,
            "average_products_per_element": round(total_products / max(total_elements, 1), 1),
            "average_searches_per_element": round(total_searches / max(total_elements, 1), 1),
            "category_distribution": category_distribution,
            "enhancements_used": [
                "Functional role extraction",
                "Search variation suggestions",
                "Coverage reflection analysis",
                "Multi-layer discovery strategy"
            ]
        }
        
        return results

# main exec
async def main():
    parser = argparse.ArgumentParser(
        description="Patent Product Discovery Agent v2.0 - Enhanced Discovery with Meta-Cognitive Tools"
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
        product_store.clear()
        checkpoint_store.clear()
        log.info("Cleared all data")
    
  
    agent = ProductDiscoveryAgent(args.company)
    await agent.initialize()
    
    try:
        log.info(f"Starting enhanced product discovery for {args.company}")
        results = await agent.discover_products(patent_data)
        
      
        if args.output:
            args.output.write_text(json.dumps(results, indent=2))
            log.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        
        print(f"\n{'='*60}")
        print(f"Product Discovery Complete (v2.0): {args.company}")
        print(f"Patent: {patent_data['patent_id']}")
        print(f"Elements Analyzed: {results['summary']['total_elements']}")
        print(f"Products Found: {results['summary']['total_products_found']}")
        print(f"Unique Products: {results['summary']['unique_products']}")
        print(f"Total Searches: {results['summary']['total_searches']}")
        print(f"Avg Products/Element: {results['summary']['average_products_per_element']}")
        print(f"Avg Searches/Element: {results['summary']['average_searches_per_element']}")
        
        if results["summary"].get("category_distribution"):
            print(f"\nProduct Category Distribution:")
            for category, count in results["summary"]["category_distribution"].items():
                print(f"  • {category}: {count}")
        
        if results.get("discovered_products"):
            print(f"\nKey Products Discovered:")
            for product in results["discovered_products"][:10]:
                print(f"  • {product['product_name']} ({product['category']})")
        
        print(f"\nEnhancements Applied:")
        for enhancement in results["summary"]["enhancements_used"]:
            print(f"  ✓ {enhancement}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        log.error(f"Discovery failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())