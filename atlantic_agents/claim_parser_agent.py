# claim_parser_agent.py 

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import openai
from jsonschema import validate as _js_validate, ValidationError
from tenacity import retry, wait_random_exponential, stop_after_attempt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.stderr.write("[ERROR] OPENAI_API_KEY not set\n")
    sys.exit(1)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMP = 0.0
AGENT_VERSION = "4.0"

client = openai.OpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("agent")

# Me di cuenta que cuando patentes mencionan "method" en el claim, son importantes, asi que que puse un flag y tb invention descriptor
TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "return_claim_structure",
            "description": "Return parsed patent claim structure with enhanced granularity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patent_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "agent_version": {"type": "string"},
                    "source_file": {"type": "string"},
                    "field_of_invention": {"type": "string", "description": "Broad technological area (1-7 words)"},
                    "invention_descriptor": {"type": "string", "description": "Precise mechanism from first claim (1-7 words)"},
                    "independent_claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_number": {"type": "integer"},
                                "is_method_claim": {"type": "boolean", "description": "True if this is a method claim"},
                                "elements": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "element_id": {"type": "string"},
                                            "text": {"type": "string"},
                                        },
                                        "required": ["element_id", "text"]
                                    }
                                },
                                "interconnections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "element_ids": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "description": {"type": "string"}
                                        },
                                        "required": ["type", "element_ids"]
                                    },
                                    "default": []
                                }
                            },
                            "required": ["claim_number", "is_method_claim", "elements"]
                        }
                    }
                },
                "required": [
                    "patent_id", "timestamp", "agent_version", "source_file",
                    "field_of_invention", "invention_descriptor", "independent_claims"
                ]
            }
        }
    }
]

# syssp
SYSTEM_PROMPT = """You are Agent 1 – Expert Patent Claim Element Parser. You must replicate expert patent analysis methodology.

CRITICAL PARSING RULES:

1. CLAIM SELECTION:
   - Process ONLY independent claims (where independent==true)
   - Ignore all dependent claims
   - Process ALL independent claims found

2. CLAIM TYPE IDENTIFICATION:
   - Method claims: Start with "A method for/of..." or contain "method comprising"
   - Set is_method_claim=true for method claims, false for others
   - Method claims are high priority for analysis

3. FIELD EXTRACTION:
   - field_of_invention: Extract broad technological area (1-7 words)
     * Remove filler: "method for", "system of", "device that", "apparatus for"
     * Keep core technological concept
     * Example: "network traffic monitoring", "power management"
   
   - invention_descriptor: Extract precise mechanism from FIRST independent claim (1-7 words)
     * Focus on what the invention introduces as solution
     * Emphasize exact mechanism or capabilities
     * Example: "selective traffic monitoring", "dynamic core control"

4. ELEMENT SPLITTING ALGORITHM:
   Step 1: Remove preamble (everything before "comprising:")
   Step 2: Split at semicolons (;) as primary delimiter
   Step 3: CRITICAL - Within conditional response blocks, further split at:
      - Each action that starts new functional step
      - "setting" actions that are separate functions
      - "determining" actions that are separate functions
   Step 4: Preserve conditional statements ending with ":" as single elements

5. ELEMENT GRANULARITY RULES:
   - Target 5-10 elements per independent claim
   - Each functional action = separate element
   - "determining X" = typically separate element
   - "setting X" = typically separate element  
   - "responsive to X" = typically separate element
   - Conditional statements ending with ":" = single element
   - Response actions after conditionals = split into separate elements

6. CONDITIONAL BLOCK HANDLING:
   - Conditional statements ending with ":" are single elements
   - Their response actions must be split into separate elements
   - Pattern recognition:
     * "in response to X, Y, and Z:" = 1 element
     * Following actions: "setting A; setting B and setting C" = 3 separate elements

7. ELEMENT ID FORMAT:
   - Use: {patent_id}_cl{claim_number}_el{index:02d}
   - Start numbering from 01
   - Ensure no duplicate IDs across all claims

8. QUALITY VALIDATION:
   - Independent claims typically have 5-10 elements
   - Each element should represent one functional step
   - Elements should flow logically
   - Verify all independent claims are processed

PARSING EXAMPLE PATTERN:
Input claim with conditional block:
"determining X; determining Y; determining Z; and in response to X, Y, and Z being true: setting A to state 1; setting B to state 2 and setting C to state 3."

Correct parsing:
Element 1: "determining X"
Element 2: "determining Y"
Element 3: "determining Z"
Element 4: "in response to X, Y, and Z being true:"
Element 5: "setting A to state 1"
Element 6: "setting B to state 2"
Element 7: "setting C to state 3"

THINK through each step methodically, showing your reasoning, then ACT with the function call."""

USER_PROMPT_TMPL = """PATENT SOURCE FILE: {fname}

TASK: Parse the independent claims from this patent following expert methodology.

Show your thinking process step by step:
1. First identify all independent claims
2. For each claim, identify if it's a method claim
3. Extract field_of_invention and invention_descriptor
4. Parse elements with proper granularity, especially for conditional blocks
5. Generate proper element IDs

Below is the JSON input (claims only):
```json
{payload}
```"""

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
def _chat_call(msgs: List[Dict[str, str]], show_thoughts: bool = False):
    """Make chat API call with optional streaming for live reasoning display"""
    if show_thoughts:
        # Use streaming to show reasoning in real-time: NO VA EN RUBY
        stream = client.chat.completions.create(
            model=MODEL,
            temperature=TEMP,
            messages=msgs,
            tools=TOOL_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "return_claim_structure"}},
            stream=True
        )
        
        
        collected_messages = []
        tool_calls = []
        current_content = ""
        
        print("\n" + "="*60)
        print("AGENT REASONING (Live):")
        print("="*60)
        
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                if delta.content:
                    print(delta.content, end='', flush=True)
                    current_content += delta.content
                    
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index is not None:
                            # Ensure we have enough slots
                            while len(tool_calls) <= tc.index:
                                tool_calls.append({
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""}
                                })
                            
                            if tc.id:
                                tool_calls[tc.index]["id"] = tc.id
                            if tc.type:
                                tool_calls[tc.index]["type"] = tc.type
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[tc.index]["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
        
        print("\n" + "="*60 + "\n")
        
        
        class MockChoice:
            def __init__(self, content, tool_calls):
                self.message = type('obj', (object,), {
                    'content': content,
                    'tool_calls': [type('obj', (object,), {
                        'function': type('obj', (object,), {
                            'arguments': tc["function"]["arguments"]
                        })
                    })() for tc in tool_calls] if tool_calls else None
                })()
        
        class MockResponse:
            def __init__(self, content, tool_calls):
                self.choices = [MockChoice(content, tool_calls)]
        
        return MockResponse(current_content, tool_calls)
    else:
        
        return client.chat.completions.create(
            model=MODEL,
            temperature=TEMP,
            messages=msgs,
            tools=TOOL_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "return_claim_structure"}},
        )

def _validate_schema(payload: Dict[str, Any]):
    """Validate the output against the schema"""
    _js_validate(instance=payload, schema=TOOL_SCHEMA[0]["function"]["parameters"])

def main():
    parser = argparse.ArgumentParser(description="Patent Claim Parser Agent v4.0")
    parser.add_argument("src", type=Path, help="Path to patent JSON file")
    parser.add_argument("--out", type=Path, help="Output path for parsed claims")
    parser.add_argument("--show-thoughts", action="store_true", help="Display agent reasoning in real-time")
    args = parser.parse_args()

    # Loadeamos patent data
    try:
        raw = json.loads(args.src.read_text())
    except Exception as e:
        log.error(f"Failed to read patent file: {e}")
        sys.exit(1)
    
    # Extract only claims and patent_id for processing
    claims_only = json.dumps({
        "claims": raw.get("claims", []),
        "patent_id": raw.get("patent_id", ""),
        "title": raw.get("title", ""),
        "abstract": raw.get("abstract", "")
    }, indent=2)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TMPL.format(
            fname=args.src.name, 
            payload=claims_only
        )}
    ]

    # API calling
    log.info(f"Processing patent {raw.get('patent_id', 'unknown')}...")
    t0 = time.time()
    
    try:
        rsp = _chat_call(messages, show_thoughts=args.show_thoughts)
    except Exception as e:
        log.error(f"API call failed: {e}")
        sys.exit(1)
    
    log.info("Processing time: %.2fs", time.time() - t0)

    # Extrae responses
    choice = rsp.choices[0]
    
    # Show thoughts if not already shown via streaming
    if not args.show_thoughts and choice.message.content:
        print("\n--- AGENT THOUGHTS ---")
        print(choice.message.content.strip())
        print("--- END THOUGHTS ---\n")

    # Check for tool call
    if not choice.message.tool_calls:
        log.error("Agent failed to call the return_claim_structure function")
        sys.exit(1)

    # Parseamos y validamos output
    out_str = choice.message.tool_calls[0].function.arguments
    try:
        parsed = json.loads(out_str)
        _validate_schema(parsed)
        log.info("Output validation successful")
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse JSON output: {e}")
        print("Raw output:")
        print(out_str)
        sys.exit(1)
    except ValidationError as e:
        log.error(f"Schema validation failed: {e}")
        print("Parsed output:")
        print(json.dumps(parsed, indent=2))
        sys.exit(1)

    # Save output
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(parsed, indent=2))
        log.info(f"Output saved to {args.out}")
    else:
        print(json.dumps(parsed, indent=2))

if __name__ == "__main__":
    main()