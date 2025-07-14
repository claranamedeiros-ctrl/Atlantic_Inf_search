# Patent Infringement Evidence Search Agent v7.0

An intelligent, domain-agnostic agent that searches for evidence of patent infringement by analyzing how companies implement patented technologies. The agent autonomously discovers relevant terminology, evaluates source quality, and builds knowledge over time.

## 🎯 What It Does

This agent takes a structured patent JSON file and a target company name, then:

- Analyzes each patent claim element
- Searches the web for evidence of how the company implements the described functionality
- Evaluates source quality (Primary/Secondary/Tertiary)
- Stores high-quality evidence with relevant quotes
- Learns effective search patterns for future use
- Produces a comprehensive report with all findings

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (for the LLM agent)
- Tavily API key (for web search)

## 🚀 Quick Start


###  Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a .env file in the project root:

```bash
touch .env
```

Add your API keys to the .env file:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here
OPENAI_MODEL=gpt-4o-mini  # Optional: defaults to gpt-4o-mini
```

Getting API Keys:

- **OpenAI**: Sign up at platform.openai.com and create an API key
- **Tavily**: Sign up at tavily.com for a search API key

###  Export Environment Variables

```bash
# On macOS/Linux:
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export TAVILY_API_KEY="tvly-your-tavily-api-key-here"

# On Windows:
set OPENAI_API_KEY=sk-your-openai-api-key-here
set TAVILY_API_KEY=tvly-your-tavily-api-key-here
```

## 📥 Input Format

The agent expects a structured patent JSON file with this format (you will find the raw patents in the patents folder and the structured patents - the output of the claim parser agent - in the 'outputs' folder):

```json
{
  "patent_id": "US8792347B2",
  "field_of_invention": "network traffic monitoring",
  "independent_claims": [
    {
      "claim_number": 1,
      "is_method_claim": true,
      "elements": [
        {
          "element_id": "US8792347B2_cl1_el01",
          "text": "receiving a notice for a beginning of a network data flow..."
        },
        {
          "element_id": "US8792347B2_cl1_el02",
          "text": "determining whether to monitor the data flow..."
        }
      ]
    }
  ]
}
```

## 🏃 Running the Agent

### Basic Usage 

**company** is a mandatory field 

To run the agent for **Intel** ( we are trying to get closer to the groundtruth file the client shared, intel is one of the companies they showed us the 'infringement flow pipeline they do manually'):

**python infringement_search_agent.py outputs/US9335805B2_structured.json --company “Intel”**

To run the agent for **Amazon** (same as above, client sent the ground truth for this company)

**python infringement_search_agent.py outputs/US8792347B2_structured.json --company “Amazon”**

```bash
python infringement_search_agent.py patent_file.json --company "Amazon"
```

### Save Results to File

```bash
python infringement_search_agent.py patent_file.json --company "Amazon" --output results.json
```

### Clear Previous Data (Fresh Start)

```bash
python infringement_search_agent.py patent_file.json --company "Amazon" --clear
```

## 📤 Output Files

The agent creates several files during operation:

### 1. patent_evidence_db.json

Stores all validated evidence found for each patent element:

```json
{
  "US8792347B2_cl1_el01": [
    {
      "url": "https://docs.aws.amazon.com/...",
      "title": "AWS Network Monitoring Guide",
      "quote": "Amazon CloudFront monitors network traffic...",
      "source_quality": "Primary",
      "relevance": "Describes how Amazon monitors network data flows",
      "confidence": "High"
    }
  ]
}
```

### 2. patent_knowledge_base.json

Stores learned patterns, successful queries, and discovered terminology:

```json
{
  "vocab_network_monitoring": ["CloudFront", "VPC Flow Logs", "Traffic Mirroring"],
  "effective_queries": ["Amazon CloudFront monitoring", "AWS network traffic analysis"],
  "product_names": ["CloudFront", "CloudWatch", "VPC"]
}
```

### 3. patent_analysis_checkpoint.json

Tracks progress and can resume if interrupted:

```json
{
  "current": {
    "claim": 1,
    "element": "US8792347B2_cl1_el03"
  },
  "searches_this_turn": 5
}
```

### 4. patent_search_agent.log

Detailed logs of all operations:

```
2025-07-14 00:09:30,717 [INFO] Loaded patent US8792347B2
2025-07-14 00:09:30,717 [INFO] Agent v7.0 initialized for Amazon
2025-07-14 00:09:33,605 [INFO] [SEARCH] Query 1: Amazon network traffic monitoring
```

### 5. Results Output (if specified with --output)

Comprehensive analysis results:

```json
{
  "patent_id": "US8792347B2",
  "company": "Amazon",
  "field": "network traffic monitoring",
  "timestamp": "2025-07-14T00:09:30.717Z",
  "claims": [
    {
      "claim_number": 1,
      "elements": [
        {
          "element_id": "US8792347B2_cl1_el01",
          "evidence_count": 6,
          "searches_executed": 8,
          "agent_summary": "Found strong evidence of Amazon implementing..."
        }
      ]
    }
  ],
  "summary": {
    "total_elements": 7,
    "total_evidence": 42,
    "total_searches": 48,
    "average_evidence_per_element": 6.0,
    "average_searches_per_element": 6.9
  }
}
```

## 📊 Understanding the Output

- **Evidence Count**: Number of high-quality sources found for each element
- **Searches Executed**: Number of web searches performed (max 8 per element)
- **Agent Summary**: Natural language explanation of findings
- **Source Quality**:
  - **Primary**: Official docs, technical specs, regulatory filings
  - **Secondary**: White papers, research papers, conference talks
  - **Tertiary**: Blogs, news articles, forums

- **Confidence Levels**: High/Medium/Low based on evidence strength

## 🔧 Configuration

Key parameters in the code:

- `MAX_SEARCHES_PER_TURN = 8`: Maximum searches per patent element
- `MAX_CONTENT_CHARS = 1500`: Maximum characters stored per search result
- `EVIDENCE_SCHEMA['min_quote_length'] = 50`: Minimum quote length for evidence

## 🐛 Troubleshooting

**"Set OPENAI_API_KEY and TAVILY_API_KEY environment variables"**  
Make sure you've exported the environment variables or added them to your .env file.

**401 Unauthorized errors**  
Check that your API keys are valid and have sufficient credits/permissions.

**No evidence found**  
The agent may need to search with different terms. Check the log file for search queries being used.

## 📝 Example Run

```bash
# Fresh analysis of Amazon's implementation of patent US8792347B2
python infringement_search_agent.py US8792347B2_structured.json --company "Amazon" --output amazon_analysis.json --clear

# Expected output:
============================================================
Analysis Complete (v7.0): Amazon
Patent: US8792347B2
Elements: 7
Evidence: 42
Searches: 48
Avg Evidence/Element: 6.0
Avg Searches/Element: 6.9
============================================================
```

## 🤝 Contributing

This agent is designed to be portable to any language (Ruby, Go, etc.). The core intelligence lives in the LLM prompt - the Python code is just a thin transaction layer.

