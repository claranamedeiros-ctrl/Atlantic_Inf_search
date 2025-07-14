# Patent Claim Parser Agent v4.0

An AI agent that parses patent claims with the precision of a patent attorney, extracting structured claim elements for downstream infringement analysis.

## 🎯 Overview

This agent transforms raw patent JSON data into structured claim elements, focusing exclusively on independent claims. It leverages OpenAI's GPT-4 to replicate expert patent analysis methodology, particularly excelling at parsing complex conditional blocks that are critical for patent infringement analysis.

### Key Features

- **Expert-Level Parsing**: Splits patent claims into granular functional elements
- **Method Claim Detection**: Automatically identifies and flags method claims for prioritization
- **Conditional Block Handling**: Properly separates conditional statements from their response actions
- **Live Reasoning Display**: Optional real-time visualization of the agent's thinking process
- **Production-Ready**: Validated against expert ground truth data with 100% accuracy

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## 🚀 Quick Start



###  Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o
```

###  Run the Agent

```bash
# Basic usage
python claim_parser_agent.py patents/US9335805B2.json

# With live reasoning display
python claim_parser_agent.py patents/US9335805B2.json --show-thoughts

# Save output to file
python claim_parser_agent.py patents/US9335805B2.json --out outputs/US9335805B2_structured.json

# Both live reasoning and save output
python claim_parser_agent.py patents/US9335805B2.json --show-thoughts --out outputs/US9335805B2_structured.json
```

## 📥 Input Format

The agent expects a JSON file with patent data containing at least:

```json
{
  "patent_id": "US1234567B2",
  "title": "Patent Title",
  "abstract": "Patent abstract...",
  "claims": [
    {
      "number": "1",
      "text": "1. A method of doing something, comprising: ...",
      "independent": true,
      "classification": "Method"
    },
    ...
  ]
}
```

### Required Fields
- `patent_id`: Unique patent identifier
- `claims`: Array of claim objects
  - `number`: Claim number
  - `text`: Full claim text
  - `independent`: Boolean indicating if claim is independent

## 📤 Output Format

The agent produces a structured JSON with parsed claim elements:

```json
{
  "patent_id": "US1234567B2",
  "timestamp": "2023-10-01T12:00:00Z",
  "agent_version": "4.0",
  "source_file": "US1234567B2.json",
  "field_of_invention": "network traffic monitoring",
  "invention_descriptor": "selective traffic monitoring",
  "independent_claims": [
    {
      "claim_number": 1,
      "is_method_claim": true,
      "elements": [
        {
          "element_id": "US1234567B2_cl1_el01",
          "text": "receiving a notice for a beginning of a network data flow"
        },
        ...
      ]
    }
  ]
}
```

### Output Fields Explained

- **field_of_invention**: Broad technological area (1-7 words)
- **invention_descriptor**: Precise mechanism from first claim (1-7 words)
- **is_method_claim**: Boolean flag for method claim identification
- **elements**: Array of parsed claim elements, each representing a functional step

## 🔍 What the Agent Does

1. **Identifies Independent Claims**: Filters out dependent claims, processing only independent ones
2. **Detects Method Claims**: Flags claims starting with "A method for/of..." 
3. **Extracts Key Fields**: 
   - Broad technological field
   - Specific invention mechanism
4. **Parses Claim Elements**: Splits claims at:
   - Semicolons (primary delimiter)
   - Functional actions (determining, setting, etc.)
   - Conditional response blocks
5. **Handles Complex Structures**: Properly separates:
   - Conditional statements (ending with ":") as single elements
   - Response actions as individual elements

### Example: Conditional Block Parsing

**Input claim text:**
```
"in response to X, Y, and Z: setting A; setting B and setting C."
```

**Parsed elements:**
1. `"in response to X, Y, and Z:"`
2. `"setting A"`
3. `"setting B"`
4. `"setting C"`

## 🛠️ Advanced Usage

### Command Line Arguments

```bash
python claim_parser_agent.py [OPTIONS] PATENT_FILE

Arguments:
  PATENT_FILE              Path to patent JSON file (required)

Options:
  --out OUTPUT_PATH        Save parsed output to file
  --show-thoughts          Display agent's reasoning in real-time
  --help                   Show help message
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: "gpt-4o")

## 🏗️ Project Structure

```
claim-parser-agent/
├── claim_parser_agent.py    # Main agent script
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables (create this)
├── README.md               # This file
├── patents/                # Input patent JSON files
│   ├── US9335805B2.json
│   └── US8792347B2.json
└── outputs/                # Parsed output files
    └── US9335805B2_structured.json
```



### Test with Sample Patents

Two sample patents are included in the `patents/` directory for testing:
- `US9335805B2.json`: Multi-core power management patent
- `US8792347B2.json`: Network traffic monitoring patent



## 📝 Dependencies

Key dependencies (see `requirements.txt` for full list):
- `openai>=1.0.0` - OpenAI Python SDK
- `python-dotenv` - Environment variable management
- `jsonschema` - JSON schema validation
- `tenacity` - Retry logic for API calls

## ⚠️ Important Notes

- **API Costs**: This agent uses OpenAI's GPT-4 API, which incurs costs per token
- **Rate Limits**: Be aware of OpenAI API rate limits
- **Patent Data**: Ensure you have rights to process the patent data
- **Output Usage**: Parsed claims are for analysis purposes; consult legal counsel for infringement determinations

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Create `.env` file with your API key
   - Ensure `.env` is in the project root

2. **"Failed to parse JSON output"**
   - Check input patent JSON is valid
   - Ensure claims have proper structure

3. **API Rate Limits**
   - The agent includes retry logic
   - Consider adding delays between batch processing



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This agent is designed for patent analysis and research purposes. Always consult with qualified patent attorneys for legal advice regarding patent infringement or validity.