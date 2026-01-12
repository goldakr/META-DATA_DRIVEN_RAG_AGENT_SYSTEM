# Insurance RAG Agent - Query Examples with Ground Truth

This directory contains comprehensive query examples with ground truth answers for testing and evaluating the Insurance RAG Agent system.

## 📁 Files Overview

- **`query_examples_metadata.json`** - Main metadata file containing 10 query examples with ground truth
- **`validate_queries.py`** - Script to validate and display query metadata
- **`test_query_example.py`** - Script to test individual queries against the RAG system
- **`QUERY_EXAMPLES_README.md`** - This documentation file

## 🎯 Query Examples Summary

The metadata file contains **10 diverse query examples** covering different aspects of insurance document analysis:

| ID | Type | Question Focus |
|----|------|----------------|
| Q001 | accident_details | Alex Jones accident details and circumstances |
| Q002 | policy_information | Insurance coverage limits and policy details |
| Q003 | payment_history | Payment transactions and amounts |
| Q004 | case_timeline | Chronological sequence of events |
| Q005 | damage_assessment | Vehicle damage evaluation and repair costs |
| Q006 | coverage_verification | Insurance policy verification |
| Q007 | settlement_amounts | Settlement payment details |
| Q008 | vehicle_information | Vehicle specifications and details |
| Q009 | incident_analysis | Accident cause and investigation analysis |
| Q010 | financial_summary | Total financial impact and breakdown |

## 📊 Data Sources

The queries are based on real content extracted from the following PDF files:

- **Alex_Jones_Accident_Report.pdf** - Detailed accident report with timeline and analysis
- **Alex_Jones_Car_Accident_Case.pdf** - Case timeline and investigation details
- **Alex_Jones_Insurance_Policy.pdf** - Policy terms, coverage limits, and conditions
- **Alex_Jones_Payment_Reports.pdf** - Payment transactions and financial records
- **Maria_Petrov_Accident_Report.pdf** - Another accident case for comparison
- **Petr_Petrov_Care_Accident_Case.pdf** - Related accident case details
- **Petr_Petrov_Insurance_Policy_Honda.pdf** - Policy information for different client
- **Petr_Petrov_Insurance_Policy_Toyota.pdf** - Additional policy documentation
- **Petr_Petrov_Payment_Reports.pdf** - Payment records for related case

## 🔍 Ground Truth Structure

Each query example includes:

```json
{
  "id": "Q001",
  "query_type": "accident_details",
  "question": "What happened in the Alex Jones accident on July 22, 2025?",
  "ground_truth": {
    "answer": "Detailed answer based on source documents...",
    "source_files": ["Alex_Jones_Accident_Report.pdf"],
    "case_id": "103",
    "incident_date": "July 22, 2025",
    "additional_metadata": "..."
  },
  "expected_metadata": {
    "FileName": "Alex_Jones_Accident_Report.pdf",
    "CaseId": "103",
    "IncidentType": "accident",
    "SectionType": ["Summary", "Timeline", "Body"]
  }
}
```

## 🛠️ Usage Instructions

### 1. Validate Query Metadata

```bash
python validate_queries.py
```

This will:
- ✅ Validate JSON structure
- ✅ Check for missing source files
- 📊 Display query summary and statistics
- 📝 Show all query examples with ground truth

### 2. Test Individual Queries

```bash
# List available queries
python test_query_example.py

# Test a specific query
python test_query_example.py Q001
```

This will:
- 🤖 Run the query against your RAG system
- ✅ Display ground truth answer
- 🔍 Show RAG system response
- 📊 Compare results

### 3. Build Index First (Required)

Before testing queries, ensure your RAG system index is built:

```bash
python -m src.main build
```

## 📈 Evaluation Criteria

The query examples are designed to test:

1. **Accuracy** - Answers should match exact information from source documents
2. **Completeness** - All relevant details should be included
3. **Source Attribution** - Correct identification of source documents
4. **Metadata Matching** - Proper metadata filtering and retrieval
5. **Context Understanding** - Understanding of insurance case context

## 🎯 Query Types Explained

### Accident Details (Q001)
Tests ability to extract and summarize accident circumstances, including:
- Date, time, and location
- Vehicles and drivers involved
- Accident sequence and cause

### Policy Information (Q002)
Tests insurance policy comprehension:
- Coverage limits and types
- Policy numbers and terms
- Premium amounts

### Payment History (Q003)
Tests financial transaction analysis:
- Payment amounts and dates
- Transaction descriptions
- Total financial impact

### Case Timeline (Q004)
Tests chronological information extraction:
- Event sequencing
- Time-based relationships
- Process flow understanding

### Damage Assessment (Q005)
Tests technical damage evaluation:
- Vehicle damage descriptions
- Repair cost estimates
- Assessment details

### Coverage Verification (Q006)
Tests policy verification capabilities:
- Policy type identification
- Coverage status
- Policy holder information

### Settlement Amounts (Q007)
Tests financial settlement analysis:
- Settlement payment amounts
- Payment dates and recipients
- Financial impact assessment

### Vehicle Information (Q008)
Tests vehicle specification extraction:
- Vehicle make, model, year
- VIN numbers
- Vehicle classification

### Incident Analysis (Q009)
Tests analytical reasoning:
- Cause and effect relationships
- Investigation findings
- Liability assessment

### Financial Summary (Q010)
Tests comprehensive financial analysis:
- Total cost calculations
- Payment breakdowns
- Financial impact assessment

## 🔧 Customization

### Adding New Queries

To add new query examples:

1. Edit `query_examples_metadata.json`
2. Add new query object with required fields
3. Update the `total_queries` count in metadata
4. Run validation script to check structure

### Modifying Ground Truth

Ground truth answers are based on actual extracted content. To update:

1. Re-run `print_chunks.py` to get latest content
2. Update ground truth answers in metadata file
3. Validate with `validate_queries.py`

## 📊 Performance Metrics

Use these queries to measure:

- **Retrieval Accuracy** - Correct document retrieval
- **Answer Quality** - Completeness and accuracy of responses
- **Metadata Filtering** - Proper use of metadata for filtering
- **Context Understanding** - Insurance domain knowledge
- **Response Time** - System performance

## 🚀 Integration with RAG Evaluation

These query examples can be integrated with RAG evaluation frameworks like:

- **RAGAS** - For automated RAG evaluation
- **Custom evaluation scripts** - For specific metrics
- **A/B testing** - For system comparison
- **Performance monitoring** - For continuous improvement

## 📝 Notes

- All ground truth answers are based on actual content from PDF files
- Queries represent realistic user questions about insurance cases
- Metadata expectations match actual document chunk metadata
- Source file references are validated against existing files
- Query diversity covers different types of insurance information requests

## 🤝 Contributing

When adding new queries:

1. Ensure they represent realistic user scenarios
2. Base ground truth on actual document content
3. Include proper metadata expectations
4. Test with validation script
5. Update documentation as needed

---

**Created:** 2025-09-02  
**Version:** 1.0  
**Total Queries:** 10  
**Source Files:** 9 PDF documents
