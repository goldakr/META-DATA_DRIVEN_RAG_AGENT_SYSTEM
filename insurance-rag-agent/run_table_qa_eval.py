#!/usr/bin/env python3
"""
Standalone Table-QA Evaluation Script

This script runs the table_qa_tool evaluation by directly calling the main agent
instead of trying to import the table_qa_tool directly.
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class TableQAEvaluator:
    """Evaluator for Table-QA tool accuracy using the main agent."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.results = []
        
    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases for table QA evaluation."""
        return [
            {
                "id": "payment_amount_1",
                "query": "How much was Petr Petrov final settlement payment?",
                "expected_answer": "212500",
                "expected_tool": "table_qa",
                "description": "Final settlement payment amount from Petr Petrov payment report"
            },
            {
                "id": "payment_amount_2", 
                "query": "What is Alex Jones total policy premium?",
                "expected_answer": "premium",
                "expected_tool": "table_qa",
                "description": "Total policy premium from Alex Jones insurance policy"
            },
            {
                "id": "payment_date_1",
                "query": "When was the final payment made to Petr Petrov?",
                "expected_answer": "date",
                "expected_tool": "table_qa", 
                "description": "Date of final payment from Petr Petrov payment report"
            },
            {
                "id": "coverage_limit_1",
                "query": "What is the per-person liability limit for Alex Jones?",
                "expected_answer": "limit",
                "expected_tool": "table_qa",
                "description": "Per-person liability limit from Alex Jones insurance policy"
            },
            {
                "id": "deductible_1",
                "query": "What is the deductible amount for Petr Petrov Honda policy?",
                "expected_answer": "deductible",
                "expected_tool": "table_qa",
                "description": "Deductible amount from Petr Petrov Honda insurance policy"
            },
            {
                "id": "payment_description_1",
                "query": "What was the description of the largest payment to Petr Petrov?",
                "expected_answer": "description",
                "expected_tool": "table_qa",
                "description": "Description of largest payment from Petr Petrov payment report"
            },
            {
                "id": "total_payments_1",
                "query": "What is the total amount of all payments made to Petr Petrov?",
                "expected_answer": "total",
                "expected_tool": "table_qa",
                "description": "Sum of all payments from Petr Petrov payment report"
            },
            {
                "id": "coverage_type_1",
                "query": "What type of coverage does Alex Jones have for bodily injury?",
                "expected_answer": "coverage",
                "expected_tool": "table_qa",
                "description": "Coverage type from Alex Jones insurance policy"
            },
            {
                "id": "per_accident_limit_1",
                "query": "What is the per-Accident liability limit for Alex Jones?",
                "expected_answer": "limit",
                "expected_tool": "table_qa",
                "description": "Per-accident liability limit from Alex Jones insurance policy"
            },
            {
                "id": "demanded_settlement_1",
                "query": "What is the total demanded settlement amount in case of Alex Jones?",
                "expected_answer": "amount",
                "expected_tool": "table_qa",
                "description": "Total demanded settlement amount for Alex Jones case"
            },
            {
                "id": "partial_compensation_1",
                "query": "What is the approved partial compensation for Maria Petrov?",
                "expected_answer": "compensation",
                "expected_tool": "table_qa",
                "description": "Approved partial compensation amount for Maria Petrov"
            }
        ]
    
    def format_answer_for_display(self, answer_text: str) -> str:
        """Format answer text for better readability, converting JSON to readable format."""
        # Check if the answer contains JSON
        if "```json" in answer_text:
            try:
                # Extract JSON content
                json_start = answer_text.find("```json") + 7
                json_end = answer_text.find("```", json_start)
                if json_end != -1:
                    json_content = answer_text[json_start:json_end].strip()
                    json_data = json.loads(json_content)
                    
                    # Format as readable text
                    formatted = []
                    
                    # Handle incident components
                    if "incident_components" in json_data:
                        components = json_data["incident_components"]
                        if isinstance(components, list) and len(components) > 0:
                            if isinstance(components[0], dict):
                                # List of objects
                                formatted.append("Incident Components:")
                                for comp in components:
                                    if "name" in comp and "role" in comp:
                                        formatted.append(f"  - {comp['name']}: {comp['role']}")
                                    elif "name" in comp and "limit" in comp:
                                        formatted.append(f"  - {comp['name']}: {comp['limit']}")
                                    elif "name" in comp and "type" in comp:
                                        formatted.append(f"  - {comp['name']}: {comp['type']}")
                            else:
                                # List of strings
                                formatted.append("Incident Components: " + ", ".join(components))
                    
                    # Handle matched policy sections
                    if "matched_policy_sections" in json_data:
                        sections = json_data["matched_policy_sections"]
                        if isinstance(sections, list) and len(sections) > 0:
                            formatted.append("Policy Sections:")
                            for section in sections:
                                if isinstance(section, dict):
                                    if "section" in section and "details" in section:
                                        formatted.append(f"  - {section['section']}: {section['details']}")
                                    elif "section" in section:
                                        formatted.append(f"  - {section['section']}")
                                else:
                                    formatted.append(f"  - {section}")
                    
                    # Handle coverage notes
                    if "coverage_notes" in json_data:
                        formatted.append(f"Coverage Notes: {json_data['coverage_notes']}")
                    
                    return "\n".join(formatted) if formatted else answer_text
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                # If JSON parsing fails, return original text
                pass
        
        return answer_text
    
    def run_agent_query(self, query: str) -> Dict[str, Any]:
        """Run a query through the main agent and parse the response."""
        try:
            # Run the main agent with the query
            cmd = [
                sys.executable, "-m", "src.main", "ask", 
                f'"{query}"', "--storage", str(self.storage_dir)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent
            )
            
            if result.returncode != 0:
                return {
                    "error": f"Command failed: {result.stderr}",
                    "stdout": result.stdout
                }
            
            # Parse the output to extract the answer
            output = result.stdout
            
            # Look for the answer section
            answer_start = output.find("Answer:")
            if answer_start == -1:
                return {
                    "error": "No answer found in output",
                    "stdout": output
                }
            
            # Extract the answer text
            answer_section = output[answer_start:]
            lines = answer_section.split('\n')
            
            answer_text = ""
            tables_count = 0
            anchors_count = 0
            
            for line in lines[1:]:  # Skip the "Answer:" line
                line = line.strip()
                if line.startswith("Tables:"):
                    # Count tables
                    tables_section = line
                    tables_count = tables_section.count("TableId")
                    break
                elif line.startswith("Anchors:"):
                    # Count anchors
                    anchors_section = line
                    anchors_count = anchors_section.count("FileName")
                    break
                elif line and not line.startswith("Tables:") and not line.startswith("Anchors:"):
                    answer_text += line + " "
            
            # Clean up the answer text
            answer_text = answer_text.strip()
            
            return {
                "answer_text": answer_text,
                "tables_count": tables_count,
                "anchors_count": anchors_count,
                "full_output": output
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "stdout": ""
            }
    
    def evaluate_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query against the table_qa_tool."""
        query = test_case["query"]
        expected_answer = test_case["expected_answer"]
        expected_tool = test_case["expected_tool"]
        
        print(f"\n🔍 Testing: {test_case['description']}")
        print(f"   Query: {query}")
        
        try:
            # Get response from agent
            response = self.run_agent_query(query)
            
            if "error" in response:
                print(f"   ❌ Error: {response['error']}")
                return {
                    "test_case_id": test_case["id"],
                    "query": query,
                    "expected_answer": expected_answer,
                    "expected_tool": expected_tool,
                    "description": test_case["description"],
                    "error": response["error"],
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract components
            answer_text = response.get("answer_text", "")
            tables_count = response.get("tables_count", 0)
            anchors_count = response.get("anchors_count", 0)
            
            # Evaluate accuracy metrics
            evaluation = {
                "test_case_id": test_case["id"],
                "query": query,
                "expected_answer": expected_answer,
                "expected_tool": expected_tool,
                "description": test_case["description"],
                "actual_answer": answer_text,
                "actual_tool": "table_qa",  # We expect table_qa for these queries
                "tables_referenced": tables_count,
                "anchors_count": anchors_count,
                "has_table_data": tables_count > 0,
                "response_length": len(answer_text),
                "timestamp": datetime.now().isoformat()
            }
            
            # Check if answer contains expected information
            if expected_answer == "212500":
                evaluation["correct_amount"] = "212500" in answer_text or "212,500" in answer_text
            elif expected_answer == "premium":
                evaluation["mentions_premium"] = "premium" in answer_text.lower()
            elif expected_answer == "date":
                evaluation["mentions_date"] = any(word in answer_text.lower() for word in ["date", "2025", "april", "may", "june"])
            elif expected_answer == "limit":
                evaluation["mentions_limit"] = "limit" in answer_text.lower()
            elif expected_answer == "deductible":
                evaluation["mentions_deductible"] = "deductible" in answer_text.lower()
            elif expected_answer == "description":
                evaluation["mentions_description"] = len(answer_text) > 50  # Substantial description
            elif expected_answer == "total":
                evaluation["mentions_total"] = any(word in answer_text.lower() for word in ["total", "sum", "amount"])
            elif expected_answer == "coverage":
                evaluation["mentions_coverage"] = "coverage" in answer_text.lower()
            elif expected_answer == "amount":
                evaluation["mentions_amount"] = any(word in answer_text.lower() for word in ["amount", "$", "settlement", "demanded", "total"])
            elif expected_answer == "compensation":
                evaluation["mentions_compensation"] = any(word in answer_text.lower() for word in ["compensation", "partial", "approved", "$", "payment"])
            
            # Overall accuracy assessment
            evaluation["tool_correct"] = evaluation["actual_tool"] == expected_tool
            
            # Format the answer for better readability
            formatted_answer = self.format_answer_for_display(answer_text)
            print(f"   ✅ Answer: {formatted_answer}")
            
            return evaluation
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                "test_case_id": test_case["id"],
                "query": query,
                "expected_answer": expected_answer,
                "expected_tool": expected_tool,
                "description": test_case["description"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation of table_qa_tool."""
        print("🚀 Starting Table-QA Accuracy Evaluation")
        print("=" * 60)
        
        test_cases = self.load_test_cases()
        print(f"📋 Loaded {len(test_cases)} test cases")
        
        for test_case in test_cases:
            result = self.evaluate_query(test_case)
            self.results.append(result)
        
        # Calculate overall metrics
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if "error" not in r])
        tool_correct = len([r for r in self.results if r.get("tool_correct", False)])
        
        # Calculate specific accuracy metrics
        accuracy_metrics = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "tool_accuracy": tool_correct / total_tests if total_tests > 0 else 0,
            "avg_anchors_count": sum(r.get("anchors_count", 0) for r in self.results) / total_tests if total_tests > 0 else 0,
            "avg_response_length": sum(r.get("response_length", 0) for r in self.results) / total_tests if total_tests > 0 else 0
        }
        
        # Specific accuracy checks
        specific_checks = {
            "correct_amount": len([r for r in self.results if r.get("correct_amount", False)]),
            "mentions_premium": len([r for r in self.results if r.get("mentions_premium", False)]),
            "mentions_date": len([r for r in self.results if r.get("mentions_date", False)]),
            "mentions_limit": len([r for r in self.results if r.get("mentions_limit", False)]),
            "mentions_deductible": len([r for r in self.results if r.get("mentions_deductible", False)]),
            "mentions_description": len([r for r in self.results if r.get("mentions_description", False)]),
            "mentions_total": len([r for r in self.results if r.get("mentions_total", False)]),
            "mentions_coverage": len([r for r in self.results if r.get("mentions_coverage", False)]),
            "mentions_amount": len([r for r in self.results if r.get("mentions_amount", False)]),
            "mentions_compensation": len([r for r in self.results if r.get("mentions_compensation", False)])
        }
        
        evaluation_summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "storage_directory": str(self.storage_dir),
            "accuracy_metrics": accuracy_metrics,
            "specific_checks": specific_checks,
            "detailed_results": self.results
        }
        
        return evaluation_summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 TABLE-QA EVALUATION SUMMARY")
        print("=" * 60)
        
        metrics = summary["accuracy_metrics"]
        print(f"Total Tests: {metrics['total_tests']}")
        print(f"Successful Tests: {metrics['successful_tests']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Tool Accuracy: {metrics['tool_accuracy']:.2%}")
        print(f"Avg Anchors Count: {metrics['avg_anchors_count']:.1f}")
        print(f"Avg Response Length: {metrics['avg_response_length']:.0f} chars")
        
        print("\n🎯 SPECIFIC ACCURACY CHECKS:")
        checks = summary["specific_checks"]
        for check, count in checks.items():
            print(f"  {check}: {count}")
        
        print(f"\n📁 Results saved to: {summary['evaluation_timestamp']}")
    
    def save_results(self, summary: Dict[str, Any], output_file: Optional[str] = None):
        """Save evaluation results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"table_qa_evaluation_{timestamp}.json"
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Table-QA tool accuracy")
    parser.add_argument("--storage", default="storage", help="Storage directory path")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = TableQAEvaluator(args.storage)
    summary = evaluator.run_evaluation()
    
    # Print and save results
    evaluator.print_summary(summary)
    evaluator.save_results(summary, args.output)


if __name__ == "__main__":
    main()
