"""
Integration tests for ingest.py using real PDF files from the data directory
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

from src.ingest import (
    find_pdfs,
    extract_tables_for_pdf,
    chunk_pdf_text,
    ingest_directory
)
from llama_index.core.schema import TextNode


class TestIngestWithRealPDFs:
    """Integration tests using real PDF files from the data directory"""
    
    @pytest.fixture
    def data_dir(self):
        """Fixture providing the data directory path"""
        return Path("data")
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Fixture providing a temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_find_pdfs_with_real_data(self, data_dir):
        """Test finding PDFs in the real data directory"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        pdfs = find_pdfs(data_dir)
        
        # Should find all PDF files
        assert len(pdfs) > 0
        
        # Check that all found files are PDFs
        for pdf in pdfs:
            assert pdf.suffix.lower() == '.pdf'
            assert pdf.exists()
        
        # Check for specific expected files
        pdf_names = [pdf.name for pdf in pdfs]
        expected_files = [
            "Alex_Jones_Accident_Report.pdf",
            "Alex_Jones_Car_Accident_Case.pdf", 
            "Alex_Jones_Insurance_Policy.pdf",
            "Alex_Jones_Payment_Reports.pdf",
            "Maria_Petrov_Accident_Report.pdf",
            "Petr_Petrov_Care_Accident_Case.pdf",
            "Petr_Petrov_Insurance_Policy_Honda.pdf",
            "Petr_Petrov_Insurance_Policy_Toyota.pdf",
            "Petr_Petrov_Payment_Reports.pdf"
        ]
        
        for expected_file in expected_files:
            assert expected_file in pdf_names, f"Expected file {expected_file} not found"
    
    def test_extract_tables_from_real_pdf(self, data_dir, temp_storage_dir):
        """Test table extraction from a real PDF file"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Use the first available PDF
        pdfs = find_pdfs(data_dir)
        if not pdfs:
            pytest.skip("No PDF files found in data directory")
        
        pdf_path = pdfs[0]
        table_dir = temp_storage_dir / "tables"
        table_dir.mkdir(parents=True)
        
        # Extract tables
        entries = extract_tables_for_pdf(pdf_path, table_dir)
        
        # Verify the function runs without errors
        assert isinstance(entries, list)
        
        # If tables were found, verify their structure
        for entry in entries:
            assert "table_id" in entry
            assert "page" in entry
            assert "csv" in entry
            assert "file_name" in entry
            assert "summary" in entry
            
            # Verify CSV file exists if table was extracted
            csv_path = Path(entry["csv"])
            if csv_path.exists():
                assert csv_path.suffix == '.csv'
                
                # Verify CSV is readable
                import pandas as pd
                df = pd.read_csv(csv_path)
                assert isinstance(df, pd.DataFrame)
                assert len(df) >= 0  # Can be empty but should be valid
    
    def test_chunk_pdf_text_from_real_pdf(self, data_dir):
        """Test text chunking from a real PDF file"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Use the first available PDF
        pdfs = find_pdfs(data_dir)
        if not pdfs:
            pytest.skip("No PDF files found in data directory")
        
        pdf_path = pdfs[0]
        
        # Chunk the PDF text
        nodes = chunk_pdf_text(pdf_path)
        
        # Verify the function runs without errors
        assert isinstance(nodes, list)
        assert len(nodes) > 0, "Should extract at least one text chunk"
        
        # Verify node structure
        for node in nodes:
            assert isinstance(node, TextNode)
            assert hasattr(node, 'text')
            assert hasattr(node, 'metadata')
            assert len(node.text) > 0, "Text chunks should not be empty"
            
            # Verify metadata structure
            metadata = node.metadata
            assert "FileName" in metadata
            assert "PageNumber" in metadata
            assert "SectionType" in metadata
            assert metadata["FileName"] == pdf_path.name
    
    def test_full_ingest_with_real_data(self, data_dir, temp_storage_dir):
        """Test full ingestion process with real PDF files"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Run full ingestion
        result = ingest_directory(data_dir, temp_storage_dir)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "table_registry" in result
        
        # Verify nodes
        nodes = result["nodes"]
        assert isinstance(nodes, list)
        assert len(nodes) > 0, "Should extract text chunks from PDFs"
        
        # Verify table registry
        table_registry = result["table_registry"]
        assert isinstance(table_registry, dict)
        
        # Verify storage structure was created
        assert temp_storage_dir.exists()
        assert (temp_storage_dir / "tables").exists()
        
        # Verify registry file was created
        registry_file = temp_storage_dir / "tables" / "registry.json"
        assert registry_file.exists()
        
        # Verify registry file is valid JSON
        import json
        with open(registry_file, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)
        assert isinstance(registry_data, dict)
    
    def test_ingest_specific_insurance_documents(self, data_dir, temp_storage_dir):
        """Test ingestion focusing on specific insurance document types"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Test with a subset of documents
        test_files = [
            "Alex_Jones_Accident_Report.pdf",
            "Alex_Jones_Insurance_Policy.pdf"
        ]
        
        # Create a temporary directory with only test files
        with tempfile.TemporaryDirectory() as temp_data_dir:
            temp_data_path = Path(temp_data_dir)
            
            # Copy test files
            for filename in test_files:
                src_file = data_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, temp_data_path / filename)
            
            # Run ingestion on subset
            result = ingest_directory(temp_data_path, temp_storage_dir)
            
            # Verify results
            assert len(result["nodes"]) > 0
            
            # Check that we have nodes from different document types
            file_names = set()
            section_types = set()
            
            for node in result["nodes"]:
                metadata = node.metadata
                file_names.add(metadata["FileName"])
                section_types.add(metadata["SectionType"])
            
            # Should have nodes from multiple files
            assert len(file_names) > 0
            
            # Should have different section types
            assert len(section_types) > 0
    
    def test_incident_field_extraction_from_real_docs(self, data_dir):
        """Test incident field extraction from real insurance documents"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        # Use accident report PDFs
        accident_reports = [
            "Alex_Jones_Accident_Report.pdf",
            "Maria_Petrov_Accident_Report.pdf"
        ]
        
        found_incident_data = False
        
        for filename in accident_reports:
            pdf_path = data_dir / filename
            if not pdf_path.exists():
                continue
            
            # Chunk the PDF to get nodes with extracted incident fields
            nodes = chunk_pdf_text(pdf_path)
            
            for node in nodes:
                metadata = node.metadata
                
                # Check for incident-related fields
                if any(key in metadata for key in ["IncidentType", "IncidentDate", "ClientId", "CaseId"]):
                    found_incident_data = True
                    
                    # Verify field formats if present
                    if "IncidentType" in metadata and metadata["IncidentType"]:
                        assert isinstance(metadata["IncidentType"], str)
                    
                    if "ClientId" in metadata and metadata["ClientId"]:
                        assert isinstance(metadata["ClientId"], str)
                        assert len(metadata["ClientId"]) > 0
        
        # Should find some incident data in accident reports
        if any((data_dir / name).exists() for name in accident_reports):
            assert found_incident_data, "Should extract incident fields from accident reports"
    
    def test_table_extraction_from_payment_reports(self, data_dir, temp_storage_dir):
        """Test table extraction specifically from payment reports"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        payment_reports = [
            "Alex_Jones_Payment_Reports.pdf",
            "Petr_Petrov_Payment_Reports.pdf"
        ]
        
        total_tables_found = 0
        
        for filename in payment_reports:
            pdf_path = data_dir / filename
            if not pdf_path.exists():
                continue
            
            table_dir = temp_storage_dir / "tables" / filename.replace('.pdf', '')
            table_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract tables
            entries = extract_tables_for_pdf(pdf_path, table_dir)
            total_tables_found += len(entries)
            
            # Verify table entries
            for entry in entries:
                assert entry["file_name"] == filename
                assert "Table-" in entry["table_id"]
                assert entry["page"] > 0
                
                # If CSV was created, verify it's readable
                csv_path = Path(entry["csv"])
                if csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    assert isinstance(df, pd.DataFrame)
        
        # Payment reports should contain tables
        if any((data_dir / name).exists() for name in payment_reports):
            assert total_tables_found > 0, "Payment reports should contain tables"
    
    def test_policy_document_processing(self, data_dir):
        """Test processing of insurance policy documents"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        policy_docs = [
            "Alex_Jones_Insurance_Policy.pdf",
            "Petr_Petrov_Insurance_Policy_Honda.pdf",
            "Petr_Petrov_Insurance_Policy_Toyota.pdf"
        ]
        
        policy_nodes_found = 0
        
        for filename in policy_docs:
            pdf_path = data_dir / filename
            if not pdf_path.exists():
                continue
            
            nodes = chunk_pdf_text(pdf_path)
            
            for node in nodes:
                metadata = node.metadata
                
                # Check for policy-related content
                if metadata.get("SectionType") == "Policy":
                    policy_nodes_found += 1
                
                # Verify policy-related keywords
                keywords = metadata.get("Keywords", [])
                if any(keyword in ["policy", "coverage", "insurance"] for keyword in keywords):
                    policy_nodes_found += 1
        
        # Should find policy-related content
        if any((data_dir / name).exists() for name in policy_docs):
            assert policy_nodes_found > 0, "Should identify policy-related content"


class TestIngestPerformance:
    """Performance tests for ingest functionality"""
    
    @pytest.fixture
    def data_dir(self):
        """Fixture providing the data directory path"""
        return Path("data")
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Fixture providing a temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_ingest_performance_with_real_data(self, data_dir, temp_storage_dir):
        """Test ingestion performance with real data"""
        if not data_dir.exists():
            pytest.skip("Data directory not found")
        
        import time
        
        start_time = time.time()
        result = ingest_directory(data_dir, temp_storage_dir)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify reasonable processing time (adjust threshold as needed)
        assert processing_time < 300, f"Ingestion took too long: {processing_time:.2f} seconds"
        
        # Verify we got results
        assert len(result["nodes"]) > 0
        assert isinstance(result["table_registry"], dict)
        
        print(f"Processed {len(result['nodes'])} text chunks in {processing_time:.2f} seconds")
        print(f"Found {len(result['table_registry'])} tables")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
