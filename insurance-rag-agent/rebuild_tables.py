#!/usr/bin/env python3
"""
Script to rebuild the table registry and CSV files from updated PDF files in the data directory.
This will clear existing table files and regenerate them based on the current PDF files.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any
import json

# Add the project root to the Python path so we can import src modules
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

# Import from the src module
try:
    from src.ingest import find_pdfs, extract_tables_for_pdf
    from src.utils import ensure_json, save_registry
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Make sure you're running this script from the insurance-rag-agent directory")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {current_dir}")
    print(f"Project root: {project_root}")
    sys.exit(1)

def clear_existing_tables(storage_dir: Path) -> None:
    """Clear existing table files and registry."""
    table_dir = storage_dir / "tables"
    
    if table_dir.exists():
        print(f"🗑️  Clearing existing table files in {table_dir}")
        
        # Remove all CSV files except registry.json
        for file_path in table_dir.glob("*.csv"):
            print(f"  Removing: {file_path.name}")
            file_path.unlink()
        
        # Clear the registry
        registry_file = table_dir / "registry.json"
        if registry_file.exists():
            print(f"  Clearing: {registry_file.name}")
            registry_file.unlink()
    else:
        print(f"📁 Creating table directory: {table_dir}")
        table_dir.mkdir(parents=True, exist_ok=True)

def rebuild_tables(data_dir: Path, storage_dir: Path, method: str = "pdfplumber") -> Dict[str, Any]:
    """
    Rebuild all tables from PDF files in the data directory.
    
    Args:
        data_dir: Directory containing PDF files
        storage_dir: Directory to store extracted tables
        method: Extraction method ("pdfplumber", "camelot", or "camelot_multi")
    """
    print(f"🔄 Rebuilding tables from PDFs in: {data_dir}")
    print(f"📁 Output directory: {storage_dir}")
    print(f"🔧 Using method: {method}")
    
    # Clear existing tables
    clear_existing_tables(storage_dir)
    
    # Find all PDF files
    pdf_files = find_pdfs(data_dir)
    print(f"📄 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    if not pdf_files:
        print("❌ No PDF files found in data directory")
        return {"table_registry": {}, "total_tables": 0}
    
    # Initialize table registry
    table_dir = storage_dir / "tables"
    table_registry = ensure_json(table_dir / "registry.json")
    total_tables = 0
    
    # Process each PDF file
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n📊 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
        
        try:
            # Extract tables from this PDF
            entries = extract_tables_for_pdf(pdf_path, table_dir, pages="all")
            
            if entries:
                print(f"  ✅ Found {len(entries)} tables")
                for entry in entries:
                    table_registry[entry["table_id"]] = entry
                    total_tables += 1
                    method_info = f"method: {entry.get('method', 'unknown')}"
                    print(f"    - {entry['table_id']} (page {entry['page']}, {method_info})")
            else:
                print(f"  ⚠️  No tables found")
                
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {str(e)}")
            continue
    
    # Save the updated registry
    save_registry(table_dir / "registry.json", table_registry)
    
    print(f"\n✅ Table rebuild complete!")
    print(f"📊 Total tables extracted: {total_tables}")
    print(f"📁 Registry saved to: {table_dir / 'registry.json'}")
    
    return {
        "table_registry": table_registry,
        "total_tables": total_tables,
        "processed_pdfs": len(pdf_files)
    }

def main():
    """Main function to run the table rebuild process."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Rebuild table registry from PDF files")
    parser.add_argument(
        "--method", 
        choices=["pdfplumber", "camelot", "camelot_multi"], 
        default="pdfplumber",
        help="Table extraction method (default: pdfplumber)"
    )
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    storage_dir = script_dir / "storage"
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure PDF files are in the data directory")
        return 1
    
    # Check if storage directory exists, create if not
    if not storage_dir.exists():
        print(f"📁 Creating storage directory: {storage_dir}")
        storage_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Rebuild tables
        result = rebuild_tables(data_dir, storage_dir, method=args.method)
        
        # Print summary
        print(f"\n📋 Summary:")
        print(f"  - Method used: {args.method}")
        print(f"  - Processed PDFs: {result['processed_pdfs']}")
        print(f"  - Total tables: {result['total_tables']}")
        print(f"  - Registry entries: {len(result['table_registry'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during table rebuild: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
