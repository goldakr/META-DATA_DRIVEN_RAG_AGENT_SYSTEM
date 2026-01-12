#!/usr/bin/env python3
"""
Script to process PDF files and print all created chunks to a text file
"""
import os
from pathlib import Path
from datetime import datetime

from src.ingest import find_pdfs, chunk_pdf_text, extract_tables_for_pdf


def clean_text(text: str) -> str:
    """
    Clean and normalize text content for better readability
    
    Args:
        text: Raw text content from PDF
        
    Returns:
        Cleaned text with proper spacing and formatting
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive line breaks and normalize
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Clean up common PDF extraction artifacts
    text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def print_chunks_to_file(data_dir: Path, output_file: Path):
    """
    Process all PDFs in data_dir and print chunks to output_file
    
    Args:
        data_dir: Directory containing PDF files
        output_file: Path to output text file
    """
    print(f"🔍 Processing PDFs from: {data_dir}")
    print(f"📝 Output file: {output_file}")
    
    # Find all PDF files
    pdf_files = find_pdfs(data_dir)
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("❌ No PDF files found!")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("INSURANCE RAG AGENT - TEXT CHUNKS EXTRACTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total PDF files: {len(pdf_files)}\n")
        f.write("=" * 80 + "\n\n")
        
        total_chunks = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"📖 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            
            # Write PDF header
            f.write(f"\n{'='*60}\n")
            f.write(f"PDF FILE {i}: {pdf_path.name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Full path: {pdf_path}\n")
            f.write(f"File size: {pdf_path.stat().st_size:,} bytes\n\n")
            
            try:
                # Extract text chunks
                chunks = chunk_pdf_text(pdf_path)
                f.write(f"📝 TEXT CHUNKS ({len(chunks)} found):\n")
                f.write("-" * 40 + "\n\n")
                
                for j, chunk in enumerate(chunks, 1):
                    f.write(f"CHUNK {j}:\n")
                    f.write("-" * 20 + "\n")
                    
                    # Write metadata
                    metadata = chunk.metadata
                    f.write("METADATA:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                    
                    # Write cleaned text content
                    f.write("TEXT CONTENT:\n")
                    cleaned_text = clean_text(chunk.text)
                    f.write(f"{cleaned_text}\n")
                    f.write("\n" + "="*60 + "\n\n")
                    
                    total_chunks += 1
                
                print(f"  ✅ Extracted {len(chunks)} text chunks")
                
            except Exception as e:
                error_msg = f"❌ Error processing {pdf_path.name}: {str(e)}"
                print(f"  {error_msg}")
                f.write(f"ERROR: {error_msg}\n\n")
        
        # Write summary
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total PDF files processed: {len(pdf_files)}\n")
        f.write(f"Total text chunks extracted: {total_chunks}\n")
        f.write(f"Average chunks per PDF: {total_chunks/len(pdf_files):.1f}\n")
        f.write(f"Generation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Successfully wrote {total_chunks} chunks to {output_file}")
    print(f"📊 Average chunks per PDF: {total_chunks/len(pdf_files):.1f}")


def print_tables_to_file(data_dir: Path, output_file: Path, print_to_terminal: bool = True):
    """
    Process all PDFs and print extracted tables to output file and optionally to terminal
    
    Args:
        data_dir: Directory containing PDF files
        output_file: Path to output text file
        print_to_terminal: Whether to also print tables to terminal
    """
    print(f"🔍 Processing PDFs for tables from: {data_dir}")
    print(f"📝 Output file: {output_file}")
    
    # Find all PDF files
    pdf_files = find_pdfs(data_dir)
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("❌ No PDF files found!")
        return
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("INSURANCE RAG AGENT - TABLE EXTRACTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total PDF files: {len(pdf_files)}\n")
        f.write("=" * 80 + "\n\n")
        
        total_tables = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"📊 Processing tables {i}/{len(pdf_files)}: {pdf_path.name}")
            
            # Write PDF header
            f.write(f"\n{'='*60}\n")
            f.write(f"PDF FILE {i}: {pdf_path.name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Full path: {pdf_path}\n\n")
            
            try:
                # Create temporary directory for tables
                temp_dir = Path("temp_tables")
                temp_dir.mkdir(exist_ok=True)
                
                # Extract tables
                table_entries = extract_tables_for_pdf(pdf_path, temp_dir)
                f.write(f"📊 TABLES FOUND ({len(table_entries)}):\n")
                f.write("-" * 40 + "\n\n")
                
                for j, entry in enumerate(table_entries, 1):
                    f.write(f"TABLE {j}:\n")
                    f.write("-" * 20 + "\n")
                    
                    # Write table metadata
                    f.write("TABLE METADATA:\n")
                    for key, value in entry.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                    
                    # Write extraction method info if available
                    if 'method' in entry:
                        f.write(f"EXTRACTION METHOD: {entry['method']}\n")
                    if 'accuracy' in entry:
                        f.write(f"EXTRACTION ACCURACY: {entry['accuracy']:.2f}\n")
                    f.write("\n")
                    
                    # Write table summary
                    f.write("TABLE SUMMARY:\n")
                    f.write(f"{entry.get('summary', 'No summary available')}\n\n")
                    
                    # Try to read and display CSV content
                    csv_path = Path(entry.get('csv', ''))
                    if csv_path.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path)
                            f.write("TABLE CONTENT (CSV):\n")
                            f.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
                            f.write(f"Columns: {list(df.columns)}\n\n")
                            
                            # Show complete table content
                            f.write("COMPLETE TABLE CONTENT:\n")
                            f.write(df.to_string(index=False))
                            f.write("\n\n")
                            
                            # Print to terminal if requested
                            if print_to_terminal:
                                print(f"\n📊 TABLE {j} from {pdf_path.name}:")
                                print(f"   Method: {entry.get('method', 'Unknown')}")
                                print(f"   Accuracy: {entry.get('accuracy', 'N/A'):.2f}" if entry.get('accuracy') else "   Accuracy: N/A")
                                print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                                print(f"   Summary: {entry.get('summary', 'No summary available')}")
                                print("   Content:")
                                print("   " + "="*50)
                                # Print table content with proper indentation
                                table_str = df.to_string(index=False)
                                for line in table_str.split('\n'):
                                    print("   " + line)
                                print("   " + "="*50)
                            
                        except Exception as e:
                            f.write(f"Error reading CSV: {str(e)}\n\n")
                            if print_to_terminal:
                                print(f"   ❌ Error reading table CSV: {str(e)}")
                    
                    f.write("="*60 + "\n\n")
                    total_tables += 1
                
                print(f"  ✅ Extracted {len(table_entries)} tables")
                
                # Clean up temp directory
                import shutil
                import time
                if temp_dir.exists():
                    try:
                        # Give a moment for file handles to be released
                        time.sleep(0.1)
                        shutil.rmtree(temp_dir)
                    except PermissionError:
                        # If we can't delete immediately, try again after a delay
                        time.sleep(1)
                        try:
                            shutil.rmtree(temp_dir)
                        except:
                            print(f"Warning: Could not clean up temp directory {temp_dir}")
                    except Exception as e:
                        print(f"Warning: Error cleaning up temp directory: {e}")
                
            except Exception as e:
                error_msg = f"❌ Error processing {pdf_path.name}: {str(e)}"
                print(f"  {error_msg}")
                f.write(f"ERROR: {error_msg}\n\n")
        
        # Write summary
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total PDF files processed: {len(pdf_files)}\n")
        f.write(f"Total tables extracted: {total_tables}\n")
        f.write(f"Average tables per PDF: {total_tables/len(pdf_files):.1f}\n")
        f.write(f"Generation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Successfully wrote {total_tables} tables to {output_file}")


def main():
    """Main function to run the chunk printing script"""
    # Set up paths
    data_dir = Path("data")
    output_dir = Path("output")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print text chunks
    chunks_file = output_dir / f"extracted_chunks_{timestamp}.txt"
    print_chunks_to_file(data_dir, chunks_file)
    
    # Print tables
    tables_file = output_dir / f"extracted_tables_{timestamp}.txt"
    print_tables_to_file(data_dir, tables_file)
    
    print(f"\n🎉 All done!")
    print(f"📝 Text chunks saved to: {chunks_file}")
    print(f"📊 Tables saved to: {tables_file}")


if __name__ == "__main__":
    main()
