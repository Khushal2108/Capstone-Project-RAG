"""
Standalone document ingestion script
Run this BEFORE starting Streamlit for faster startup
"""

import os
import sys
from pathlib import Path
from config import Config
from document_processor import DocumentProcessor
from image_processor import ImageProcessor
from vector_store import VectorStore

def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print("🤖 BYTE SIZE - Document Ingestion")
    print("="*70 + "\n")

def check_data_directory():
    """Check if data directory exists and has documents"""
    data_dir = Path(Config.DATA_DIR)
    
    if not data_dir.exists():
        print(f"❌ Error: Data directory '{Config.DATA_DIR}' not found")
        print(f"💡 Creating directory...")
        Config.ensure_directories()
        print(f"✅ Created '{Config.DATA_DIR}' directory")
        print(f"\n📝 Please add PDF, DOC, or DOCX files to this directory and run again.")
        return False
    
    doc_files = (
        list(data_dir.glob("*.pdf")) + 
        list(data_dir.glob("*.docx")) + 
        list(data_dir.glob("*.doc"))
    )
    
    if not doc_files:
        print(f"⚠️ Warning: No documents found in '{Config.DATA_DIR}'")
        print(f"📝 Please add PDF, DOC, or DOCX files to this directory.")
        return False
    
    print(f"✅ Found {len(doc_files)} document(s) in '{Config.DATA_DIR}':")
    for doc in doc_files:
        file_size = doc.stat().st_size / 1024
        print(f"   • {doc.name} ({file_size:.1f} KB)")
    print()
    
    return True

def initialize_vector_store():
    """Initialize the vector store"""
    print("⚙️ Initializing vector store...")
    try:
        vector_store = VectorStore()
        print("✅ Vector store initialized\n")
        return vector_store
    except Exception as e:
        print(f"❌ Error initializing vector store: {str(e)}")
        sys.exit(1)

def process_documents(vector_store):
    """Process and ingest all documents"""
    print("="*70)
    print("📄 STEP 1: Processing Text Documents")
    print("="*70 + "\n")
    
    try:
        doc_processor = DocumentProcessor()
        text_chunks, document_texts = doc_processor.process_all_documents(Config.DATA_DIR)
        
        if not text_chunks:
            print("⚠️ No text content extracted from documents")
            return None, None
        
        print(f"\n✅ Text processing complete:")
        print(f"   • Documents processed: {len(document_texts)}")
        print(f"   • Text chunks created: {len(text_chunks)}")
        
        return text_chunks, document_texts
        
    except Exception as e:
        print(f"❌ Error processing documents: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

def process_images(document_texts):
    """Process images from documents"""
    print("\n" + "="*70)
    print("🖼️ STEP 2: Processing Images and Visual Content")
    print("="*70 + "\n")
    
    if not document_texts:
        print("⚠️ No documents to process for images")
        return []
    
    try:
        image_processor = ImageProcessor()
        all_image_data = []
        
        for idx, (file_path, doc_text) in enumerate(document_texts.items(), start=1):
            print(f"Processing document {idx}/{len(document_texts)}: {Path(file_path).name}")
            image_data = image_processor.process_document_images(file_path, doc_text)
            all_image_data.extend(image_data)
            print()
        
        print(f"✅ Image processing complete:")
        print(f"   • Total images processed: {len(all_image_data)}")
        
        return all_image_data
        
    except Exception as e:
        print(f"❌ Error processing images: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

def store_embeddings(vector_store, text_chunks, image_data):
    """Store embeddings in vector database"""
    print("\n" + "="*70)
    print("💾 STEP 3: Creating and Storing Embeddings")
    print("="*70 + "\n")
    
    try:
        if text_chunks:
            print("📝 Storing text embeddings...")
            vector_store.add_text_chunks(text_chunks)
            print()
        
        if image_data:
            print("🖼️ Storing image description embeddings...")
            vector_store.add_image_descriptions(image_data)
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing embeddings: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def display_statistics(vector_store):
    """Display final statistics"""
    print("\n" + "="*70)
    print("📊 INGESTION SUMMARY")
    print("="*70 + "\n")
    
    stats = vector_store.get_statistics()
    
    print(f"✅ Successfully ingested documents!")
    print(f"\n📈 Statistics:")
    print(f"   • Text chunks: {stats['text_chunks']}")
    print(f"   • Image descriptions: {stats['image_descriptions']}")
    print(f"   • Total embeddings: {stats['total']}")
    print(f"\n💾 Storage location: {Config.CHROMA_DB_DIR}")
    
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETE!")
    print("="*70 + "\n")

def clear_database(vector_store):
    """Clear the vector database"""
    print("\n🗑️ Clearing existing database...")
    try:
        vector_store.clear_all()
        print("✅ Database cleared successfully\n")
    except Exception as e:
        print(f"⚠️ Error clearing database: {str(e)}\n")

def main():
    """Main ingestion pipeline"""
    print_banner()
    
    # Check for clear flag
    if "--clear" in sys.argv:
        print("🗑️ Clear database mode activated")
        Config.ensure_directories()
        vector_store = initialize_vector_store()
        clear_database(vector_store)
        print("✅ Database cleared. Exiting.")
        return
    
    # Check data directory
    if not check_data_directory():
        sys.exit(1)
    
    # Initialize
    Config.ensure_directories()
    vector_store = initialize_vector_store()
    
    # Ask user if they want to clear existing data
    try:
        stats = vector_store.get_statistics()
        if stats['total'] > 0:
            print(f"⚠️ Database already contains {stats['total']} embeddings")
            response = input("Do you want to clear existing data? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                clear_database(vector_store)
            else:
                print("ℹ️ Adding to existing database...\n")
    except:
        pass
    
    # Process documents
    text_chunks, document_texts = process_documents(vector_store)
    
    if text_chunks is None:
        print("\n❌ Ingestion failed. Please check the errors above.")
        sys.exit(1)
    
    # Process images
    image_data = process_images(document_texts)
    
    # Store in vector database
    success = store_embeddings(vector_store, text_chunks, image_data)
    
    if not success:
        print("\n❌ Failed to store embeddings. Please check the errors above.")
        sys.exit(1)
    
    # Display results
    display_statistics(vector_store)
    
    # Final instructions
    print("🚀 Next Steps:")
    print("   1. Run: streamlit run app.py")
    print("   2. OR Run: python auto_ingest_and_run.py (automated)")
    print("   3. System will auto-load your data!")
    print("\n💡 Tips:")
    print("   • Upload images in chat for analysis")
    print("   • Ask about charts, diagrams, or any visuals")
    print("   • LangGraph workflow is active!")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Ingestion cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
