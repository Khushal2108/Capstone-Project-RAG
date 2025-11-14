import os
from typing import List, Tuple
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    """Process and chunk documents (PDF, DOC, DOCX)"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num}]\n{page_text}\n"
            print(f"✅ Extracted {len(text)} characters from PDF: {file_path}")
        except Exception as e:
            print(f"❌ Error reading PDF {file_path}: {str(e)}")
        return text
    
    def read_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        text = ""
        try:
            doc = Document(file_path)
            for para_num, paragraph in enumerate(doc.paragraphs, start=1):
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            print(f"✅ Extracted {len(text)} characters from DOCX: {file_path}")
        except Exception as e:
            print(f"❌ Error reading DOCX {file_path}: {str(e)}")
        return text
    
    def read_doc(self, file_path: str) -> str:
        """Extract text from DOC (legacy format)"""
        # For .doc files, we'll try to read as DOCX (works for some files)
        try:
            return self.read_docx(file_path)
        except Exception as e:
            print(f"⚠️ Could not read .doc file {file_path}: {str(e)}")
            print("💡 Tip: Convert .doc to .docx for better compatibility")
            return ""
    
    def read_document(self, file_path: str) -> str:
        """Read document based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.read_pdf(file_path)
        elif ext == '.docx':
            return self.read_docx(file_path)
        elif ext == '.doc':
            return self.read_doc(file_path)
        else:
            print(f"⚠️ Unsupported file format: {ext}")
            return ""
    
    def chunk_text(self, text: str, source: str) -> List[Tuple[str, dict]]:
        """
        Split text into chunks with metadata
        Returns: List of (chunk_text, metadata)
        """
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata to each chunk
        chunked_data = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                'source': source,
                'chunk_id': idx,
                'total_chunks': len(chunks)
            }
            chunked_data.append((chunk, metadata))
        
        print(f"📝 Created {len(chunks)} chunks from {source}")
        return chunked_data
    
    def process_all_documents(self, data_dir: str) -> Tuple[List[Tuple[str, dict]], dict]:
        """
        Process all documents in the data directory
        Returns: (list of text chunks, dict of full document texts)
        """
        all_chunks = []
        document_texts = {}
        
        supported_extensions = ['.pdf', '.docx', '.doc']
        
        print(f"\n📂 Scanning directory: {data_dir}")
        
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in supported_extensions:
                    print(f"\n📄 Processing: {filename}")
                    
                    # Read document
                    text = self.read_document(file_path)
                    
                    if text:
                        # Store full text for image context
                        document_texts[file_path] = text
                        
                        # Chunk text
                        chunks = self.chunk_text(text, filename)
                        all_chunks.extend(chunks)
        
        print(f"\n✅ Processed {len(document_texts)} documents")
        print(f"✅ Created {len(all_chunks)} total text chunks")
        
        return all_chunks, document_texts
