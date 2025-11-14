import streamlit as st
import os
from pathlib import Path
import time
from PIL import Image
from config import Config
from document_processor import DocumentProcessor
from image_processor import ImageProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
from graph_workflow import RAGWorkflow

# Page configuration
st.set_page_config(
    page_title="Byte Size - Multimodal RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1557a0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #666;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vector_store = None
    st.session_state.rag_chain = None
    st.session_state.workflow = None
    st.session_state.chat_history = []
    st.session_state.ingestion_complete = False
    st.session_state.processing_stats = None
    st.session_state.uploaded_image = None
    st.session_state.image_processor = None

def initialize_system():
    """Initialize the RAG system"""
    try:
        with st.spinner("⚙️ Initializing system components..."):
            # Ensure directories exist
            Config.ensure_directories()
            
            # Initialize vector store
            st.session_state.vector_store = VectorStore()
            
            # Initialize RAG chain
            st.session_state.rag_chain = RAGChain(st.session_state.vector_store)
            
            # Initialize LangGraph workflow
            st.session_state.workflow = RAGWorkflow(st.session_state.rag_chain)
            
            # Initialize image processor
            st.session_state.image_processor = ImageProcessor()
            st.session_state.workflow.set_image_processor(st.session_state.image_processor)
            
            st.session_state.initialized = True
            
        return True
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return False

def ingest_documents():
    """Ingest documents from data directory with enhanced progress tracking"""
    with st.spinner("📄 Processing documents..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process documents (30%)
            status_text.text("📄 Reading and chunking documents...")
            doc_processor = DocumentProcessor()
            text_chunks, document_texts = doc_processor.process_all_documents(Config.DATA_DIR)
            progress_bar.progress(30)
            
            if not text_chunks:
                st.warning("⚠️ No documents found in the data directory. Please add PDF, DOC, or DOCX files.")
                return False
            
            # Step 2: Process images (50%)
            status_text.text("🖼️ Extracting and analyzing images...")
            image_processor = ImageProcessor()
            
            all_image_data = []
            for idx, (file_path, doc_text) in enumerate(document_texts.items(), start=1):
                status_text.text(f"🖼️ Processing images from document {idx}/{len(document_texts)}...")
                image_data = image_processor.process_document_images(file_path, doc_text)
                all_image_data.extend(image_data)
            
            progress_bar.progress(50)
            
            # Step 3: Add to vector store (80%)
            status_text.text("💾 Creating embeddings and storing in ChromaDB...")
            st.session_state.vector_store.add_text_chunks(text_chunks)
            progress_bar.progress(70)
            
            if all_image_data:
                status_text.text("💾 Storing image descriptions...")
                st.session_state.vector_store.add_image_descriptions(all_image_data)
            progress_bar.progress(90)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Ingestion complete!")
            
            # Show statistics
            stats = st.session_state.vector_store.get_statistics()
            st.session_state.processing_stats = stats
            
            st.success(f"""
            ✅ **Ingestion Complete!**
            - Documents processed: {len(document_texts)}
            - Text chunks: {stats['text_chunks']}
            - Image descriptions: {stats['image_descriptions']}
            - Total embeddings: {stats['total']}
            """)
            
            st.session_state.ingestion_complete = True
            time.sleep(2)
            status_text.empty()
            progress_bar.empty()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error during ingestion: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return False

def display_chat_history():
    """Display chat history with enhanced formatting"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            content = message["content"]
            if message.get("has_image"):
                content = f"🖼️ [With Image] {content}"
            
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<p class="main-header">🤖 Byte Size - Multimodal RAG Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by LangChain, LangGraph & Google Gemini</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize System", use_container_width=True):
                if initialize_system():
                    st.success("✅ System initialized!")
                    st.rerun()
        else:
            st.markdown('<div class="success-box">✅ System Ready</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Ingestion section
        if st.session_state.initialized:
            st.header("📂 Document Management")
            
            # Check for documents
            data_dir = Path(Config.DATA_DIR)
            if data_dir.exists():
                doc_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx")) + list(data_dir.glob("*.doc"))
                
                if doc_files:
                    st.markdown(f'<div class="info-box">📄 Found {len(doc_files)} document(s)</div>', unsafe_allow_html=True)
                    
                    with st.expander("📋 View Documents", expanded=False):
                        for doc in doc_files:
                            file_size = doc.stat().st_size / 1024  # KB
                            st.text(f"• {doc.name} ({file_size:.1f} KB)")
                else:
                    st.markdown('<div class="warning-box">⚠️ No documents found</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Data directory not found")
            
            st.divider()
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Ingest Documents", use_container_width=True):
                    ingest_documents()
            
            with col2:
                if st.button("🗑️ Clear Database", use_container_width=True):
                    if st.session_state.vector_store:
                        with st.spinner("Clearing database..."):
                            st.session_state.vector_store.clear_all()
                            st.session_state.ingestion_complete = False
                            st.session_state.chat_history = []
                            st.session_state.processing_stats = None
                        st.success("✅ Database cleared!")
                        time.sleep(1)
                        st.rerun()
            
            st.divider()
            
            # Statistics
            if st.session_state.vector_store and st.session_state.ingestion_complete:
                st.header("📊 Statistics")
                stats = st.session_state.vector_store.get_statistics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Text Chunks", stats['text_chunks'])
                with col2:
                    st.metric("Images", stats['image_descriptions'])
                with col3:
                    st.metric("Total", stats['total'])
                
                # API Key status
                st.divider()
                st.header("🔑 API Status")
                available_keys = Config.api_key_manager.get_available_keys_count()
                total_keys = len(Config.api_key_manager.api_keys)
                
                # Color-coded API key status
                if available_keys == total_keys:
                    status_color = "🟢"
                elif available_keys > 0:
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                
                st.metric("Available Keys", f"{status_color} {available_keys}/{total_keys}")
        
        st.divider()
        
        # Instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            ### Quick Start Guide
            
            1. **Initialize System** 
               - Click the 'Initialize System' button above
            
            2. **Add Documents** 
               - Place PDF, DOC, or DOCX files in the `data/` folder
            
            3. **Ingest Documents** 
               - Click 'Ingest Documents' to process files
               - Extracts text, images, charts, and tables
            
            4. **Ask Questions** 
               - Text-only or with uploaded images
            
            ### 🆕 Multimodal Features
            - ✅ Upload images and ask about them
            - ✅ System relates images to document content
            - ✅ Compare uploaded images with document visuals
            - ✅ Extract images from DOCX files
            
            ### Example Questions
            **Text Only:**
            - "Summarize the main points"
            - "What does page 5 discuss?"
            
            **With Uploaded Image:**
            - "What does this image show?"
            - "Is this similar to any charts in the documents?"
            - "Analyze this diagram in context of my docs"
            """)
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Byte Size RAG Chatbot**
            
            A multimodal Retrieval-Augmented Generation system:
            - 🖼️ **Image Upload Support**: Ask about uploaded images
            - 🔍 **Visual Search**: CLIP embeddings for image-text matching
            - 📄 **DOCX Image Extraction**: Now extracts images from Word docs
            - 🤖 **Context-Aware**: Relates images to document content
            - 💾 **ChromaDB**: Persistent vector storage
            
            Built with ❤️ using LangChain, CLIP & Streamlit
            """)
    
    # Main chat area
    if not st.session_state.initialized:
        st.markdown("""
        <div class="info-box">
            <h3>👈 Welcome to Byte Size RAG Chatbot!</h3>
            <p>Please initialize the system using the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.ingestion_complete:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ No Documents Ingested</h3>
            <p>Please follow these steps:</p>
            <ol>
                <li>Add PDF, DOC, or DOCX files to the <code>data/</code> folder</li>
                <li>Click 'Ingest Documents' in the sidebar</li>
                <li>Wait for processing to complete</li>
                <li>Start chatting!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat interface
    st.header("💬 Chat with Your Documents")
    
    # Display chat history
    display_chat_history()
    
    # Image upload section
    with st.expander("🖼️ Upload Image (Optional)", expanded=False):
        st.markdown("""
        **Upload an image to ask questions about it** in the context of your documents.
        
        Examples:
        - Upload a chart and ask "What does this show?"
        - Upload a diagram and ask "Is this similar to any in my documents?"
        - Upload a screenshot and ask "Explain this in context of my docs"
        """)
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.uploaded_image = image
            st.success("✅ Image uploaded! Ask a question about it below.")
        else:
            st.session_state.uploaded_image = None
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your documents...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "has_image": st.session_state.uploaded_image is not None
        })
        
        # Display user message immediately
        user_display = user_input
        if st.session_state.uploaded_image:
            user_display = f"🖼️ [With Image] {user_input}"
        
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {user_display}
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Get document context for image analysis
                doc_context = ""
                if st.session_state.uploaded_image:
                    # Get some context from vector store
                    query_results = st.session_state.vector_store.query(user_input, n_results=3)
                    doc_context = st.session_state.rag_chain._format_context(query_results)
                
                # Use workflow with optional image
                response = st.session_state.workflow.run(
                    user_input,
                    uploaded_image=st.session_state.uploaded_image,
                    document_context=doc_context
                )
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Display assistant message
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Clear uploaded image after use
                st.session_state.uploaded_image = None
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
        
        # Rerun to update chat
        st.rerun()
    
    # Clear chat button at bottom
    if st.session_state.chat_history:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
