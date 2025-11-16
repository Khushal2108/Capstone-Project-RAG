from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List, Optional
from config import Config
import time
from PIL import Image

class RAGChain:
    """LangChain RAG implementation with multimodal support"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = None
        self._initialize_llm()
        
        # Standard prompt for text-only queries
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an intelligent assistant with deep knowledge of the provided documents.
Your task is to answer questions accurately based on the retrieved context.

Guidelines:
1. Answer based ONLY on the provided context
2. If the context contains image descriptions, use them to answer visual questions
3. Be specific and cite sources when possible
4. If information is not in the context, say so clearly
5. For charts/figures/tables, describe what you see in the context
6. Provide detailed, informative answers

Context:
{context}

Question: {question}

Answer:""")
        ])
        
        # Enhanced prompt for queries with uploaded images
        self.image_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an intelligent assistant analyzing an uploaded image along with document knowledge.

The user has uploaded an image and wants to understand it. Even if the exact image is not in the documents, use your knowledge to:
1. Analyze and describe the uploaded image in detail
2. Compare it with similar content from the documents if available
3. Provide insights based on the image content
4. Connect it to relevant document information when possible

IMPORTANT: Always provide a detailed analysis of the uploaded image, even if it's not directly in the documents.

Document Context (for reference):
{context}

Uploaded Image Analysis:
{image_description}

User Question: {question}

Provide a comprehensive answer about the uploaded image:""")
        ])
    
    def _initialize_llm(self):
        """Initialize LLM with current API key"""
        try:
            api_key = Config.api_key_manager.get_current_key()
            self.llm = ChatGoogleGenerativeAI(
                model=Config.LLM_MODEL,
                google_api_key=api_key,
                temperature=Config.TEMPERATURE,
                max_output_tokens=Config.MAX_OUTPUT_TOKENS,
                convert_system_message_to_human=True
            )
            print("✅ LLM initialized")
        except Exception as e:
            print(f"❌ Error initializing LLM: {str(e)}")
            raise
    
    def _format_context(self, query_results: dict) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        
        if query_results.get('text_results', {}).get('documents'):
            context_parts.append("=== TEXT CONTENT ===")
            docs = query_results['text_results'].get('documents', [[]])
            metas = query_results['text_results'].get('metadatas', [[]])
            
            if docs and docs[0]:
                for idx, (doc, metadata) in enumerate(zip(docs[0], metas[0])):
                    source = metadata.get('source', 'Unknown')
                    context_parts.append(f"\n[Source: {source}]\n{doc}\n")
        
        if query_results.get('image_results', {}).get('documents'):
            context_parts.append("\n=== VISUAL CONTENT (Charts, Diagrams, Tables) ===")
            docs = query_results['image_results'].get('documents', [[]])
            metas = query_results['image_results'].get('metadatas', [[]])
            
            if docs and docs[0]:
                for idx, (doc, metadata) in enumerate(zip(docs[0], metas[0])):
                    source = metadata.get('source', 'Unknown')
                    page = metadata.get('page', 'Unknown')
                    context_parts.append(f"\n[Visual from {source}, Page {page}]\n{doc}\n")
        
        return "\n".join(context_parts) if context_parts else "No specific document context available."
    
    def generate_response(
        self, 
        question: str, 
        uploaded_image: Optional[Image.Image] = None,
        image_description: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """Generate response with optional image support"""
        
        for attempt in range(max_retries):
            try:
                # Query vector store (with or without image)
                if uploaded_image:
                    # For uploaded images, still get document context but don't fail if empty
                    try:
                        query_results = self.vector_store.query_with_uploaded_image(
                            question, 
                            uploaded_image, 
                            n_results=Config.TOP_K
                        )
                    except:
                        # Fallback to text-only query
                        query_results = self.vector_store.query(question, n_results=Config.TOP_K)
                else:
                    query_results = self.vector_store.query(question, n_results=Config.TOP_K)
                
                # Format context
                context = self._format_context(query_results)
                
                # For uploaded images, proceed even with minimal context
                if uploaded_image and image_description:
                    chain = (
                        {
                            "context": lambda x: context,
                            "image_description": lambda x: image_description,
                            "question": RunnablePassthrough()
                        }
                        | self.image_prompt
                        | self.llm
                        | StrOutputParser()
                    )
                else:
                    # Only fail if no context for text-only queries
                    if not context.strip() or context == "No specific document context available.":
                        return "I don't have enough information in my knowledge base to answer this question. Please make sure documents are properly ingested."
                    
                    chain = (
                        {
                            "context": lambda x: context,
                            "question": RunnablePassthrough()
                        }
                        | self.prompt
                        | self.llm
                        | StrOutputParser()
                    )
                
                # Generate response
                response = chain.invoke(question)
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if 'quota' in error_msg or 'rate limit' in error_msg or '429' in error_msg:
                    print(f"⚠️ LLM API quota/rate limit hit, cycling key... (Attempt {attempt + 1}/{max_retries})")
                    Config.api_key_manager.mark_key_failed()
                    self._initialize_llm()
                    time.sleep(2)
                else:
                    print(f"❌ Error: {str(e)}")
                    return f"Error generating response: {str(e)}"
                
                if attempt == max_retries - 1:
                    return "All API keys exhausted. Please check your API key configuration."
        
        return "Failed to generate response after multiple attempts."
