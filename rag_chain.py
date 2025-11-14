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
        
        # Prompt for queries with uploaded images
        self.image_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an intelligent assistant analyzing documents and images together.
The user has uploaded an image and is asking a question about it in the context of the provided documents.

Guidelines:
1. Analyze both the uploaded image description and the document context
2. Explain how the uploaded image relates to the documents
3. Answer the user's specific question about the image
4. If the image appears in the documents, reference that connection
5. Be specific and detailed in your analysis

Document Context:
{context}

Uploaded Image Analysis:
{image_description}

User Question: {question}

Answer:""")
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
        
        if query_results['text_results'].get('documents'):
            context_parts.append("=== TEXT CONTENT ===")
            for idx, (doc, metadata) in enumerate(zip(
                query_results['text_results']['documents'][0],
                query_results['text_results']['metadatas'][0]
            )):
                source = metadata.get('source', 'Unknown')
                context_parts.append(f"\n[Source: {source}]\n{doc}\n")
        
        if query_results['image_results'].get('documents'):
            context_parts.append("\n=== VISUAL CONTENT (Charts, Diagrams, Tables) ===")
            for idx, (doc, metadata) in enumerate(zip(
                query_results['image_results']['documents'][0],
                query_results['image_results']['metadatas'][0]
            )):
                source = metadata.get('source', 'Unknown')
                page = metadata.get('page', 'Unknown')
                context_parts.append(f"\n[Visual from {source}, Page {page}]\n{doc}\n")
        
        return "\n".join(context_parts)
    
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
                    query_results = self.vector_store.query_with_uploaded_image(
                        question, 
                        uploaded_image, 
                        n_results=Config.TOP_K
                    )
                else:
                    query_results = self.vector_store.query(question, n_results=Config.TOP_K)
                
                # Format context
                context = self._format_context(query_results)
                
                if not context.strip():
                    return "I don't have enough information in my knowledge base to answer this question. Please make sure documents are properly ingested."
                
                # Choose appropriate prompt and create chain
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
