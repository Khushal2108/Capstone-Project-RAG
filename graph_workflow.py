from typing import Optional
from PIL import Image

class RAGWorkflow:
    """Simplified RAG workflow with multimodal support"""
    
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.image_processor = None
    
    def set_image_processor(self, image_processor):
        """Set image processor for handling uploaded images"""
        self.image_processor = image_processor
    
    def run(
        self, 
        question: str, 
        uploaded_image: Optional[Image.Image] = None,
        document_context: str = ""
    ) -> str:
        """Execute the workflow with optional image support"""
        try:
            print(f"🚀 Processing {'multimodal ' if uploaded_image else ''}question: {question[:50]}...")
            
            # If image is uploaded, generate description
            image_description = None
            if uploaded_image and self.image_processor:
                print("🖼️ Processing uploaded image...")
                image_description = self.image_processor.process_uploaded_image(
                    uploaded_image,
                    document_context,
                    question
                )
            
            # Generate response
            print("🤖 Generating response...")
            response = self.rag_chain.generate_response(
                question, 
                uploaded_image=uploaded_image,
                image_description=image_description
            )
            
            print("✅ Response generated successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
