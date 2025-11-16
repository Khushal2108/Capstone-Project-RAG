from typing import Optional, Dict, Any
from PIL import Image
import os

# Try to import LangGraph - if it fails, we'll use simple workflow
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph imported successfully")
except Exception as e:
    print(f"⚠️ LangGraph import failed: {str(e)}")
    print("ℹ️ Will use simple workflow mode")

# LangGraph State Definition (only if available)
if LANGGRAPH_AVAILABLE:
    class GraphState(TypedDict):
        """State for the LangGraph workflow"""
        question: str
        uploaded_image: Optional[Image.Image]
        document_context: str
        image_description: str
        response: str
        has_image: bool
        error: Optional[str]

class RAGWorkflow:
    """Hybrid RAG workflow with LangGraph support and automatic fallback"""
    
    def __init__(self, rag_chain, use_langgraph: bool = True):
        """
        Initialize workflow with optional LangGraph
        
        Args:
            rag_chain: The RAG chain instance
            use_langgraph: Whether to try using LangGraph (default: True)
        """
        self.rag_chain = rag_chain
        self.image_processor = None
        self.workflow = None
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        
        # Try to create LangGraph workflow
        if self.use_langgraph:
            try:
                self._create_langgraph_workflow()
            except Exception as e:
                print(f"⚠️ LangGraph workflow creation failed: {str(e)}")
                print("ℹ️ Falling back to simple workflow")
                self.use_langgraph = False
        
        if not self.use_langgraph:
            print("✅ Using simple workflow mode (stable)")
    
    def set_image_processor(self, image_processor):
        """Set image processor for handling uploaded images"""
        self.image_processor = image_processor
        print("✅ Image processor attached to workflow")
    
    def _create_langgraph_workflow(self):
        """Create LangGraph workflow (only called if LangGraph is available)"""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph not available")
        
        print("🔄 Creating LangGraph workflow...")
        
        # Create state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("process_image", self._process_image_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_edge("analyze_query", "process_image")
        workflow.add_edge("process_image", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Compile
        self.workflow = workflow.compile()
        print("✅ LangGraph workflow created successfully!")
    
    # ==================== LangGraph Nodes ====================
    
    def _analyze_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the query (LangGraph node)"""
        print(f"📊 Analyzing query: {state['question'][:50]}...")
        
        # Detect if query is about visuals
        visual_keywords = [
            'chart', 'graph', 'figure', 'image', 'diagram', 'table',
            'picture', 'photo', 'visual', 'show', 'display'
        ]
        
        question_lower = state['question'].lower()
        state['has_image'] = (
            state.get('uploaded_image') is not None or
            any(keyword in question_lower for keyword in visual_keywords)
        )
        
        print(f"   • Has image: {state.get('uploaded_image') is not None}")
        print(f"   • Visual query: {state['has_image']}")
        
        return state
    
    def _process_image_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process uploaded image if present (LangGraph node)"""
        if state.get('uploaded_image') and self.image_processor:
            print("🖼️ Processing uploaded image in LangGraph node...")
            try:
                image_description = self.image_processor.process_uploaded_image(
                    state['uploaded_image'],
                    state.get('document_context', ''),
                    state['question']
                )
                
                if image_description:
                    state['image_description'] = image_description
                    print("✅ Image analysis complete")
                else:
                    state['image_description'] = "Image analysis returned no results"
                    print("⚠️ Image analysis empty")
                    
            except Exception as e:
                print(f"❌ Image processing error: {str(e)}")
                state['image_description'] = f"Error processing image: {str(e)}"
                state['error'] = str(e)
        else:
            state['image_description'] = ""
        
        return state
    
    def _generate_response_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response (LangGraph node)"""
        print("🤖 Generating response in LangGraph node...")
        
        try:
            response = self.rag_chain.generate_response(
                state['question'],
                uploaded_image=state.get('uploaded_image'),
                image_description=state.get('image_description', '')
            )
            state['response'] = response
            print("✅ Response generated")
            
        except Exception as e:
            print(f"❌ Response generation error: {str(e)}")
            state['response'] = f"Error generating response: {str(e)}"
            state['error'] = str(e)
        
        return state
    
    # ==================== Simple Workflow ====================
    
    def _run_simple_workflow(
        self,
        question: str,
        uploaded_image: Optional[Image.Image] = None,
        document_context: str = ""
    ) -> str:
        """Simple workflow without LangGraph"""
        print("🚀 Running simple workflow...")
        
        try:
            # Step 1: Process image if present
            image_description = None
            if uploaded_image and self.image_processor:
                print("🖼️ Processing uploaded image...")
                try:
                    image_description = self.image_processor.process_uploaded_image(
                        uploaded_image,
                        document_context,
                        question
                    )
                    
                    if image_description:
                        print("✅ Image analysis complete")
                    else:
                        print("⚠️ Image analysis empty")
                        image_description = "Image uploaded for analysis"
                        
                except Exception as img_error:
                    print(f"⚠️ Image processing error: {str(img_error)}")
                    image_description = "Image processing encountered an error"
            
            # Step 2: Generate response
            print("🤖 Generating response...")
            response = self.rag_chain.generate_response(
                question,
                uploaded_image=uploaded_image,
                image_description=image_description
            )
            
            print("✅ Response generated successfully")
            return response
            
        except Exception as e:
            error_msg = f"Error in simple workflow: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error processing your request.\n\nError: {str(e)}"
    
    # ==================== Main Run Method ====================
    
    def run(
        self,
        question: str,
        uploaded_image: Optional[Image.Image] = None,
        document_context: str = ""
    ) -> str:
        """
        Execute the workflow with automatic LangGraph/Simple mode selection
        
        Args:
            question: User's question
            uploaded_image: Optional uploaded image
            document_context: Context from documents
            
        Returns:
            Generated response string
        """
        
        # Try LangGraph workflow first if available
        if self.use_langgraph and self.workflow:
            try:
                print("🔄 Using LangGraph workflow...")
                
                initial_state = {
                    "question": question,
                    "uploaded_image": uploaded_image,
                    "document_context": document_context,
                    "image_description": "",
                    "response": "",
                    "has_image": uploaded_image is not None,
                    "error": None
                }
                
                result = self.workflow.invoke(initial_state)
                
                # Check if there was an error
                if result.get('error'):
                    print(f"⚠️ LangGraph workflow error: {result['error']}")
                    print("ℹ️ Falling back to simple workflow...")
                    return self._run_simple_workflow(question, uploaded_image, document_context)
                
                return result.get("response", "No response generated")
                
            except Exception as e:
                print(f"❌ LangGraph execution error: {str(e)}")
                print("ℹ️ Falling back to simple workflow...")
                import traceback
                traceback.print_exc()
                
                # Disable LangGraph for future requests in this session
                self.use_langgraph = False
                
                # Fall through to simple workflow
        
        # Use simple workflow (either as primary or fallback)
        return self._run_simple_workflow(question, uploaded_image, document_context)
    
    # ==================== Utility Methods ====================
    
    def get_workflow_mode(self) -> str:
        """Get current workflow mode"""
        if self.use_langgraph and self.workflow:
            return "LangGraph"
        return "Simple"
    
    def force_simple_mode(self):
        """Force switch to simple workflow mode"""
        self.use_langgraph = False
        self.workflow = None
        print("✅ Switched to simple workflow mode")
    
    def try_enable_langgraph(self):
        """Try to re-enable LangGraph mode"""
        if not LANGGRAPH_AVAILABLE:
            print("❌ LangGraph not available in environment")
            return False
        
        try:
            self._create_langgraph_workflow()
            self.use_langgraph = True
            print("✅ LangGraph mode enabled")
            return True
        except Exception as e:
            print(f"❌ Failed to enable LangGraph: {str(e)}")
            return False
