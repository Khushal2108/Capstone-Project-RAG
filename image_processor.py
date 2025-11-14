import io
from PIL import Image
from typing import List, Tuple, Optional
import google.generativeai as genai
import PyPDF2
from docx import Document
from docx.oxml import parse_xml
from config import Config
import time
import base64

class ImageProcessor:
    """Extract and process images from documents with context-aware descriptions"""
    
    def __init__(self):
        self.vision_model = None
        self._initialize_vision_model()
    
    def _initialize_vision_model(self):
        """Initialize Gemini vision model with retry logic"""
        for attempt in range(3):
            try:
                api_key = Config.api_key_manager.get_current_key()
                genai.configure(api_key=api_key)
                self.vision_model = genai.GenerativeModel(Config.VISION_MODEL)
                print("✅ Vision model initialized successfully")
                return
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                Config.api_key_manager.mark_key_failed()
                if attempt == 2:
                    raise Exception("Failed to initialize vision model after 3 attempts")
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[Image.Image, int]]:
        """Extract images from PDF pages"""
        images = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    
                    try:
                        if '/XObject' in page['/Resources']:
                            xObject = page['/Resources']['/XObject'].get_object()
                            
                            for obj in xObject:
                                if xObject[obj]['/Subtype'] == '/Image':
                                    try:
                                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                                        data = xObject[obj].get_data()
                                        
                                        if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                                            mode = "RGB"
                                        else:
                                            mode = "P"
                                        
                                        image = Image.frombytes(mode, size, data)
                                        image.thumbnail(Config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                                        images.append((image, page_num + 1))
                                        
                                    except Exception as img_error:
                                        continue
                    except:
                        continue
            
            if images:
                print(f"📄 Extracted {len(images)} embedded images from {pdf_path}")
            else:
                print(f"ℹ️ No embedded images found in {pdf_path}")
            
        except Exception as e:
            print(f"⚠️ Error extracting images from PDF: {str(e)}")
        
        return images
    
    def extract_images_from_docx(self, docx_path: str) -> List[Tuple[Image.Image, int]]:
        """Extract images from DOCX files"""
        images = []
        
        try:
            doc = Document(docx_path)
            
            # Extract images from document relationships
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        image = Image.open(io.BytesIO(image_data))
                        image.thumbnail(Config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                        
                        # Try to determine which paragraph/section contains this image
                        # For simplicity, we'll mark as page 1 for now
                        images.append((image, 1))
                        
                    except Exception as e:
                        print(f"⚠️ Could not process image: {str(e)}")
                        continue
            
            if images:
                print(f"📄 Extracted {len(images)} images from {docx_path}")
            else:
                print(f"ℹ️ No images found in {docx_path}")
                
        except Exception as e:
            print(f"❌ Error extracting images from DOCX: {str(e)}")
        
        return images
    
    def generate_contextual_description(
        self, 
        image: Image.Image, 
        document_context: str,
        page_num: int,
        user_query: str = None
    ) -> Optional[str]:
        """Generate context-aware description of image using Gemini Vision"""
        
        max_retries = len(Config.api_key_manager.api_keys)
        
        for attempt in range(max_retries):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG', quality=Config.IMAGE_QUALITY)
                img_byte_arr.seek(0)
                
                # Create prompt based on whether it's from document or user upload
                if user_query:
                    prompt = f"""You are analyzing an image uploaded by a user in the context of a document Q&A system.

User's Question: {user_query}

Document Context (for reference):
{document_context[:800]}...

Please analyze this image and provide:
1. What type of visual is this (photo, chart, diagram, screenshot, etc.)
2. Detailed description of what's shown
3. Key information, data points, or text visible
4. How this image relates to the user's question
5. How this connects to the document context (if applicable)

Be specific and detailed - the user is asking about this image in relation to their documents."""
                else:
                    prompt = f"""Analyze this image from page {page_num} of a technical document.

Document Context:
{document_context[:500]}...

Please provide a detailed description of this image focusing on:
1. Type of visual (chart, diagram, table, flowchart, graph, illustration, etc.)
2. Key data points, labels, and values if present
3. Relationships and flows shown
4. Main insights or purpose of the visual
5. Any text, annotations, or legends visible
6. How this relates to the document context

Be specific and detailed - imagine you're describing it to someone who can't see it.
If it's a chart, describe the axes, data series, trends.
If it's a diagram, describe the components and their relationships.
If it's a table, describe the structure and key data.

Description:"""
                
                api_key = Config.api_key_manager.get_current_key()
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(Config.VISION_MODEL)
                
                response = model.generate_content([prompt, image])
                description = response.text
                
                if not user_query:
                    full_description = f"[Page {page_num}] {description}"
                else:
                    full_description = f"[User Uploaded Image] {description}"
                
                return full_description
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if 'quota' in error_msg or 'rate limit' in error_msg or '429' in error_msg:
                    print(f"⚠️ API quota/rate limit hit, cycling key... (Attempt {attempt + 1}/{max_retries})")
                    Config.api_key_manager.mark_key_failed()
                    time.sleep(2)
                else:
                    print(f"❌ Error generating image description: {str(e)}")
                    return None
        
        print("❌ All API keys exhausted for this image")
        return None
    
    def process_document_images(
        self, 
        file_path: str, 
        document_text: str
    ) -> List[Tuple[str, str, int]]:
        """
        Process all images from a document
        Returns: List of (description, source_file, page_num)
        """
        results = []
        doc_context = document_text[:1000] if document_text else "Technical documentation"
        
        # Extract images based on file type
        if file_path.lower().endswith('.pdf'):
            images = self.extract_images_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            images = self.extract_images_from_docx(file_path)
        elif file_path.lower().endswith('.doc'):
            # Try DOCX method for .doc files
            try:
                images = self.extract_images_from_docx(file_path)
            except:
                print(f"⚠️ Could not extract images from .doc file: {file_path}")
                images = []
        else:
            images = []
        
        if not images:
            print(f"ℹ️ No images to process from {file_path}")
            return results
        
        print(f"📄 Generating descriptions for {len(images)} images...")
        
        for idx, (image, page_num) in enumerate(images, start=1):
            print(f"  Processing image {idx}/{len(images)} from page {page_num}...")
            
            description = self.generate_contextual_description(
                image, 
                doc_context,
                page_num
            )
            
            if description:
                results.append((description, file_path, page_num))
                print(f"  ✅ Generated description for page {page_num}")
            else:
                print(f"  ⚠️ Failed to generate description for page {page_num}")
            
            time.sleep(0.5)
        
        print(f"✅ Processed {len(results)} images with descriptions")
        return results
    
    def process_uploaded_image(
        self,
        image: Image.Image,
        document_context: str,
        user_query: str
    ) -> Optional[str]:
        """
        Process a user-uploaded image in the context of documents and query
        Returns: Image description
        """
        print(f"🖼️ Processing uploaded image for query: {user_query[:50]}...")
        
        description = self.generate_contextual_description(
            image,
            document_context,
            page_num=0,  # Not from a page
            user_query=user_query
        )
        
        if description:
            print("✅ Generated description for uploaded image")
        else:
            print("⚠️ Failed to generate description for uploaded image")
        
        return description
