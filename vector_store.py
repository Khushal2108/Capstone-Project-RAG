import chromadb
from chromadb.config import Settings
from typing import List, Tuple, Optional
from config import Config
import time
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

class VectorStore:
    """Manage ChromaDB vector store for text and image embeddings using CLIP"""
    
    def __init__(self):
        print("📄 Loading CLIP embedding model...")
        self.embedding_model = SentenceTransformer(Config.CLIP_MODEL)
        print("✅ CLIP model loaded successfully")
        
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.text_collection = self.client.get_or_create_collection(
            name=Config.TEXT_COLLECTION,
            metadata={"description": "Document text chunks with CLIP embeddings"}
        )
        
        self.image_collection = self.client.get_or_create_collection(
            name=Config.IMAGE_COLLECTION,
            metadata={"description": "Image descriptions with CLIP embeddings"}
        )
        
        print("✅ ChromaDB initialized")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using CLIP (local, no API limits!)"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error generating CLIP embedding: {str(e)}")
            return None
    
    def _generate_image_embedding(self, image: Image.Image) -> Optional[List[float]]:
        """Generate embedding directly from image using CLIP"""
        try:
            # CLIP can encode images directly
            embedding = self.embedding_model.encode(image, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Error generating image embedding: {str(e)}")
            return None
    
    def add_text_chunks(self, chunks: List[Tuple[str, dict]]):
        """Add text chunks to vector store"""
        
        if not chunks:
            print("⚠️ No text chunks to add")
            return
        
        print(f"\n📄 Adding {len(chunks)} text chunks to vector store...")
        
        texts = []
        metadatas = []
        ids = []
        embeddings = []
        
        for idx, (chunk_text, metadata) in enumerate(chunks):
            embedding = self._generate_embedding(chunk_text)
            
            if embedding:
                texts.append(chunk_text)
                metadatas.append(metadata)
                ids.append(f"text_{idx}")
                embeddings.append(embedding)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(chunks)} chunks...")
        
        if embeddings:
            self.text_collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Added {len(embeddings)} text chunks to vector store")
        else:
            print("❌ Failed to generate embeddings for text chunks")
    
    def add_image_descriptions(self, image_data: List[Tuple[str, str, int]]):
        """Add image descriptions to vector store"""
        
        if not image_data:
            print("⚠️ No image descriptions to add")
            return
        
        print(f"\n📄 Adding {len(image_data)} image descriptions to vector store...")
        
        texts = []
        metadatas = []
        ids = []
        embeddings = []
        
        for idx, (description, source, page_num) in enumerate(image_data):
            embedding = self._generate_embedding(description)
            
            if embedding:
                texts.append(description)
                metadatas.append({
                    'source': source,
                    'page': page_num,
                    'type': 'image_description'
                })
                ids.append(f"image_{idx}")
                embeddings.append(embedding)
                
                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx + 1}/{len(image_data)} images...")
        
        if embeddings:
            self.image_collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Added {len(embeddings)} image descriptions to vector store")
        else:
            print("❌ Failed to generate embeddings for image descriptions")
    
    def query(self, query_text: str, n_results: int = 5, query_image: Image.Image = None) -> dict:
        """Query both text and image collections with text or image"""
        
        # Generate query embedding based on input type
        if query_image:
            print("🖼️ Generating query embedding from uploaded image...")
            query_embedding = self._generate_image_embedding(query_image)
        else:
            query_embedding = self._generate_embedding(query_text)
        
        if not query_embedding:
            return {'text_results': {}, 'image_results': {}}
        
        # Query text collection
        try:
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        except Exception as e:
            print(f"⚠️ Error querying text collection: {str(e)}")
            text_results = {}
        
        # Query image collection
        try:
            image_results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, 3)
            )
        except Exception as e:
            print(f"⚠️ Error querying image collection: {str(e)}")
            image_results = {}
        
        return {
            'text_results': text_results,
            'image_results': image_results
        }
    
    def query_with_uploaded_image(
        self,
        query_text: str,
        uploaded_image: Image.Image,
        n_results: int = 5
    ) -> dict:
        """Query with both text and uploaded image for multimodal search"""
        
        print("🔍 Performing multimodal query...")
        
        # Generate embeddings for both text and image
        text_embedding = self._generate_embedding(query_text)
        image_embedding = self._generate_image_embedding(uploaded_image)
        
        if not text_embedding or not image_embedding:
            return {'text_results': {}, 'image_results': {}}
        
        # Average the embeddings for combined query
        import numpy as np
        combined_embedding = (np.array(text_embedding) + np.array(image_embedding)) / 2
        combined_embedding = combined_embedding.tolist()
        
        # Query both collections with combined embedding
        try:
            text_results = self.text_collection.query(
                query_embeddings=[combined_embedding],
                n_results=n_results
            )
        except Exception as e:
            print(f"⚠️ Error querying text collection: {str(e)}")
            text_results = {}
        
        try:
            image_results = self.image_collection.query(
                query_embeddings=[combined_embedding],
                n_results=min(n_results, 3)
            )
        except Exception as e:
            print(f"⚠️ Error querying image collection: {str(e)}")
            image_results = {}
        
        return {
            'text_results': text_results,
            'image_results': image_results
        }
    
    def clear_all(self):
        """Clear all collections"""
        try:
            self.client.delete_collection(Config.TEXT_COLLECTION)
            self.client.delete_collection(Config.IMAGE_COLLECTION)
            
            self.text_collection = self.client.get_or_create_collection(
                name=Config.TEXT_COLLECTION,
                metadata={"description": "Document text chunks with CLIP embeddings"}
            )
            
            self.image_collection = self.client.get_or_create_collection(
                name=Config.IMAGE_COLLECTION,
                metadata={"description": "Image descriptions with CLIP embeddings"}
            )
            
            print("✅ Cleared all collections")
        except Exception as e:
            print(f"⚠️ Error clearing collections: {str(e)}")
    
    def get_statistics(self) -> dict:
        """Get collection statistics"""
        try:
            text_count = self.text_collection.count()
            image_count = self.image_collection.count()
            
            return {
                'text_chunks': text_count,
                'image_descriptions': image_count,
                'total': text_count + image_count
            }
        except Exception as e:
            print(f"⚠️ Error getting statistics: {str(e)}")
            return {'text_chunks': 0, 'image_descriptions': 0, 'total': 0}
