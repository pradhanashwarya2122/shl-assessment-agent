# agent/retrieval.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class HybridRetriever:
    def __init__(self, catalog_path='data/assessments.json', cache_dir='data/cache'):
        with open(catalog_path) as f:
            self.assessments = json.load(f)
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try loading from cache first
        if not self._load_cache():
            self._build_indices()
            self._save_cache()
    
    def _build_indices(self):
        """Build both embedding and BM25 indices"""
        # Prepare documents for indexing
        self.documents = []
        for a in self.assessments:
            doc = f"{a['name']} {a['description']} {a['test_type']}"
            if a.get('job_level'):
                doc += f" {a['job_level']}"
            if a.get('languages'):
                doc += f" {' '.join(a['languages'])}"
            self.documents.append(doc.lower())
        
        # BM25 Index
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Embedding Model & Index
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        # Normalize for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # ID to index mapping for fast lookup
        self.name_to_idx = {a['name'].lower(): i for i, a in enumerate(self.assessments)}
    
    def _load_cache(self):
        """Load pre-built indices from disk"""
        try:
            with open(f'{self.cache_dir}/embeddings.pkl', 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(f'{self.cache_dir}/documents.pkl', 'rb') as f:
                self.documents = pickle.load(f)
            with open(f'{self.cache_dir}/bm25.pkl', 'rb') as f:
                self.bm25 = pickle.load(f)
            print("✅ Loaded indices from cache")
            return True
        except:
            return False
    
    def _save_cache(self):
        """Save indices to disk"""
        with open(f'{self.cache_dir}/embeddings.pkl', 'wb') as f:
            pickle.dump(self.embeddings, f)
        with open(f'{self.cache_dir}/documents.pkl', 'wb') as f:
            pickle.dump(self.documents, f)
        with open(f'{self.cache_dir}/bm25.pkl', 'wb') as f:
            pickle.dump(self.bm25, f)
        print("✅ Saved indices to cache")
    
    def build_query(self, context):
        """Construct optimized search query from context"""
        query_parts = []
        
        # Role/skills are highest priority
        if context.get('role'):
            query_parts.append(context['role'])
        if context.get('skills'):
            query_parts.extend(context['skills'][:3])  # Top 3 skills
        
        # Add constraints with lower weight
        if context.get('seniority'):
            query_parts.append(context['seniority'])
        if context.get('test_types'):
            type_names = {
                'K': 'knowledge technical skills',
                'P': 'personality behavioral',
                'C': 'cognitive ability aptitude',
                'S': 'situational judgment',
                'B': 'behavioral'
            }
            for t in context['test_types']:
                if t in type_names:
                    query_parts.append(type_names[t])
        
        return ' '.join(query_parts)
    
    def search(self, context, k=10):
        """Hybrid search combining BM25 and embeddings"""
        query = self.build_query(context)
        
        if not query.strip():
            return []
        
        # BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Embedding Search
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        embedding_scores = np.dot(query_embedding, self.embeddings.T)[0]
        
        # Normalize scores
        scaler = MinMaxScaler()
        bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        embedding_norm = scaler.fit_transform(embedding_scores.reshape(-1, 1)).flatten()
        
        # Weighted combination (favor exact matches for keyword queries)
        if any(skill.lower() in query.lower() for skill in ['java', 'python', 'sql', 'react']):
            weights = (0.6, 0.4)  # More BM25 weight for technical skills
        else:
            weights = (0.4, 0.6)  # More embedding weight for conceptual queries
        
        combined_scores = weights[0] * bm25_norm + weights[1] * embedding_norm
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-k*2:][::-1]  # Get more for filtering
        
        # Apply filters
        results = []
        for idx in top_indices:
            if combined_scores[idx] < 0.1:  # Minimum relevance threshold
                continue
            
            assessment = self.assessments[idx].copy()
            assessment['score'] = float(combined_scores[idx])
            
            # Filter by test type if specified
            if context.get('test_types'):
                if assessment['test_type'] in context['test_types']:
                    results.append(assessment)
            else:
                results.append(assessment)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_by_name(self, name):
        """Direct lookup by assessment name"""
        idx = self.name_to_idx.get(name.lower())
        if idx is not None:
            return self.assessments[idx]
        return None
    
    def get_by_names(self, names):
        """Lookup multiple assessments by name"""
        return [a for name in names if (a := self.get_by_name(name))]