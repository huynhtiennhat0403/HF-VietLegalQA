# src/models/retrieval_model.py
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

class SmartRetriever:
    """Retrieval với layer và intent filtering"""
    
    def __init__(self, model_name='keepitreal/vietnamese-sbert'):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.embeddings = None
    
    def index(self, df: pd.DataFrame):
        """Create embeddings cho tất cả questions"""
        self.df = df.reset_index(drop=True)
        
        print("🔄 Encoding questions...")
        self.embeddings = self.model.encode(
            df['question'].tolist(),
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        print(f"✅ Indexed {len(df)} Q&A pairs")
    
    def retrieve(self, query, intent=None, layer=None, 
                 answer_type=None, top_k=3):
        """
        Retrieve với filtering
        
        Args:
            query: User question
            intent: Filter by intent (optional)
            layer: Filter by layer (optional)
            answer_type: 'summary', 'full', 'overview' (optional)
            top_k: Number of results
        """
        # Filter dataset
        filtered_df = self.df.copy()
        
        if intent:
            filtered_df = filtered_df[filtered_df['intent'] == intent]
        
        if layer:
            filtered_df = filtered_df[filtered_df['layer'] == layer]
        
        if answer_type:
            filtered_df = filtered_df[filtered_df['answer_type'] == answer_type]
        
        if len(filtered_df) == 0:
            print("⚠️ No matches with filters, using full dataset")
            filtered_df = self.df
        
        # Get embeddings của filtered questions
        filtered_indices = filtered_df.index.tolist()
        filtered_embeddings = self.embeddings[filtered_indices]
        
        # Semantic search
        query_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(
            query_emb, 
            filtered_embeddings, 
            top_k=min(top_k, len(filtered_df))
        )[0]
        
        # Map back to original indices
        results = []
        for hit in hits:
            original_idx = filtered_indices[hit['corpus_id']]
            row = self.df.iloc[original_idx]
            results.append({
                'question': row['question'],
                'answer': row['answer'],
                'score': hit['score'],
                'metadata': {
                    'intent': row['intent'],
                    'layer': row['layer'],
                    'answer_type': row['answer_type'],
                    'category': row['category'],
                    'subtopic': row['subtopic']
                }
            })
        
        return results
    
    def save_embeddings(self, path='models/embeddings'):
        """Save embeddings for fast loading"""
        import os
        os.makedirs(path, exist_ok=True)
        
        np.save(f'{path}/embeddings.npy', self.embeddings.cpu().numpy())
        self.df.to_csv(f'{path}/indexed_data.csv', index=False)
        print(f"✅ Saved embeddings to {path}")