# scripts/index_retrieval_data.py
"""Create embeddings for retrieval model"""

import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.retrieval_model import SmartRetriever

def main():
    print("="*60)
    print("🚀 Indexing Data for Retrieval")
    print("="*60)
    
    # Load full dataset
    print("\n📂 Loading data...")
    df = pd.read_csv('data/processed/csv/vn_admin_qa_full.csv')
    print(f"✅ Loaded {len(df)} Q&A pairs")
    
    # Show distribution
    print("\n📊 Data distribution:")
    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    print(f"  Layers: {df['layer'].value_counts().sort_index().to_dict()}")
    print(f"  Intents: {df['intent'].value_counts().to_dict()}")
    
    # Create retriever
    print("\n🔧 Creating embeddings...")
    retriever = SmartRetriever(model_name='keepitreal/vietnamese-sbert')
    retriever.index(df)
    
    # Save embeddings
    print("\n💾 Saving embeddings...")
    retriever.save_embeddings()
    
    # Test retrieval
    print("\n🧪 Testing retrieval:")
    test_queries = [
        ("Điều kiện để kết hôn là gì?", 'conditions', 1),
        ("Tôi muốn biết chi phí ly hôn", 'cost', 1),
        ("Hướng dẫn toàn bộ thủ tục kết hôn", 'procedure', 3)
    ]
    
    for query, intent, layer in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"Filters: intent={intent}, layer={layer}")
        
        results = retriever.retrieve(
            query=query,
            intent=intent,
            layer=layer,
            top_k=2
        )
        
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i} (score: {r['score']:.3f}):")
            print(f"    Q: {r['question']}")
            
            # ✅ FIX: Truncate answer và replace newlines
            answer_preview = r['answer'].replace('\n', ' ')[:150]
            print(f"    A: {answer_preview}...")
            
            # ✅ FIX: Pretty print metadata
            meta = r['metadata']
            print(f"    Meta:")
            print(f"      - Category: {meta['category']}")
            print(f"      - Intent: {meta['intent']}")
            print(f"      - Layer: {meta['layer']}")
            print(f"      - Answer type: {meta['answer_type']}")
    
    print("\n" + "="*60)
    print("✨ Indexing completed!")
    print("="*60)

if __name__ == "__main__":
    main()