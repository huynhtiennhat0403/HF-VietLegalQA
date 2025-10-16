# scripts/train_query_analyzer.py
"""Train query analyzer (intent + layer classifier)"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.query_analyzer import QueryAnalyzer

def main():
    print("="*60)
    print("🚀 Training Query Analyzer")
    print("="*60)
    
    # Load training data
    print("\n📂 Loading data...")
    df_train = pd.read_csv('data/processed/csv/vn_admin_qa_train.csv')
    df_val = pd.read_csv('data/processed/csv/vn_admin_qa_validation.csv')
    
    print(f"✅ Train: {len(df_train)} samples")
    print(f"✅ Validation: {len(df_val)} samples")
    
    # Train analyzer
    print("\n🔧 Training classifiers...")
    analyzer = QueryAnalyzer()
    analyzer.train(df_train)
    
    # Evaluate on validation
    print("\n📊 Validation Performance:")
    X_val = analyzer.vectorizer.transform(df_val['question'])
    
    intent_acc = analyzer.intent_clf.score(X_val, df_val['intent'])
    layer_acc = analyzer.layer_clf.score(X_val, df_val['layer'])
    category_acc = analyzer.category_clf.score(X_val, df_val['category'])
    
    print(f"  Intent accuracy: {intent_acc:.3f}")
    print(f"  Layer accuracy: {layer_acc:.3f}")
    print(f"  Category accuracy: {category_acc:.3f}")
    
    # Save models
    print("\n💾 Saving models...")
    analyzer.save()
    print("✅ Saved to models/query_analyzer/")
    
    # Test examples
    print("\n🧪 Testing examples:")
    test_questions = [
        "Mình cần giấy tờ gì để kết hôn?",
        "Quy trình ly hôn như thế nào?",
        "Chi phí đăng ký kết hôn là bao nhiêu?"
    ]
    
    for q in test_questions:
        result = analyzer.analyze(q)
        print(f"\nQ: {q}")
        print(f"   → Intent: {result['intent']} (conf: {result['intent_confidence']:.2f})")
        print(f"   → Layer: {result['layer']} (conf: {result['layer_confidence']:.2f})")
    
    print("\n" + "="*60)
    print("✨ Training completed!")
    print("="*60)

if __name__ == "__main__":
    main()