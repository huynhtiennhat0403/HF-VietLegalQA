# scripts/run_chatbot.py
"""Run interactive chatbot"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.query_analyzer import QueryAnalyzer
from src.models.retrieval_model import SmartRetriever
import torch

class VNAdminChatbot:
    """Vietnamese Administrative Chatbot for Marriage & Divorce"""
    
    def __init__(self):
        print("🔄 Loading chatbot components...")
        
        # Load Query Analyzer
        self.analyzer = QueryAnalyzer()
        self.analyzer.vectorizer = joblib.load('models/query_analyzer/vectorizer.pkl')
        self.analyzer.intent_clf = joblib.load('models/query_analyzer/intent_clf.pkl')
        self.analyzer.layer_clf = joblib.load('models/query_analyzer/layer_clf.pkl')
        self.analyzer.category_clf = joblib.load('models/query_analyzer/category_clf.pkl')  # ← NEW
        print("  ✅ Query Analyzer loaded")
        
        # Load Retriever
        self.retriever = SmartRetriever()
        df = pd.read_csv('models/embeddings/indexed_data.csv')
        self.retriever.df = df
        
        # Load embeddings
        embeddings_np = np.load('models/embeddings/embeddings.npy')
        self.retriever.embeddings = torch.from_numpy(embeddings_np)
        print("  ✅ Retriever loaded")
        
        print("✨ Chatbot ready!\n")
    
    def answer(self, question, detail_level='auto', top_k=1, verbose=False):
        """
        Answer user question
        
        Args:
            question: User query
            detail_level: 'auto', 'brief', 'detailed', 'overview'
            top_k: Number of answers
            verbose: Show debug info
        """
        # Stage 1: Analyze query
        analysis = self.analyzer.analyze(question)
        
        intent = analysis['intent']
        layer = analysis['layer']
        category = analysis['category']  # ← NEW
        
        if verbose:
            print(f"🔍 Analysis: category={category}, intent={intent}, layer={layer}")
        
        # Override layer if specified
        if detail_level != 'auto':
            layer_map = {'brief': 1, 'detailed': 2, 'overview': 3}
            layer = layer_map.get(detail_level, layer)
        
        # Map layer to answer_type
        answer_type_map = {1: 'summary', 2: 'full', 3: 'overview'}
        answer_type = answer_type_map[layer]
        
        # Stage 2: Retrieve
        results = self.retriever.retrieve(
            query=question,
            intent=intent,
            layer=layer,
            answer_type=answer_type,
            category=category,  # ← NEW: Filter by category
            top_k=top_k
        )
        
        if len(results) == 0:
            return {
                'answer': "Xin lỗi, tôi không tìm thấy thông tin phù hợp. Bạn có thể hỏi lại với cách diễn đạt khác không?",
                'confidence': 0.0,
                'metadata': None
            }
        
        best = results[0]
        
        return {
            'answer': best['answer'],
            'confidence': best['score'],
            'metadata': {
                **best['metadata'],
                'detected_intent': intent,
                'detected_layer': layer,
                'alternatives': results[1:] if len(results) > 1 else []
            }
        }
    
    def chat(self):
        """Interactive chat mode"""
        print("=" * 60)
        print("🤖 VN ADMIN CHATBOT - Tư vấn Kết hôn & Ly hôn")
        print("=" * 60)
        print("\n📋 Hướng dẫn:")
        print("  • Hỏi bình thường: 'Điều kiện kết hôn là gì?'")
        print("  • Chi tiết hơn: 'chi tiết: <câu hỏi>'")
        print("  • Tóm tắt: 'tóm tắt: <câu hỏi>'")
        print("  • Thoát: 'quit' hoặc 'exit'")
        print()
        
        while True:
            try:
                question = input("👤 Bạn: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'thoát', 'q']:
                    print("\n👋 Cảm ơn bạn đã sử dụng chatbot! Hẹn gặp lại!")
                    break
                
                # Parse commands
                detail_level = 'auto'
                if question.lower().startswith('chi tiết:'):
                    detail_level = 'detailed'
                    question = question[9:].strip()
                elif question.lower().startswith('tóm tắt:'):
                    detail_level = 'brief'
                    question = question[8:].strip()
                elif question.lower().startswith('tổng quan:'):
                    detail_level = 'overview'
                    question = question[10:].strip()
                
                # Get answer
                response = self.answer(
                    question, 
                    detail_level=detail_level,
                    top_k=2
                )
                
                # Display answer
                print(f"\n🤖 Bot:")
                print(f"{response['answer']}")
                
                # Show metadata
                if response['metadata']:
                    meta = response['metadata']
                    confidence_emoji = "🟢" if response['confidence'] > 0.8 else "🟡" if response['confidence'] > 0.6 else "🔴"
                    
                    print(f"\n{confidence_emoji} Độ tin cậy: {response['confidence']:.2%}")
                    print(f"📊 [Chủ đề: {meta['category']} | "
                          f"Intent: {meta['detected_intent']} | "
                          f"Chi tiết: Layer {meta['detected_layer']}]")
                    
                    # Show alternatives
                    if meta['alternatives']:
                        print(f"\n💡 Có {len(meta['alternatives'])} câu trả lời khác liên quan")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {e}")
                print("Vui lòng thử lại!")
                print()

def main():
    """Main entry point"""
    try:
        # Initialize chatbot
        bot = VNAdminChatbot()
        
        # Run interactive chat
        bot.chat()
        
    except FileNotFoundError as e:
        print(f"\n❌ Lỗi: Không tìm thấy file cần thiết")
        print(f"Chi tiết: {e}")
        print("\n💡 Vui lòng chạy các script sau trước:")
        print("  1. python scripts/train_query_analyzer.py")
        print("  2. python scripts/index_retrieval_data.py")
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()