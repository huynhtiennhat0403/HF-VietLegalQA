# src/inference/chatbot.py
from src.models.query_analyzer import QueryAnalyzer
from src.models.retrieval_model import SmartRetriever
import joblib
import pandas as pd
import numpy as np

class VNAdminChatbot:
    """Main chatbot pipeline"""
    
    def __init__(self):
        # Load components
        self.analyzer = QueryAnalyzer()
        self.analyzer.vectorizer = joblib.load('models/query_analyzer/vectorizer.pkl')
        self.analyzer.intent_clf = joblib.load('models/query_analyzer/intent_clf.pkl')
        self.analyzer.layer_clf = joblib.load('models/query_analyzer/layer_clf.pkl')
        
        self.retriever = SmartRetriever()
        df = pd.read_csv('models/embeddings/indexed_data.csv')
        self.retriever.df = df
        self.retriever.embeddings = np.load('models/embeddings/embeddings.npy')
    
    def answer(self, question, detail_level='auto', top_k=1):
        """
        Answer user question
        
        Args:
            question: User query
            detail_level: 'auto', 'brief', 'detailed', 'overview'
            top_k: Number of answers to return
        """
        # Stage 1: Analyze query
        analysis = self.analyzer.analyze(question)
        
        intent = analysis['intent']
        layer = analysis['layer']
        
        # Override layer nếu user specify
        if detail_level != 'auto':
            layer_map = {'brief': 1, 'detailed': 2, 'overview': 3}
            layer = layer_map.get(detail_level, layer)
        
        # Determine answer_type based on layer
        answer_type_map = {1: 'summary', 2: 'full', 3: 'overview'}
        answer_type = answer_type_map[layer]
        
        # Stage 2: Retrieve answers
        results = self.retriever.retrieve(
            query=question,
            intent=intent,
            layer=layer,
            answer_type=answer_type,
            top_k=top_k
        )
        
        # Format response
        if len(results) == 0:
            return {
                'answer': "Xin lỗi, tôi không tìm thấy thông tin phù hợp.",
                'confidence': 0.0,
                'metadata': None
            }
        
        best_result = results[0]
        
        return {
            'answer': best_result['answer'],
            'confidence': best_result['score'],
            'metadata': {
                **best_result['metadata'],
                'detected_intent': intent,
                'detected_layer': layer,
                'alternatives': results[1:] if len(results) > 1 else []
            }
        }
    
    def chat(self):
        """Interactive chat mode"""
        print("🤖 VN Admin Chatbot - Tư vấn Kết hôn & Ly hôn")
        print("=" * 50)
        print("Gõ 'quit' để thoát\n")
        
        while True:
            question = input("👤 Bạn: ")
            
            if question.lower() in ['quit', 'exit', 'thoát']:
                print("👋 Tạm biệt!")
                break
            
            response = self.answer(question)
            
            print(f"\n🤖 Bot (Confidence: {response['confidence']:.2f}):")
            print(f"{response['answer']}\n")
            
            # Show metadata
            meta = response['metadata']
            if meta:
                print(f"📊 [Intent: {meta['detected_intent']}, "
                      f"Layer: {meta['detected_layer']}, "
                      f"Category: {meta['category']}]\n")