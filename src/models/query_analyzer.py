"""Query Analyzer - Classify intent and predict answer layer"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class QueryAnalyzer:
    """
    Analyze user queries to determine:
    - Intent: What type of info they want (rights, documents, cost, etc.)
    - Layer: How detailed the answer should be (1=brief, 2=detailed, 3=overview)
    """
    
    def __init__(self):
        self.intent_clf = None
        self.layer_clf = None
        self.category_clf = None  # ← THÊM CATEGORY CLASSIFIER
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            min_df=1
        )
    
    def train(self, df):
        """
        Train classifiers on labeled data
        
        Args:
            df: DataFrame with columns ['question', 'intent', 'layer', 'category']
        """
        # Vectorize questions
        X = self.vectorizer.fit_transform(df['question'])
        
        # Train intent classifier
        print("  🔹 Training intent classifier...")
        self.intent_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.intent_clf.fit(X, df['intent'])
        intent_acc = self.intent_clf.score(X, df['intent'])
        
        # Train layer predictor
        print("  🔹 Training layer predictor...")
        self.layer_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.layer_clf.fit(X, df['layer'])
        layer_acc = self.layer_clf.score(X, df['layer'])
        
        # Train category classifier (NEW!)
        print("  🔹 Training category classifier...")
        self.category_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self.category_clf.fit(X, df['category'])
        category_acc = self.category_clf.score(X, df['category'])
        
        print(f"  ✅ Training complete!")
        print(f"     Intent accuracy: {intent_acc:.3f}")
        print(f"     Layer accuracy: {layer_acc:.3f}")
        print(f"     Category accuracy: {category_acc:.3f}")
    
    def analyze(self, question):
        """
        Analyze a user question
        
        Args:
            question: User's question string
            
        Returns:
            dict: {
                'intent': predicted intent,
                'intent_confidence': probability,
                'layer': predicted layer (1, 2, or 3),
                'layer_confidence': probability,
                'category': predicted category (ly_hon or ket_hon),
                'category_confidence': probability
            }
        """
        X = self.vectorizer.transform([question])
        
        # Predict intent
        intent = self.intent_clf.predict(X)[0]
        intent_proba = self.intent_clf.predict_proba(X)[0]
        intent_conf = intent_proba.max()
        
        # Predict layer
        layer = self.layer_clf.predict(X)[0]
        layer_proba = self.layer_clf.predict_proba(X)[0]
        layer_conf = layer_proba.max()
        
        # Predict category (NEW!)
        category = self.category_clf.predict(X)[0]
        category_proba = self.category_clf.predict_proba(X)[0]
        category_conf = category_proba.max()
        
        return {
            'intent': intent,
            'intent_confidence': intent_conf,
            'layer': layer,
            'layer_confidence': layer_conf,
            'category': category,
            'category_confidence': category_conf
        }
    
    def save(self, path='models/query_analyzer'):
        """
        Save trained models and vectorizer to disk
        
        Args:
            path: Directory path to save models (default: 'models/query_analyzer')
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Validate models are trained
            if not all([self.vectorizer, self.intent_clf, self.layer_clf]):
                print("❌ Error: Models must be trained before saving")
                return False
                
            # Save all components
            print(f"💾 Saving models to {path}...")
            joblib.dump(self.vectorizer, f'{path}/vectorizer.pkl')
            joblib.dump(self.intent_clf, f'{path}/intent_clf.pkl')
            joblib.dump(self.layer_clf, f'{path}/layer_clf.pkl')
            joblib.dump(self.category_clf, f'{path}/category_clf.pkl')
            
            print("✅ Models saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False