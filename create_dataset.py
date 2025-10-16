import re
import json
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import random

class VNAdminQAGenerator:
    """Generate Q&A dataset from Vietnamese administrative procedure texts"""
    
    def __init__(self):
        self.category_map = {
            'lyhon': 'ly_hon',
            'kethon': 'ket_hon'
        }
        
        # Intent classification keywords
        self.intent_keywords = {
            'rights': ['quyền', 'có quyền', 'được quyền', 'ai có thể'],
            'conditions': ['điều kiện', 'yêu cầu', 'cần', 'phải'],
            'documents': ['giấy tờ', 'hồ sơ', 'tài liệu', 'chuẩn bị'],
            'location': ['nộp', 'đâu', 'nơi', 'địa điểm', 'ở đâu'],
            'cost': ['phí', 'chi phí', 'giá', 'tiền', 'bao nhiêu'],
            'duration': ['bao lâu', 'thời gian', 'nhanh', 'lâu'],
            'procedure': ['quy trình', 'thủ tục', 'cách', 'làm thế nào', 'bước']
        }
    
    def parse_file(self, filepath: str, category: str) -> Dict:
        """
        Parse file với structure phức tạp (có và không có subtopics)
        
        Returns:
            {
                'category': 'ly_hon',
                'main_topics': [
                    {
                        'id': '1',
                        'title': 'Hướng dẫn chi tiết...',
                        'has_subtopics': True,
                        'subtopics': [...] hoặc None,
                        'content': '...' (nếu không có subtopics)
                    }
                ]
            }
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        parsed = {
            'category': self.category_map.get(category, category),
            'main_topics': []
        }
        
        # Split by main topics (1., 2., 3., ...)
        main_pattern = r'^(\d+)\.\s+([^\n]+)'
        main_matches = list(re.finditer(main_pattern, text, re.MULTILINE))
        
        for i, match in enumerate(main_matches):
            topic_id = match.group(1)
            topic_title = match.group(2).strip()
            
            # Extract content between this topic and next topic
            start_pos = match.end()
            end_pos = main_matches[i + 1].start() if i + 1 < len(main_matches) else len(text)
            topic_content = text[start_pos:end_pos].strip()
            
            # Check if has subtopics (1.1, 1.2, ...)
            subtopic_pattern = rf'^{topic_id}\.(\d+)\.?\s+([^\n]+)'
            subtopic_matches = list(re.finditer(subtopic_pattern, topic_content, re.MULTILINE))
            
            main_topic = {
                'id': topic_id,
                'title': topic_title,
                'has_subtopics': len(subtopic_matches) > 0
            }
            
            if len(subtopic_matches) > 0:
                # Has subtopics
                subtopics = []
                for j, sub_match in enumerate(subtopic_matches):
                    sub_id = f"{topic_id}.{sub_match.group(1)}"
                    sub_title = sub_match.group(2).strip()
                    
                    # Extract subtopic content
                    sub_start = sub_match.end()
                    sub_end = subtopic_matches[j + 1].start() if j + 1 < len(subtopic_matches) else len(topic_content)
                    sub_content = topic_content[sub_start:sub_end].strip()
                    
                    subtopics.append({
                        'id': sub_id,
                        'title': sub_title,
                        'content': sub_content
                    })
                
                main_topic['subtopics'] = subtopics
                main_topic['content'] = None
            else:
                # No subtopics, just content
                main_topic['subtopics'] = None
                main_topic['content'] = topic_content
            
            parsed['main_topics'].append(main_topic)
        
        return parsed
    
    def generate_question_variants(self, title: str, is_subtopic: bool = True) -> List[str]:
        """
        Generate formal + casual question variants
        
        Args:
            title: Section title (e.g., "Ai có quyền yêu cầu ly hôn?")
            is_subtopic: True if subtopic, False if main topic
        """
        questions = []
        title_lower = title.lower().strip()
        
        # If already a question
        if '?' in title:
            # Add original
            questions.append(title)
            
            # Remove question mark for variants
            title_no_q = title.rstrip('?').strip()
            
            # Formal variants
            questions.extend([
                f"Cho tôi biết {title_no_q.lower()}?",
                f"Xin hướng dẫn {title_no_q.lower()}?",
            ])
            
            # Casual variants (first-person)
            questions.extend([
                f"Mình muốn hỏi {title_no_q.lower()}?",
                f"{title_no_q}?",  # Variant with different capitalization
            ])
            
            # Very casual
            if 'có quyền' in title_lower or 'được' in title_lower:
                questions.append(f"Mình có thể {title_no_q.lower().replace('có quyền', '').replace('được', '')} không?")
        
        else:
            # Not a question - convert to questions
            # Formal
            questions.extend([
                f"{title}?",
                f"Cho tôi biết về {title.lower()}?",
                f"Hướng dẫn về {title.lower()}?",
            ])
            
            # Casual
            questions.extend([
                f"{title} như thế nào?",
                f"Mình muốn hỏi về {title.lower()}?",
            ])
        
        # Limit to 6 variants for subtopics, 3 for main topics
        limit = 6 if is_subtopic else 3
        return list(set(questions))[:limit]
    
    def summarize_content(self, content: str, max_words: int = 100) -> str:
        """
        Simple summarization (extractive)
        Takes first few sentences up to max_words
        """
        sentences = re.split(r'[.!?]\s+', content)
        summary = []
        word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) <= max_words:
                summary.append(sentence)
                word_count += len(words)
            else:
                break
        
        result = '. '.join(summary)
        if not result.endswith('.'):
            result += '.'
        
        return result.strip()
    
    def classify_intent(self, title: str) -> str:
        """Classify intent based on title keywords"""
        title_lower = title.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return intent
        
        return 'general'
    
    def estimate_difficulty(self, content: str) -> str:
        """Estimate difficulty based on content length and complexity"""
        word_count = len(content.split())
        
        # Check for legal terms complexity
        legal_terms = ['luật', 'điều', 'khoản', 'nghị định', 'bộ luật', 'quy định']
        legal_count = sum(1 for term in legal_terms if term in content.lower())
        
        if word_count < 100 and legal_count < 3:
            return 'easy'
        elif word_count < 250 and legal_count < 5:
            return 'medium'
        else:
            return 'hard'
    
    def generate_layer1_qa(self, parsed_data: Dict) -> List[Dict]:
        """
        Layer 1: Granular Q&A from subtopics
        """
        qa_pairs = []
        category = parsed_data['category']
        
        for main_topic in parsed_data['main_topics']:
            if main_topic['has_subtopics']:
                for subtopic in main_topic['subtopics']:
                    # Generate questions
                    questions = self.generate_question_variants(
                        subtopic['title'], 
                        is_subtopic=True
                    )
                    
                    # Full answer
                    answer_full = subtopic['content']
                    
                    # Summarized answer
                    answer_summary = self.summarize_content(subtopic['content'], max_words=100)
                    
                    # Metadata
                    intent = self.classify_intent(subtopic['title'])
                    difficulty = self.estimate_difficulty(subtopic['content'])
                    
                    # Create pairs for each question
                    for question in questions:
                        # Full version
                        qa_pairs.append({
                            'question': question,
                            'answer': answer_full,
                            'answer_type': 'full',
                            'category': category,
                            'main_topic': main_topic['title'],
                            'main_topic_id': main_topic['id'],
                            'subtopic': subtopic['title'],
                            'subtopic_id': subtopic['id'],
                            'intent': intent,
                            'difficulty': difficulty,
                            'layer': 1,
                            'answer_word_count': len(answer_full.split())
                        })
                        
                        # Summary version
                        qa_pairs.append({
                            'question': question,
                            'answer': answer_summary,
                            'answer_type': 'summary',
                            'category': category,
                            'main_topic': main_topic['title'],
                            'main_topic_id': main_topic['id'],
                            'subtopic': subtopic['title'],
                            'subtopic_id': subtopic['id'],
                            'intent': intent,
                            'difficulty': difficulty,
                            'layer': 1,
                            'answer_word_count': len(answer_summary.split())
                        })
        
        return qa_pairs
    
    def generate_layer2_qa(self, parsed_data: Dict) -> List[Dict]:
        """
        Layer 2: Topic-level Q&A from sections without subtopics
        """
        qa_pairs = []
        category = parsed_data['category']
        
        for main_topic in parsed_data['main_topics']:
            if not main_topic['has_subtopics']:
                # Generate questions
                questions = self.generate_question_variants(
                    main_topic['title'],
                    is_subtopic=False
                )
                
                # Answers
                answer_full = main_topic['content']
                answer_summary = self.summarize_content(main_topic['content'], max_words=120)
                
                # Metadata
                intent = self.classify_intent(main_topic['title'])
                difficulty = self.estimate_difficulty(main_topic['content'])
                
                for question in questions:
                    # Full version
                    qa_pairs.append({
                        'question': question,
                        'answer': answer_full,
                        'answer_type': 'full',
                        'category': category,
                        'main_topic': main_topic['title'],
                        'main_topic_id': main_topic['id'],
                        'subtopic': None,
                        'subtopic_id': None,
                        'intent': intent,
                        'difficulty': difficulty,
                        'layer': 2,
                        'answer_word_count': len(answer_full.split())
                    })
                    
                    # Summary version
                    qa_pairs.append({
                        'question': question,
                        'answer': answer_summary,
                        'answer_type': 'summary',
                        'category': category,
                        'main_topic': main_topic['title'],
                        'main_topic_id': main_topic['id'],
                        'subtopic': None,
                        'subtopic_id': None,
                        'intent': intent,
                        'difficulty': difficulty,
                        'layer': 2,
                        'answer_word_count': len(answer_summary.split())
                    })
        
        return qa_pairs
    
    def generate_layer3_qa(self, parsed_data: Dict) -> List[Dict]:
        """
        Layer 3: Overview Q&A for main topics with subtopics
        """
        qa_pairs = []
        category = parsed_data['category']
        
        for main_topic in parsed_data['main_topics']:
            if main_topic['has_subtopics']:
                # Generate overview questions
                overview_questions = [
                    f"Hướng dẫn chi tiết về {main_topic['title'].lower()}?",
                    f"{main_topic['title']}?",
                    f"Tổng quan về {main_topic['title'].lower()}?",
                ]
                
                # Create overview answer from all subtopics
                overview_parts = []
                for i, subtopic in enumerate(main_topic['subtopics'], 1):
                    summary = self.summarize_content(subtopic['content'], max_words=50)
                    overview_parts.append(f"{i}. {subtopic['title']}: {summary}")
                
                answer_overview = f"{main_topic['title']} bao gồm:\n\n" + "\n\n".join(overview_parts)
                
                for question in overview_questions:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer_overview,
                        'answer_type': 'overview',
                        'category': category,
                        'main_topic': main_topic['title'],
                        'main_topic_id': main_topic['id'],
                        'subtopic': None,
                        'subtopic_id': None,
                        'intent': 'general',
                        'difficulty': 'medium',
                        'layer': 3,
                        'answer_word_count': len(answer_overview.split())
                    })
        
        return qa_pairs
    
    def process_files(self, files: Dict[str, str]) -> pd.DataFrame:
        """
        Process multiple files and generate complete dataset
        
        Args:
            files: {'lyhon': 'path/to/lyhon.txt', 'kethon': 'path/to/kethon.txt'}
        """
        all_qa = []
        
        for category, filepath in files.items():
            print(f"\n{'='*60}")
            print(f"📄 Processing: {category.upper()}")
            print(f"{'='*60}")
            
            # Parse
            parsed = self.parse_file(filepath, category)
            print(f"✅ Parsed {len(parsed['main_topics'])} main topics")
            
            # Layer 1: Granular
            layer1_qa = self.generate_layer1_qa(parsed)
            print(f"✅ Layer 1 (Granular): {len(layer1_qa)} Q&A pairs")
            
            # Layer 2: Topic-level
            layer2_qa = self.generate_layer2_qa(parsed)
            print(f"✅ Layer 2 (Topic-level): {len(layer2_qa)} Q&A pairs")
            
            # Layer 3: Overview
            layer3_qa = self.generate_layer3_qa(parsed)
            print(f"✅ Layer 3 (Overview): {len(layer3_qa)} Q&A pairs")
            
            all_qa.extend(layer1_qa + layer2_qa + layer3_qa)
            print(f"📊 Total for {category}: {len(layer1_qa) + len(layer2_qa) + len(layer3_qa)}")
        
        df = pd.DataFrame(all_qa)
        return df
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Split dataset with stratification on category
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train vs temp (val+test)
        train, temp = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df['category'],
            random_state=random_state
        )
        
        # Second split: val vs test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            temp,
            test_size=(1 - val_test_ratio),
            stratify=temp['category'],
            random_state=random_state
        )
        
        return {
            'train': train.reset_index(drop=True),
            'validation': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)
        }
    
    def generate_statistics(self, df: pd.DataFrame, splits: Dict = None) -> str:
        """Generate comprehensive statistics report"""
        stats = []
        stats.append("="*70)
        stats.append("📊 DATASET STATISTICS REPORT")
        stats.append("="*70)
        
        # Overall stats
        stats.append(f"\n📈 OVERALL STATISTICS:")
        stats.append(f"  Total samples: {len(df)}")
        stats.append(f"  Unique questions: {df['question'].nunique()}")
        
        # Category distribution
        stats.append(f"\n📂 CATEGORY DISTRIBUTION:")
        for cat, count in df['category'].value_counts().items():
            stats.append(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")
        
        # Layer distribution
        stats.append(f"\n🏗️ LAYER DISTRIBUTION:")
        for layer, count in sorted(df['layer'].value_counts().items()):
            stats.append(f"  Layer {layer}: {count} ({count/len(df)*100:.1f}%)")
        
        # Answer type distribution
        stats.append(f"\n📝 ANSWER TYPE DISTRIBUTION:")
        for atype, count in df['answer_type'].value_counts().items():
            stats.append(f"  {atype}: {count} ({count/len(df)*100:.1f}%)")
        
        # Intent distribution
        stats.append(f"\n🎯 INTENT DISTRIBUTION:")
        for intent, count in df['intent'].value_counts().items():
            stats.append(f"  {intent}: {count} ({count/len(df)*100:.1f}%)")
        
        # Difficulty distribution
        stats.append(f"\n⭐ DIFFICULTY DISTRIBUTION:")
        for diff, count in df['difficulty'].value_counts().items():
            stats.append(f"  {diff}: {count} ({count/len(df)*100:.1f}%)")
        
        # Answer length statistics
        stats.append(f"\n📏 ANSWER LENGTH STATISTICS (words):")
        stats.append(f"  Mean: {df['answer_word_count'].mean():.1f}")
        stats.append(f"  Median: {df['answer_word_count'].median():.1f}")
        stats.append(f"  Min: {df['answer_word_count'].min()}")
        stats.append(f"  Max: {df['answer_word_count'].max()}")
        
        # Split statistics
        if splits:
            stats.append(f"\n✂️ TRAIN/VAL/TEST SPLIT:")
            for split_name, split_df in splits.items():
                stats.append(f"  {split_name}: {len(split_df)} samples")
                for cat in df['category'].unique():
                    cat_count = len(split_df[split_df['category'] == cat])
                    stats.append(f"    - {cat}: {cat_count}")
        
        stats.append("\n" + "="*70)
        
        return "\n".join(stats)
    
    def save_dataset(self, df: pd.DataFrame, output_prefix: str = 'vn_admin_qa'):
        """Save dataset in multiple formats"""
        
        # Split dataset
        splits = self.split_dataset(df)
        
        # Generate statistics
        stats_report = self.generate_statistics(df, splits)
        print(f"\n{stats_report}")
        
        # Save statistics to file
        with open(f'{output_prefix}_statistics.txt', 'w', encoding='utf-8') as f:
            f.write(stats_report)
        
        # Save full dataset
        df.to_csv(f'{output_prefix}_full.csv', index=False, encoding='utf-8-sig')
        df.to_json(f'{output_prefix}_full.json', orient='records', 
                   force_ascii=False, indent=2)
        
        # Save splits
        for split_name, split_df in splits.items():
            # CSV
            split_df.to_csv(f'{output_prefix}_{split_name}.csv', 
                           index=False, encoding='utf-8-sig')
            # JSON
            split_df.to_json(f'{output_prefix}_{split_name}.json', 
                            orient='records', force_ascii=False, indent=2)
        
        # Save in Hugging Face Dataset format
        hf_dataset = {
            'train': splits['train'].to_dict('records'),
            'validation': splits['validation'].to_dict('records'),
            'test': splits['test'].to_dict('records')
        }
        with open(f'{output_prefix}_hf_format.json', 'w', encoding='utf-8') as f:
            json.dump(hf_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 FILES SAVED:")
        print(f"  ✅ {output_prefix}_full.csv")
        print(f"  ✅ {output_prefix}_full.json")
        print(f"  ✅ {output_prefix}_train.csv/json")
        print(f"  ✅ {output_prefix}_validation.csv/json")
        print(f"  ✅ {output_prefix}_test.csv/json")
        print(f"  ✅ {output_prefix}_hf_format.json")
        print(f"  ✅ {output_prefix}_statistics.txt")


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    # Initialize generator
    generator = VNAdminQAGenerator()
    
    # Define file paths
    files = {
        'lyhon': 'lyhon.txt',
        'kethon': 'kethon.txt'
    }
    
    # Process files and generate dataset
    print("\n🚀 Starting Vietnamese Administrative Q&A Generation...")
    print("="*70)
    
    df = generator.process_files(files)
    
    # Save dataset
    generator.save_dataset(df, output_prefix='vn_admin_qa')
    
    print("\n✨ Dataset generation completed successfully!")
    print("\n📌 NEXT STEPS:")
    print("  1. Review the statistics file to check data quality")
    print("  2. Manually check some samples in the CSV files")
    print("  3. Use vn_admin_qa_hf_format.json for training with HuggingFace")
    print("  4. Consider adding more synthetic variants if needed")