import torch
import json
import requests
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple
import random
import numpy as np
from pathlib import Path

class MultiModalDataPipeline:
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_instruction_datasets(self) -> List[Dict]:
        """Load various instruction-following datasets"""
        datasets = []
        
        try:
            # Alpaca-like instruction dataset
            alpaca_data = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
            for item in alpaca_data:
                if item['input']:
                    text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nResponse: {item['output']}"
                else:
                    text = f"Instruction: {item['instruction']}\nResponse: {item['output']}"
                datasets.append({
                    'text': text,
                    'source': 'alpaca',
                    'difficulty': len(text.split()) // 20 + 1  # Simple difficulty metric
                })
        except Exception as e:
            print(f"Could not load Alpaca dataset: {e}")
            
        try:
            # OpenAssistant conversations
            oa_data = load_dataset("OpenAssistant/oasst1", split="train[:500]")
            for item in oa_data:
                if item['role'] == 'assistant':
                    datasets.append({
                        'text': item['text'],
                        'source': 'openassistant',
                        'difficulty': len(item['text'].split()) // 25 + 1
                    })
        except Exception as e:
            print(f"Could not load OpenAssistant dataset: {e}")
            
        return datasets
    
    def create_diverse_prompts(self) -> List[Dict]:
        """Create diverse training prompts for different capabilities"""
        prompts = [
            # Reasoning and logic
            {
                'text': 'Question: If all roses are flowers and some flowers are red, can we conclude that some roses are red?\nAnswer: No, we cannot conclude that some roses are red from the given information.',
                'source': 'reasoning',
                'difficulty': 3
            },
            {
                'text': 'Solve this step by step: What is 15% of 240?\nStep 1: Convert percentage to decimal: 15% = 0.15\nStep 2: Multiply: 0.15 × 240 = 36\nTherefore, 15% of 240 is 36.',
                'source': 'math',
                'difficulty': 2
            },
            
            # General knowledge
            {
                'text': 'Question: What is photosynthesis?\nAnswer: Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.',
                'source': 'science',
                'difficulty': 2
            },
            {
                'text': 'Explain the water cycle in simple terms.\nThe water cycle describes how water moves through Earth: water evaporates from oceans and lakes, forms clouds, falls as rain or snow, and flows back to bodies of water.',
                'source': 'science',
                'difficulty': 2
            },
            
            # Creative tasks
            {
                'text': 'Write a short poem about friendship.\nFriends are like stars in the night,\nGuiding us through dark and light.\nWith laughter, tears, and memories shared,\nThey show us how much we are cared.',
                'source': 'creative',
                'difficulty': 3
            },
            
            # Conversational
            {
                'text': 'Human: How are you today?\nAssistant: I appreciate you asking! I\'m functioning well and ready to help with any questions or tasks you might have.',
                'source': 'conversation',
                'difficulty': 1
            },
            {
                'text': 'Human: Can you help me understand machine learning?\nAssistant: Certainly! Machine learning is a type of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each specific task.',
                'source': 'explanation',
                'difficulty': 3
            },
            
            # Problem solving
            {
                'text': 'Problem: You have 8 coins and one is counterfeit (lighter). How can you find it using a balance scale in just 2 weighings?\nSolution: Divide coins into groups of 3, 3, and 2. First weighing: compare the two groups of 3. If balanced, the counterfeit is in the group of 2. If unbalanced, it\'s in the lighter group.',
                'source': 'problem_solving',
                'difficulty': 4
            }
        ]
        
        return prompts
    
    def augment_data(self, data: List[Dict]) -> List[Dict]:
        """Apply data augmentation techniques"""
        augmented = []
        
        for item in data:
            # Original item
            augmented.append(item)
            
            # Variation 1: Rephrase question
            text = item['text']
            if 'Question:' in text and 'Answer:' in text:
                rephrased = text.replace('Question:', 'Q:').replace('Answer:', 'A:')
                augmented.append({
                    'text': rephrased,
                    'source': item['source'] + '_rephrased',
                    'difficulty': item['difficulty']
                })
            
            # Variation 2: Add context markers
            if 'Human:' not in text and 'Assistant:' not in text:
                contextual = f"Human: {text.split('Answer:')[0].replace('Question:', '').strip()}\nAssistant: {text.split('Answer:')[1].strip() if 'Answer:' in text else text}"
                augmented.append({
                    'text': contextual,
                    'source': item['source'] + '_contextual',
                    'difficulty': item['difficulty']
                })
        
        return augmented
    
    def create_curriculum_stages(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize data into curriculum learning stages"""
        stages = {
            'foundation': [],    # Difficulty 1-2
            'intermediate': [],  # Difficulty 3-4
            'advanced': []       # Difficulty 5+
        }
        
        for item in data:
            difficulty = item['difficulty']
            if difficulty <= 2:
                stages['foundation'].append(item)
            elif difficulty <= 4:
                stages['intermediate'].append(item)
            else:
                stages['advanced'].append(item)
        
        # Ensure minimum samples per stage
        min_samples = 50
        for stage_name, stage_data in stages.items():
            if len(stage_data) < min_samples:
                # Duplicate existing samples to meet minimum
                while len(stage_data) < min_samples:
                    stage_data.extend(stage_data[:min_samples - len(stage_data)])
        
        return stages
    
    def prepare_training_data(self) -> Dict[str, List[str]]:
        """Prepare complete training dataset"""
        print("Loading instruction datasets...")
        instruction_data = self.load_instruction_datasets()
        
        print("Creating diverse prompts...")
        diverse_prompts = self.create_diverse_prompts()
        
        print("Combining datasets...")
        all_data = instruction_data + diverse_prompts
        
        print(f"Augmenting {len(all_data)} samples...")
        augmented_data = self.augment_data(all_data)
        
        print("Creating curriculum stages...")
        curriculum = self.create_curriculum_stages(augmented_data)
        
        # Convert to text lists for training
        training_data = {}
        for stage_name, stage_data in curriculum.items():
            training_data[stage_name] = [item['text'] for item in stage_data]
            random.shuffle(training_data[stage_name])
        
        print(f"Training data prepared:")
        for stage, texts in training_data.items():
            print(f"  {stage}: {len(texts)} samples")
        
        return training_data
    
    def save_training_data(self, training_data: Dict[str, List[str]], filename: str = "training_curriculum.json"):
        """Save training data to file"""
        filepath = self.cache_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to {filepath}")
    
    def load_training_data(self, filename: str = "training_curriculum.json") -> Dict[str, List[str]]:
        """Load training data from file"""
        filepath = self.cache_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Training data file not found at {filepath}")
            return {}

class TeacherResponseCollector:
    def __init__(self, teacher_models: List[str]):
        self.teacher_models = teacher_models
        self.responses_cache = {}
        
    def collect_teacher_responses(self, prompts: List[str], max_responses: int = 100):
        """Collect responses from teacher models for given prompts"""
        print(f"Collecting responses from {len(self.teacher_models)} teacher models...")
        
        collected_data = []
        
        for prompt in prompts[:max_responses]:
            teacher_responses = {}
            
            # This would integrate with the actual teacher models
            # For now, we'll create placeholder responses
            for teacher in self.teacher_models:
                # In real implementation, this would call the actual teacher model
                teacher_responses[teacher] = {
                    'response': f"Response from {teacher} to: {prompt[:50]}...",
                    'confidence': random.uniform(0.7, 0.95)
                }
            
            collected_data.append({
                'prompt': prompt,
                'teacher_responses': teacher_responses
            })
        
        return collected_data
    
    def create_consensus_targets(self, collected_data: List[Dict]) -> List[Dict]:
        """Create consensus targets from multiple teacher responses"""
        consensus_data = []
        
        for item in collected_data:
            # Simple consensus: use response from most confident teacher
            best_teacher = max(
                item['teacher_responses'].items(),
                key=lambda x: x[1]['confidence']
            )
            
            consensus_data.append({
                'prompt': item['prompt'],
                'target_response': best_teacher[1]['response'],
                'consensus_confidence': best_teacher[1]['confidence'],
                'source_teacher': best_teacher[0]
            })
        
        return consensus_data

if __name__ == "__main__":
    # Initialize data pipeline
    pipeline = MultiModalDataPipeline()
    
    # Prepare training data
    training_data = pipeline.prepare_training_data()
    
    # Save for later use
    pipeline.save_training_data(training_data)
    
    # Example usage with teacher response collection
    teacher_collector = TeacherResponseCollector([
        'mistral-7b', 'phi-3-mini', 'gemma-2b'
    ])
    
    sample_prompts = training_data['foundation'][:10]
    teacher_responses = teacher_collector.collect_teacher_responses(sample_prompts)
    consensus_targets = teacher_collector.create_consensus_targets(teacher_responses)
    
    print(f"Created {len(consensus_targets)} consensus training targets")