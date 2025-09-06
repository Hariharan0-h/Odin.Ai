#!/usr/bin/env python3
"""
Comprehensive evaluation suite for Odin model
Benchmarks performance against Claude Opus and other models
"""

import torch
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

from model import Odin
from server import BPETokenizer, OdinServer

@dataclass
class EvaluationMetrics:
    perplexity: float = 0.0
    coherbleu_score: float = 0.0
    rouge_l: float = 0.0
    coherence_score: float = 0.0
    factual_accuracy: float = 0.0
    response_quality: float = 0.0
    response_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'perplexity': self.perplexity,
            'bleu_score': self.bleu_score,
            'rouge_l': self.rouge_l,
            'coherence_score': self.coherence_score,
            'factual_accuracy': self.factual_accuracy,
            'response_quality': self.response_quality,
            'response_time': self.response_time
        }

class ModelEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def generate_response(self, prompt: str, max_length: int = 100) -> Tuple[str, float]:
        """Generate response and measure time"""
        raise NotImplementedError
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate model perplexity on given text"""
        raise NotImplementedError

class OdinEvaluator(ModelEvaluator):
    def __init__(self):
        super().__init__("Odin")
        self.odin_server = OdinServer()
        
    def generate_response(self, prompt: str, max_length: int = 100) -> Tuple[str, float]:
        start_time = time.time()
        response = self.odin_server.generate_response(prompt, max_length)
        end_time = time.time()
        return response, end_time - start_time
    
    def calculate_perplexity(self, text: str) -> float:
        if not self.odin_server.model or not self.odin_server.tokenizer:
            return float('inf')
            
        try:
            tokens = self.odin_server.tokenizer.encode(text)
            input_ids = torch.tensor([tokens]).to(self.odin_server.device)
            
            with torch.no_grad():
                outputs = self.odin_server.model(input_ids)
                log_likelihood = torch.nn.functional.log_softmax(outputs, dim=-1)
                
                # Calculate perplexity
                perplexity = torch.exp(-log_likelihood.mean())
                return perplexity.item()
                
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')

class HuggingFaceEvaluator(ModelEvaluator):
    def __init__(self, model_id: str, model_name: str = None):
        super().__init__(model_name or model_id.split('/')[-1])
        self.model_id = model_id
        self.load_model()
        
    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Failed to load {self.model_id}: {e}")
            
    def generate_response(self, prompt: str, max_length: int = 100) -> Tuple[str, float]:
        if not self.model or not self.tokenizer:
            return "Model not available", 0.0
            
        try:
            start_time = time.time()
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            end_time = time.time()
            
            return response.strip(), end_time - start_time
            
        except Exception as e:
            return f"Error: {e}", 0.0

class ClaudeOpusEvaluator(ModelEvaluator):
    def __init__(self, api_key: str = None):
        super().__init__("Claude-Opus")
        self.api_key = api_key
        
    def generate_response(self, prompt: str, max_length: int = 100) -> Tuple[str, float]:
        # Placeholder for Claude Opus API integration
        # In real implementation, this would call Anthropic's API
        return "Claude Opus response placeholder", 0.1

class ComprehensiveEvaluationSuite:
    def __init__(self):
        self.evaluators = {}
        self.benchmark_questions = self.load_benchmark_questions()
        
    def add_evaluator(self, evaluator: ModelEvaluator):
        self.evaluators[evaluator.model_name] = evaluator
        
    def load_benchmark_questions(self) -> List[Dict]:
        """Load comprehensive benchmark questions"""
        questions = [
            # General Knowledge
            {
                'prompt': 'What is the capital of France?',
                'category': 'factual',
                'expected_keywords': ['Paris'],
                'difficulty': 1
            },
            {
                'prompt': 'Explain the theory of relativity in simple terms.',
                'category': 'explanation',
                'expected_keywords': ['Einstein', 'space', 'time', 'speed of light'],
                'difficulty': 3
            },
            
            # Reasoning
            {
                'prompt': 'If all roses are flowers and some flowers are red, can we conclude that some roses are red?',
                'category': 'logic',
                'expected_keywords': ['No', 'cannot conclude', 'insufficient'],
                'difficulty': 4
            },
            
            # Math
            {
                'prompt': 'What is 15% of 240?',
                'category': 'math',
                'expected_keywords': ['36'],
                'difficulty': 2
            },
            {
                'prompt': 'Solve for x: 2x + 5 = 15',
                'category': 'math',
                'expected_keywords': ['5', 'x = 5'],
                'difficulty': 3
            },
            
            # Creative
            {
                'prompt': 'Write a short poem about artificial intelligence.',
                'category': 'creative',
                'expected_keywords': ['AI', 'machine', 'intelligence', 'future'],
                'difficulty': 3
            },
            
            # Conversational
            {
                'prompt': 'How can I improve my study habits?',
                'category': 'advice',
                'expected_keywords': ['schedule', 'organize', 'break', 'focus'],
                'difficulty': 2
            },
            
            # Complex reasoning
            {
                'prompt': 'A farmer has 100 animals: cows and chickens. If there are 260 legs total, how many cows are there?',
                'category': 'word_problem',
                'expected_keywords': ['30', 'thirty'],
                'difficulty': 4
            },
            
            # Science
            {
                'prompt': 'What happens during photosynthesis?',
                'category': 'science',
                'expected_keywords': ['chlorophyll', 'sunlight', 'CO2', 'oxygen', 'glucose'],
                'difficulty': 3
            },
            
            # Ethics/Philosophy
            {
                'prompt': 'Is it ethical to use AI for making important decisions about people?',
                'category': 'ethics',
                'expected_keywords': ['bias', 'fairness', 'transparency', 'human oversight'],
                'difficulty': 5
            }
        ]
        
        return questions
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Simple BLEU score approximation"""
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        
        if not cand_words:
            return 0.0
            
        intersection = ref_words.intersection(cand_words)
        return len(intersection) / len(cand_words)
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Simple ROUGE-L approximation"""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        # Longest common subsequence
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(ref_words, cand_words)
        
        if len(ref_words) == 0 and len(cand_words) == 0:
            return 1.0
        elif len(ref_words) == 0 or len(cand_words) == 0:
            return 0.0
        else:
            precision = lcs_len / len(cand_words) if len(cand_words) > 0 else 0
            recall = lcs_len / len(ref_words) if len(ref_words) > 0 else 0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
    
    def calculate_factual_accuracy(self, response: str, expected_keywords: List[str]) -> float:
        """Check if response contains expected factual information"""
        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        return matches / len(expected_keywords) if expected_keywords else 0.0
    
    def calculate_coherence_score(self, response: str) -> float:
        """Simple coherence score based on sentence structure and length"""
        if not response or len(response.strip()) == 0:
            return 0.0
            
        sentences = response.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if not sentence_lengths:
            return 0.0
        
        # Penalize very short or very long responses
        avg_length = np.mean(sentence_lengths)
        if avg_length < 3:
            return 0.3
        elif avg_length > 50:
            return 0.6
        else:
            return min(1.0, avg_length / 20)
    
    def evaluate_response(self, prompt: str, response: str, expected_keywords: List[str], 
                         response_time: float) -> EvaluationMetrics:
        """Evaluate a single response comprehensively"""
        metrics = EvaluationMetrics()
        
        # Response time
        metrics.response_time = response_time
        
        # Factual accuracy
        metrics.factual_accuracy = self.calculate_factual_accuracy(response, expected_keywords)
        
        # Coherence
        metrics.coherence_score = self.calculate_coherence_score(response)
        
        # Response quality (composite score)
        if response and response.strip():
            quality_factors = [
                len(response) > 10,  # Not too short
                len(response) < 500,  # Not too long
                '.' in response,  # Has sentences
                not response.startswith('Error'),  # No errors
                any(keyword.lower() in response.lower() for keyword in expected_keywords) if expected_keywords else True
            ]
            metrics.response_quality = sum(quality_factors) / len(quality_factors)
        
else:
            metrics.response_quality = 0.0
        
        return metrics
    
    def run_comprehensive_evaluation(self) -> Dict[str, Dict]:
        """Run evaluation on all models with all benchmark questions"""
        results = {}
        
        print("🔍 Starting Comprehensive Model Evaluation")
        print("=" * 50)
        
        for model_name, evaluator in self.evaluators.items():
            print(f"\nEvaluating {model_name}...")
            model_results = []
            
            for i, question in enumerate(self.benchmark_questions):
                print(f"  Question {i+1}/{len(self.benchmark_questions)}: {question['category']}")
                
                try:
                    response, response_time = evaluator.generate_response(question['prompt'])
                    
                    metrics = self.evaluate_response(
                        question['prompt'],
                        response,
                        question.get('expected_keywords', []),
                        response_time
                    )
                    
                    question_result = {
                        'question': question,
                        'response': response,
                        'metrics': metrics.to_dict()
                    }
                    model_results.append(question_result)
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
            
            results[model_name] = model_results
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Dict]) -> Dict:
        """Generate comparative analysis report"""
        print("\n📊 Generating Comparison Report...")
        
        summary = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
                
            # Aggregate metrics
            all_metrics = [result['metrics'] for result in model_results]
            
            summary[model_name] = {
                'avg_factual_accuracy': np.mean([m['factual_accuracy'] for m in all_metrics]),
                'avg_coherence': np.mean([m['coherence_score'] for m in all_metrics]),
                'avg_quality': np.mean([m['response_quality'] for m in all_metrics]),
                'avg_response_time': np.mean([m['response_time'] for m in all_metrics]),
                'total_questions': len(model_results)
            }
            
            # Category breakdown
            categories = {}
            for result in model_results:
                cat = result['question']['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result['metrics'])
            
            summary[model_name]['category_performance'] = {}
            for cat, cat_metrics in categories.items():
                summary[model_name]['category_performance'][cat] = {
                    'accuracy': np.mean([m['factual_accuracy'] for m in cat_metrics]),
                    'quality': np.mean([m['response_quality'] for m in cat_metrics])
                }
        
        return summary
    
    def save_results(self, results: Dict, summary: Dict, filename_prefix: str = "evaluation"):
        """Save evaluation results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = f"{filename_prefix}_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = f"{filename_prefix}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {detailed_file} and {summary_file}")
        return detailed_file, summary_file
    
    def create_visualization(self, summary: Dict, save_path: str = None):
        """Create visualization of model comparison"""
        models = list(summary.keys())
        metrics = ['avg_factual_accuracy', 'avg_coherence', 'avg_quality']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Overall metrics comparison
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [summary[model][metric] for model in models]
            ax1.bar(x + i * width, values, width, label=metric.replace('avg_', ''))
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Response time comparison
        ax2 = axes[0, 1]
        response_times = [summary[model]['avg_response_time'] for model in models]
        ax2.bar(models, response_times)
        ax2.set_ylabel('Response Time (s)')
        ax2.set_title('Response Time Comparison')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Category performance heatmap
        ax3 = axes[1, 0]
        categories = set()
        for model_data in summary.values():
            categories.update(model_data.get('category_performance', {}).keys())
        categories = sorted(list(categories))
        
        if categories:
            heatmap_data = []
            for model in models:
                model_row = []
                for cat in categories:
                    cat_perf = summary[model].get('category_performance', {}).get(cat, {})
                    model_row.append(cat_perf.get('quality', 0))
                heatmap_data.append(model_row)
            
            im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax3.set_xticks(range(len(categories)))
            ax3.set_xticklabels(categories, rotation=45)
            ax3.set_yticks(range(len(models)))
            ax3.set_yticklabels(models)
            ax3.set_title('Performance by Category')
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(categories)):
                    text = ax3.text(j, i, f'{heatmap_data[i][j]:.2f}',
                                  ha="center", va="center", color="black")
        
        # Overall ranking
        ax4 = axes[1, 1]
        overall_scores = []
        for model in models:
            score = (summary[model]['avg_factual_accuracy'] + 
                    summary[model]['avg_coherence'] + 
                    summary[model]['avg_quality']) / 3
            overall_scores.append(score)
        
        sorted_indices = sorted(range(len(models)), key=lambda i: overall_scores[i], reverse=True)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [overall_scores[i] for i in sorted_indices]
        
        ax4.barh(range(len(sorted_models)), sorted_scores)
        ax4.set_yticks(range(len(sorted_models)))
        ax4.set_yticklabels(sorted_models)
        ax4.set_xlabel('Overall Score')
        ax4.set_title('Model Ranking')
        ax4.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()

def main():
    # Initialize evaluation suite
    suite = ComprehensiveEvaluationSuite()
    
    # Add Odin evaluator
    odin_evaluator = OdinEvaluator()
    suite.add_evaluator(odin_evaluator)
    
    # Add comparison models (if available)
    comparison_models = [
        ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3-Mini'),
        ('google/gemma-2b-it', 'Gemma-2B'),
    ]
    
    for model_id, model_name in comparison_models:
        try:
            evaluator = HuggingFaceEvaluator(model_id, model_name)
            suite.add_evaluator(evaluator)
            print(f"Added {model_name} for comparison")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
    
    # Run evaluation
    results = suite.run_comprehensive_evaluation()
    
    # Generate report
    summary = suite.generate_comparison_report(results)
    
    # Print summary
    print("\n🏆 EVALUATION SUMMARY")
    print("=" * 50)
    for model_name, metrics in summary.items():
        print(f"\n{model_name}:")
        print(f"  Factual Accuracy: {metrics['avg_factual_accuracy']:.3f}")
        print(f"  Coherence Score:  {metrics['avg_coherence']:.3f}")
        print(f"  Response Quality: {metrics['avg_quality']:.3f}")
        print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
    
    # Save results
    detailed_file, summary_file = suite.save_results(results, summary)
    
    # Create visualization
    suite.create_visualization(summary, f"evaluation_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png")
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Detailed results: {detailed_file}")
    print(f"📊 Summary: {summary_file}")

if __name__ == "__main__":
    main()