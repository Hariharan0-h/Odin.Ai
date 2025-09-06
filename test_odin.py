#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Odin Model
Tests model quality, performance, and Opus-level capabilities
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from model import Odin
from train_odin import AdvancedTokenizer
import argparse
import requests
from typing import List, Dict, Any

class OdinTester:
    """Comprehensive testing framework for Odin model"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AdvancedTokenizer.load(tokenizer_path)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = Odin(
            vocab_size=self.tokenizer.vocab_size,
            d_model=1024,
            num_heads=16,
            num_layers=24,
            d_ff=4096,
            max_seq_len=2048
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def generate_response(self, prompt: str, max_length: int = 150, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response with advanced sampling"""
        with torch.no_grad():
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)]).to(self.device)
            original_length = input_ids.size(1)
            
            # Generate tokens
            for _ in range(max_length):
                # Forward pass
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append token
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for end token
                if next_token.item() in [self.tokenizer.word_to_id.get('<eos>', 3), self.tokenizer.word_to_id.get('<pad>', 0)]:
                    break
                    
                # Prevent infinite loops
                if input_ids.size(1) > 2000:
                    break
            
            # Decode response
            generated_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            
            # Extract only the generated part
            prompt_text = self.tokenizer.decode(input_ids[0][:original_length].tolist(), skip_special_tokens=True)
            if generated_text.startswith(prompt_text):
                response = generated_text[len(prompt_text):].strip()
            else:
                response = generated_text.strip()
            
            return response if response else "I'm still learning to respond better."
    
    def run_quality_tests(self) -> Dict[str, Any]:
        """Run comprehensive quality tests"""
        print("\n=== QUALITY TESTS ===")
        
        test_categories = {
            "reasoning": [
                "If John has 5 apples and gives 2 to Mary, how many does he have left?",
                "What comes next in this sequence: 2, 4, 8, 16, ?",
                "If all cats are mammals and some mammals are pets, can we conclude that some cats are pets?",
            ],
            "knowledge": [
                "Who was the first person to walk on the moon?",
                "What is the capital of France?",
                "Explain photosynthesis in simple terms.",
            ],
            "creativity": [
                "Write a short poem about the ocean.",
                "Create a story about a robot who learns to paint.",
                "Describe a new invention that could help the environment.",
            ],
            "conversation": [
                "Hello, how are you doing today?",
                "What's your favorite type of music and why?",
                "Can you help me plan a birthday party?",
            ],
            "analysis": [
                "What are the pros and cons of renewable energy?",
                "Compare democracy and monarchy as forms of government.",
                "Analyze the impact of social media on society.",
            ]
        }
        
        results = {}
        
        for category, prompts in test_categories.items():
            print(f"\n--- {category.upper()} TESTS ---")
            category_results = []
            
            for prompt in prompts:
                start_time = time.time()
                response = self.generate_response(prompt, max_length=200)
                response_time = time.time() - start_time
                
                result = {
                    "prompt": prompt,
                    "response": response,
                    "response_time": response_time,
                    "quality_score": self._assess_response_quality(prompt, response)
                }
                
                category_results.append(result)
                
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print(f"Time: {response_time:.2f}s, Quality: {result['quality_score']}/10")
                print("-" * 50)
            
            results[category] = category_results
        
        return results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("\n=== PERFORMANCE TESTS ===")
        
        # Test different sequence lengths
        test_lengths = [50, 100, 200, 500]
        performance_results = {}
        
        for length in test_lengths:
            print(f"\nTesting generation length: {length} tokens")
            
            prompt = "The future of artificial intelligence"
            times = []
            
            # Run multiple iterations for accurate timing
            for i in range(5):
                start_time = time.time()
                response = self.generate_response(prompt, max_length=length)
                end_time = time.time()
                
                response_time = end_time - start_time
                times.append(response_time)
                
                print(f"Iteration {i+1}: {response_time:.2f}s")
            
            avg_time = np.mean(times)
            tokens_per_second = length / avg_time
            
            performance_results[f"length_{length}"] = {
                "avg_time": avg_time,
                "std_time": np.std(times),
                "tokens_per_second": tokens_per_second,
                "sample_response": response
            }
            
            print(f"Average time: {avg_time:.2f}s")
            print(f"Tokens/second: {tokens_per_second:.1f}")
        
        return performance_results
    
    def run_opus_comparison_tests(self) -> Dict[str, Any]:
        """Run tests comparing against Claude Opus-level expectations"""
        print("\n=== OPUS-LEVEL COMPARISON TESTS ===")
        
        opus_level_tasks = [
            {
                "task": "complex_reasoning",
                "prompt": "A farmer has chickens and cows. Together they have 30 heads and 74 legs. How many chickens and how many cows are there?",
                "expected_approach": "algebraic solution"
            },
            {
                "task": "creative_writing",
                "prompt": "Write a compelling opening paragraph for a science fiction novel set on Mars.",
                "expected_approach": "engaging narrative"
            },
            {
                "task": "code_explanation",
                "prompt": "Explain what this Python code does: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "expected_approach": "technical accuracy"
            },
            {
                "task": "ethical_reasoning",
                "prompt": "Should AI systems be allowed to make decisions about human healthcare? Discuss both sides.",
                "expected_approach": "balanced analysis"
            },
            {
                "task": "language_understanding",
                "prompt": "Explain the difference between 'affect' and 'effect' with examples.",
                "expected_approach": "clear explanation"
            }
        ]
        
        comparison_results = {}
        
        for task_info in opus_level_tasks:
            task = task_info["task"]
            prompt = task_info["prompt"]
            
            print(f"\n--- {task.upper()} ---")
            print(f"Prompt: {prompt}")
            
            # Generate response
            start_time = time.time()
            response = self.generate_response(prompt, max_length=300)
            response_time = time.time() - start_time
            
            # Assess opus-level qualities
            opus_score = self._assess_opus_level_quality(task, response)
            
            comparison_results[task] = {
                "prompt": prompt,
                "response": response,
                "response_time": response_time,
                "opus_score": opus_score,
                "expected_approach": task_info["expected_approach"]
            }
            
            print(f"Response: {response}")
            print(f"Opus-level score: {opus_score}/10")
            print(f"Time: {response_time:.2f}s")
        
        return comparison_results
    
    def _assess_response_quality(self, prompt: str, response: str) -> int:
        """Assess response quality on a scale of 1-10"""
        score = 5  # Base score
        
        # Length check
        if len(response.split()) < 5:
            score -= 2
        elif len(response.split()) > 10:
            score += 1
        
        # Coherence check
        if response.strip() and not response.startswith("I'm still learning"):
            score += 1
        
        # Relevance check (simple keyword matching)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        
        if overlap > 0:
            score += 1
        if overlap > 2:
            score += 1
        
        # Grammar check (very basic)
        if response.endswith(('.', '!', '?')):
            score += 1
        
        return min(10, max(1, score))
    
    def _assess_opus_level_quality(self, task: str, response: str) -> int:
        """Assess if response meets Opus-level quality standards"""
        base_score = self._assess_response_quality("", response)
        
        # Task-specific assessments
        task_bonus = 0
        
        if task == "complex_reasoning":
            if any(word in response.lower() for word in ["equation", "solve", "algebra", "variables"]):
                task_bonus += 2
            if any(word in response.lower() for word in ["chicken", "cow", "legs", "heads"]):
                task_bonus += 1
        
        elif task == "creative_writing":
            if len(response.split()) > 30:
                task_bonus += 1
            if any(word in response.lower() for word in ["mars", "red", "space", "colony"]):
                task_bonus += 1
            if "." in response and "," in response:  # Basic structure
                task_bonus += 1
        
        elif task == "code_explanation":
            if any(word in response.lower() for word in ["fibonacci", "recursive", "function"]):
                task_bonus += 2
            if any(word in response.lower() for word in ["return", "calls", "itself"]):
                task_bonus += 1
        
        return min(10, base_score + task_bonus)
    
    def generate_test_report(self, results: Dict[str, Any], output_path: str = "odin_test_report.json"):
        """Generate comprehensive test report"""
        print(f"\n=== GENERATING TEST REPORT ===")
        
        # Calculate overall scores
        quality_scores = []
        opus_scores = []
        
        for category, tests in results.get("quality_tests", {}).items():
            for test in tests:
                quality_scores.append(test["quality_score"])
        
        for task, test in results.get("opus_comparison", {}).items():
            opus_scores.append(test["opus_score"])
        
        # Create summary
        summary = {
            "overall_quality_score": np.mean(quality_scores) if quality_scores else 0,
            "opus_level_score": np.mean(opus_scores) if opus_scores else 0,
            "avg_response_time": np.mean([test["response_time"] for category in results.get("quality_tests", {}).values() for test in category]),
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "vocab_size": self.tokenizer.vocab_size,
                "device": str(self.device)
            }
        }
        
        # Add summary to results
        results["summary"] = summary
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Test report saved to: {output_path}")
        print(f"Overall Quality Score: {summary['overall_quality_score']:.2f}/10")
        print(f"Opus-level Score: {summary['opus_level_score']:.2f}/10")
        print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        
        return summary
    
    def run_full_test_suite(self):
        """Run all tests and generate comprehensive report"""
        print("Starting Odin Comprehensive Test Suite...")
        
        all_results = {}
        
        # Run quality tests
        all_results["quality_tests"] = self.run_quality_tests()
        
        # Run performance tests
        all_results["performance_tests"] = self.run_performance_tests()
        
        # Run Opus comparison tests
        all_results["opus_comparison"] = self.run_opus_comparison_tests()
        
        # Generate report
        summary = self.generate_test_report(all_results)
        
        return all_results, summary

def main():
    parser = argparse.ArgumentParser(description="Test Odin model")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer file")
    parser.add_argument("--output", default="odin_test_report.json", help="Output report path")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.tokenizer).exists():
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        return
    
    # Initialize tester
    tester = OdinTester(args.model, args.tokenizer)
    
    if args.quick:
        # Quick test with just a few prompts
        results = tester.run_quality_tests()
        summary = tester.generate_test_report({"quality_tests": results}, args.output)
    else:
        # Full test suite
        results, summary = tester.run_full_test_suite()
    
    print("\n=== TEST COMPLETE ===")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()