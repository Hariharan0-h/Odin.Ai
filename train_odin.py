#!/usr/bin/env python3
"""
Advanced Odin Training Pipeline
Designed to create a Claude Opus-level language model using freely available models and techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import math
import wandb
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
from model import Odin
import argparse
from pathlib import Path

class OdinTrainingConfig:
    """Configuration for advanced Odin training"""
    
    # Model Architecture
    VOCAB_SIZE = 32000
    D_MODEL = 1024
    NUM_HEADS = 16  
    NUM_LAYERS = 24
    D_FF = 4096
    MAX_SEQ_LEN = 2048
    DROPOUT = 0.1
    
    # Training Hyperparameters
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 16
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 2000
    MAX_STEPS = 100000
    SAVE_EVERY = 5000
    EVAL_EVERY = 1000
    
    # Advanced Techniques
    USE_KNOWLEDGE_DISTILLATION = True
    TEACHER_MODEL = "microsoft/DialoGPT-large"
    DISTILLATION_ALPHA = 0.7
    DISTILLATION_TEMPERATURE = 4.0
    
    # Data and Paths
    OUTPUT_DIR = "odin_opus_model"
    TOKENIZER_PATH = "odin_tokenizer.json"
    CHECKPOINT_PREFIX = "odin_opus"

class AdvancedTokenizer:
    """Enhanced tokenizer with better vocabulary coverage"""
    
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<sep>': 4,
            '<mask>': 5
        }
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.word_freq = {}
        
    def build_vocabulary(self, texts):
        """Build vocabulary from training texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in tqdm(texts, desc="Counting words"):
            words = self._tokenize_text(text)
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Select most frequent words
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        current_id = len(self.special_tokens)
        for word, freq in sorted_words:
            if current_id >= self.vocab_size:
                break
            if word not in self.word_to_id and freq > 2:  # Minimum frequency threshold
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        print(f"Built vocabulary with {len(self.word_to_id)} tokens")
        
    def _tokenize_text(self, text):
        """Simple word-level tokenization with better handling"""
        # Basic preprocessing
        text = text.lower().strip()
        
        # Handle punctuation
        for punct in '.,!?;:()[]{}"-':
            text = text.replace(punct, f' {punct} ')
        
        # Split and clean
        words = [word.strip() for word in text.split() if word.strip()]
        return words
    
    def encode(self, text, max_length=None, add_special_tokens=True):
        """Encode text to token IDs"""
        words = self._tokenize_text(text)
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.word_to_id['<bos>'])
        
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id['<unk>'])
            tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.word_to_id['<eos>'])
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.word_to_id['<eos>']]
        elif max_length:
            tokens.extend([self.word_to_id['<pad>']] * (max_length - len(tokens)))
            
        return tokens
    
    def decode(self, tokens, skip_special_tokens=True):
        """Decode token IDs back to text"""
        words = []
        for token in tokens:
            if token == self.word_to_id['<pad>']:
                break
            word = self.id_to_word.get(token, '<unk>')
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)
        return ' '.join(words)
    
    def save(self, path):
        """Save tokenizer to JSON file"""
        data = {
            'vocab_size': self.vocab_size,
            'word_to_id': self.word_to_id,
            'word_freq': self.word_freq
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.word_to_id = data['word_to_id']
        tokenizer.id_to_word = {int(k): v for v, k in tokenizer.word_to_id.items()}
        tokenizer.word_freq = data.get('word_freq', {})
        
        return tokenizer

class OpusQualityDataset(Dataset):
    """Dataset designed to train Opus-level conversational AI"""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print("Loading high-quality datasets...")
        self._load_datasets()
        
    def _load_datasets(self):
        """Load and prepare high-quality datasets"""
        datasets_to_load = [
            # Instruction-following datasets
            ("Anthropic/hh-rlhf", "helpful-base"),
            # Conversation datasets
            ("microsoft/DialoGPT-medium", None),
            # Knowledge datasets
            ("squad", None),
            ("natural_questions", None),
            # Creative writing
            ("roneneldan/TinyStories", None)
        ]
        
        all_texts = []
        
        for dataset_name, config in datasets_to_load:
            try:
                print(f"Loading {dataset_name}...")
                if config:
                    dataset = load_dataset(dataset_name, config, split="train[:10000]")
                else:
                    dataset = load_dataset(dataset_name, split="train[:10000]")
                
                # Extract text based on dataset structure
                if "text" in dataset.features:
                    texts = [item["text"] for item in dataset]
                elif "chosen" in dataset.features and "rejected" in dataset.features:
                    texts = [item["chosen"] for item in dataset]
                elif "question" in dataset.features and "answer" in dataset.features:
                    texts = [f"Q: {item['question']} A: {item['answer']}" for item in dataset]
                else:
                    # Try to find any text field
                    text_fields = [k for k in dataset.features.keys() if 'text' in k.lower()]
                    if text_fields:
                        texts = [item[text_fields[0]] for item in dataset]
                    else:
                        continue
                
                all_texts.extend(texts)
                print(f"Loaded {len(texts)} samples from {dataset_name}")
                
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                continue
        
        # Build tokenizer vocabulary if not exists
        if len(self.tokenizer.word_to_id) <= len(self.tokenizer.special_tokens):
            self.tokenizer.build_vocabulary(all_texts)
        
        # Tokenize all texts
        print("Tokenizing datasets...")
        for text in tqdm(all_texts):
            if isinstance(text, str) and len(text.strip()) > 10:
                tokens = self.tokenizer.encode(text, max_length=self.max_length)
                if len(tokens) > 20:  # Filter very short sequences
                    self.data.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"Prepared {len(self.data)} training samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Input is all tokens except last, target is all tokens except first
        return tokens[:-1], tokens[1:]

class KnowledgeDistillationTrainer:
    """Advanced trainer with knowledge distillation and Opus-level techniques"""
    
    def __init__(self, config: OdinTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        tokenizer_path = Path(config.TOKENIZER_PATH)
        if tokenizer_path.exists():
            self.tokenizer = AdvancedTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = AdvancedTokenizer(config.VOCAB_SIZE)
        
        # Initialize models
        self.model = self._create_model()
        if config.USE_KNOWLEDGE_DISTILLATION:
            self.teacher_model = self._load_teacher_model()
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize wandb for tracking
        if not os.getenv('WANDB_DISABLED'):
            wandb.init(
                project="odin-opus-training",
                config=vars(config),
                tags=["knowledge-distillation", "opus-level"]
            )
    
    def _create_model(self):
        """Create the Odin model with enhanced architecture"""
        model = Odin(
            vocab_size=self.config.VOCAB_SIZE,
            d_model=self.config.D_MODEL,
            num_heads=self.config.NUM_HEADS,
            num_layers=self.config.NUM_LAYERS,
            d_ff=self.config.D_FF,
            max_seq_len=self.config.MAX_SEQ_LEN
        ).to(self.device)
        
        # Initialize weights properly
        self._init_weights(model)
        
        print(f"Created Odin model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _init_weights(self, model):
        """Initialize model weights using best practices"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def _load_teacher_model(self):
        """Load teacher model for knowledge distillation"""
        try:
            print(f"Loading teacher model: {self.config.TEACHER_MODEL}")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.TEACHER_MODEL,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
            return None
    
    def _create_optimizer(self):
        """Create optimizer with weight decay"""
        no_decay = ["bias", "LayerNorm.weight", "norm1.weight", "norm2.weight", "ln_f.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optim.AdamW(optimizer_grouped_parameters, lr=self.config.LEARNING_RATE)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.WARMUP_STEPS:
                return step / self.config.WARMUP_STEPS
            return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - self.config.WARMUP_STEPS) / (self.config.MAX_STEPS - self.config.WARMUP_STEPS))))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute knowledge distillation loss"""
        # Standard language modeling loss
        lm_loss = nn.CrossEntropyLoss()(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        if teacher_logits is None:
            return lm_loss
        
        # Distillation loss
        student_log_probs = nn.LogSoftmax(dim=-1)(student_logits / self.config.DISTILLATION_TEMPERATURE)
        teacher_probs = nn.Softmax(dim=-1)(teacher_logits / self.config.DISTILLATION_TEMPERATURE)
        
        distill_loss = nn.KLDivLoss(reduction='batchmean')(
            student_log_probs,
            teacher_probs
        ) * (self.config.DISTILLATION_TEMPERATURE ** 2)
        
        # Combine losses
        total_loss = (
            self.config.DISTILLATION_ALPHA * distill_loss +
            (1 - self.config.DISTILLATION_ALPHA) * lm_loss
        )
        
        return total_loss
    
    def train(self):
        """Main training loop"""
        print("Starting Opus-level training...")
        
        # Create dataset and dataloader
        dataset = OpusQualityDataset(self.tokenizer, self.config.MAX_SEQ_LEN)
        
        # Save tokenizer
        self.tokenizer.save(self.config.TOKENIZER_PATH)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        self.model.train()
        step = 0
        total_loss = 0
        
        for epoch in range(100):  # Large number, we'll break based on steps
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if step >= self.config.MAX_STEPS:
                    break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                student_logits = self.model(inputs)
                
                # Get teacher logits if using distillation
                teacher_logits = None
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(inputs)
                        teacher_logits = teacher_outputs.logits
                
                # Compute loss
                loss = self.compute_distillation_loss(student_logits, teacher_logits, targets)
                loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass
                loss.backward()
                total_loss += loss.item()
                
                # Update weights
                if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    avg_loss = total_loss / self.config.GRADIENT_ACCUMULATION_STEPS
                    total_loss = 0
                    
                    # Log metrics
                    if step % 100 == 0:
                        print(f"Step {step}, Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
                        
                        if not os.getenv('WANDB_DISABLED'):
                            wandb.log({
                                "loss": avg_loss,
                                "learning_rate": self.scheduler.get_last_lr()[0],
                                "step": step
                            })
                
                # Save checkpoint
                if step % self.config.SAVE_EVERY == 0 and step > 0:
                    self._save_checkpoint(step)
                
                # Evaluate
                if step % self.config.EVAL_EVERY == 0 and step > 0:
                    self._evaluate(step)
                
                step += 1
            
            if step >= self.config.MAX_STEPS:
                break
        
        # Save final model
        self._save_final_model()
        print("Training completed!")
    
    def _save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = f"{self.config.CHECKPOINT_PREFIX}_step_{step}.pth"
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final trained model"""
        final_path = f"{self.config.CHECKPOINT_PREFIX}_final.pth"
        torch.save(self.model.state_dict(), final_path)
        print(f"Saved final model: {final_path}")
    
    def _evaluate(self, step):
        """Run evaluation"""
        self.model.eval()
        test_prompts = [
            "What is the meaning of life?",
            "Explain quantum physics in simple terms.",
            "Write a creative story about space exploration.",
            "How can AI benefit humanity?",
        ]
        
        print(f"\n--- Evaluation at step {step} ---")
        for prompt in test_prompts:
            response = self._generate_response(prompt, max_length=100)
            print(f"Prompt: {prompt}")
            print(f"Response: {response}\n")
        
        self.model.train()
    
    def _generate_response(self, prompt, max_length=100):
        """Generate response for evaluation"""
        with torch.no_grad():
            input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            
            for _ in range(max_length):
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :] / 0.8  # Temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.word_to_id.get('<eos>', 0):
                    break
            
            generated_text = self.tokenizer.decode(input_ids[0].tolist())
            # Extract response (remove prompt)
            prompt_words = len(self.tokenizer._tokenize_text(prompt))
            response_words = generated_text.split()[prompt_words:]
            return ' '.join(response_words)

def main():
    parser = argparse.ArgumentParser(description="Train Odin to Opus level")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    config = OdinTrainingConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()