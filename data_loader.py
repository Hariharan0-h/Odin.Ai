import requests
import json
import os
from pathlib import Path
import re

class DataLoader:
    """Utility class for loading and preparing training data from various sources"""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_gutenberg_books(self, book_ids=[1342, 11, 84, 74, 2701]):
        """Download books from Project Gutenberg
        Default books: Pride and Prejudice, Alice in Wonderland, Frankenstein, Tom Sawyer, Moby Dick
        """
        texts = []
        base_url = "https://www.gutenberg.org/files/{}/{}-0.txt"
        
        for book_id in book_ids:
            cache_file = self.cache_dir / f"gutenberg_{book_id}.txt"
            
            if cache_file.exists():
                print(f"Loading cached book {book_id}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                print(f"Downloading book {book_id}...")
                try:
                    response = requests.get(base_url.format(book_id, book_id), timeout=30)
                    response.raise_for_status()
                    text = response.text
                    
                    # Cache the downloaded text
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                except Exception as e:
                    print(f"Failed to download book {book_id}: {e}")
                    continue
            
            # Clean and split text
            cleaned_text = self.clean_gutenberg_text(text)
            texts.extend(self.split_into_chunks(cleaned_text, chunk_size=200))
        
        return texts
    
    def clean_gutenberg_text(self, text):
        """Clean Project Gutenberg text"""
        # Remove Gutenberg header and footer
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
            "START OF THE PROJECT GUTENBERG"
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "END OF THE PROJECT GUTENBERG"
        ]
        
        # Find content start
        start_idx = 0
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                start_idx = text.find('\n', idx) + 1
                break
        
        # Find content end
        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                end_idx = idx
                break
        
        # Extract main content
        content = text[start_idx:end_idx]
        
        # Clean up formatting
        content = re.sub(r'\n\n+', '\n\n', content)  # Multiple newlines to double
        content = re.sub(r'[ \t]+', ' ', content)     # Multiple spaces to single
        content = content.strip()
        
        return content
    
    def split_into_chunks(self, text, chunk_size=200):
        """Split text into smaller chunks for training"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def load_conversation_data(self):
        """Generate conversation-style training data"""
        conversations = [
            # Q&A patterns
            ("What is artificial intelligence?", "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn."),
            ("How does machine learning work?", "Machine learning uses algorithms to analyze data, identify patterns, and make predictions without being explicitly programmed."),
            ("What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns."),
            ("Explain neural networks", "Neural networks are computing systems inspired by biological neural networks that process information through interconnected nodes."),
            
            # Instructional content
            ("How do I learn programming?", "Start with basics, practice regularly, build projects, and learn from others. Choose a beginner-friendly language like Python."),
            ("What makes a good programmer?", "Problem-solving skills, attention to detail, continuous learning, and the ability to break down complex problems into smaller parts."),
            ("How to improve at coding?", "Practice coding daily, read others' code, participate in coding challenges, and work on personal projects."),
            
            # General knowledge
            ("Tell me about space exploration", "Space exploration involves the discovery and exploration of celestial structures in outer space using space technology."),
            ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily due to human activities."),
            ("Explain renewable energy", "Renewable energy comes from sources that naturally replenish themselves, like solar, wind, and hydroelectric power."),
        ]
        
        texts = []
        for question, answer in conversations:
            # Create different formats
            texts.append(f"Question: {question}\nAnswer: {answer}")
            texts.append(f"User: {question}\nAssistant: {answer}")
            texts.append(f"{question} {answer}")
        
        return texts
    
    def load_educational_content(self):
        """Generate educational content for training"""
        topics = {
            "science": [
                "The scientific method involves observation, hypothesis formation, experimentation, and analysis.",
                "Physics studies matter, energy, and their interactions in the universe.",
                "Chemistry explores the properties and behavior of atoms and molecules.",
                "Biology examines living organisms and their interactions with the environment."
            ],
            "technology": [
                "Computers process information using binary code consisting of ones and zeros.",
                "The internet connects billions of devices worldwide through interconnected networks.",
                "Algorithms are step-by-step procedures for solving problems or performing tasks.",
                "Software engineering involves designing, developing, and maintaining computer programs."
            ],
            "mathematics": [
                "Mathematics is the study of numbers, shapes, patterns, and logical reasoning.",
                "Algebra uses symbols and letters to represent unknown values in equations.",
                "Geometry deals with shapes, sizes, angles, and spatial relationships.",
                "Statistics involves collecting, analyzing, and interpreting numerical data."
            ]
        }
        
        texts = []
        for category, content_list in topics.items():
            texts.extend(content_list)
            # Add category introductions
            texts.append(f"Let me tell you about {category}. {' '.join(content_list[:2])}")
        
        return texts
    
    def create_comprehensive_dataset(self):
        """Create a comprehensive training dataset from all sources"""
        all_texts = []
        
        print("Loading conversation data...")
        all_texts.extend(self.load_conversation_data())
        
        print("Loading educational content...")
        all_texts.extend(self.load_educational_content())
        
        print("Attempting to download books...")
        try:
            book_texts = self.download_gutenberg_books()
            all_texts.extend(book_texts)
            print(f"Added {len(book_texts)} book chunks")
        except Exception as e:
            print(f"Could not download books: {e}")
        
        print(f"Total training samples: {len(all_texts)}")
        
        # Save dataset
        dataset_file = self.cache_dir / "comprehensive_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved to {dataset_file}")
        return all_texts

def main():
    """Example usage"""
    loader = DataLoader()
    dataset = loader.create_comprehensive_dataset()
    
    print(f"Created dataset with {len(dataset)} samples")
    print("\nSample texts:")
    for i, text in enumerate(dataset[:5]):
        print(f"{i+1}. {text[:100]}...")

if __name__ == "__main__":
    main()