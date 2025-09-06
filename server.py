from flask import Flask, request, jsonify, render_template_string, send_from_directory
import torch
import json
import os
import time
import logging
from pathlib import Path
from model import Odin
from train_odin import AdvancedTokenizer

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdinServer:
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        # Auto-detect best available model
        self.model_path = model_path or self._find_best_model()
        self.tokenizer_path = tokenizer_path or self._find_tokenizer()
        
        self.load_model()
    
    def _find_best_model(self):
        """Find the best available model file"""
        model_candidates = [
            'odin_opus_final.pth',
            'odin_gpu_final.pth', 
            'odin_best_model.pth',
            'odin_advanced_model.pth'
        ]
        
        for model_file in model_candidates:
            if Path(model_file).exists():
                logger.info(f"Found model: {model_file}")
                return model_file
        
        raise FileNotFoundError("No trained model found. Please run training first.")
    
    def _find_tokenizer(self):
        """Find the best available tokenizer"""
        tokenizer_candidates = [
            'odin_tokenizer.json',
            'advanced_tokenizer.json'
        ]
        
        for tokenizer_file in tokenizer_candidates:
            if Path(tokenizer_file).exists():
                logger.info(f"Found tokenizer: {tokenizer_file}")
                return tokenizer_file
        
        raise FileNotFoundError("No tokenizer found. Please run training first.")
    
    def load_model(self):
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AdvancedTokenizer.load(self.tokenizer_path)
            
            # Determine model configuration based on tokenizer vocab size
            if self.tokenizer.vocab_size > 20000:
                # Opus-level configuration
                self.model_config = {
                    'vocab_size': self.tokenizer.vocab_size,
                    'd_model': 1024,
                    'num_heads': 16,
                    'num_layers': 24,
                    'd_ff': 4096,
                    'max_seq_len': 2048
                }
                logger.info("Using Opus-level model configuration")
            else:
                # Standard configuration
                self.model_config = {
                    'vocab_size': self.tokenizer.vocab_size,
                    'd_model': 512,
                    'num_heads': 8,
                    'num_layers': 6,
                    'd_ff': 2048,
                    'max_seq_len': 512
                }
                logger.info("Using standard model configuration")
            
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = Odin(**self.model_config).to(self.device)
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Enable optimizations
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled with torch.compile for better performance")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
            
            model_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Odin model loaded successfully!")
            logger.info(f"Parameters: {model_params:,}")
            logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("Make sure you've trained the model first by running: python train_odin.py")
            raise
    
    def generate_response(self, prompt, max_length=150, temperature=0.8, top_p=0.9):
        """Generate response with advanced sampling techniques"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please train the model first."
        
        try:
            start_time = time.time()
            
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)]).to(self.device)
            original_length = input_ids.size(1)
            
            with torch.no_grad():
                for step in range(max_length):
                    # Manage sequence length for memory efficiency
                    if input_ids.size(1) > self.model_config['max_seq_len'] - 50:
                        # Keep the last tokens to maintain context
                        keep_length = self.model_config['max_seq_len'] // 2
                        input_ids = input_ids[:, -keep_length:]
                    
                    # Forward pass
                    logits = self.model(input_ids)
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Apply top-p (nucleus) sampling
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
                    
                    # Check for end tokens
                    if next_token.item() in [
                        self.tokenizer.word_to_id.get('<eos>', 3),
                        self.tokenizer.word_to_id.get('<pad>', 0)
                    ]:
                        break
                    
                    # Early stopping for repetitive content
                    if step > 10 and self._is_repetitive(input_ids[0].tolist()):
                        break
            
            # Decode response
            generated_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            
            # Extract only the generated part
            prompt_text = self.tokenizer.decode(input_ids[0][:original_length].tolist(), skip_special_tokens=True)
            if generated_text.startswith(prompt_text):
                response = generated_text[len(prompt_text):].strip()
            else:
                response = generated_text.strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s")
            
            return response if response else "I'm still learning to provide better responses."
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    def _is_repetitive(self, tokens):
        """Check if the sequence is becoming repetitive"""
        if len(tokens) < 20:
            return False
        
        # Check for immediate repetition
        last_5 = tokens[-5:]
        prev_5 = tokens[-10:-5]
        
        return last_5 == prev_5
    
    def _clean_response(self, response):
        """Clean and improve response quality"""
        if not response:
            return ""
        
        # Remove excessive whitespace
        response = ' '.join(response.split())
        
        # Remove partial sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure reasonable length
        if len(response) > 500:
            # Cut at last complete sentence
            sentences = response[:500].split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
            else:
                response = response[:500] + '...'
        
        return response

# Initialize Odin server (will be done in main)
odin = None

@app.route('/')
def home():
    return send_from_directory('web-chat', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('web-chat', filename)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if odin is None:
            return jsonify({
                'error': 'Model not initialized',
                'response': 'Odin is starting up. Please try again in a moment.'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get generation parameters
        max_length = data.get('max_length', 150)
        temperature = data.get('temperature', 0.8)
        top_p = data.get('top_p', 0.9)
        
        # Validate parameters
        max_length = min(max(max_length, 10), 500)
        temperature = max(min(temperature, 2.0), 0.1)
        top_p = max(min(top_p, 1.0), 0.1)
        
        start_time = time.time()
        response = odin.generate_response(message, max_length, temperature, top_p)
        response_time = time.time() - start_time
        
        return jsonify({
            'response': response,
            'status': 'success',
            'response_time': round(response_time, 2),
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'top_p': top_p
            }
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'response': 'Sorry, I encountered an error processing your message.'
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Advanced generation endpoint with full parameter control"""
    try:
        if odin is None:
            return jsonify({'error': 'Model not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Advanced generation parameters
        params = {
            'max_length': data.get('max_length', 150),
            'temperature': data.get('temperature', 0.8),
            'top_p': data.get('top_p', 0.9)
        }
        
        # Validate parameters
        params['max_length'] = min(max(params['max_length'], 10), 500)
        params['temperature'] = max(min(params['temperature'], 2.0), 0.1)
        params['top_p'] = max(min(params['top_p'], 1.0), 0.1)
        
        start_time = time.time()
        response = odin.generate_response(prompt, **params)
        response_time = time.time() - start_time
        
        return jsonify({
            'prompt': prompt,
            'response': response,
            'response_time': round(response_time, 2),
            'parameters': params,
            'model_info': {
                'vocab_size': odin.tokenizer.vocab_size,
                'model_config': odin.model_config
            }
        })
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    if odin is None:
        return jsonify({
            'model_loaded': False,
            'status': 'initializing'
        })
    
    return jsonify({
        'model_loaded': odin.model is not None,
        'device': str(odin.device),
        'vocab_size': odin.tokenizer.vocab_size if odin.tokenizer else 0,
        'model_path': odin.model_path,
        'tokenizer_path': odin.tokenizer_path,
        'model_config': odin.model_config,
        'model_parameters': sum(p.numel() for p in odin.model.parameters()) if odin.model else 0,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'status': 'ready'
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if odin and odin.model else 'starting',
        'timestamp': time.time()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def initialize_odin():
    """Initialize Odin server with error handling"""
    global odin
    try:
        logger.info("Initializing Odin server...")
        odin = OdinServer()
        logger.info("Odin initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Odin: {e}")
        logger.error("Make sure you've trained a model first by running: python train_odin.py")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Odin AI Server")
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--tokenizer', help='Path to tokenizer file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🤖 ODIN AI SERVER")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print("=" * 50)
    
    # Initialize Odin with custom paths if provided
    if args.model or args.tokenizer:
        try:
            odin = OdinServer(args.model, args.tokenizer)
        except Exception as e:
            logger.error(f"Failed to initialize with custom paths: {e}")
            exit(1)
    else:
        # Try to initialize with auto-detection
        if not initialize_odin():
            print("\n❌ Failed to initialize Odin.")
            print("\n🔧 To get started:")
            print("1. Run: python train_odin.py")
            print("2. Wait for training to complete")
            print("3. Restart this server")
            print("\n📚 Or specify model paths:")
            print("python server.py --model path/to/model.pth --tokenizer path/to/tokenizer.json")
            exit(1)
    
    print("\n✅ Odin is ready!")
    print(f"🌐 Open your browser and go to: http://localhost:{args.port}")
    print(f"🔗 API endpoints:")
    print(f"   • POST /chat - Simple chat interface")
    print(f"   • POST /generate - Advanced generation with parameters")
    print(f"   • GET /status - Model status and info")
    print(f"   • GET /health - Health check")
    print()
    
    try:
        app.run(debug=args.debug, host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        exit(1)