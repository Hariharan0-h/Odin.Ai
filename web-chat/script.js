class OdinChat {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messagesContainer = document.getElementById('messagesContainer');
        this.statusIndicator = document.getElementById('statusIndicator');
        
        this.initEventListeners();
        this.removeWelcomeMessage();
    }
    
    initEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }
    
    removeWelcomeMessage() {
        setTimeout(() => {
            const welcome = this.messagesContainer.querySelector('.welcome-message');
            if (welcome && this.messagesContainer.children.length > 1) {
                welcome.style.opacity = '0';
                setTimeout(() => welcome.remove(), 300);
            }
        }, 100);
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        this.removeWelcomeMessage();
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.setLoading(true);
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.addMessage(data.response, 'bot');
            this.updateStatus('Ready', 'success');
            
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('I encountered an error. Please try again.', 'bot');
            this.updateStatus('Connection Error', 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.innerHTML = `<div class="message-bubble">${content}</div>`;
        
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    setLoading(isLoading) {
        this.sendButton.disabled = isLoading;
        this.messageInput.disabled = isLoading;
        
        if (isLoading) {
            this.updateStatus('Thinking...', 'loading');
        }
    }
    
    updateStatus(message, type = 'success') {
        const dot = this.statusIndicator.querySelector('.status-dot');
        const text = this.statusIndicator.querySelector('span:last-child');
        
        text.textContent = message;
        dot.style.background = type === 'error' ? '#ef4444' : '#22c55e';
    }
}

document.addEventListener('DOMContentLoaded', () => new OdinChat());