document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    
    // Generate unique session ID
    const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    
    // Initialize chat with welcome message
    addMessage('bot', 'আসসালামু আলাইকুম! আমি বাংলা RAG চ্যাটবট। আপনি কী জানতে চান?', []);
    
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        addMessage('user', message, []);
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message
                })
            });
            
            const data = await response.json();
            addMessage('bot', data.bot_reply, data.retrieved_contexts);
        } catch (error) {
            addMessage('bot', 'দুঃখিত, একটি সমস্যা হয়েছে। পরে আবার চেষ্টা করুন।', []);
        }
        
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
    
    function addMessage(sender, text, contexts) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.textContent = text;
        
        if (contexts.length > 0) {
            const contextDiv = document.createElement('div');
            contextDiv.className = 'context-tag';
            contextDiv.textContent = `${contexts.length} প্রাসঙ্গিক তথ্য ব্যবহার করা হয়েছে`;
            messageDiv.appendChild(contextDiv);
        }
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});