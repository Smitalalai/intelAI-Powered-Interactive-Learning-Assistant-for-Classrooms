// AI Features JavaScript for EduAI Pro

// Global variables
let isAIChatOpen = false;
let isRecording = false;
let recognition = null;

// Initialize AI features when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeAIFeatures();
    initializeSpeechRecognition();
    initializeNotifications();
});

// Initialize AI features
function initializeAIFeatures() {
    // Auto-scroll chat messages
    const chatMessages = document.getElementById('aiChatMessages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Enter key support for chat input
    const chatInput = document.getElementById('aiChatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendAIMessage();
            }
        });
    }
    
    // Initialize flashcard flip functionality
    initializeFlashcards();
    
    // Initialize quiz enhancements
    initializeQuizEnhancements();
}

// Toggle AI Chat Panel
function toggleAIChat() {
    const panel = document.getElementById('aiChatPanel');
    if (panel) {
        isAIChatOpen = !isAIChatOpen;
        panel.style.display = isAIChatOpen ? 'flex' : 'none';
        
        if (isAIChatOpen) {
            document.getElementById('aiChatInput').focus();
        }
    }
}

// Send AI Message
async function sendAIMessage() {
    const input = document.getElementById('aiChatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessageToChat('user', message);
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/ai_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                context: getCurrentContext()
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator and add AI response
        removeTypingIndicator();
        addMessageToChat('ai', data.response);
        
    } catch (error) {
        removeTypingIndicator();
        addMessageToChat('ai', 'Sorry, I encountered an error. Please try again.');
        console.error('AI Chat error:', error);
    }
}

// Add message to chat
function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('aiChatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = sender === 'user' ? 'user-message' : 'ai-message';
    
    if (sender === 'ai') {
        messageDiv.innerHTML = `<i class="fas fa-robot me-2"></i>${message}`;
    } else {
        messageDiv.textContent = message;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('aiChatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'ai-message typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = '<i class="fas fa-robot me-2"></i>Thinking...';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

// Get current context for AI
function getCurrentContext() {
    return {
        page: window.location.pathname,
        user_id: document.body.getAttribute('data-user-id'),
        timestamp: new Date().toISOString()
    };
}

// Initialize Speech Recognition
function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('aiChatInput').value = transcript;
            stopVoiceRecording();
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            stopVoiceRecording();
        };
        
        recognition.onend = function() {
            stopVoiceRecording();
        };
    }
}

// Start voice recording
function startVoiceRecording() {
    if (recognition && !isRecording) {
        isRecording = true;
        const voiceBtn = document.getElementById('voiceInputBtn');
        voiceBtn.classList.add('recording');
        voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
        recognition.start();
    }
}

// Stop voice recording
function stopVoiceRecording() {
    if (recognition && isRecording) {
        isRecording = false;
        const voiceBtn = document.getElementById('voiceInputBtn');
        voiceBtn.classList.remove('recording');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        recognition.stop();
    }
}

// Voice input button click handler
document.addEventListener('click', function(e) {
    if (e.target.closest('#voiceInputBtn')) {
        e.preventDefault();
        if (isRecording) {
            stopVoiceRecording();
        } else {
            startVoiceRecording();
        }
    }
});

// Initialize flashcards
function initializeFlashcards() {
    const flashcards = document.querySelectorAll('.flashcard');
    flashcards.forEach(card => {
        card.addEventListener('click', function() {
            this.classList.toggle('flipped');
        });
    });
}

// Initialize quiz enhancements
function initializeQuizEnhancements() {
    // Add hint functionality to quiz questions
    const hintButtons = document.querySelectorAll('.hint-button');
    hintButtons.forEach(button => {
        button.addEventListener('click', function() {
            const questionId = this.getAttribute('data-question-id');
            getHintForQuestion(questionId);
        });
    });
    
    // Enhance quiz option selection
    const quizOptions = document.querySelectorAll('.quiz-option');
    quizOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selected class from siblings
            const siblings = this.parentNode.querySelectorAll('.quiz-option');
            siblings.forEach(sibling => sibling.classList.remove('selected'));
            
            // Add selected class to clicked option
            this.classList.add('selected');
            
            // Update hidden input value
            const input = this.querySelector('input[type="radio"]');
            if (input) {
                input.checked = true;
            }
        });
    });
}

// Get hint for question
async function getHintForQuestion(questionId) {
    try {
        const response = await fetch(`/api/get_hint/${questionId}`);
        const data = await response.json();
        
        // Show hint in a modal or tooltip
        showHint(data.hint);
        
    } catch (error) {
        console.error('Error getting hint:', error);
        showHint('Sorry, unable to get hint at the moment.');
    }
}

// Show hint to user
function showHint(hintText) {
    // Create and show hint modal
    const hintModal = document.createElement('div');
    hintModal.className = 'modal fade';
    hintModal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"><i class="fas fa-lightbulb text-warning me-2"></i>AI Hint</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p>${hintText}</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Got it!</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(hintModal);
    const modal = new bootstrap.Modal(hintModal);
    modal.show();
    
    // Remove modal from DOM when closed
    hintModal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(hintModal);
    });
}

// Text-to-Speech functionality
function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.8;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
    }
}

// Add speak buttons to quiz questions
function addSpeakButtons() {
    const questionTexts = document.querySelectorAll('.quiz-question-text');
    questionTexts.forEach(questionText => {
        const speakBtn = document.createElement('button');
        speakBtn.className = 'btn btn-sm btn-outline-primary ms-2';
        speakBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
        speakBtn.title = 'Listen to question';
        speakBtn.type = 'button';
        
        speakBtn.addEventListener('click', function() {
            speakText(questionText.textContent);
        });
        
        questionText.appendChild(speakBtn);
    });
}

// Initialize notifications
function initializeNotifications() {
    // Simulate real-time notifications
    setInterval(updateNotifications, 30000); // Check every 30 seconds
}

// Update notifications
function updateNotifications() {
    // This would normally fetch from server
    const notificationCount = document.getElementById('notificationCount');
    if (notificationCount) {
        // Simulate new notifications
        const currentCount = parseInt(notificationCount.textContent);
        if (Math.random() > 0.7) { // 30% chance of new notification
            notificationCount.textContent = currentCount + 1;
            notificationCount.style.animation = 'pulse 1s';
        }
    }
}

// Global search functionality
function performGlobalSearch() {
    const searchInput = document.getElementById('globalSearch');
    const query = searchInput.value.trim();
    
    if (query) {
        // Implement search functionality
        console.log('Searching for:', query);
        // This would redirect to search results or show search modal
    }
}

// Search button click handler
document.addEventListener('click', function(e) {
    if (e.target.closest('#searchBtn')) {
        performGlobalSearch();
    }
});

// Search input enter key handler
document.addEventListener('keypress', function(e) {
    if (e.target.id === 'globalSearch' && e.key === 'Enter') {
        performGlobalSearch();
    }
});

// Create flashcards from quiz results
async function createFlashcardsFromQuiz(topic) {
    try {
        const response = await fetch('/api/create_flashcards', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                topic: topic,
                count: 5
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess(`Created ${data.created} flashcards for ${topic}!`);
        }
        
    } catch (error) {
        console.error('Error creating flashcards:', error);
        showError('Failed to create flashcards.');
    }
}

// Show success message
function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alert, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

// Show error message
function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alert, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

// Timer functionality for quizzes
function initializeQuizTimer(duration) {
    let timeRemaining = duration * 60; // Convert minutes to seconds
    const timerElement = document.getElementById('quizTimer');
    
    if (!timerElement) return;
    
    const timer = setInterval(() => {
        const minutes = Math.floor(timeRemaining / 60);
        const seconds = timeRemaining % 60;
        
        timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        // Add warning class when time is running low
        if (timeRemaining <= 300) { // Last 5 minutes
            timerElement.classList.add('timer-warning');
        }
        
        if (timeRemaining <= 0) {
            clearInterval(timer);
            autoSubmitQuiz();
        }
        
        timeRemaining--;
    }, 1000);
}

// Auto-submit quiz when time runs out
function autoSubmitQuiz() {
    const quizForm = document.querySelector('form');
    if (quizForm) {
        showError('Time\'s up! Quiz submitted automatically.');
        quizForm.submit();
    }
}

// Export functions for global use
window.toggleAIChat = toggleAIChat;
window.sendAIMessage = sendAIMessage;
window.createFlashcardsFromQuiz = createFlashcardsFromQuiz;
window.speakText = speakText;
window.initializeQuizTimer = initializeQuizTimer;
