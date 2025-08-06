import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import postaLogo from './assets/posta_logo.png';

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Initialize session and load history
  useEffect(() => {
    if (!localStorage.getItem('session_id')) {
      localStorage.setItem('session_id', crypto.randomUUID());
    }
    
    const savedHistory = localStorage.getItem('posta_chat_history');
    if (savedHistory) {
      try {
        setMessages(JSON.parse(savedHistory));
      } catch (e) {
        console.error("Error loading history:", e);
      }
    }
  }, []);

  // Save history and auto-scroll
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('posta_chat_history', JSON.stringify(messages));
    }
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!question.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      text: question,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          question,
          session_id: localStorage.getItem('session_id')
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      const botMessage = {
        id: Date.now() + 1,
        role: 'bot',
        text: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('API Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: '⚠️ Eroare la comunicare cu serverul. Încercați din nou.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleSend();
    }
  };

  const formatText = (text) => {
    if (!text) return null;
    return text.split('\n').map((paragraph, i) => (
      <p key={i}>{paragraph}</p>
    ));
  };

  const commonQuestions = [
    "Care sunt orele de funcționare?",
    "Cum trimit un colet urgent?",
    "Ce documente sunt necesare pentru un plic recomandat?",
    "Cum pot urmări un colet?"
  ];

  const resetConversation = () => {
    localStorage.removeItem('posta_chat_history');
    setMessages([]);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <img src={postaLogo} alt="Poșta Română" className="logo" />
        <h1>Chatbot Poșta Română</h1>
        {messages.length > 0 && (
          <button className="reset-button" onClick={resetConversation}>
            🗑️ Șterge conversația
          </button>
        )}
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h3>Bun venit la Chatbot-ul Poștei Române</h3>
            <p>Cu ce vă pot ajuta astăzi?</p>

            <div className="common-questions">
              <h4>Întrebări frecvente:</h4>
              <div className="question-grid">
                {commonQuestions.map((q, i) => (
                  <button
                    key={i}
                    className="question-chip"
                    onClick={() => setQuestion(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <div className="message-header">
                  <span className="sender">
                    {msg.role === 'bot' ? 'PoștaBot' : 'Dvs.'}
                  </span>
                  <span className="timestamp">{msg.timestamp}</span>
                </div>
                <div className="message-content">
                  {formatText(msg.text)}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}
      </main>

      <footer className="input-area">
        <input
          type="text"
          placeholder="Scrieți întrebarea aici..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button 
          onClick={handleSend} 
          disabled={loading || !question.trim()}
        >
          {loading ? 'Se încarcă...' : 'Trimite'}
        </button>
      </footer>
    </div>
  );
}

export default App;