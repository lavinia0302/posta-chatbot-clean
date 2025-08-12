// app.jsx
import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import postaLogo from './assets/posta_logo.png';

// URL-ul backendului (ngrok). Dacă e gol, cade pe localhost:5000.
const API_BASE =
  'https://68d90a4696c0.ngrok-free.app' || 'http://localhost:5000';

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // setează un session_id unic în localStorage
  useEffect(() => {
    if (!localStorage.getItem('session_id')) {
      localStorage.setItem('session_id', crypto.randomUUID());
    }
  }, []);

  // încărcare istoric
  useEffect(() => {
    const savedHistory = localStorage.getItem('posta_chat_history');
    if (savedHistory) {
      try {
        setMessages(JSON.parse(savedHistory));
      } catch {}
    }
  }, []);

  // salvare istoric
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('posta_chat_history', JSON.stringify(messages));
    }
  }, [messages]);

  // autoscroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const cleanText = (text) => {
    if (!text) return '';
    return text
      .replace(/(\[[^\]]+\]\([^)]+\))(\([^)]+\))/g, '$1')
      .replace(/\[(https?:\/\/[^\]]+)\]\(https?:\/\/[^)]+\)/g, '$1')
      .replace(/(https?:\/\/[^\s]+)\)\(https?:\/\/[^\s]+\)/g, '$1');
  };

  const handleSend = async () => {
    if (!question.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      text: question,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          session_id: localStorage.getItem('session_id')
        })
      });

      if (!response.ok) throw new Error('Eroare server');

      const data = await response.json();
      const cleanedAnswer = cleanText(data.answer);

      const botMessage = {
        id: Date.now() + 1,
        role: 'bot',
        text: cleanedAnswer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'bot',
          text: '⚠️ Eroare la conectare cu serverul. Vă rugăm încercați mai târziu.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
      console.error('Eroare API:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) handleSend();
  };

  const formatText = (text) => {
    if (!text) return null;
    return text.split('\n').map((paragraph, i) => {
      if (paragraph.includes('Nu am găsit un răspuns clar')) {
        return (
          <div key={i} className="not-found">
            <p>{paragraph}</p>
            <div className="suggestions">
              <p>Puteți găsi informația:</p>
              <ul>
                <li>
                  Pe{' '}
                  <a href="https://www.posta-romana.ro" target="_blank" rel="noopener noreferrer">
                    site-ul oficial
                  </a>
                </li>
                <li>
                  Prin apel la <strong>021 9393</strong>
                </li>
                <li>
                  La orice <strong>oficiu poștal</strong>
                </li>
              </ul>
            </div>
          </div>
        );
      }
      const html = paragraph
        .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n•/g, '<br/>•');
      return <p key={i} dangerouslySetInnerHTML={{ __html: html }} />;
    });
  };

  const commonQuestions = [
    { text: 'Care sunt orele de funcționare?', details: 'program oficiu poștal' },
    { text: 'Cum trimit un colet urgent?', details: 'tarife și timpi de livrare' },
    { text: 'Unde găsesc cel mai apropiat oficiu?', details: 'căutare după localitate/adresă' },
    { text: 'Cum pot urmări un colet?', details: 'folosind numărul de tracking' }
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
                  <button key={i} className="question-chip" onClick={() => setQuestion(q.text)}>
                    <div className="question-text">{q.text}</div>
                    <div className="question-details">{q.details}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            <h4 className="continue-label">Puteți continua conversația în același fir 🧠</h4>
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <div className="message-header">
                  <span className="sender">{msg.role === 'bot' ? 'PoștaBot' : 'Dvs.'}</span>
                  <span className="timestamp">{msg.timestamp}</span>
                </div>
                <div className="message-content">{formatText(msg.text)}</div>
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
        <button onClick={handleSend} disabled={loading || !question.trim()}>
          {loading ? <span className="loading-spinner"></span> : 'Trimite'}
        </button>
      </footer>
    </div>
  );
}

export default App;
