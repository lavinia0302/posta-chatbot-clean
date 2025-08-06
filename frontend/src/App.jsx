import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import postaLogo from './assets/posta_logo.png';

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

<<<<<<< HEAD
  // Initialize session and load history
=======
  // Setează un session_id unic dacă nu există deja
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
  useEffect(() => {
    if (!localStorage.getItem('session_id')) {
      localStorage.setItem('session_id', crypto.randomUUID());
    }
<<<<<<< HEAD
    
=======
  }, []);

  // Încărcare istoric din localStorage
  useEffect(() => {
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
    const savedHistory = localStorage.getItem('posta_chat_history');
    if (savedHistory) {
      try {
        setMessages(JSON.parse(savedHistory));
      } catch (e) {
<<<<<<< HEAD
        console.error("Error loading history:", e);
=======
        console.error("Eroare la încărcarea istoricului:", e);
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
      }
    }
  }, []);

<<<<<<< HEAD
  // Save history and auto-scroll
=======
  // Salvarea istoricului
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('posta_chat_history', JSON.stringify(messages));
    }
<<<<<<< HEAD
=======
  }, [messages]);

  // Scroll automat la mesaje noi
  useEffect(() => {
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

<<<<<<< HEAD
=======
  const cleanText = (text) => {
    if (!text) return '';
    return text
      .replace(/(\[[^\]]+\]\([^)]+\))(\([^)]+\))/g, '$1')
      .replace(/\[(https?:\/\/[^\]]+)\]\(https?:\/\/[^)]+\)/g, '$1')
      .replace(/(https?:\/\/[^\s]+)\)\(https?:\/\/[^\s]+\)/g, '$1');
  };

>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
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
<<<<<<< HEAD
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
=======
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          session_id: localStorage.getItem('session_id')  // trimite ID-ul sesiunii
        })
      });

      if (!response.ok) throw new Error('Eroare server');

      const data = await response.json();
      const cleanedAnswer = cleanText(data.answer);

      const botMessage = {
        id: Date.now() + 1,
        role: 'bot',
        text: cleanedAnswer,
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
<<<<<<< HEAD
      console.error('API Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        text: '⚠️ Eroare la comunicare cu serverul. Încercați din nou.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
=======
      const errorMessage = {
        id: Date.now() + 1,
        role: 'bot',
        text: '⚠️ Eroare la conectare cu serverul. Vă rugăm încercați mai târziu.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Eroare API:', error);
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
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
<<<<<<< HEAD
    return text.split('\n').map((paragraph, i) => (
      <p key={i}>{paragraph}</p>
    ));
  };

  const commonQuestions = [
    "Care sunt orele de funcționare?",
    "Cum trimit un colet urgent?",
    "Ce documente sunt necesare pentru un plic recomandat?",
    "Cum pot urmări un colet?"
=======

    return text.split('\n').map((paragraph, i) => {
      if (paragraph.includes("Nu am găsit un răspuns clar")) {
        return (
          <div key={i} className="not-found">
            <p>{paragraph}</p>
            <div className="suggestions">
              <p>Puteți găsi informația:</p>
              <ul>
                <li>Pe <a href="https://www.posta-romana.ro" target="_blank" rel="noopener noreferrer">site-ul oficial</a></li>
                <li>Prin apel la <strong>021 9393</strong></li>
                <li>La orice <strong>oficiu poștal</strong></li>
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
    { text: "Care sunt orele de funcționare?", details: "program oficiu poștal" },
    { text: "Cum trimit un colet urgent?", details: "tarife și timpi de livrare" },
    { text: "Ce documente sunt necesare pentru un plic recomandat?", details: "acte de identitate necesare" },
    { text: "Cum pot urmări un colet?", details: "folosind numărul de tracking" }
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
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
<<<<<<< HEAD
                    onClick={() => setQuestion(q)}
                  >
                    {q}
=======
                    onClick={() => setQuestion(q.text)}
                  >
                    <div className="question-text">{q.text}</div>
                    <div className="question-details">{q.details}</div>
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="messages-list">
<<<<<<< HEAD
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
=======
            <h4 className="continue-label">Puteți continua conversația în același fir 🧠</h4>
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <div className="message-header">
                  <span className="sender">{msg.role === 'bot' ? 'PoștaBot' : 'Dvs.'}</span>
                  <span className="timestamp">{msg.timestamp}</span>
                </div>
                <div className="message-content">{formatText(msg.text)}</div>
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
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
<<<<<<< HEAD
        <button 
          onClick={handleSend} 
          disabled={loading || !question.trim()}
        >
          {loading ? 'Se încarcă...' : 'Trimite'}
=======
        <button onClick={handleSend} disabled={loading || !question.trim()}>
          {loading ? <span className="loading-spinner"></span> : 'Trimite'}
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
        </button>
      </footer>
    </div>
  );
}

<<<<<<< HEAD
export default App;
=======
export default App;
>>>>>>> 94ddae98afdebfa829d3bbe77ebe44d667258306
