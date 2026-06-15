import { useState, useRef, useEffect } from 'react'

const BACKEND_URL = 'http://tinywebapp-alb-935751383.us-east-1.elb.amazonaws.com'

function parseCitations(text, sources) {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/)
    if (match) {
      const num = parseInt(match[1])
      const source = sources.find(s => s.citation_number === num)
      return (
        <a
          key={i}
          href={source?.url || '#'}
          target="_blank"
          rel="noreferrer"
          className="citation"
          title={source?.title || ''}
        >
          [{num}]
        </a>
      )
    }
    return <span key={i}>{part}</span>
  })
}

function SourceList({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null
  return (
    <div className="sources-container">
      <button className="sources-toggle" onClick={() => setOpen(o => !o)}>
        {open ? '▾' : '▸'} {sources.length} source{sources.length > 1 ? 's' : ''}
      </button>
      {open && (
        <ul className="sources-list">
          {sources.map(s => (
            <li key={s.citation_number}>
              <span className="citation-badge">[{s.citation_number}]</span>
              <a href={s.url} target="_blank" rel="noreferrer">{s.title || s.url}</a>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
        <p>{isUser ? msg.content : parseCitations(msg.content, msg.sources || [])}</p>
        {!isUser && <SourceList sources={msg.sources} />}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { role: 'user', content: text }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      })
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        sources: data.sources,
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.',
        sources: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="logo">CA Benefits Assistant</span>
          <span className="subtitle">Helping Californians navigate unemployment resources</span>
        </div>
      </header>

      <main className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>How can I help you today?</h2>
            <p>Ask me about unemployment benefits, CalFresh, Medi-Cal, job training, and more.</p>
            <div className="suggestions">
              {[
                'What unemployment benefits am I entitled to?',
                'How do I apply for CalFresh food assistance?',
                'What healthcare options do I have after being laid off?',
              ].map(s => (
                <button key={s} className="suggestion" onClick={() => setInput(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => <Message key={i} msg={msg} />)}

        {loading && (
          <div className="message-row assistant">
            <div className="bubble assistant loading">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <form className="input-form" onSubmit={sendMessage}>
          <input
            className="input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about California benefits..."
            disabled={loading}
          />
          <button className="send-btn" type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </footer>
    </div>
  )
}
