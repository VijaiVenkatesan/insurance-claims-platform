import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  MessageCircle, X, Send, Bot, User,
  Loader, ChevronRight
} from 'lucide-react'
import { useLocation } from 'react-router-dom'
import api from '../utils/api'

const PAGE_NAMES = {
  '/':          'Dashboard',
  '/submit':    'Submit Claim',
  '/claims':    'Claims List',
  '/hitl':      'HITL Review',
  '/analytics': 'Analytics',
  '/policies':  'Policy Admin',
  '/medical':   'Medical AI',
  '/admin':     'Admin Panel',
  '/help':      'Help',
  '/about':     'About',
}

const WELCOME = `Hi! I'm **MediSure AI Assistant** 👋

I can help you with:
• Submitting and tracking claims
• Understanding AI decisions  
• HITL review process
• Medical AI features
• India-specific insurance rules

What can I help you with today?`

function MessageBubble({ msg }) {
  const isBot = msg.role === 'assistant'
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        display: 'flex', gap: 8, marginBottom: 12,
        flexDirection: isBot ? 'row' : 'row-reverse',
        alignItems: 'flex-start',
      }}
    >
      {/* Avatar */}
      <div style={{
        width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
        background: isBot ? 'var(--brand-dim)' : 'var(--bg-elevated)',
        border: `1px solid ${isBot ? 'var(--brand-glow)' : 'var(--border-default)'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {isBot
          ? <Bot size={14} color="var(--brand)" />
          : <User size={14} color="var(--text-secondary)" />}
      </div>

      {/* Bubble */}
      <div style={{
        maxWidth: '80%',
        background: isBot ? 'var(--bg-elevated)' : 'var(--brand-dim)',
        border: `1px solid ${isBot ? 'var(--border-subtle)' : 'var(--brand-glow)'}`,
        borderRadius: isBot ? '4px 12px 12px 12px' : '12px 4px 12px 12px',
        padding: '10px 14px',
        fontSize: 13,
        lineHeight: 1.65,
        color: 'var(--text-primary)',
        whiteSpace: 'pre-wrap',
      }}>
        {msg.content.replace(/\*\*(.*?)\*\*/g, '$1')}
      </div>
    </motion.div>
  )
}

export default function SupportChatbot() {
  const location = useLocation()
  const [open,        setOpen]        = useState(false)
  const [messages,    setMessages]    = useState([
    { role: 'assistant', content: WELCOME }
  ])
  const [input,       setInput]       = useState('')
  const [loading,     setLoading]     = useState(false)
  const [suggestions, setSuggestions] = useState([
    'How to submit a claim',
    'What is HITL review?',
    'India-specific features',
  ])
  const [unread, setUnread] = useState(0)
  const bottomRef = useRef(null)
  const inputRef  = useRef(null)

  const currentPage = PAGE_NAMES[location.pathname] || 'MediSure AI'

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  useEffect(() => {
    if (open) {
      setUnread(0)
      setTimeout(() => inputRef.current?.focus(), 200)
    }
  }, [open])

  const sendMessage = useCallback(async (text) => {
    const userMsg = text || input.trim()
    if (!userMsg || loading) return

    setInput('')
    setSuggestions([])
    const newMessages = [...messages, { role: 'user', content: userMsg }]
    setMessages(newMessages)
    setLoading(true)

    try {
      const res = await api.post('/support/chat', {
        messages: newMessages,
        context: currentPage,
      })
      const { reply, suggestions: newSuggestions } = res.data
      setMessages(prev => [...prev, { role: 'assistant', content: reply }])
      setSuggestions(newSuggestions || [])
      if (!open) setUnread(u => u + 1)
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Sorry, I'm having trouble connecting. Please check the Help page or try again.",
      }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, open, currentPage])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // Don't show chatbot on login page
  if (location.pathname === '/login') return null

  return (
    <>
      {/* ── FLOATING BUTTON ── */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(o => !o)}
        className="chatbot-fab"
      style={{
          width: 52, height: 52, borderRadius: '50%',
          background: open ? 'var(--bg-elevated)' : 'linear-gradient(135deg, var(--brand), #007aff)',
          border: open ? '1px solid var(--border-default)' : 'none',
          cursor: 'pointer', boxShadow: open ? 'var(--shadow-md)' : 'var(--shadow-brand)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: open ? 'var(--text-primary)' : '#000',
          transition: 'background 0.2s',
        }}
      >
        <AnimatePresence mode="wait">
          {open
            ? <motion.div key="x" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }}>
                <X size={20} />
              </motion.div>
            : <motion.div key="chat" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }}>
                <MessageCircle size={20} />
              </motion.div>}
        </AnimatePresence>
        {/* Unread badge */}
        {unread > 0 && !open && (
          <div style={{
            position: 'absolute', top: -2, right: -2,
            width: 18, height: 18, borderRadius: '50%',
            background: 'var(--reject)', color: 'white',
            fontSize: 11, fontWeight: 700,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>{unread}</div>
        )}
      </motion.button>

      {/* ── CHAT WINDOW ── */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 16 }}
            transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            className="chatbot-panel"
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-xl)',
              boxShadow: 'var(--shadow-lg)',
              display: 'flex', flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Header */}
            <div style={{
              padding: '14px 16px',
              background: 'linear-gradient(135deg, var(--bg-elevated), var(--bg-surface))',
              borderBottom: '1px solid var(--border-subtle)',
              display: 'flex', alignItems: 'center', gap: 10,
            }}>
              <div style={{
                width: 36, height: 36, borderRadius: '50%',
                background: 'var(--brand-dim)',
                border: '1px solid var(--brand-glow)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Bot size={18} color="var(--brand)" />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 700 }}>MediSure AI Assistant</div>
                <div style={{ fontSize: 11, color: 'var(--approve)', display: 'flex',
                  alignItems: 'center', gap: 4 }}>
                  <div style={{ width: 6, height: 6, borderRadius: '50%',
                    background: 'var(--approve)' }} />
                  Online · AI powered by Groq
                </div>
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textAlign: 'right' }}>
                {currentPage}
              </div>
            </div>

            {/* Messages */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '14px 12px' }}>
              {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
              {loading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  style={{ display: 'flex', gap: 8, alignItems: 'center',
                    padding: '8px 0', marginBottom: 12 }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: '50%',
                    background: 'var(--brand-dim)', border: '1px solid var(--brand-glow)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <Bot size={14} color="var(--brand)" />
                  </div>
                  <div style={{ display: 'flex', gap: 4, padding: '8px 12px',
                    background: 'var(--bg-elevated)', borderRadius: '4px 12px 12px 12px',
                    border: '1px solid var(--border-subtle)' }}>
                    {[0, 0.2, 0.4].map((d, i) => (
                      <motion.div key={i}
                        animate={{ y: [0, -4, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8, delay: d }}
                        style={{ width: 6, height: 6, borderRadius: '50%',
                          background: 'var(--brand)' }} />
                    ))}
                  </div>
                </motion.div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Suggestions */}
            {suggestions.length > 0 && !loading && (
              <div style={{
                padding: '6px 12px',
                borderTop: '1px solid var(--border-subtle)',
                display: 'flex', gap: 6, flexWrap: 'wrap',
              }}>
                {suggestions.map((s, i) => (
                  <button key={i} onClick={() => sendMessage(s)}
                    style={{
                      background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
                      borderRadius: 20, padding: '4px 10px', fontSize: 11,
                      color: 'var(--text-secondary)', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 4,
                      transition: 'all var(--transition)',
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.borderColor = 'var(--brand)'
                      e.currentTarget.style.color = 'var(--brand)'
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.borderColor = ''
                      e.currentTarget.style.color = ''
                    }}>
                    <ChevronRight size={10} /> {s}
                  </button>
                ))}
              </div>
            )}

            {/* Input */}
            <div style={{
              padding: '10px 12px',
              borderTop: '1px solid var(--border-subtle)',
              display: 'flex', gap: 8, alignItems: 'flex-end',
            }}>
              <textarea
                ref={inputRef}
                rows={1}
                className="form-textarea"
                placeholder="Ask anything about MediSure AI…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                style={{
                  flex: 1, minHeight: 36, maxHeight: 100, resize: 'none',
                  fontSize: 13, padding: '8px 12px', borderRadius: 'var(--radius-md)',
                }}
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => sendMessage()}
                disabled={!input.trim() || loading}
                style={{
                  width: 36, height: 36, borderRadius: 'var(--radius-md)',
                  background: input.trim() && !loading ? 'var(--brand)' : 'var(--bg-elevated)',
                  border: `1px solid ${input.trim() && !loading ? 'var(--brand)' : 'var(--border-default)'}`,
                  cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: input.trim() && !loading ? '#000' : 'var(--text-muted)',
                  flexShrink: 0, transition: 'all var(--transition)',
                }}
              >
                {loading ? <Loader size={15} style={{ animation: 'spin 0.7s linear infinite' }} />
                         : <Send size={15} />}
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
