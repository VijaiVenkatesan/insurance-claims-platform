import { Routes, Route, NavLink, useLocation, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Upload, List, ShieldCheck, BarChart2, BookOpen,
  Activity, AlertCircle, X, Stethoscope, Shield, HelpCircle,
  Info, LogOut, User, ChevronDown, FileSearch, Menu
} from 'lucide-react'
import { useState, useEffect, useRef } from 'react'
import { useAuth } from './hooks/useAuth'
import { getPendingHITL, getHealth } from './utils/api'

import LoginPage    from './pages/Login.jsx'
import Dashboard    from './pages/Dashboard.jsx'
import Submit       from './pages/Submit.jsx'
import Claims       from './pages/Claims.jsx'
import ClaimDetail  from './pages/ClaimDetail.jsx'
import HITLReview   from './pages/HITLReview.jsx'
import Analytics    from './pages/Analytics.jsx'
import PolicyAdmin  from './pages/PolicyAdmin.jsx'
import MedicalAI    from './pages/MedicalAI.jsx'
import AdminPanel   from './pages/AdminPanel.jsx'
import HelpPage     from './pages/HelpPage.jsx'
import AboutPage    from './pages/AboutPage.jsx'
import Underwriting from './pages/Underwriting.jsx'
import ClinicalDecisionSupport from './pages/ClinicalDecisionSupport.jsx'
import SupportChatbot from './components/SupportChatbot.jsx'

const IS_PRODUCTION = typeof window !== 'undefined' && window.location.hostname !== 'localhost'
const RENDER_URL = 'https://medisure-api-vyx1.onrender.com'

// ── USER MENU ──────────────────────────────────────────────────────
function UserMenu() {
  const { user, logout } = useAuth()
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  const roleColor = {
    ADMIN: 'var(--brand)', REVIEWER: 'var(--info)', USER: 'var(--approve)'
  }[user?.role] || 'var(--text-muted)'

  return (
    <div ref={ref} style={{ position: 'relative', padding: '12px 12px 0' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', background: 'var(--bg-elevated)',
        border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)',
        padding: '9px 12px', cursor: 'pointer', display: 'flex',
        alignItems: 'center', gap: 8, color: 'inherit',
        transition: 'border-color var(--transition)',
      }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-strong)'}
        onMouseLeave={e => e.currentTarget.style.borderColor = ''}>
        <div style={{
          width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
          background: roleColor + '20', color: roleColor,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 700,
        }}>
          {user?.username?.charAt(0).toUpperCase()}
        </div>
        <div style={{ flex: 1, textAlign: 'left', overflow: 'hidden' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {user?.full_name || user?.username}
          </div>
          <div style={{ fontSize: 10, color: roleColor, fontWeight: 700 }}>{user?.role}</div>
        </div>
        <ChevronDown size={12} style={{
          color: 'var(--text-muted)',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
        }} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            style={{
              position: 'absolute', bottom: '100%', left: 12, right: 12,
              background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-md)', overflow: 'hidden',
              boxShadow: 'var(--shadow-md)', zIndex: 100, marginBottom: 4,
            }}>
            <div style={{ padding: '10px 12px 8px', borderBottom: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: 12, fontWeight: 600 }}>{user?.username}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{user?.role} account</div>
            </div>
            <button onClick={() => { logout(); setOpen(false) }} style={{
              width: '100%', background: 'none', border: 'none', cursor: 'pointer',
              padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 8,
              color: 'var(--reject)', fontSize: 13, textAlign: 'left',
              transition: 'background var(--transition)',
            }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--reject-dim)'}
              onMouseLeave={e => e.currentTarget.style.background = 'none'}>
              <LogOut size={13} /> Sign Out
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── MAIN APP ───────────────────────────────────────────────────────
export default function App() {
  const location   = useLocation()

  // ── ALL HOOKS MUST BE AT THE TOP — before any conditional returns ──
  // Single useAuth() call — no duplicate hook calls
  const { user, loading, isAdmin, isReviewer } = useAuth()

  const [pendingCount, setPendingCount] = useState(0)
  const [systemStatus, setSystemStatus] = useState('checking')
  const [dismissed,    setDismissed]    = useState(false)
  const [sidebarOpen,  setSidebarOpen]  = useState(false)

  useEffect(() => {
    if (!user) return
    const checkHealth = async () => {
      try { const d = await getHealth(); setSystemStatus(d.status) }
      catch { setSystemStatus('sleeping') }
    }
    const loadPending = async () => {
      try { const d = await getPendingHITL(); setPendingCount(d.total) } catch {}
    }
    checkHealth()
    loadPending()
    const iv = setInterval(() => { checkHealth(); loadPending() }, 20000)
    return () => clearInterval(iv)
  }, [user])

  // Close sidebar when route changes (mobile)
  useEffect(() => { setSidebarOpen(false) }, [location.pathname])

  // ── CONDITIONAL RETURNS — after all hooks ──
  if (loading) return (
    <div style={{
      height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 14, background: 'var(--bg-base)',
    }}>
      <div style={{
        width: 52, height: 52, borderRadius: 16,
        background: 'linear-gradient(135deg, var(--brand), #007aff)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 22, fontWeight: 800, color: '#000',
      }}>C</div>
      <div className="spinner" />
    </div>
  )

  if (!user) return <LoginPage />

  const bannerVisible = IS_PRODUCTION && !dismissed && systemStatus !== 'healthy'

  return (
    <div className="app-layout">

      {/* ── WAKE-UP BANNER ── */}
      <AnimatePresence>
        {bannerVisible && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            style={{
              position: 'fixed', top: 0, left: 0, right: 0, zIndex: 200,
              background: 'linear-gradient(90deg, #1a1f2e, #1e2535)',
              borderBottom: '1px solid var(--investigate)',
              padding: '10px 20px', display: 'flex', alignItems: 'center',
              gap: 10, fontSize: 13,
            }}>
            <AlertCircle size={15} color="var(--investigate)" style={{ flexShrink: 0 }} />
            <span style={{ color: 'var(--text-secondary)', flex: 1 }}>
              <strong style={{ color: 'var(--investigate)' }}>Backend waking up</strong>
              {' '}— Render free tier sleeps after 15 min. First request ~30 seconds.
            </span>
            <a href={`${RENDER_URL}/api/v1/health`} target="_blank" rel="noopener noreferrer"
              style={{ color: 'var(--brand)', fontSize: 12, textDecoration: 'none', marginRight: 8 }}>
              Check API ↗
            </a>
            <button onClick={() => setDismissed(true)} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-muted)', padding: 2,
            }}>
              <X size={14} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── MOBILE HAMBURGER ── */}
      <button
        className="mobile-menu-btn"
        onClick={() => setSidebarOpen(o => !o)}
        aria-label="Toggle menu"
        style={{ top: bannerVisible ? 53 : 12 }}
      >
        {sidebarOpen ? <X size={18} /> : <Menu size={18} />}
      </button>

      {/* ── SIDEBAR OVERLAY (mobile) ── */}
      <div
        className={`sidebar-overlay${sidebarOpen ? ' open' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* ── SIDEBAR ── */}
      <aside
        className={`sidebar${sidebarOpen ? ' open' : ''}`}
        style={{ top: bannerVisible ? 41 : 0 }}
      >
        <div className="sidebar-logo">
          <div className="logo-mark">M</div>
          <div className="logo-text">Medi<span>Sure AI</span></div>
        </div>

        <nav className="sidebar-nav" style={{ flex: 1 }}>
          <div className="nav-section-label">Platform</div>
          <NavLink to="/" end className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <LayoutDashboard size={15} /> Dashboard
          </NavLink>
          <NavLink to="/submit" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <Upload size={15} /> Submit Claim
          </NavLink>
          <NavLink to="/claims" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <List size={15} /> All Claims
          </NavLink>

          {isReviewer && (
            <>
              <div className="nav-section-label" style={{ marginTop: 8 }}>Review</div>
              <NavLink to="/hitl" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
                <ShieldCheck size={15} /> HITL Review
                {pendingCount > 0 && <span className="nav-badge">{pendingCount}</span>}
              </NavLink>
            </>
          )}

          <div className="nav-section-label" style={{ marginTop: 8 }}>Intelligence</div>
          <NavLink to="/analytics" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <BarChart2 size={15} /> Analytics
          </NavLink>
          <NavLink to="/policies" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <BookOpen size={15} /> Policy Admin
          </NavLink>

          <div className="nav-section-label" style={{ marginTop: 8 }}>Healthcare AI</div>
          <NavLink to="/medical" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <Stethoscope size={15} /> Medical AI
            <span style={{
              fontSize: 9, background: 'var(--brand-dim)', color: 'var(--brand)',
              padding: '1px 5px', borderRadius: 8, fontWeight: 700, marginLeft: 4,
            }}>NEW</span>
          </NavLink>
          <NavLink to="/underwriting" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <FileSearch size={15} /> Underwriting
          </NavLink>
          <NavLink to="/clinical" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <Activity size={15} /> Clinical DSS
          </NavLink>

          {isAdmin && (
            <>
              <div className="nav-section-label" style={{ marginTop: 8 }}>System</div>
              <NavLink to="/admin" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
                <Shield size={15} /> Admin Panel
              </NavLink>
            </>
          )}

          <div className="nav-section-label" style={{ marginTop: 8 }}>Support</div>
          <NavLink to="/help" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <HelpCircle size={15} /> Help & Docs
          </NavLink>
          <NavLink to="/about" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <Info size={15} /> About
          </NavLink>
        </nav>

        {/* System status */}
        <div className="sidebar-footer">
          <div className="flex gap-8" style={{ marginBottom: 4 }}>
            <Activity size={12} />
            <span>API: </span>
            <span style={{
              color: systemStatus === 'healthy'  ? 'var(--approve)'
                   : systemStatus === 'sleeping' ? 'var(--investigate)'
                   : 'var(--text-muted)',
            }}>
              {systemStatus === 'sleeping' ? 'waking…' : systemStatus}
            </span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text-disabled)', marginBottom: 2 }}>
            {IS_PRODUCTION ? '🌐 cloud' : '💻 local'} · MediSure AI v2.0
          </div>
        </div>

        <UserMenu />
        <div style={{ height: 12 }} />
      </aside>

      {/* ── MAIN CONTENT ── */}
      <main className="main-content" style={{ marginTop: bannerVisible ? 41 : 0 }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
          >
            <Routes>
              <Route path="/"             element={<Dashboard />} />
              <Route path="/submit"       element={<Submit />} />
              <Route path="/claims"       element={<Claims />} />
              <Route path="/claims/:id"   element={<ClaimDetail />} />
              <Route path="/hitl"         element={isReviewer ? <HITLReview />  : <Navigate to="/" />} />
              <Route path="/analytics"    element={<Analytics />} />
              <Route path="/policies"     element={<PolicyAdmin />} />
              <Route path="/medical"      element={<MedicalAI />} />
              <Route path="/underwriting" element={<Underwriting />} />
              <Route path="/clinical"     element={<ClinicalDecisionSupport />} />
              <Route path="/admin"        element={isAdmin ? <AdminPanel /> : <Navigate to="/" />} />
              <Route path="/help"         element={<HelpPage />} />
              <Route path="/about"        element={<AboutPage />} />
              <Route path="*"             element={<Navigate to="/" />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* ── AI SUPPORT CHATBOT ── */}
      <SupportChatbot />

    </div>
  )
}
