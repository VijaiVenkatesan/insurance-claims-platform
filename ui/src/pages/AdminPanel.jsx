import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Shield, Trash2, Edit3, RefreshCw, Search,
  AlertTriangle, CheckCircle, XCircle, Eye,
  Download, Filter, X, Save, RotateCcw
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import {
  adminListClaims, adminUpdateClaim, adminDeleteClaim,
  adminReprocessClaim, adminGetLogs, adminGetStats
} from '../utils/api'
import { fmtCurrency, fmtDate, statusConfig } from '../utils/helpers'

const TABS = ['Claims', 'Audit Logs', 'Statistics']
const STATUSES = ['ALL','RECEIVED','OCR_PROCESSING','EXTRACTING','VALIDATING',
  'POLICY_CHECK','FRAUD_ANALYSIS','DECISION_PENDING','HITL_REVIEW',
  'APPROVED','REJECTED','INVESTIGATING','ERROR']

// ── EDIT MODAL ──────────────────────────────────────────────────────
function EditModal({ claim, onClose, onSave }) {
  const [form, setForm] = useState({
    status:               claim.status || '',
    fraud_score:          claim.fraud_score ?? '',
    decision:             claim.decision || '',
    decision_explanation: claim.decision_explanation || '',
    approved_amount:      claim.approved_amount ?? '',
    claimant_name:        claim.claimant_name || '',
  })
  const [saving, setSaving] = useState(false)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSave = async () => {
    setSaving(true)
    try {
      const payload = Object.fromEntries(
        Object.entries(form).filter(([, v]) => v !== '' && v !== null)
      )
      if (payload.fraud_score !== undefined)
        payload.fraud_score = parseFloat(payload.fraud_score)
      if (payload.approved_amount !== undefined)
        payload.approved_amount = parseFloat(payload.approved_amount)

      await adminUpdateClaim(claim.id, payload)
      toast.success('Claim updated')
      onSave()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 300, padding: 20,
      }}
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-xl)', width: '100%', maxWidth: 560,
          maxHeight: '90vh', overflow: 'auto',
        }}
      >
        <div style={{
          padding: '20px 24px', borderBottom: '1px solid var(--border-subtle)',
          display: 'flex', alignItems: 'center', gap: 10
        }}>
          <Edit3 size={16} color="var(--brand)" />
          <div style={{ flex: 1, fontWeight: 700, fontSize: 15 }}>
            Edit Claim
          </div>
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)',
            color: 'var(--text-muted)' }}>{claim.id}</span>
          <button className="btn btn-secondary btn-sm btn-icon" onClick={onClose}>
            <X size={14} />
          </button>
        </div>

        <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
          {[
            { label: 'Claimant Name', key: 'claimant_name', type: 'text' },
            { label: 'Status', key: 'status', type: 'select',
              options: STATUSES.filter(s => s !== 'ALL') },
            { label: 'Decision', key: 'decision', type: 'select',
              options: ['', 'APPROVE', 'REJECT', 'INVESTIGATE', 'PENDING'] },
            { label: 'Fraud Score (0.0 – 1.0)', key: 'fraud_score', type: 'number',
              step: '0.01', min: '0', max: '1' },
            { label: 'Approved Amount (₹)', key: 'approved_amount',
              type: 'number', min: '0' },
            { label: 'Decision Explanation', key: 'decision_explanation',
              type: 'textarea' },
          ].map(f => (
            <div key={f.key} className="form-group">
              <label className="form-label">{f.label}</label>
              {f.type === 'select' ? (
                <select className="form-select" value={form[f.key]}
                  onChange={e => set(f.key, e.target.value)}>
                  {f.options.map(o => (
                    <option key={o} value={o}>{o || '— no change —'}</option>
                  ))}
                </select>
              ) : f.type === 'textarea' ? (
                <textarea className="form-textarea" rows={3}
                  value={form[f.key]}
                  onChange={e => set(f.key, e.target.value)} />
              ) : (
                <input className="form-input" type={f.type}
                  step={f.step} min={f.min} max={f.max}
                  value={form[f.key]}
                  onChange={e => set(f.key, e.target.value)} />
              )}
            </div>
          ))}

          <div style={{ display: 'flex', gap: 10, paddingTop: 8 }}>
            <button className="btn btn-secondary" style={{ flex: 1 }}
              onClick={onClose} disabled={saving}>Cancel</button>
            <button className="btn btn-primary" style={{ flex: 2 }}
              onClick={handleSave} disabled={saving}>
              {saving
                ? <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Saving…</>
                : <><Save size={14} /> Save Changes</>}
            </button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

// ── CONFIRM DELETE MODAL ────────────────────────────────────────────
function DeleteConfirm({ claimId, onClose, onConfirm }) {
  const [deleting, setDeleting] = useState(false)
  const handle = async () => {
    setDeleting(true)
    try {
      await adminDeleteClaim(claimId)
      toast.success(`Claim ${claimId} deleted`)
      onConfirm()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setDeleting(false)
    }
  }
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 300
      }}
      onClick={e => e.target === e.currentTarget && onClose()}>
      <motion.div initial={{ scale: 0.95 }} animate={{ scale: 1 }}
        style={{
          background: 'var(--bg-surface)', borderRadius: 'var(--radius-xl)',
          border: '1px solid var(--border-default)', padding: 28, maxWidth: 400, width: '90%'
        }}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          <AlertTriangle size={24} color="var(--reject)" style={{ flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Delete Claim?</div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              This permanently deletes <span style={{ fontFamily: 'var(--font-mono)',
                color: 'var(--reject)' }}>{claimId}</span> and all related data.
              This cannot be undone.
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-secondary" style={{ flex: 1 }}
            onClick={onClose} disabled={deleting}>Cancel</button>
          <button className="btn btn-reject" style={{ flex: 1 }}
            onClick={handle} disabled={deleting}>
            {deleting ? 'Deleting…' : <><Trash2 size={14} /> Delete</>}
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}

// ── CLAIMS TAB ──────────────────────────────────────────────────────
function ClaimsTab() {
  const navigate = useNavigate()
  const [data,     setData]     = useState({ claims: [], total: 0 })
  const [loading,  setLoading]  = useState(true)
  const [page,     setPage]     = useState(1)
  const [status,   setStatus]   = useState('ALL')
  const [search,   setSearch]   = useState('')
  const [editing,  setEditing]  = useState(null)
  const [deleting, setDeleting] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = { page, page_size: 25 }
      if (status !== 'ALL') params.status = status
      if (search) params.search = search
      const d = await adminListClaims(params)
      setData(d)
    } catch (e) { toast.error(e.message) }
    finally { setLoading(false) }
  }, [page, status, search])

  useEffect(() => { load() }, [load])

  const handleReprocess = async (id) => {
    try {
      await adminReprocessClaim(id)
      toast.success('Reprocessing started')
      load()
    } catch (e) { toast.error(e.message) }
  }

  return (
    <div>
      {/* Filters */}
      <div className="flex gap-12 mb-16" style={{ flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
          <Search size={14} style={{
            position: 'absolute', left: 11, top: '50%',
            transform: 'translateY(-50%)', color: 'var(--text-muted)'
          }} />
          <input className="form-input" placeholder="Search ID, claimant, policy…"
            style={{ paddingLeft: 33 }} value={search}
            onChange={e => { setSearch(e.target.value); setPage(1) }} />
        </div>
        <select className="form-select" style={{ width: 200 }}
          value={status} onChange={e => { setStatus(e.target.value); setPage(1) }}>
          {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {/* Claims list — card on mobile, table on desktop */}
      {loading ? (
        <div className="loading-overlay"><div className="spinner" /></div>
      ) : data.claims.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📋</div>
          <div className="empty-state-title">No claims found</div>
        </div>
      ) : (
        <>
          {/* ── MOBILE CARDS ── */}
          <div className="mobile-claim-card-list">
            {data.claims.map(c => {
              const sc = statusConfig[c.status] || { label: c.status, cls: 'badge-pending', dot: '#888' }
              const score = c.fraud_score
              const lvl = score != null ? (score >= 0.75 ? 'high' : score >= 0.45 ? 'medium' : 'low') : null
              return (
                <div key={c.id} className="mobile-claim-card"
                  onClick={() => navigate(`/claims/${c.id}`)}>
                  <div className="mobile-claim-card-row">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11,
                      color: 'var(--brand)', fontWeight: 700 }}>{c.id}</span>
                    <span className={`badge ${sc.cls}`}>
                      <span className="badge-dot" style={{ background: sc.dot }} />
                      {sc.label}
                    </span>
                  </div>
                  <div className="mobile-claim-card-row">
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{c.claimant_name || '—'}</span>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                      {c.insurance_type || '—'}
                    </span>
                  </div>
                  <div className="mobile-claim-card-row">
                    <span style={{ fontSize: 13 }}>
                      {c.claimed_amount ? fmtCurrency(c.claimed_amount, c.currency) : '—'}
                    </span>
                    {score != null && (
                      <span style={{
                        fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 700,
                        color: lvl === 'high' ? 'var(--risk-high)'
                             : lvl === 'medium' ? 'var(--risk-medium)' : 'var(--risk-low)'
                      }}>Fraud: {(score * 100).toFixed(0)}%</span>
                    )}
                  </div>
                  {/* Actions row — stop propagation to prevent card click */}
                  <div style={{ display: 'flex', gap: 8, marginTop: 10,
                    paddingTop: 10, borderTop: '1px solid var(--border-subtle)' }}
                    onClick={e => e.stopPropagation()}>
                    <button className="btn btn-secondary btn-sm" style={{ flex: 1 }}
                      onClick={() => navigate(`/claims/${c.id}`)}>
                      <Eye size={13} /> View
                    </button>
                    <button className="btn btn-secondary btn-sm" style={{ flex: 1 }}
                      onClick={() => setEditing(c)}>
                      <Edit3 size={13} /> Edit
                    </button>
                    <button className="btn btn-secondary btn-sm" style={{ flex: 1 }}
                      onClick={() => handleReprocess(c.id)}>
                      <RotateCcw size={13} /> Retry
                    </button>
                    <button className="btn btn-reject btn-sm btn-icon"
                      onClick={() => setDeleting(c.id)}>
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>

          {/* ── DESKTOP TABLE ── */}
          <div className="table-container table-hide-mobile">
            <table>
              <thead>
                <tr>
                  <th>Claim ID</th>
                  <th>Claimant</th>
                  <th>Type</th>
                  <th>Amount</th>
                  <th>Fraud</th>
                  <th>Status</th>
                  <th>Decision</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.claims.map(c => {
                  const sc = statusConfig[c.status] || { label: c.status, cls: 'badge-pending', dot: '#888' }
                  const score = c.fraud_score
                  const lvl = score != null ? (score >= 0.75 ? 'high' : score >= 0.45 ? 'medium' : 'low') : null
                  return (
                    <tr key={c.id}>
                      <td>
                        <span className="claim-id" style={{ cursor: 'pointer', color: 'var(--brand)' }}
                          onClick={() => navigate(`/claims/${c.id}`)}>
                          {c.id}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-primary)' }}>{c.claimant_name || '—'}</td>
                      <td className="text-secondary">{c.insurance_type || '—'}</td>
                      <td>{c.claimed_amount ? fmtCurrency(c.claimed_amount, c.currency) : '—'}</td>
                      <td>
                        {score != null ? (
                          <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 700,
                            color: lvl === 'high' ? 'var(--risk-high)'
                                 : lvl === 'medium' ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                            {(score * 100).toFixed(0)}%
                          </span>
                        ) : '—'}
                      </td>
                      <td>
                        <span className={`badge ${sc.cls}`}>
                          <span className="badge-dot" style={{ background: sc.dot }} />
                          {sc.label}
                        </span>
                      </td>
                      <td>
                        {c.decision
                          ? <span className={`badge badge-${c.decision.toLowerCase()}`}>{c.decision}</span>
                          : '—'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 5, justifyContent: 'flex-end' }}>
                          <button className="btn btn-secondary btn-sm btn-icon" title="View"
                            onClick={() => navigate(`/claims/${c.id}`)}>
                            <Eye size={13} />
                          </button>
                          <button className="btn btn-secondary btn-sm btn-icon" title="Edit"
                            onClick={() => setEditing(c)}>
                            <Edit3 size={13} />
                          </button>
                          <button className="btn btn-secondary btn-sm btn-icon" title="Reprocess"
                            onClick={() => handleReprocess(c.id)}>
                            <RotateCcw size={13} />
                          </button>
                          <button className="btn btn-reject btn-sm btn-icon" title="Delete"
                            onClick={() => setDeleting(c.id)}>
                            <Trash2 size={13} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Pagination */}
      {data.total > 25 && (
        <div className="flex-between mt-16">
          <span className="text-sm text-secondary">
            {data.total} total · Page {page} of {data.pages}
          </span>
          <div className="flex gap-8">
            <button className="btn btn-secondary btn-sm"
              disabled={page <= 1} onClick={() => setPage(p => p - 1)}>←</button>
            <button className="btn btn-secondary btn-sm"
              disabled={page >= data.pages} onClick={() => setPage(p => p + 1)}>→</button>
          </div>
        </div>
      )}

      {/* Modals */}
      <AnimatePresence>
        {editing && (
          <EditModal claim={editing} onClose={() => setEditing(null)}
            onSave={() => { setEditing(null); load() }} />
        )}
        {deleting && (
          <DeleteConfirm claimId={deleting} onClose={() => setDeleting(null)}
            onConfirm={() => { setDeleting(null); load() }} />
        )}
      </AnimatePresence>
    </div>
  )
}

// ── LOGS TAB ────────────────────────────────────────────────────────
function LogsTab() {
  const [logs,    setLogs]    = useState([])
  const [total,   setTotal]   = useState(0)
  const [loading, setLoading] = useState(true)
  const [page,    setPage]    = useState(1)
  const [filter,  setFilter]  = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = { page, page_size: 50 }
      if (filter) params.event_type = filter
      const d = await adminGetLogs(params)
      setLogs(d.logs)
      setTotal(d.total)
    } catch (e) { toast.error(e.message) }
    finally { setLoading(false) }
  }, [page, filter])

  useEffect(() => { load() }, [load])

  const eventColor = (type) => {
    if (type?.includes('APPROVE')) return 'var(--approve)'
    if (type?.includes('REJECT'))  return 'var(--reject)'
    if (type?.includes('ERROR'))   return 'var(--risk-high)'
    if (type?.includes('ADMIN'))   return 'var(--investigate)'
    return 'var(--brand)'
  }

  return (
    <div>
      <div className="flex gap-12 mb-16">
        <input className="form-input" placeholder="Filter by event type (e.g. CLAIM_SUBMITTED)…"
          style={{ flex: 1 }} value={filter}
          onChange={e => { setFilter(e.target.value); setPage(1) }} />
        <button className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>
          <RefreshCw size={13} />
        </button>
      </div>

      <div className="card">
        <div className="card-body" style={{ padding: 0 }}>
          {loading ? (
            <div className="loading-overlay"><div className="spinner" /></div>
          ) : (
            <div className="agent-trace" style={{ padding: '8px 20px' }}>
              {logs.length === 0
                ? <div className="empty-state">No logs found</div>
                : logs.map((log, i) => (
                  <div key={log.id} className="trace-item">
                    <div className="trace-dot" style={{ background: eventColor(log.event_type) }} />
                    <div className="trace-meta">
                      <div className="trace-agent">{log.event_type}</div>
                      <div className="trace-detail">
                        Claim: <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                          {log.claim_id}
                        </span>
                        {' · '}Actor: <strong>{log.actor}</strong>
                        {log.details && Object.keys(log.details).length > 0 && (
                          <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
                            {Object.entries(log.details).slice(0, 3)
                              .map(([k, v]) => `${k}: ${String(v).slice(0, 30)}`)
                              .join(' · ')}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="trace-time">{fmtDate(log.timestamp)}</div>
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>

      {total > 50 && (
        <div className="flex-between mt-16">
          <span className="text-sm text-secondary">{total} total logs</span>
          <div className="flex gap-8">
            <button className="btn btn-secondary btn-sm"
              disabled={page <= 1} onClick={() => setPage(p => p - 1)}>←</button>
            <button className="btn btn-secondary btn-sm"
              disabled={page * 50 >= total} onClick={() => setPage(p => p + 1)}>→</button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── STATS TAB ───────────────────────────────────────────────────────
function StatsTab() {
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    adminGetStats()
      .then(setStats)
      .catch(e => toast.error(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading-overlay"><div className="spinner" /></div>
  if (!stats)  return <div className="empty-state">No statistics available</div>

  const { claims } = stats

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      {/* By Status */}
      <div className="card">
        <div className="card-header"><span className="card-title">Claims by Status</span></div>
        <div className="card-body">
          {Object.entries(claims.by_status || {}).map(([status, count]) => {
            const sc = statusConfig[status] || { dot: '#888' }
            return (
              <div key={status} className="flex-between"
                style={{ padding: '7px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                <div className="flex gap-8">
                  <div style={{ width: 8, height: 8, borderRadius: '50%',
                    background: sc.dot, marginTop: 5 }} />
                  <span style={{ fontSize: 13 }}>{status}</span>
                </div>
                <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)',
                  fontSize: 14 }}>{count}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* By Type */}
      <div className="card">
        <div className="card-header"><span className="card-title">Claims by Type</span></div>
        <div className="card-body">
          {Object.entries(claims.by_type || {}).map(([type, count]) => (
            <div key={type} className="flex-between"
              style={{ padding: '7px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: 13 }}>{type}</span>
              <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 14 }}>{count}</span>
            </div>
          ))}
          {Object.keys(claims.by_type || {}).length === 0 && (
            <div className="text-muted text-sm">No data yet</div>
          )}
        </div>
      </div>

      {/* Financial */}
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="card-header"><span className="card-title">Financial Summary</span></div>
        <div className="card-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 20 }}>
          {[
            { label: 'Total Claims', value: claims.total, color: 'var(--brand)' },
            { label: 'Avg Fraud Score', value: `${(claims.avg_fraud_score * 100).toFixed(1)}%`,
              color: 'var(--investigate)' },
            { label: 'Total Claimed', value: fmtCurrency(claims.total_claimed_inr), color: 'var(--text-primary)' },
            { label: 'Total Approved', value: fmtCurrency(claims.total_approved_inr), color: 'var(--approve)' },
          ].map(s => (
            <div key={s.label} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 800, color: s.color,
                fontFamily: 'var(--font-mono)', letterSpacing: -1 }}>{s.value}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── MAIN PAGE ────────────────────────────────────────────────────────
export default function AdminPanel() {
  const [tab, setTab] = useState(0)

  return (
    <div>
      <div className="page-header">
        <div style={{ paddingBottom: 0 }}>
          <div className="flex gap-8" style={{ marginBottom: 6 }}>
            <Shield size={18} color="var(--brand)" />
            <h1 className="page-title" style={{ margin: 0 }}>Admin Panel</h1>
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
            Full system control — edit, delete, reprocess claims, view logs and statistics
          </p>
          {/* Tabs */}
          <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border-subtle)' }}>
            {TABS.map((t, i) => (
              <button key={t} onClick={() => setTab(i)}
                style={{
                  padding: '10px 20px', background: 'none', border: 'none',
                  borderBottom: tab === i ? '2px solid var(--brand)' : '2px solid transparent',
                  color: tab === i ? 'var(--brand)' : 'var(--text-secondary)',
                  fontWeight: tab === i ? 600 : 400, cursor: 'pointer',
                  fontSize: 13, fontFamily: 'var(--font-sans)',
                  transition: 'all var(--transition)',
                }}>
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="page-content">
        <AnimatePresence mode="wait">
          <motion.div key={tab}
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
            {tab === 0 && <ClaimsTab />}
            {tab === 1 && <LogsTab />}
            {tab === 2 && <StatsTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
