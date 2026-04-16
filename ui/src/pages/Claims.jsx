import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Search, Filter, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react'
import { listClaims } from '../utils/api'
import { fmtCurrency, fmtRelative, statusConfig } from '../utils/helpers'

const STATUS_OPTIONS = [
  'ALL', 'RECEIVED', 'OCR_PROCESSING', 'EXTRACTING', 'VALIDATING',
  'POLICY_CHECK', 'FRAUD_ANALYSIS', 'DECISION_PENDING',
  'HITL_REVIEW', 'APPROVED', 'REJECTED', 'INVESTIGATING', 'ERROR',
]

export default function Claims() {
  const navigate = useNavigate()
  const [claims,     setClaims]     = useState([])
  const [total,      setTotal]      = useState(0)
  const [page,       setPage]       = useState(1)
  const [loading,    setLoading]    = useState(true)
  const [statusFilter, setStatusFilter] = useState('ALL')
  const [search,     setSearch]     = useState('')
  const PAGE_SIZE = 15

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = { page, page_size: PAGE_SIZE }
      if (statusFilter !== 'ALL') params.status_filter = statusFilter
      const data = await listClaims(params)
      let claims = data.claims
      if (search) {
        const q = search.toLowerCase()
        claims = claims.filter(c =>
          c.claim_id.toLowerCase().includes(q) ||
          (c.claimant_name || '').toLowerCase().includes(q) ||
          (c.policy_type || '').toLowerCase().includes(q)
        )
      }
      setClaims(claims)
      setTotal(data.total)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [page, statusFilter, search])

  useEffect(() => { load() }, [load])

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div>
      <div className="page-header">
        <div className="flex-between" style={{ paddingBottom: 20 }}>
          <div>
            <h1 className="page-title">All Claims</h1>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
              {total} total claims
            </p>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>
            <RefreshCw size={13} className={loading ? 'spinning' : ''} /> Refresh
          </button>
        </div>
      </div>

      <div className="page-content">
        {/* FILTERS */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="flex gap-12 mb-16" style={{ flexWrap: 'wrap' }}>
          <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
            <Search size={14} style={{ position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input className="form-input" placeholder="Search claim ID, claimant…"
              style={{ paddingLeft: 33 }} value={search}
              onChange={e => { setSearch(e.target.value); setPage(1) }} />
          </div>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Filter size={14} style={{ color: 'var(--text-muted)' }} />
            <select className="form-select" style={{ width: 180 }}
              value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(1) }}>
              {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s === 'ALL' ? 'All Statuses' : s.replace(/_/g, ' ')}</option>)}
            </select>
          </div>
        </motion.div>

        {/* TABLE + MOBILE CARDS */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>

          {/* Mobile cards */}
          {!loading && claims.length > 0 && (
            <div className="mobile-claim-card-list">
              {claims.map(c => {
                const sc = statusConfig[c.status] || { label: c.status, cls: 'badge-pending', dot: '#888' }
                const score = c.fraud_score
                const lvl = score != null ? (score >= 0.75 ? 'high' : score >= 0.45 ? 'medium' : 'low') : null
                return (
                  <div key={c.claim_id} className="mobile-claim-card"
                    onClick={() => navigate(`/claims/${c.claim_id}`)}>
                    <div className="mobile-claim-card-row">
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11,
                        color: 'var(--brand)', fontWeight: 700 }}>{c.claim_id}</span>
                      <span className={`badge ${sc.cls}`}>
                        <span className="badge-dot" style={{ background: sc.dot }} />
                        {sc.label}
                      </span>
                    </div>
                    <div className="mobile-claim-card-row">
                      <span style={{ fontWeight: 600, fontSize: 14 }}>
                        {c.claimant_name || <span style={{ color: 'var(--text-muted)' }}>Processing…</span>}
                      </span>
                      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {c.policy_type || '—'}
                      </span>
                    </div>
                    <div className="mobile-claim-card-row">
                      <span style={{ fontSize: 13 }}>
                        {c.claimed_amount ? fmtCurrency(c.claimed_amount, c.currency) : '—'}
                      </span>
                      {score != null && (
                        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 700,
                          color: lvl === 'high' ? 'var(--risk-high)'
                               : lvl === 'medium' ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                          Fraud: {(score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'flex-end',
                      marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                      {fmtRelative(c.created_at)}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Desktop table */}
          <div className="table-container table-hide-mobile">
            {loading ? (
              <div className="loading-overlay"><div className="spinner" /><span>Loading claims…</span></div>
            ) : claims.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">🔍</div>
                <div className="empty-state-title">No claims found</div>
                <div className="empty-state-sub">Try adjusting your filters</div>
              </div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Claim ID</th>
                    <th>Claimant</th>
                    <th>Type</th>
                    <th>Amount</th>
                    <th>Fraud Risk</th>
                    <th>Decision</th>
                    <th>Status</th>
                    <th>Submitted</th>
                  </tr>
                </thead>
                <tbody>
                  {claims.map(c => {
                    const sc  = statusConfig[c.status] || { label: c.status, cls: 'badge-pending', dot: '#888' }
                    const score = c.fraud_score
                    const lvl = score != null ? (score >= 0.75 ? 'high' : score >= 0.45 ? 'medium' : 'low') : null

                    return (
                      <tr key={c.claim_id} onClick={() => navigate(`/claims/${c.claim_id}`)}>
                        <td><span className="claim-id">{c.claim_id}</span></td>
                        <td style={{ color: 'var(--text-primary)' }}>{c.claimant_name || <span className="text-muted">—</span>}</td>
                        <td className="text-secondary">{c.policy_type || '—'}</td>
                        <td>{c.claimed_amount ? fmtCurrency(c.claimed_amount, c.currency) : <span className="text-muted">—</span>}</td>
                        <td>
                          {score != null ? (
                            <div className={`fraud-gauge fraud-${lvl}`} style={{ maxWidth: 120 }}>
                              <div className="fraud-bar">
                                <div className="fraud-fill" style={{ width: `${score * 100}%` }} />
                              </div>
                              <span className="fraud-score-text" style={{
                                color: lvl === 'high' ? 'var(--risk-high)' : lvl === 'medium' ? 'var(--risk-medium)' : 'var(--risk-low)'
                              }}>{(score * 100).toFixed(0)}%</span>
                            </div>
                          ) : <span className="text-muted">—</span>}
                        </td>
                        <td>
                          {c.decision ? (
                            <span className={`badge badge-${c.decision.toLowerCase()}`}>{c.decision}</span>
                          ) : <span className="text-muted">—</span>}
                        </td>
                        <td>
                          <span className={`badge ${sc.cls}`}>
                            <span className="badge-dot" style={{ background: sc.dot }} />
                            {sc.label}
                          </span>
                        </td>
                        <td className="text-muted text-sm">{fmtRelative(c.created_at)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </div>{/* end desktop table */}

          {/* PAGINATION */}
          {totalPages > 1 && (
            <div className="flex-between mt-16">
              <span className="text-sm text-secondary">
                Page {page} of {totalPages} — {total} total
              </span>
              <div className="flex gap-8">
                <button className="btn btn-secondary btn-sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                  <ChevronLeft size={14} />
                </button>
                <button className="btn btn-secondary btn-sm" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
                  <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
