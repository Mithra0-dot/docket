import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const api = async (path, options) => {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) throw new Error((await response.json()).detail || `Request failed (${response.status})`)
  return response.json()
}
const money = (value) => `$${Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

function Memo({ text }) {
  return <div className="memo">{text.split('\n').map((line, index) => {
    line = line.replace(/\*\*/g, '').replace(/^\* /, '- ')
    if (!line.trim()) return <div className="memo-gap" key={index} />
    if (/^(Suspicious signals|What precedents suggest|Recommendation)$/i.test(line.trim())) return <h3 key={index}>{line}</h3>
    if (line.startsWith('- ')) return <p className="memo-point" key={index}>{line.slice(2)}</p>
    if (/^Template-based investigation memo$/i.test(line)) return <p className="memo-kicker" key={index}>{line}</p>
    return <p key={index}>{line}</p>
  })}</div>
}

function App() {
  const [page, setPage] = useState('investigation')
  const [cases, setCases] = useState([])
  const [total, setTotal] = useState(0)
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [chat, setChat] = useState([])
  const [question, setQuestion] = useState('')
  const [cost, setCost] = useState(null)
  const [graph, setGraph] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api('/cases?limit=100').then((data) => { setCases(data.cases); setTotal(data.total); setSelectedId(data.cases[0]?.transaction_id) }).catch((reason) => setError(reason.message))
    api('/risk-overview/cost').then(setCost).catch((reason) => setError(reason.message))
    api('/risk-overview/graph?max_nodes=1').then(setGraph).catch((reason) => setError(reason.message))
  }, [])
  useEffect(() => {
    if (!selectedId) return
    setLoading(true); setError('')
    Promise.all([api(`/transactions/${selectedId}/explanation`), api(`/transactions/${selectedId}/investigate`, { method: 'POST' }), api(`/transactions/${selectedId}/chat`)]).then(([explanation, investigation, history]) => { setDetail({ ...explanation, ...investigation }); setChat(history) }).catch((reason) => setError(reason.message)).finally(() => setLoading(false))
  }, [selectedId])
  const sendQuestion = async (event) => {
    event.preventDefault(); if (!question.trim() || !selectedId) return
    const asked = question.trim(); setQuestion(''); setError('')
    try { const answer = await api(`/transactions/${selectedId}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: asked }) }); setChat((history) => [...history, { role: 'user', content: asked }, { role: 'assistant', content: answer.answer }]) } catch (reason) { setError(reason.message) }
  }
  return <div className="app-shell"><header className="topbar"><div className="brand"><span className="brand-mark">D</span><span>DOCKET</span></div><div className="system-state"><span /> LOCAL RISK SYSTEM <b>OPERATIONAL</b></div></header><div className="layout"><aside className="sidebar"><p className="eyebrow">RISK OPERATIONS</p><nav><button className={page === 'investigation' ? 'nav-active' : ''} onClick={() => setPage('investigation')}><span>01</span> Case investigation</button><button className={page === 'overview' ? 'nav-active' : ''} onClick={() => setPage('overview')}><span>02</span> Risk overview</button></nav><div className="sidebar-note"><span className="live-dot" /> Live database<br /><small>PostgreSQL + pgvector</small></div></aside><main className="content">{error && <div className="error-banner">{error}</div>}{page === 'investigation' ? <Investigation cases={cases} total={total} selectedId={selectedId} setSelectedId={setSelectedId} detail={detail} chat={chat} question={question} setQuestion={setQuestion} sendQuestion={sendQuestion} loading={loading} /> : <Overview cost={cost} graph={graph} />}</main></div></div>
}

function Investigation({ cases, total, selectedId, setSelectedId, detail, chat, question, setQuestion, sendQuestion, loading }) {
  return <><div className="page-heading"><div><p className="eyebrow">01 / CASE INVESTIGATION</p><h1>Ambiguous queue</h1><p className="subhead">Cases where model confidence and evidence require an analyst decision.</p></div><div className="metric"><strong>{total.toLocaleString()}</strong><span>OPEN CASES</span></div></div><div className="investigation-grid"><section className="queue panel"><div className="panel-heading"><h2>Priority queue</h2><span>Score descending</span></div>{cases.map((item) => <button className={`case-row ${item.transaction_id === selectedId ? 'selected' : ''}`} key={item.transaction_id} onClick={() => setSelectedId(item.transaction_id)}><span className="case-id">#{item.transaction_id}</span><span>{money(item.amount)}</span><strong>{percent(item.score)}</strong></button>)}</section><section className="case-detail">{loading && <div className="loading">Loading case evidence...</div>}{detail && !loading && <><div className="detail-header"><div><p className="eyebrow">CASE / #{detail.transaction_id}</p><h2>Transaction review</h2></div><span className="tier-badge">AMBIGUOUS</span></div><div className="stat-strip"><div><span>AMOUNT</span><strong>{money(cases.find((item) => item.transaction_id === detail.transaction_id)?.amount)}</strong></div><div><span>FRAUD PROBABILITY</span><strong className="risk-text">{percent(detail.fraud_probability)}</strong></div><div><span>SHAP CONFLICT</span><strong>{detail.shap_conflict ? 'Detected' : 'None'}</strong></div></div><div className="evidence-grid"><section className="panel chart-panel"><div className="panel-heading"><h2>Evidence signals</h2><span>SHAP contribution</span></div>{detail.top_contributors.map((item) => <div className="bar-row" key={item.feature}><div className="bar-label"><span>{item.feature}</span><b className={item.shap_value > 0 ? 'fraud' : 'legit'}>{item.shap_value > 0 ? '+' : ''}{Number(item.shap_value).toFixed(4)}</b></div><div className="bar-track"><i className={item.shap_value > 0 ? 'fraud' : 'legit'} style={{ width: `${Math.max(4, Math.min(100, Math.abs(item.shap_value) * 100))}%` }} /></div></div>)}</section><Precedents precedents={detail.precedents} /></div><section className="panel memo-panel"><div className="panel-heading"><h2>Investigation memo</h2><span className="generator">{detail.model}</span></div><Memo text={detail.memo} /></section><section className="panel chat-panel"><div className="panel-heading"><h2>Analyst follow-up</h2><span>Grounded conversation</span></div><div className="chat-history">{chat.length === 0 && <p className="empty">No questions asked for this case.</p>}{chat.map((message, index) => <div className={`message ${message.role}`} key={index}><span>{message.role === 'user' ? 'YOU' : 'DOCKET'}</span><p>{message.content}</p></div>)}</div><form onSubmit={sendQuestion}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about the evidence or recommendation..." /><button type="submit">Send question</button></form></section></>}{!detail && !loading && <div className="empty-state">Select a case to inspect its evidence.</div>}</section></div></>
}
function Precedents({ precedents }) { return <section className="panel precedent-panel"><div className="panel-heading"><h2>Retrieved precedents</h2><span>Top 5 by cosine similarity</span></div><table><thead><tr><th>CASE</th><th>SIMILARITY</th><th>OUTCOME</th></tr></thead><tbody>{precedents.map((item) => <tr key={item.transaction_id}><td>#{item.transaction_id}</td><td>{(1 - item.distance).toFixed(4)}</td><td><span className={item.fraud_label ? 'label-fraud' : 'label-clear'}>{item.fraud_label ? 'FRAUD' : 'LEGITIMATE'}</span></td></tr>)}</tbody></table></section> }
function Overview({ cost, graph }) { const ambiguous = cost?.ambiguous_tier_value; return <><div className="page-heading"><div><p className="eyebrow">02 / RISK OVERVIEW</p><h1>Decision economics</h1><p className="subhead">A transparent view of where the current policy spends attention.</p></div></div>{cost && <><div className="cost-grid">{[['Naive 0.5 threshold', cost.naive_flat_0_5], ['Current tiering', cost.current_tiering], ['Equal-recall baseline', cost.naive_at_current_tiering_recall]].map(([label, data]) => <section className="panel cost-card" key={label}><span>{label}</span><strong>{money(data.total_cost)}</strong><small>Total modeled cost</small><div>{data.false_negative_count || 0} false negatives <b>·</b> {data.false_positive_count || 0} false positives</div></section>)}</div><section className="insight"><span className="insight-tag">ANALYSIS INSIGHT</span><h2>Ambiguous-tier review costs {money(ambiguous.review_cost)} to catch {ambiguous.fraud_count_caught} fraud cases worth {money(ambiguous.fraud_amount_caught)}.</h2><p>Under the current thresholds, broad ambiguous-tier review costs more than the fraud it catches. That is a real, self-diagnosed inefficiency, not a number to hide.</p></section></>}{graph && <section className="panel ring-summary"><div><p className="eyebrow">FRAUD-RING CHECK</p><h2>{graph.total_components_at_least_minimum_size} candidate clusters analyzed</h2><p>{graph.flagged_clusters.length === 0 ? 'None met the fraud-concentration threshold, as expected for random synthetic identifiers.' : `${graph.flagged_clusters.length} clusters met the review threshold.`}</p></div><span className="ring-number">{graph.flagged_clusters.length}<small>FLAGGED</small></span></section>}</> }

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
