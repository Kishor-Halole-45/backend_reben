import { useEffect, useMemo, useRef, useState } from 'react'
import { Aperture, ArrowRight, Check, LoaderCircle, Radar, ScanSearch, Upload, X } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL
const PAGES = {
  orchestrate: { path: '', number: '00', label: 'Auto pilot', title: 'Let the image choose the model', description: 'Upload one image and ask naturally, or add a second date. The routing layer selects the right specialist automatically.', query: '', button: 'Analyze automatically' },
  grounding: { path: 'grounding', number: '01', label: 'Grounding', title: 'Find objects in orbit', description: 'Describe what you need to locate and the detector will tune itself to the image.', query: 'buildings', button: 'Find objects' },
  classification: { path: 'classification', number: '02', label: 'Classification', title: 'Read the land cover', description: 'Rank the land-cover signals present in a satellite scene.', button: 'Classify scene' },
  caption: { path: 'caption', number: '03', label: 'Captioning', title: 'Give the scene a voice', description: 'Turn the visual texture of the image into a clear natural-language description.', button: 'Write caption' },
  vqa: { path: 'vqa', number: '04', label: 'Visual Q&A', title: 'Ask the image directly', description: 'Ask a visual question and receive an answer with evidence when detection applies.', query: 'How many buildings are there?', button: 'Ask question' },
  change_detection: { path: 'change-detection', number: '05', label: 'Change detection', title: 'See what moved over time', description: 'Compare two dates of the same area and get a visual change report.', button: 'Compare dates' },
}

function pageFromHash() {
  const path = window.location.hash.replace(/^#\/?/, '')
  return Object.keys(PAGES).find((key) => PAGES[key].path === path) || 'orchestrate'
}

function App() {
  const [page, setPage] = useState(pageFromHash)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [beforeFile, setBeforeFile] = useState(null)
  const [afterFile, setAfterFile] = useState(null)
  const [beforePreview, setBeforePreview] = useState('')
  const [afterPreview, setAfterPreview] = useState('')
  const [query, setQuery] = useState(PAGES[page].query || '')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const inputRef = useRef(null)
  const beforeRef = useRef(null)
  const afterRef = useRef(null)
  const config = PAGES[page]
  const detections = result?.detections || []
  const isCompare = page === 'change_detection'
  const isOrchestrator = page === 'orchestrate'
  const imageMeta = result?.image

  useEffect(() => { const sync = () => setPage(pageFromHash()); window.addEventListener('hashchange', sync); return () => window.removeEventListener('hashchange', sync) }, [])
  useEffect(() => { setQuery(PAGES[page].query || ''); setResult(null); setError('') }, [page])

  const summary = useMemo(() => {
    if (!result) return 'Waiting for an image'
    if (page === 'grounding') return detections.length ? `${detections.length} detected ${detections.length === 1 ? 'object' : 'objects'}` : 'No confident objects found'
    if (page === 'change_detection' || result?.mode === 'change_detection') return result.changes?.length ? `${result.changes.length} changed regions` : 'No significant change'
    return 'Analysis complete'
  }, [result, page, detections.length])

  function chooseFile(nextFile, side = 'single') {
    if (!nextFile || !nextFile.type.startsWith('image/')) return
    const objectUrl = URL.createObjectURL(nextFile)
    if (side === 'before') { setBeforeFile(nextFile); setBeforePreview(objectUrl) } else if (side === 'after') { setAfterFile(nextFile); setAfterPreview(objectUrl) } else { setFile(nextFile); setPreview(objectUrl) }
    setResult(null); setError('')
  }

  async function runModel(event) {
    event.preventDefault()
    if (isCompare ? (!beforeFile || !afterFile) : (isOrchestrator ? (!file && (!beforeFile || !afterFile)) : (!file || (['grounding', 'vqa'].includes(page) && !query.trim())))) return
    setStatus('loading'); setError('')
    const body = new FormData()
    body.append('file', file || beforeFile); body.append('mode', isOrchestrator ? 'orchestrate' : page); body.append('query', query)
    if (isCompare || isOrchestrator) { if (beforeFile) body.append('before_file', beforeFile); if (afterFile) body.append('after_file', afterFile) }
    try { const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Model request failed'); setResult(data); setStatus('success') } catch (requestError) { setError(requestError.message); setStatus('error') }
  }

  function reset() { setFile(null); setPreview(''); setBeforeFile(null); setAfterFile(null); setBeforePreview(''); setAfterPreview(''); setResult(null); setError(''); setStatus('idle'); if (inputRef.current) inputRef.current.value = ''; if (beforeRef.current) beforeRef.current.value = ''; if (afterRef.current) afterRef.current.value = '' }
  const ready = isCompare ? beforeFile && afterFile : isOrchestrator ? (file || (beforeFile && afterFile)) : file

  return <main className="app-shell">
    <header className="topbar"><a className="brand" href="#/grounding"><span className="brand-mark"><Radar size={18} /></span><span>FIELDNOTE</span></a><div className="topbar-status"><span className="status-dot" /> satellite intelligence <span className="slash">/</span> five focused models</div><button className="icon-button" onClick={reset} title="Reset workspace" aria-label="Reset workspace"><X size={17} /></button></header>
    <nav className="page-nav" aria-label="Model pages">{Object.entries(PAGES).map(([key, value]) => <a className={page === key ? 'active' : ''} href={value.path ? `#/${value.path}` : '#/'} key={key}><span>{value.number}</span>{value.label}</a>)}</nav>
    <section className="page-intro"><div className="eyebrow"><span className="eyebrow-line" /> {config.number} / {config.label.toUpperCase()}</div><h1>{config.title}</h1><p>{config.description}</p></section>
    <section className={`workspace page-${page}`}>
      <aside className="control-panel"><div className="panel-label"><span>INPUT</span> {isCompare ? 'TWO DATES' : 'IMAGE'}</div>
        {isCompare ? <div className="change-uploads"><UploadCard label="Earlier image" file={beforeFile} onClick={() => beforeRef.current?.click()} /><UploadCard label="Later image" file={afterFile} onClick={() => afterRef.current?.click()} /></div> : isOrchestrator ? <div className="change-uploads"><UploadCard label="Primary image" file={file} onClick={() => inputRef.current?.click()} /><UploadCard label="Optional earlier date" file={beforeFile} onClick={() => beforeRef.current?.click()} /><UploadCard label="Optional later date" file={afterFile} onClick={() => afterRef.current?.click()} /></div> : <UploadCard label="Upload satellite image" file={file} onClick={() => inputRef.current?.click()} />}
        <input ref={inputRef} hidden type="file" accept="image/*" onChange={(event) => chooseFile(event.target.files[0])} /><input ref={beforeRef} hidden type="file" accept="image/*" onChange={(event) => chooseFile(event.target.files[0], 'before')} /><input ref={afterRef} hidden type="file" accept="image/*" onChange={(event) => chooseFile(event.target.files[0], 'after')} />
        <div className="panel-label query-label"><span>REQUEST</span> {page === 'vqa' ? 'QUESTION' : page === 'grounding' ? 'OBJECTS TO FIND' : isOrchestrator ? 'OPTIONAL QUESTION' : 'OPTIONAL'}</div>
        <form onSubmit={runModel}><div className="query-box"><span className="query-mark">{page === 'vqa' ? '?' : page === 'grounding' ? '⌖' : '✦'}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={isOrchestrator ? 'e.g. How many buildings are visible?' : page === 'grounding' ? 'buildings, roads, water' : 'No request needed'} disabled={!['grounding', 'vqa', 'orchestrate'].includes(page)} /><kbd>↵</kbd></div><button className="analyze-button" disabled={!ready || (['grounding', 'vqa'].includes(page) && !query.trim()) || status === 'loading'}>{status === 'loading' ? <LoaderCircle className="spin" size={18} /> : <ScanSearch size={18} />} {status === 'loading' ? 'Analyzing...' : config.button}</button></form>
        {error && <div className="error-message">{error}</div>}<div className="model-note"><span className="model-index">{config.number}</span><div><strong>{config.label}</strong><span>{isCompare ? 'Aligned visual comparison' : 'Specialist vision model'}</span></div></div>
      </aside>
      <section className="canvas-panel"><div className="canvas-heading"><div><div className="panel-label"><span>OUTPUT</span> {config.label.toUpperCase()}</div><h2>{summary}</h2></div><div className="canvas-meta">{imageMeta ? `${imageMeta.width} × ${imageMeta.height} px` : ready ? 'Ready to analyze' : 'No input'}</div></div>
        <ResultVisual page={page} preview={preview} beforePreview={beforePreview} afterPreview={afterPreview} result={result} detections={detections} imageMeta={imageMeta} />
        {result?.selected_model && <div className="response-card route-card"><small>AUTO ROUTING</small><strong>{PAGES[result.selected_model]?.label || result.selected_model}</strong><span>{result.input_image_count || 1} input image{result.input_image_count === 1 ? '' : 's'} · {result.routing_reason}</span></div>}{result?.explanation && <div className="response-card"><small>ANALYSIS</small><strong>{result.explanation}</strong></div>}{result?.answer && <div className="response-card"><small>{page === 'vqa' ? 'ANSWER' : 'CAPTION'}</small><strong>{result.answer}</strong></div>}{result?.interpretation && <div className="response-card"><small>EVIDENCE</small><strong>{result.interpretation}</strong></div>}{result?.predictions && <div className="response-card prediction-list">{result.predictions.map((prediction) => <div key={prediction.label}><span>{prediction.label}</span><b>{Math.round(prediction.score * 100)}%</b></div>)}</div>}
        <div className="canvas-footer"><span><span className="legend-box" /> {result ? `Model: ${result.model || config.label}` : 'Awaiting analysis'}</span><span>{result ? <><Check size={14} /> Complete</> : <><ArrowRight size={14} /> Submit when ready</>}</span></div>
      </section>
    </section>
  </main>
}

function UploadCard({ label, file, onClick }) { return <button className="upload-card" onClick={onClick}><span className="upload-icon"><Upload size={19} /></span><strong>{file ? file.name : label}</strong><span>{file ? 'Click to replace' : 'PNG · JPG · TIFF · WEBP'}</span></button> }

function ResultVisual({ page, preview, beforePreview, afterPreview, result, detections, imageMeta }) {
  if (page === 'change_detection' || result?.mode === 'change_detection') return <div className={`image-stage comparison-grid ${beforePreview || afterPreview ? 'has-image' : ''}`}>{[['Earlier', beforePreview], ['Later', afterPreview]].map(([label, source]) => <div className="comparison-image" key={label}><small>{label}</small>{source ? <div className="image-wrap"><img src={source} alt={`${label} satellite scene`} />{result?.changes?.map((change, index) => <div className="box change-box" key={`${label}-${index}`} style={{ left: `${change.x / result.image.width * 100}%`, top: `${change.y / result.image.height * 100}%`, width: `${change.width / result.image.width * 100}%`, height: `${change.height / result.image.height * 100}%` }} />)}</div> : <div className="comparison-empty">Upload image</div>}</div>)}</div>
  return <div className={`image-stage ${preview ? 'has-image' : ''}`}>{preview ? <div className="image-wrap"><img src={preview} alt="Uploaded satellite scene" />{(page === 'grounding' || result?.selected_model === 'grounding') && imageMeta && detections.map((detection, index) => <div className="box" key={`${detection.label}-${index}`} style={{ left: `${detection.box.x / imageMeta.width * 100}%`, top: `${detection.box.y / imageMeta.height * 100}%`, width: `${detection.box.width / imageMeta.width * 100}%`, height: `${detection.box.height / imageMeta.height * 100}%` }}><span>{detection.label} <b>{Math.round(detection.score * 100)}%</b></span></div>)}</div> : <div className="empty-state"><div className="empty-orbit"><Aperture size={28} /></div><strong>Your result will appear here</strong><span>Upload an image to begin.</span></div>}</div>
}

export default App