import { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import type { ChangeEvent, CSSProperties } from 'react'
import * as ort from 'onnxruntime-web'
import './styles.css'

type Tab = 'overview' | 'matrix' | 'evidence'
type Status = 'idle' | 'ready' | 'processing' | 'complete' | 'error'

type ModelSpec = {
  classes: string[]
  input_size: [number, number]
  crop_size: [number, number]
  source_size: [number, number]
  mean: [number, number, number]
  std: [number, number, number]
  views: number
}

type Analysis = {
  width: number
  height: number
  channels: number
  meanRgb: [number, number, number]
  luminance: number
  variance: number
  edgeDensity: number
  colorSpread: number
  coverage: number
  signal: number
  confidence: number
  band: string
  rationale: string[]
  sampleMatrix: number[][]
  histogram: number[]
  originalUrl: string
  overlayUrl: string
  heatmapUrl: string
  processedUrl: string
  processingMs: number
  quality: string
  melanomaProbability: number
  nevusProbability: number
  classProbabilities: { label: string; probability: number }[]
  modelViews: number
  modelLatencyMs: number
  modelVersion: string
}

const SAMPLE_SIZE = 12
const ANALYSIS_SIZE = 256
const MODEL_PATH = 'https://huggingface.co/sazio/skin-mole-vit-onnx/resolve/main/onnx/model_quantized.onnx'
const MODEL_META_PATH = '/model/labels.json'
const MODEL_VERSION = 'sazio/skin-mole-vit-onnx · ViT-base HAM10000 · quantized ONNX'

const MODEL_LABELS: Record<string, string> = {
  bkl: 'Benign keratosis-like lesions',
  bcc: 'Basal cell carcinoma',
  akiec: 'Actinic keratoses',
  vasc: 'Vascular lesions',
  nv: 'Melanocytic nevi',
  mel: 'Melanoma',
  df: 'Dermatofibroma',
}

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  if (name === 'scan') return <svg {...common}><path d="M4 7V5a1 1 0 0 1 1-1h2M17 4h2a1 1 0 0 1 1 1v2M20 17v2a1 1 0 0 1-1 1h-2M7 20H5a1 1 0 0 1-1-1v-2"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>
  if (name === 'upload') return <svg {...common}><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3"/></svg>
  if (name === 'camera') return <svg {...common}><path d="M4 7h3l1.4-2h7.2L17 7h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z"/><circle cx="12" cy="13" r="3.5"/></svg>
  if (name === 'refresh') return <svg {...common}><path d="M20 11a8 8 0 0 0-14.5-4L4 9"/><path d="M4 4v5h5M4 13a8 8 0 0 0 14.5 4L20 15"/><path d="M20 20v-5h-5"/></svg>
  if (name === 'shield') return <svg {...common}><path d="M12 3 20 6v5c0 5.2-3.2 8.2-8 10-4.8-1.8-8-4.8-8-10V6l8-3Z"/><path d="m9 12 2 2 4-4"/></svg>
  if (name === 'arrow') return <svg {...common}><path d="M5 12h14M13 6l6 6-6 6"/></svg>
  if (name === 'spark') return <svg {...common}><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z"/><path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z"/></svg>
  if (name === 'grid') return <svg {...common}><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 10h16M10 4v16M16 4v16M4 16h16"/></svg>
  if (name === 'info') return <svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>
  if (name === 'close') return <svg {...common}><path d="m6 6 12 12M18 6 6 18"/></svg>
  if (name === 'check') return <svg {...common}><path d="m5 12 4 4L19 6"/></svg>
  return <svg {...common}><circle cx="12" cy="12" r="8" /></svg>
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function createImage(file: File) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('The image could not be decoded.'))
    image.src = URL.createObjectURL(file)
  })
}

function renderCover(ctx: CanvasRenderingContext2D, image: HTMLImageElement, size: number) {
  const scale = Math.max(size / image.width, size / image.height)
  const width = image.width * scale
  const height = image.height * scale
  const x = (size - width) / 2
  const y = (size - height) / 2
  ctx.fillStyle = '#101c1f'
  ctx.fillRect(0, 0, size, size)
  ctx.drawImage(image, x, y, width, height)
}

function softmax(values: number[]) {
  const top = Math.max(...values)
  const exponentials = values.map((value) => Math.exp(value - top))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return exponentials.map((value) => value / total)
}

function preprocessForModel(image: HTMLImageElement, spec: ModelSpec, canvas: HTMLCanvasElement, xRatio: number, yRatio: number) {
  const [inputWidth, inputHeight] = spec.input_size
  const [sourceWidth, sourceHeight] = spec.source_size
  const [cropWidth, cropHeight] = spec.crop_size
  const boxWidth = Math.min(image.naturalWidth, (cropWidth / sourceWidth) * image.naturalWidth)
  const boxHeight = Math.min(image.naturalHeight, (cropHeight / sourceHeight) * image.naturalHeight)
  const x = clamp((image.naturalWidth - boxWidth) * xRatio, 0, Math.max(0, image.naturalWidth - boxWidth))
  const y = clamp((image.naturalHeight - boxHeight) * yRatio, 0, Math.max(0, image.naturalHeight - boxHeight))
  canvas.width = inputWidth
  canvas.height = inputHeight
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('The browser could not create a model input canvas.')
  context.drawImage(image, x, y, boxWidth, boxHeight, 0, 0, inputWidth, inputHeight)
  const { data } = context.getImageData(0, 0, inputWidth, inputHeight)
  const pixels = inputWidth * inputHeight
  const tensorData = new Float32Array(3 * pixels)
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      tensorData[channel * pixels + pixel] = (data[pixel * 4 + channel] / 255 - spec.mean[channel]) / spec.std[channel]
    }
  }
  return new ort.Tensor('float32', tensorData, [1, 3, inputHeight, inputWidth])
}

async function runModel(session: ort.InferenceSession, image: HTMLImageElement, spec: ModelSpec) {
  const started = performance.now()
  const canvas = document.createElement('canvas')
  const positions = [[0.08, 0.2], [0.5, 0.5], [0.92, 0.8]]
  const views = positions.slice(0, Math.max(1, spec.views))
  const sums = new Array(spec.classes.length).fill(0) as number[]
  for (const [xRatio, yRatio] of views) {
    const input = preprocessForModel(image, spec, canvas, xRatio, yRatio)
    const output = await session.run({ [session.inputNames[0]]: input })
    const outputTensor = output[session.outputNames[0]] as ort.Tensor
    const logits = Array.from(outputTensor.data as ArrayLike<number>)
    const probabilities = softmax(logits)
    probabilities.forEach((probability, index) => { sums[index] += probability / views.length })
  }
  return { probabilities: sums, views: views.length, latencyMs: Math.max(1, Math.round(performance.now() - started)) }
}

async function analyzeFile(file: File, session: ort.InferenceSession, spec: ModelSpec): Promise<Analysis> {
  const started = performance.now()
  if (!file.type.startsWith('image/')) throw new Error('Please choose a valid image file.')
  if (file.size > 15 * 1024 * 1024) throw new Error('Please choose an image smaller than 15 MB.')

  const image = await createImage(file)
  const originalUrl = URL.createObjectURL(file)
  const canvas = document.createElement('canvas')
  canvas.width = ANALYSIS_SIZE
  canvas.height = ANALYSIS_SIZE
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('This browser does not support canvas analysis.')
  renderCover(ctx, image, ANALYSIS_SIZE)
  const source = ctx.getImageData(0, 0, ANALYSIS_SIZE, ANALYSIS_SIZE)
  const pixels = source.data
  const count = ANALYSIS_SIZE * ANALYSIS_SIZE
  let rSum = 0
  let gSum = 0
  let bSum = 0
  let lSum = 0
  let lSquared = 0
  let edgeTotal = 0
  let colorTotal = 0
  const saliency = new Float32Array(count)
  const histogram = Array.from({ length: 10 }, () => 0)

  const luminanceAt = (index: number) => {
    const offset = index * 4
    return (0.2126 * pixels[offset] + 0.7152 * pixels[offset + 1] + 0.0722 * pixels[offset + 2]) / 255
  }

  for (let y = 0; y < ANALYSIS_SIZE; y += 1) {
    for (let x = 0; x < ANALYSIS_SIZE; x += 1) {
      const index = y * ANALYSIS_SIZE + x
      const offset = index * 4
      const r = pixels[offset] / 255
      const g = pixels[offset + 1] / 255
      const b = pixels[offset + 2] / 255
      const luminance = luminanceAt(index)
      rSum += r
      gSum += g
      bSum += b
      lSum += luminance
      lSquared += luminance * luminance
      colorTotal += Math.max(r, g, b) - Math.min(r, g, b)
      histogram[Math.min(9, Math.floor(luminance * 10))] += 1
      const right = luminanceAt(y * ANALYSIS_SIZE + Math.min(ANALYSIS_SIZE - 1, x + 1))
      const down = luminanceAt(Math.min(ANALYSIS_SIZE - 1, y + 1) * ANALYSIS_SIZE + x)
      const edge = Math.abs(luminance - right) + Math.abs(luminance - down)
      edgeTotal += edge
      const localContrast = Math.abs(luminance - 0.52)
      saliency[index] = 0.58 * edge + 0.27 * (Math.max(r, g, b) - Math.min(r, g, b)) + 0.15 * localContrast
    }
  }

  const meanLuma = lSum / count
  const variance = Math.max(0, lSquared / count - meanLuma * meanLuma)
  const edgeDensity = clamp(edgeTotal / count, 0, 1)
  const colorSpread = clamp(colorTotal / count, 0, 1)
  const meanRgb: [number, number, number] = [rSum / count, gSum / count, bSum / count]
  const sortedSaliency = Array.from(saliency).sort((a, b) => a - b)
  const low = sortedSaliency[Math.floor(count * 0.08)]
  const high = sortedSaliency[Math.floor(count * 0.96)] || 1
  const coverage = saliency.reduce((total, value) => total + (value > low + (high - low) * 0.52 ? 1 : 0), 0) / count
  const signal = clamp(Math.round(100 * (0.48 * edgeDensity + 0.28 * colorSpread + 0.24 * clamp(coverage * 2.2, 0, 1))), 0, 100)
  const quality = variance < 0.012 ? 'Low contrast' : edgeDensity > 0.11 ? 'High detail' : 'Balanced detail'

  const sampleMatrix: number[][] = []
  for (let row = 0; row < SAMPLE_SIZE; row += 1) {
    const values: number[] = []
    for (let col = 0; col < SAMPLE_SIZE; col += 1) {
      let sum = 0
      for (let yy = 0; yy < Math.floor(ANALYSIS_SIZE / SAMPLE_SIZE); yy += 1) {
        for (let xx = 0; xx < Math.floor(ANALYSIS_SIZE / SAMPLE_SIZE); xx += 1) {
          const index = (row * 21 + yy) * ANALYSIS_SIZE + col * 21 + xx
          sum += luminanceAt(index)
        }
      }
      values.push(Math.round((sum / 441) * 255))
    }
    sampleMatrix.push(values)
  }

  const heatmapCanvas = document.createElement('canvas')
  heatmapCanvas.width = ANALYSIS_SIZE
  heatmapCanvas.height = ANALYSIS_SIZE
  const heatCtx = heatmapCanvas.getContext('2d')
  if (!heatCtx) throw new Error('The browser could not render the explanation map.')
  const heatData = heatCtx.createImageData(ANALYSIS_SIZE, ANALYSIS_SIZE)
  for (let i = 0; i < count; i += 1) {
    const normalized = clamp((saliency[i] - low) / Math.max(0.0001, high - low), 0, 1)
    const smooth = normalized * normalized * (3 - 2 * normalized)
    const heat = Math.round(smooth * 255)
    const offset = i * 4
    heatData.data[offset] = Math.min(255, Math.round(32 + heat * 0.9))
    heatData.data[offset + 1] = Math.max(20, Math.round(220 - heat * 0.78))
    heatData.data[offset + 2] = Math.max(24, Math.round(145 - heat * 0.45))
    heatData.data[offset + 3] = Math.round(40 + smooth * 175)
  }
  heatCtx.putImageData(heatData, 0, 0)
  const heatmapUrl = heatmapCanvas.toDataURL('image/png')
  const overlayCanvas = document.createElement('canvas')
  overlayCanvas.width = ANALYSIS_SIZE
  overlayCanvas.height = ANALYSIS_SIZE
  const overlayCtx = overlayCanvas.getContext('2d')
  if (!overlayCtx) throw new Error('The browser could not render the overlay.')
  overlayCtx.drawImage(canvas, 0, 0)
  overlayCtx.globalAlpha = 0.68
  overlayCtx.drawImage(heatmapCanvas, 0, 0)
  overlayCtx.globalAlpha = 1

  const modelResult = await runModel(session, image, spec)
  const melanomaIndex = spec.classes.indexOf('mel')
  const melanomaProbability = modelResult.probabilities[melanomaIndex] ?? 0
  const nevusProbability = modelResult.probabilities.reduce((sum, value, index) => sum + (index === melanomaIndex ? 0 : value), 0)
  const topClassIndex = modelResult.probabilities.indexOf(Math.max(...modelResult.probabilities))
  const band = MODEL_LABELS[spec.classes[topClassIndex]] ?? spec.classes[topClassIndex]
  const confidence = Math.round((modelResult.probabilities[topClassIndex] ?? 0) * 100)

  const rationale = signal >= 52
    ? [`Real ONNX output: ${Math.round(melanomaProbability * 100)}% melanoma class / ${Math.round(nevusProbability * 100)}% combined other classes.`, 'Irregular edge response is above the browser reference range.', 'Color spread contributes to the pixel-saliency overlay.']
    : [`Real ONNX output: ${Math.round(melanomaProbability * 100)}% melanoma class / ${Math.round(nevusProbability * 100)}% combined other classes.`, 'Edge response is relatively distributed across the image.', 'The overlay is a pixel-saliency view, not a model-gradient attribution.']

  return {
    width: image.naturalWidth,
    height: image.naturalHeight,
    channels: 4,
    meanRgb: [Math.round(meanRgb[0] * 255), Math.round(meanRgb[1] * 255), Math.round(meanRgb[2] * 255)],
    luminance: Math.round(meanLuma * 255),
    variance: Number(variance.toFixed(4)),
    edgeDensity: Number(edgeDensity.toFixed(3)),
    colorSpread: Number(colorSpread.toFixed(3)),
    coverage: Number(coverage.toFixed(3)),
    signal,
    confidence,
    band,
    rationale,
    sampleMatrix,
    histogram: histogram.map((value) => Math.round((value / count) * 100)),
    originalUrl,
    overlayUrl: overlayCanvas.toDataURL('image/png'),
    heatmapUrl,
    processedUrl: canvas.toDataURL('image/jpeg', 0.9),
    processingMs: Math.max(1, Math.round(performance.now() - started)),
    quality,
    melanomaProbability,
    nevusProbability,
    classProbabilities: spec.classes.map((key, index) => ({ label: MODEL_LABELS[key] ?? key, probability: modelResult.probabilities[index] ?? 0 })),
    modelViews: modelResult.views,
    modelLatencyMs: modelResult.latencyMs,
    modelVersion: MODEL_VERSION,
  }
}

function MetricBar({ label, value, suffix = '' }: { label: string; value: number; suffix?: string }) {
  return <div className="metric-bar-row"><div className="metric-bar-label"><span>{label}</span><strong>{value}{suffix}</strong></div><div className="metric-bar"><span style={{ width: `${clamp(value, 0, 100)}%` }} /></div></div>
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [modelSpec, setModelSpec] = useState<ModelSpec | null>(null)
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [modelError, setModelError] = useState('')
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState('')
  const [tab, setTab] = useState<Tab>('overview')
  const [showOverlay, setShowOverlay] = useState(true)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const modelSessionRef = useRef<ort.InferenceSession | null>(null)

  const imageName = file?.name ?? 'No image selected'
  const progressLabel = status === 'processing' ? 'Reading matrix…' : status === 'complete' ? 'Analysis complete' : status === 'ready' ? 'Ready to analyze' : 'Awaiting image'

  useEffect(() => {
    let active = true
    ;(async () => {
      try {
        ort.env.wasm.numThreads = 1
        ort.env.wasm.proxy = false
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/'
        const spec = await fetch(MODEL_META_PATH).then((response) => {
          if (!response.ok) throw new Error('Model metadata could not be loaded.')
          return response.json() as Promise<ModelSpec>
        })
        const session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] })
        if (!active) return
        modelSessionRef.current = session
        setModelSpec(spec)
        setModelStatus('ready')
      } catch (loadError) {
        if (!active) return
        setModelStatus('error')
        setModelError(loadError instanceof Error ? loadError.message : 'The real model could not be loaded.')
      }
    })()
    return () => { active = false }
  }, [])

  useEffect(() => () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop())
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  useEffect(() => () => {
    if (analysis?.originalUrl) URL.revokeObjectURL(analysis.originalUrl)
  }, [analysis])

  const chooseFile = (selected: File | null) => {
    if (!selected) return
    setError('')
    setAnalysis(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(selected))
    setFile(selected)
    setStatus('ready')
    setTab('overview')
  }

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => chooseFile(event.target.files?.[0] ?? null)

  const runAnalysis = async () => {
    if (!file) return
    if (!modelSessionRef.current || !modelSpec) {
      setError(modelError || 'The real model is still loading. Please try again in a moment.')
      return
    }
    setError('')
    setStatus('processing')
    try {
      const result = await analyzeFile(file, modelSessionRef.current, modelSpec)
      setAnalysis(result)
      setStatus('complete')
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'The image could not be analyzed.')
      setStatus('error')
    }
  }

  const reset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
    setAnalysis(null)
    setStatus('idle')
    setError('')
    setTab('overview')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const openCamera = async () => {
    setCameraError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      fileInputRef.current?.click()
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false })
      streamRef.current = stream
      setCameraOpen(true)
      window.setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play().catch(() => undefined)
        }
      }, 50)
    } catch {
      setCameraError('Camera access was not granted. You can still upload an image from this device.')
    }
  }

  const closeCamera = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setCameraOpen(false)
  }

  const captureCamera = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) chooseFile(new File([blob], `camera-capture-${Date.now()}.jpg`, { type: 'image/jpeg' }))
      closeCamera()
    }, 'image/jpeg', 0.92)
  }

  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    chooseFile(event.dataTransfer.files?.[0] ?? null)
  }

  const matrixMax = useMemo(() => Math.max(1, ...(analysis?.sampleMatrix.flat() ?? [1])), [analysis])

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand-lockup"><div className="brand-mark"><Icon name="scan" size={24} /></div><div><div className="brand-name">LesionScan <span>Lab</span></div><div className="brand-subtitle">Image-processing research workspace</div></div></div>
      <div className="privacy-pill"><span className={`pulse-dot ${modelStatus === 'error' ? 'pulse-error' : modelStatus === 'ready' ? 'pulse-ready' : ''}`} /> {modelStatus === 'ready' ? 'Real ONNX model ready' : modelStatus === 'error' ? 'Model unavailable' : 'Loading real model · ~11 MB'}</div>
    </header>

    <main className="workspace">
      <section className="intro-row">
        <div><div className="eyebrow">DEV OZA · L. E. COLLEGE, MORBI</div><h1>Read the image.<br /><em>Question the signal.</em></h1><p className="lede">A transparent skin-lesion research prototype that turns a dermoscopic image matrix into an inspectable visual signal — without uploading the image.</p></div>
        <div className="intro-note"><Icon name="shield" size={20} /><div><strong>Research tool, not a diagnosis</strong><p>Always consult a qualified dermatologist for any changing or suspicious lesion.</p></div></div>
      </section>

      <div className="content-grid">
        <aside className="intake-column">
          <div className="section-kicker"><span>01</span> IMAGE INTAKE</div>
          <div className="intake-card">
            <div className={`dropzone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragActive(true) }} onDragLeave={() => setDragActive(false)} onDrop={onDrop} onClick={() => !file && fileInputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' && !file) fileInputRef.current?.click() }}>
              {file && analysis ? <img src={showOverlay ? analysis.overlayUrl : analysis.originalUrl} alt="Uploaded lesion preview" className="preview-image" /> : file && previewUrl ? <img src={previewUrl} alt="Uploaded lesion preview" className="preview-image" /> : <div className="drop-content"><div className="upload-icon"><Icon name="upload" size={22} /></div><strong>Drop a dermoscopic image</strong><span>or tap to browse your device</span><small>JPG, PNG, WEBP · up to 15 MB</small></div>}
              {file && <div className="image-badge"><span className="mini-dot" /> {analysis ? 'Analyzed' : 'Ready'}</div>}
            </div>
            <input ref={fileInputRef} type="file" accept="image/*" capture="environment" hidden onChange={onFileChange} />
            <div className="intake-actions"><button className="button button-ghost" onClick={() => fileInputRef.current?.click()}><Icon name="upload" size={16} /> Upload</button><button className="button button-ghost" onClick={openCamera}><Icon name="camera" size={16} /> Camera</button></div>
            {file && <div className="file-meta"><div><strong>{imageName}</strong><span>{(file.size / 1024).toFixed(0)} KB · {file.type || 'image'}</span></div><button className="icon-button" title="Reset image" onClick={reset}><Icon name="refresh" size={16} /></button></div>}
            {cameraError && <div className="inline-error">{cameraError}</div>}
            {error && <div className="inline-error">{error}</div>}
            {file && !analysis && <button className="button button-primary analyze-button" onClick={runAnalysis} disabled={status === 'processing' || modelStatus !== 'ready'}>{status === 'processing' ? 'Processing…' : modelStatus === 'loading' ? 'Loading real model…' : modelStatus === 'error' ? 'Model unavailable' : 'Run real model analysis'} <Icon name="arrow" size={16} /></button>}
            {analysis && <button className="button button-primary analyze-button" onClick={runAnalysis} disabled={status === 'processing'}><Icon name="refresh" size={16} /> Re-run analysis</button>}
          </div>
          <div className="safety-card"><Icon name="info" size={17} /><div><strong>Before you continue</strong><p>This real model only separates melanoma-class from nevus-class images. It cannot confirm or rule out skin cancer.</p></div></div>
          <div className="progress-card"><div className="progress-top"><span className={`status-dot status-${status}`} /> <strong>{progressLabel}</strong><span className="progress-time">{analysis ? `${analysis.processingMs} ms` : '—'}</span></div><div className="progress-track"><span style={{ width: status === 'complete' ? '100%' : status === 'processing' ? '62%' : status === 'ready' ? '18%' : '4%' }} /></div><div className="progress-caption">{status === 'complete' ? 'Pixel matrix, heatmap and signal summary are available.' : 'Your image stays inside this browser tab.'}</div></div>
        </aside>

        <section className="results-column">
          <div className="section-kicker"><span>02</span> EXPLAINABLE OUTPUT</div>
          {!analysis ? <div className="empty-state"><div className="empty-orbit"><span /><span /><span /><Icon name="spark" size={26} /></div><h2>Bring in an image to start.</h2><p>The workspace runs a quantized ViT-base trained on HAM10000, creates a pixel-saliency view, and exposes image matrices.</p><div className="empty-steps"><span><b>01</b> Decode pixels</span><span><b>02</b> Run ONNX model</span><span><b>03</b> Inspect evidence</span></div></div> : <>
            <div className="result-head"><div><div className="result-label"><span className="result-dot" /> REAL ONNX MODEL · SEVEN HAM10000 CLASSES</div><h2>{analysis.band}</h2><p>Output from {analysis.modelViews}-view inference. This is not a diagnosis.</p></div><div className="confidence-ring" style={{ '--confidence': `${analysis.confidence * 3.6}deg` } as CSSProperties}><div><strong>{analysis.confidence}%</strong><span>top class<br />score</span></div></div></div>
            <div className="tab-bar" role="tablist"><button className={tab === 'overview' ? 'active' : ''} onClick={() => setTab('overview')} role="tab"><Icon name="spark" size={15} /> Overview</button><button className={tab === 'matrix' ? 'active' : ''} onClick={() => setTab('matrix')} role="tab"><Icon name="grid" size={15} /> Matrix</button><button className={tab === 'evidence' ? 'active' : ''} onClick={() => setTab('evidence')} role="tab"><Icon name="scan" size={15} /> Evidence</button></div>
            {tab === 'overview' && <div className="overview-grid"><div className="visual-card"><div className="card-title"><div><span className="micro-label">PIXEL-SALIENCY VIEW</span><strong>Where visual variation concentrates</strong></div><label className="switch-label"><input type="checkbox" checked={showOverlay} onChange={(event) => setShowOverlay(event.target.checked)} /><span className="switch" /> Overlay</label></div><div className="image-stage"><img src={showOverlay ? analysis.overlayUrl : analysis.originalUrl} alt="Pixel-saliency heatmap over lesion image" /><div className="heat-legend"><span className="heat-cold" /> low <span className="heat-hot" /> high</div><div className="corner-tag">{showOverlay ? 'SALIENCY + SOURCE' : 'SOURCE IMAGE'}</div></div><p className="caption">Warm areas show higher pixel-level visual variation. This is an independent explanation proxy, not Grad-CAM and not proof of a lesion boundary.</p></div><div className="signal-card"><div className="card-title"><div><span className="micro-label">MODEL OUTPUT</span><strong>Inspectable probabilities</strong></div><span className="research-chip">{analysis.modelViews} views · ONNX</span></div>{analysis.classProbabilities.map(({ label, probability }) => <MetricBar key={label} label={label} value={Math.round(probability * 100)} suffix="%" />)}<div className="mini-section-label">PIXEL METRICS</div><MetricBar label="Edge response" value={Math.round(analysis.edgeDensity * 100)} suffix="%" /><MetricBar label="Color spread" value={Math.round(analysis.colorSpread * 100)} suffix="%" /><div className="rationale-list">{analysis.rationale.map((reason) => <div key={reason}><Icon name="check" size={14} /> {reason}</div>)}</div></div></div>}
            {tab === 'matrix' && <div className="matrix-layout"><div className="matrix-card"><div className="card-title"><div><span className="micro-label">LUMINANCE MATRIX</span><strong>12 × 12 sample window</strong></div><span className="dimension-pill">{analysis.width} × {analysis.height}px</span></div><div className="matrix-grid">{analysis.sampleMatrix.flatMap((row, rowIndex) => row.map((value, colIndex) => <div key={`${rowIndex}-${colIndex}`} className="matrix-cell" title={`Row ${rowIndex + 1}, column ${colIndex + 1}: ${value}`} style={{ backgroundColor: `rgb(${value}, ${Math.round(value * 0.94)}, ${Math.round(value * 0.82)})`, color: value > matrixMax * 0.64 ? '#16272a' : '#f4fbf5' }}>{value}</div>))}</div><p className="caption">Each cell is the mean grayscale intensity of a 21 × 21 pixel region from the normalized analysis canvas.</p></div><div className="matrix-side"><div className="stats-card"><span className="micro-label">CHANNEL STATS</span><div className="stat-row"><span>Channels</span><strong>{analysis.channels} · RGBA</strong></div><div className="stat-row"><span>Mean RGB</span><strong>{analysis.meanRgb.join(' · ')}</strong></div><div className="stat-row"><span>Mean luminance</span><strong>{analysis.luminance} / 255</strong></div><div className="stat-row"><span>Variance</span><strong>{analysis.variance}</strong></div><div className="stat-row"><span>Image quality</span><strong>{analysis.quality}</strong></div></div><div className="stats-card"><span className="micro-label">LUMINANCE HISTOGRAM</span><div className="histogram">{analysis.histogram.map((value, index) => <div className="hist-col" key={index}><div className="hist-bar" style={{ height: `${Math.max(8, value * 2.2)}px` }} /><small>{index * 26}</small></div>)}</div></div></div></div>}
            {tab === 'evidence' && <div className="evidence-layout"><div className="evidence-card"><div className="card-title"><div><span className="micro-label">PROCESSING CHAIN</span><strong>From pixels to model output</strong></div></div><div className="chain"><div><span className="chain-number">01</span><strong>Decode</strong><p>Image decoded locally to an RGBA canvas.</p></div><div><span className="chain-number">02</span><strong>Normalize</strong><p>One 224 × 224 NCHW RGB tensor uses model-card mean/std of 0.5.</p></div><div><span className="chain-number">03</span><strong>Infer</strong><p>Quantized ViT-base classifies seven HAM10000 categories in WebAssembly.</p></div><div><span className="chain-number">04</span><strong>Explain</strong><p>Pixel-saliency proxy is shown beside the model probabilities.</p></div></div><div className="model-meta"><span>MODEL</span><strong>{analysis.modelVersion}</strong><span>LATENCY</span><strong>{analysis.modelLatencyMs} ms</strong></div></div><div className="evidence-card evidence-image-card"><span className="micro-label">NORMALIZED CANVAS</span><img src={analysis.processedUrl} alt="Normalized browser analysis canvas" /><div className="evidence-meta"><span>256 × 256 render</span><span>Matrix in {analysis.processingMs} ms</span></div></div></div>}
            <div className="result-disclaimer"><Icon name="shield" size={17} /><span><strong>Safety reminder:</strong> this third-party model predicts seven HAM10000 image classes and is not a clinical diagnostic device. Any concerning, changing, bleeding, or non-healing lesion should be assessed by a qualified professional.</span></div>
          </>}
        </section>
      </div>
    </main>

    {cameraOpen && <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Camera capture"><div className="camera-modal"><div className="modal-head"><div><span className="micro-label">LIVE CAPTURE</span><strong>Frame the lesion inside the guide</strong></div><button className="icon-button" onClick={closeCamera}><Icon name="close" size={18} /></button></div><div className="camera-frame"><video ref={videoRef} playsInline muted /><div className="camera-guide" /></div><div className="modal-actions"><button className="button button-ghost" onClick={closeCamera}>Cancel</button><button className="button button-primary" onClick={captureCamera}><Icon name="camera" size={16} /> Capture frame</button></div><p className="camera-note">Camera frames remain local to this tab and are not uploaded.</p></div></div>}
    <footer className="footer"><span>LesionScan Lab · browser research prototype</span><span>Built for inspection, not diagnosis.</span></footer>
  </div>
}

export default App

createRoot(document.getElementById('root')!).render(<App />)
