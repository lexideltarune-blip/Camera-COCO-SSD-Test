import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { Camera, Check, CircleAlert, Cpu, Gauge, LockKeyhole, Play, RotateCcw, Square } from 'lucide-react';

type AppState = 'idle' | 'model-loading' | 'camera-starting' | 'camera-live' | 'running' | 'error';
type CameraError = { title: string; detail: string; action: string };
type Detection = cocoSsd.DetectedObject;
type PositionedDetection = Detection & {
  display: { left: number; top: number; width: number; height: number };
};

const SCAN_INTERVAL = 350;
const INFERENCE_LONG_EDGE = 640;
const RELATION_NEAR_THRESHOLD = 0.08;

function describeSpatialRelationship(detections: Detection[], sourceWidth: number, sourceHeight: number) {
  if (detections.length !== 2 || !sourceWidth || !sourceHeight) return '';

  const [first, second] = detections;
  const firstCenter = {
    x: first.bbox[0] + first.bbox[2] / 2,
    y: first.bbox[1] + first.bbox[3] / 2,
  };
  const secondCenter = {
    x: second.bbox[0] + second.bbox[2] / 2,
    y: second.bbox[1] + second.bbox[3] / 2,
  };
  const horizontalDistance = Math.abs(firstCenter.x - secondCenter.x) / sourceWidth;
  const verticalDistance = Math.abs(firstCenter.y - secondCenter.y) / sourceHeight;

  if (horizontalDistance < RELATION_NEAR_THRESHOLD && verticalDistance < RELATION_NEAR_THRESHOLD) {
    return `The ${first.class} is near the ${second.class}`;
  }
  if (verticalDistance >= horizontalDistance) {
    return firstCenter.y < secondCenter.y
      ? `The ${first.class} is on the top of the ${second.class}`
      : `The ${second.class} is on the top of the ${first.class}`;
  }
  return firstCenter.x < secondCenter.x
    ? `The ${first.class} is on the left of the ${second.class}`
    : `The ${second.class} is on the left of the ${first.class}`;
}

function formatTime(value: number | null) {
  if (!value) return '—';
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
  }).format(value);
}

function readableCameraError(error: unknown): CameraError {
  const name = error instanceof DOMException ? error.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return {
      title: 'Camera permission was not granted',
      detail: 'Allow camera access in your browser settings, then try again. This test only asks for access when you press the control below.',
      action: 'Check browser permissions',
    };
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return {
      title: 'No rear camera was found',
      detail: 'Connect a camera or switch to a device with a rear-facing camera. The detector is configured to prefer the environment-facing lens.',
      action: 'Check camera hardware',
    };
  }
  if (name === 'NotReadableError' || name === 'AbortError') {
    return {
      title: 'Camera is busy',
      detail: 'Another app or browser tab may be using the camera. Close it and try again.',
      action: 'Release camera and retry',
    };
  }
  if (!window.isSecureContext) {
    return {
      title: 'Secure context required',
      detail: 'Camera access needs HTTPS or localhost. Open this test from a secure origin, then try again.',
      action: 'Open a secure URL',
    };
  }
  return {
    title: 'Camera could not start',
    detail: 'The browser did not provide a usable camera stream. Check the device and browser permissions, then try again.',
    action: 'Check device and retry',
  };
}

function StatusDot({ state }: { state: AppState }) {
  const active = state === 'running';
  const loading = state === 'model-loading' || state === 'camera-starting';
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-2 w-2 rounded-full ${active ? 'bg-[hsl(var(--accent))]' : loading ? 'status-pulse bg-[hsl(var(--primary))]' : state === 'error' ? 'bg-[hsl(var(--destructive))]' : 'bg-[hsl(var(--muted-foreground))]'}`}
    />
  );
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const inferenceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const timerRef = useRef<number | null>(null);
  const detectingRef = useRef(false);
  const [state, setState] = useState<AppState>('idle');
  const [error, setError] = useState<CameraError | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [lastScan, setLastScan] = useState<number | null>(null);
  const [videoDimensions, setVideoDimensions] = useState('—');
  const [stageSize, setStageSize] = useState({ width: 1, height: 1 });

  const stopCamera = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    detectingRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  }, []);

  const runDetection = useCallback(async () => {
    const video = videoRef.current;
    const detector = modelRef.current;
    if (!video || !detector || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || detectingRef.current) return;
    detectingRef.current = true;
    try {
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      if (!videoWidth || !videoHeight) return;

      const scale = Math.min(1, INFERENCE_LONG_EDGE / Math.max(videoWidth, videoHeight));
      const inferenceWidth = Math.max(1, Math.round(videoWidth * scale));
      const inferenceHeight = Math.max(1, Math.round(videoHeight * scale));
      const canvas = inferenceCanvasRef.current ?? document.createElement('canvas');
      inferenceCanvasRef.current = canvas;
      if (canvas.width !== inferenceWidth || canvas.height !== inferenceHeight) {
        canvas.width = inferenceWidth;
        canvas.height = inferenceHeight;
      }
      const context = canvas.getContext('2d', { alpha: false });
      if (!context) return;
      context.drawImage(video, 0, 0, inferenceWidth, inferenceHeight);

      const detectedOnSmallFrame = await detector.detect(canvas);
      const coordinateScale = 1 / scale;
      const nextDetections = detectedOnSmallFrame.map((detection) => ({
        ...detection,
        bbox: [
          detection.bbox[0] * coordinateScale,
          detection.bbox[1] * coordinateScale,
          detection.bbox[2] * coordinateScale,
          detection.bbox[3] * coordinateScale,
        ] as [number, number, number, number],
      }));
      setDetections(nextDetections);
      setLastScan(Date.now());
    } catch (detectionError) {
      console.warn('COCO-SSD scan failed', detectionError);
    } finally {
      detectingRef.current = false;
    }
  }, []);

  const startCamera = useCallback(async () => {
    if (state === 'model-loading' || state === 'camera-starting') return;
    stopCamera();
    setError(null);
    setDetections([]);
    setLastScan(null);
    setState('camera-starting');
    try {
      // Deliberately called inside this user-action path. There is no camera request on mount.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Video element is unavailable');
      video.srcObject = stream;
      await video.play();
      if (!video.videoWidth) {
        await new Promise<void>((resolve) => {
          video.addEventListener('loadedmetadata', () => resolve(), { once: true });
        });
      }
      setVideoDimensions(`${video.videoWidth} × ${video.videoHeight}`);
      setState('camera-live');
    } catch (cameraError) {
      stopCamera();
      setError(readableCameraError(cameraError));
      setState('error');
      return;
    }

    try {
      if (tf.getBackend() !== 'webgl') {
        try {
          await tf.setBackend('webgl');
        } catch {
          // TensorFlow.js will use its available fallback backend.
        }
      }
      await tf.ready();
      const loadedModel = modelRef.current ?? await cocoSsd.load({ base: 'lite_mobilenet_v2' });
      modelRef.current = loadedModel;
      setState('running');
      timerRef.current = window.setInterval(runDetection, SCAN_INTERVAL);
      void runDetection();
    } catch {
      setError({
        title: 'COCO-SSD model could not load',
        detail: 'The camera is live, but the detector model did not finish loading. Check the connection used to load the model, then retry the test.',
        action: 'Retry model load',
      });
      setState('error');
    }
  }, [runDetection, state, stopCamera]);

  const stop = useCallback(() => {
    stopCamera();
    setDetections([]);
    setLastScan(null);
    setVideoDimensions('—');
    setState('idle');
  }, [stopCamera]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(([entry]) => {
      setStageSize({ width: entry.contentRect.width || 1, height: entry.contentRect.height || 1 });
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const statusText = state === 'running'
    ? 'Detection active'
    : state === 'model-loading'
      ? 'Loading COCO-SSD model'
      : state === 'camera-starting'
        ? 'Starting rear camera'
        : state === 'camera-live'
          ? 'Camera live'
        : state === 'error'
          ? 'Needs attention'
          : 'Ready to test';

  const statusDescription = state === 'running'
    ? 'Camera feed is live and scanning'
    : state === 'model-loading'
      ? 'Preparing the local object detector'
      : state === 'camera-starting'
        ? 'Waiting for a camera frame'
        : state === 'camera-live'
          ? 'Camera is live; preparing the detector'
        : state === 'error'
          ? 'The test could not start'
          : 'Start a local camera and model check';
  const modelStatus = state === 'model-loading' ? 'LOADING' : modelRef.current ? 'READY' : 'NOT LOADED';

  const positionedDetections = useMemo<PositionedDetection[]>(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return [];
    const sourceRatio = video.videoWidth / video.videoHeight;
    const stageRatio = stageSize.width / stageSize.height;
    const scale = stageRatio > sourceRatio ? stageSize.width / video.videoWidth : stageSize.height / video.videoHeight;
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const offsetX = (stageSize.width - renderedWidth) / 2;
    const offsetY = (stageSize.height - renderedHeight) / 2;
    return detections.map((detection) => ({
      ...detection,
      display: {
        left: offsetX + detection.bbox[0] * scale,
        top: offsetY + detection.bbox[1] * scale,
        width: detection.bbox[2] * scale,
        height: detection.bbox[3] * scale,
      },
    }));
  }, [detections, stageSize]);

  const relationshipText = describeSpatialRelationship(
    detections,
    videoRef.current?.videoWidth ?? 0,
    videoRef.current?.videoHeight ?? 0,
  );

  return (
    <main className="lab-app flex h-full min-h-[100dvh] flex-col">
      <header className="relative z-10 flex shrink-0 items-center justify-between border-b border-[hsl(var(--border))] bg-[hsl(222_18%_9%/0.88)] px-4 py-3 backdrop-blur-md sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center border border-[hsl(var(--primary)/0.55)] bg-[hsl(var(--primary)/0.08)] text-[hsl(var(--primary))]">
            <Camera size={17} strokeWidth={1.7} />
          </div>
          <div>
            <div className="instrument-mono uppercase text-[hsl(var(--muted-foreground))]">FIELD TEST / 01</div>
            <h1 className="text-sm font-semibold tracking-[0.13em] text-[hsl(var(--foreground))]">CAMERA LAB</h1>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-2 text-[hsl(var(--muted-foreground))] sm:flex">
            <span className="instrument-mono uppercase">Runtime</span>
            <span className="instrument-mono text-[hsl(var(--foreground))]">browser / local</span>
          </div>
          <div className="flex items-center gap-2 border-l border-[hsl(var(--border))] pl-4" data-testid="status-header" aria-live="polite">
            <StatusDot state={state} />
            <span className="instrument-mono uppercase text-[hsl(var(--foreground))]">{state === 'running' ? 'LIVE' : state === 'error' ? 'FAULT' : state === 'model-loading' ? 'MODEL' : state === 'camera-starting' ? 'CAMERA' : 'STANDBY'}</span>
          </div>
        </div>
      </header>

      <section className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
        <div ref={stageRef} className="camera-stage h-[46dvh] min-h-[46dvh] flex-none lg:h-auto lg:min-h-0 lg:flex-1" data-testid="camera-stage">
          <video ref={videoRef} playsInline autoPlay muted data-facing="environment" aria-label="Live rear camera preview" />
          <div className="camera-grid" aria-hidden="true" />
          <div className="viewfinder" aria-hidden="true" />
          {state === 'running' && <div className="scan-line" aria-hidden="true" />}
          {relationshipText && (
            <div className="relationship-hud" role="status" aria-live="polite" data-testid="relationship-hud">
              {relationshipText}
            </div>
          )}
          {positionedDetections.map((detection, index) => {
            return (
              <div
                key={`${detection.class}-${index}`}
                className="detection-box"
                style={{ left: detection.display.left, top: detection.display.top, width: detection.display.width, height: detection.display.height }}
                data-testid={`detection-box-${index}`}
              >
                <span className="detection-label">{detection.class} · {Math.round(detection.score * 100)}%</span>
              </div>
            );
          })}
          {state !== 'running' && !streamRef.current && (
            <div className="absolute inset-0 z-[5] flex items-center justify-center p-6">
              <div className="max-w-sm text-center">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center border border-[hsl(var(--foreground)/0.2)] bg-[hsl(220_18%_8%/0.72)] text-[hsl(var(--primary))]">
                  {state === 'error' ? <CircleAlert size={23} strokeWidth={1.4} /> : state === 'idle' ? <Camera size={23} strokeWidth={1.4} /> : <Cpu className="status-pulse" size={23} strokeWidth={1.4} />}
                </div>
                <p className="instrument-mono mb-2 uppercase text-[hsl(var(--primary))]" data-testid="text-stage-state">{statusText}</p>
                <p className="text-sm leading-6 text-[hsl(var(--foreground)/0.68)]">{state === 'error' && error ? error.detail : statusDescription}</p>
              </div>
            </div>
          )}
          <div className="absolute bottom-4 left-4 z-[6] flex items-center gap-2 bg-[hsl(220_18%_7%/0.62)] px-2.5 py-1.5 backdrop-blur-sm">
            <span className={`h-1.5 w-1.5 rounded-full ${streamRef.current ? 'bg-[hsl(var(--accent))]' : 'bg-[hsl(var(--muted-foreground))]'}`} />
            <span className="instrument-mono uppercase text-[hsl(var(--foreground)/0.68)]">{streamRef.current ? 'Environment camera' : 'Preview offline'}</span>
          </div>
          {state === 'running' && (
            <div className="absolute right-4 top-4 z-[6] flex items-center gap-2 bg-[hsl(220_18%_7%/0.62)] px-2.5 py-1.5 backdrop-blur-sm">
              <span className="instrument-mono text-[hsl(var(--accent))]" data-testid="text-detection-count">{detections.length.toString().padStart(2, '0')}</span>
              <span className="instrument-mono uppercase text-[hsl(var(--foreground)/0.68)]">objects</span>
            </div>
          )}
        </div>

        <aside className="relative z-10 flex w-full shrink-0 flex-col border-t border-[hsl(var(--border))] bg-[hsl(220_17%_12%/0.96)] lg:w-[22rem] lg:border-l lg:border-t-0 xl:w-[24rem]">
          <div className="flex-1 overflow-y-auto p-4 sm:p-6">
            <div className="mb-7">
              <div className="instrument-mono mb-3 flex items-center gap-2 uppercase text-[hsl(var(--muted-foreground))]"><span className="h-px w-5 bg-[hsl(var(--primary))]" /> Test instrument</div>
              <h2 className="max-w-[19rem] text-2xl font-medium leading-tight tracking-[-0.035em] text-[hsl(var(--foreground))]">Camera + object detection</h2>
              <p className="mt-3 text-sm leading-6 text-[hsl(var(--muted-foreground))]">A direct read on browser camera access and local COCO-SSD inference.</p>
            </div>

            <div className="mb-5 border border-[hsl(var(--border))] bg-[hsl(220_18%_9%/0.55)]" data-testid="diagnostics-panel">
              <div className="flex items-center justify-between border-b border-[hsl(var(--border))] px-3 py-2.5">
                <span className="instrument-mono uppercase text-[hsl(var(--foreground))]">Diagnostics</span>
                <Gauge size={14} className="text-[hsl(var(--muted-foreground))]" />
              </div>
              <dl className="divide-y divide-[hsl(var(--border))]">
                <div className="flex items-center justify-between px-3 py-3">
                  <dt className="instrument-mono text-[hsl(var(--muted-foreground))]">Secure context</dt>
                  <dd className="flex items-center gap-1.5 instrument-mono text-[hsl(var(--foreground))]" data-testid="diagnostic-secure-context">
                    {window.isSecureContext ? <Check size={13} className="text-[hsl(var(--accent))]" /> : <CircleAlert size={13} className="text-[hsl(var(--destructive))]" />}
                    {window.isSecureContext ? 'YES' : 'NO'}
                  </dd>
                </div>
                <div className="flex items-center justify-between px-3 py-3">
                  <dt className="instrument-mono text-[hsl(var(--muted-foreground))]">Video dimensions</dt>
                  <dd className="instrument-mono text-[hsl(var(--foreground))]" data-testid="diagnostic-video-dimensions">{videoDimensions}</dd>
                </div>
                <div className="flex items-center justify-between px-3 py-3">
                  <dt className="instrument-mono text-[hsl(var(--muted-foreground))]">COCO-SSD model</dt>
                  <dd className={`instrument-mono ${modelStatus === 'READY' ? 'text-[hsl(var(--accent))]' : 'text-[hsl(var(--foreground))]'}`} data-testid="diagnostic-model-status">{modelStatus}</dd>
                </div>
                <div className="flex items-center justify-between px-3 py-3">
                  <dt className="instrument-mono text-[hsl(var(--muted-foreground))]">Detected objects</dt>
                  <dd className="instrument-mono text-[hsl(var(--accent))]" data-testid="diagnostic-object-count">{detections.length}</dd>
                </div>
                <div className="flex items-center justify-between px-3 py-3">
                  <dt className="instrument-mono text-[hsl(var(--muted-foreground))]">Last scan</dt>
                  <dd className="instrument-mono text-right text-[hsl(var(--foreground))]" data-testid="diagnostic-last-scan">{formatTime(lastScan)}</dd>
                </div>
              </dl>
            </div>

            <div className="mb-5 flex items-center justify-between border border-[hsl(var(--border))] px-3 py-3">
              <div>
                <div className="instrument-mono uppercase text-[hsl(var(--muted-foreground))]">Scan cadence</div>
                <div className="mt-1 text-sm text-[hsl(var(--foreground))]">Continuous local inference</div>
              </div>
              <div className="text-right">
                <div className="font-mono text-lg text-[hsl(var(--primary))]" data-testid="text-scan-cadence">350 <span className="text-xs">ms</span></div>
                <div className="instrument-mono uppercase text-[hsl(var(--muted-foreground))]">interval</div>
              </div>
            </div>

            {state === 'error' && error && (
              <div className="mb-5 border border-[hsl(var(--destructive)/0.45)] bg-[hsl(var(--destructive)/0.07)] p-3.5" role="alert" data-testid="status-error">
                <div className="flex gap-3">
                  <CircleAlert size={16} className="mt-0.5 shrink-0 text-[hsl(var(--destructive))]" />
                  <div>
                    <p className="text-sm font-medium text-[hsl(var(--foreground))]">{error.title}</p>
                    <p className="mt-1.5 text-xs leading-5 text-[hsl(var(--muted-foreground))]">{error.detail}</p>
                    <p className="instrument-mono mt-3 uppercase text-[hsl(var(--destructive))]">{error.action}</p>
                  </div>
                </div>
              </div>
            )}

            <div className="hidden border-t border-[hsl(var(--border))] pt-5 sm:block">
              <div className="mb-2 flex items-center gap-2">
                <LockKeyhole size={13} className="text-[hsl(var(--muted-foreground))]" />
                <span className="instrument-mono uppercase text-[hsl(var(--muted-foreground))]">Local-only test</span>
              </div>
              <p className="text-xs leading-5 text-[hsl(var(--muted-foreground))]">Frames are read in this browser. No capture, upload, or backend connection is used.</p>
            </div>
          </div>

          <div className="shrink-0 border-t border-[hsl(var(--border))] bg-[hsl(220_18%_10%/0.86)] p-4 sm:p-6">
            <button
              type="button"
              onClick={state === 'running' || state === 'camera-live' ? stop : startCamera}
              disabled={state === 'model-loading' || state === 'camera-starting'}
              className={`flex min-h-12 w-full items-center justify-center gap-2.5 border px-4 text-sm font-semibold tracking-[0.02em] transition-colors disabled:cursor-wait disabled:opacity-65 ${state === 'running' || state === 'camera-live' ? 'border-[hsl(var(--border))] bg-[hsl(var(--secondary))] text-[hsl(var(--foreground))] hover:bg-[hsl(var(--secondary)/0.7)]' : 'border-[hsl(var(--primary))] bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:bg-[hsl(var(--primary)/0.88)]'}`}
              data-testid="button-camera-control"
            >
              {state === 'running' || state === 'camera-live' ? <Square size={15} fill="currentColor" /> : state === 'error' ? <RotateCcw size={16} /> : <Play size={16} fill="currentColor" />}
              {state === 'running' || state === 'camera-live' ? 'Stop test' : state === 'error' ? 'Retry camera test' : state === 'model-loading' ? 'Loading model…' : state === 'camera-starting' ? 'Starting camera…' : 'Start camera test'}
            </button>
            <p className="mt-3 flex items-center justify-center gap-2 text-center text-[11px] leading-4 text-[hsl(var(--muted-foreground))]">
              <span className="h-1 w-1 rounded-full bg-[hsl(var(--accent))]" /> Rear camera preferred · permission requested on start
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}

export default App;
