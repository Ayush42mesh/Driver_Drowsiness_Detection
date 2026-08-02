import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, StopCircle, Volume2, Shield, AlertTriangle, RefreshCw, 
  Settings, FileText, Camera, Maximize, CheckCircle, HelpCircle 
} from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import './App.css';

// Register Chart.js modules
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5000";

export default function App() {
  // Streaming and Webcam States
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState("--");
  const [confidence, setConfidence] = useState(0);
  const [consecutiveCount, setConsecutiveCount] = useState(0);
  const [consecutiveThreshold, setConsecutiveThreshold] = useState(15);
  const [fps, setFps] = useState(0);
  const [alarmActive, setAlarmActive] = useState(false);
  
  // Model and Configuration States
  const [modelStatus, setModelStatus] = useState("Checking...");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  
  // Form Settings States
  const [threshold, setThreshold] = useState(0.5);
  const [framesLimit, setFramesLimit] = useState(15);
  const [volume, setVolume] = useState(0.8);
  const [modelPath, setModelPath] = useState("C:\\Users\\Ayush Meshram\\.gemini\\antigravity-ide\\scratch\\DDD\\model\\DDD.keras");
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  
  // Chart and Log history States
  const [chartValues, setChartValues] = useState(Array(30).fill(0));
  const [logs, setLogs] = useState([]);
  
  // References
  const videoRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const displayCanvasRef = useRef(null);
  const localStreamRef = useRef(null);
  const intervalRef = useRef(null);
  const audioRef = useRef(null);
  const fpsCounterRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());
  const processingRef = useRef(false);

  // Initialize Web Audio Object
  useEffect(() => {
    const audio = new Audio(`${BACKEND_URL}/alarm/alarm.wav`);
    audio.loop = true;
    audio.volume = volume;
    audioRef.current = audio;

    // Load initial settings and logs from Flask
    fetchSettings();
    fetchLogs();

    return () => {
      stopMonitor();
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  // Sync volume changes with audio ref
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);

  // Fetch Settings from backend
  const fetchSettings = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/get_settings`);
      const data = await res.json();
      
      setThreshold(data.confidence_threshold);
      setFramesLimit(data.consecutive_frames);
      setVolume(data.alarm_volume);
      setModelPath(data.model_path);
      setModelLoaded(data.model_loaded);
      setIsDemoMode(data.is_demo_mode);

      if (data.model_loaded) {
        setModelStatus("Model Loaded");
      } else if (data.is_demo_mode) {
        setModelStatus("Demo Mode");
      } else {
        setModelStatus("Error loading model");
      }
    } catch (err) {
      console.error("Failed to load settings from server:", err);
      setModelStatus("Offline");
    }
  };

  // Fetch Logs from backend
  const fetchLogs = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/get_logs`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setLogs(data.reverse()); // Latest logs first
      }
    } catch (err) {
      console.error("Failed to fetch logs:", err);
    }
  };

  // Save Config parameters to Flask backend
  const handleSaveSettings = async () => {
    setIsSavingSettings(true);
    try {
      const res = await fetch(`${BACKEND_URL}/save_settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          confidence_threshold: threshold,
          consecutive_frames: framesLimit,
          alarm_volume: volume,
          model_path: modelPath
        })
      });
      const data = await res.json();
      if (data.success) {
        alert("Settings saved successfully!" + (data.reload_msg ? `\nModel: ${data.reload_msg}` : ""));
        fetchSettings();
      } else {
        alert("Error saving settings: " + data.error);
      }
    } catch (err) {
      alert("Failed to save settings: " + err);
    } finally {
      setIsSavingSettings(false);
    }
  };

  // Start Monitor webcam
  const startMonitor = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 20 } }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        localStreamRef.current = stream;
        
        // Explicitly play the video element to guarantee stream capture
        try {
          await videoRef.current.play();
        } catch (e) {
          console.warn("Auto-play blocked or failed, waiting for user interaction:", e);
        }
        
        setIsStreaming(true);
        
        // Match canvas dimensions
        if (displayCanvasRef.current) {
          displayCanvasRef.current.width = 640;
          displayCanvasRef.current.height = 480;
        }
        if (hiddenCanvasRef.current) {
          hiddenCanvasRef.current.width = 640;
          hiddenCanvasRef.current.height = 480;
        }

        fpsCounterRef.current = 0;
        lastFpsTimeRef.current = Date.now();
        processingRef.current = false;

        // Capture interval: ~15 FPS -> 66ms
        intervalRef.current = setInterval(captureAndSendFrame, 66);
      }
    } catch (err) {
      console.error("Webcam access error:", err);
      alert("Failed to access camera. Please allow webcam permissions.");
    }
  };

  // Stop Monitor webcam
  const stopMonitor = () => {
    setIsStreaming(false);
    clearInterval(intervalRef.current);
    
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
      localStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setPrediction("--");
    setConfidence(0);
    setConsecutiveCount(0);
    setFps(0);
    
    silenceAlarm();
  };

  // Capture frame from video stream and POST to backend
  const captureAndSendFrame = async () => {
    if (processingRef.current || !videoRef.current || !hiddenCanvasRef.current) return;
    processingRef.current = true;

    const canvas = hiddenCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Draw current frame to hidden canvas
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    
    // Extract base64 image data string
    const base64Data = canvas.toDataURL('image/jpeg', 0.85);

    try {
      const res = await fetch(`${BACKEND_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data })
      });
      const data = await res.json();

      if (data.success) {
        // Draw bounding box and prediction text onto the transparent overlay canvas
        if (displayCanvasRef.current) {
          const displayCtx = displayCanvasRef.current.getContext('2d');
          displayCtx.clearRect(0, 0, 640, 480);

          if (data.bbox) {
            const [xmin, ymin, width, height] = data.bbox;
            // Draw box outline: Red if alarm active, green otherwise
            displayCtx.strokeStyle = data.alarm_active ? "#ef4444" : "#10b981";
            displayCtx.lineWidth = 3;
            displayCtx.strokeRect(xmin, ymin, width, height);

            // Draw status text labels
            displayCtx.fillStyle = data.alarm_active ? "#ef4444" : "#10b981";
            displayCtx.font = "bold 16px Inter, sans-serif";
            const text = `${data.label} (${Math.round(data.confidence * 100)}%)`;
            displayCtx.fillText(text, xmin, ymin > 20 ? ymin - 8 : ymin + height + 20);
          }
        }

        // Set state values
        setPrediction(data.label);
        setConfidence(data.confidence);
        setConsecutiveCount(data.count);
        setConsecutiveThreshold(data.consecutive_frames_threshold);

        // Update chart history values
        let indexVal = 0;
        if (data.label === "Drowsy") {
          indexVal = data.confidence;
        } else if (data.label === "Non Drowsy") {
          indexVal = 1.0 - data.confidence;
        }
        setChartValues(prev => [...prev.slice(1), indexVal]);

        // Manage audio alarm state
        if (data.alarm_active) {
          triggerAlarm();
        } else {
          silenceAlarm();
        }

        // Calculate actual frame FPS
        fpsCounterRef.current++;
        const now = Date.now();
        if (now - lastFpsTimeRef.current >= 1000) {
          setFps(fpsCounterRef.current);
          fpsCounterRef.current = 0;
          lastFpsTimeRef.current = now;
        }
      }
    } catch (err) {
      console.error("Frame inference error:", err);
    } finally {
      processingRef.current = false;
    }
  };

  // Silence Alarm Client-side
  const silenceAlarm = () => {
    setAlarmActive(false);
    if (audioRef.current && !audioRef.current.paused) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  // Trigger Alarm Client-side
  const triggerAlarm = () => {
    setAlarmActive(true);
    if (audioRef.current && audioRef.current.paused) {
      audioRef.current.play().catch(e => console.log("Audio play delayed waiting for interaction:", e));
    }
  };

  // Reset Alarm in Backend
  const handleSilenceAlarm = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/reset_alarm`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        silenceAlarm();
        fetchLogs();
      }
    } catch (err) {
      console.error("Failed to reset alarm on server:", err);
    }
  };

  // Take a canvas screenshot
  const handleTakeScreenshot = () => {
    if (!isStreaming || !displayCanvasRef.current) {
      alert("Please start the monitor camera first!");
      return;
    }
    const link = document.createElement('a');
    link.download = `ddd_screenshot_${Date.now()}.jpg`;
    link.href = displayCanvasRef.current.toDataURL('image/jpeg');
    link.click();
  };

  // Chart configs
  const chartData = {
    labels: Array(30).fill(''),
    datasets: [{
      label: 'Drowsiness Index',
      data: chartValues,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      borderWidth: 2,
      tension: 0.4,
      fill: true,
      pointRadius: 0
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      x: { display: false },
      y: {
        min: 0,
        max: 1.0,
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 9 } }
      }
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <Shield className="brand-icon" />
          <h1>DDD Driver Monitor <span className="version-pill">v3.0 React</span></h1>
        </div>
        <div className="header-status">
          <div className="status-pill">
            Model: <span className="model-status" style={{
              color: modelLoaded ? 'var(--accent-green)' : isDemoMode ? 'var(--accent-amber)' : 'var(--accent-red)'
            }}>{modelStatus}</span>
          </div>
          <div className="status-pill">
            State: 
            <span className={`status-dot ${isStreaming ? alarmActive ? 'alarm' : 'active' : ''}`}></span>
            <span style={{ fontWeight: 600 }}>{isStreaming ? alarmActive ? "ALARM" : "ACTIVE" : "OFF"}</span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        <div className="main-content">
          {/* Warning Banner */}
          {alarmActive && (
            <div className="warning-banner">
              <div className="warning-info">
                <AlertTriangle className="warning-icon animate-pulse" />
                <div className="warning-text">
                  <h3>Drowsiness Alert Triggered!</h3>
                  <p>Eyes closed or high drowsiness probability detected. Pull over safely.</p>
                </div>
              </div>
              <button className="silence-btn" onClick={handleSilenceAlarm}>Silence Alarm</button>
            </div>
          )}

          {/* Cards Row */}
          <div className="dashboard-row">
            {/* Camera View Card */}
            <div className={`card camera-card ${alarmActive ? 'alarm-glow' : ''}`}>
              <div className="card-header">
                <h3>Live Monitoring</h3>
                {isStreaming && <span className="fps-label">FPS: {fps}</span>}
              </div>
              <div className="video-viewport">
                <video ref={videoRef} className="live-video" autoPlay playsInline muted style={{ display: isStreaming ? 'block' : 'none' }}></video>
                <canvas ref={hiddenCanvasRef} style={{ display: 'none' }}></canvas>
                <canvas ref={displayCanvasRef} className="display-canvas" style={{ display: isStreaming ? 'block' : 'none' }}></canvas>
                
                {!isStreaming && (
                  <div className="camera-placeholder">
                    <Camera size={48} className="placeholder-icon" />
                    <p>Camera feed is currently offline.</p>
                    <button className="start-btn" onClick={startMonitor}>Start Monitor</button>
                  </div>
                )}
              </div>
              <div className="camera-actions">
                {isStreaming && (
                  <button className="stop-btn" onClick={stopMonitor}>
                    <StopCircle size={16} /> Stop Monitor
                  </button>
                )}
                <button className="screenshot-btn" onClick={handleTakeScreenshot} disabled={!isStreaming}>
                  Take Local Screenshot
                </button>
              </div>
            </div>

            {/* Metrics & Graph Card */}
            <div className="card telemetry-card">
              <div className="card-header">
                <h3>Telemetry Info</h3>
              </div>
              <div className="metrics-grid">
                <div className="metric-box">
                  <span className="metric-label">Inference Status</span>
                  <span className="metric-val" style={{
                    color: prediction === "Drowsy" ? 'var(--accent-red)' : prediction === "Non Drowsy" ? 'var(--accent-green)' : 'var(--accent-amber)'
                  }}>{prediction}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-label">Confidence Score</span>
                  <span className="metric-val">{Math.round(confidence * 100)}%</span>
                </div>
              </div>

              {/* Progress Count */}
              <div className="progress-section">
                <div className="progress-label">
                  <span>Consecutive Warnings</span>
                  <span>{consecutiveCount} / {consecutiveThreshold}</span>
                </div>
                <div className="progress-track">
                  <div 
                    className={`progress-fill ${consecutiveCount >= consecutiveThreshold * 0.8 ? 'critical' : ''}`}
                    style={{ width: `${Math.min(100, (consecutiveCount / consecutiveThreshold) * 100)}%` }}
                  ></div>
                </div>
              </div>

              {/* Drowsy index line chart */}
              <div className="chart-section">
                <span className="section-title">Live Drowsiness Index</span>
                <div className="chart-wrapper">
                  <Line data={chartData} options={chartOptions} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <aside className="sidebar">
          {/* Settings Card */}
          <div className="card settings-card">
            <div className="card-header">
              <h3>System Settings</h3>
              <Settings size={18} className="text-slate-400" />
            </div>
            <div className="settings-controls">
              <div className="control-group">
                <label>Confidence Threshold: {threshold}</label>
                <input 
                  type="range" min="0.1" max="0.9" step="0.05" 
                  value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} 
                />
                <div className="label-row">
                  <span>0.1 (Strict)</span>
                  <span>0.9 (Lenient)</span>
                </div>
              </div>

              <div className="control-group">
                <label>Consecutive Alert Limit: {framesLimit}</label>
                <input 
                  type="range" min="5" max="30" step="1" 
                  value={framesLimit} onChange={(e) => setFramesLimit(parseInt(e.target.value))} 
                />
                <div className="label-row">
                  <span>5 frames</span>
                  <span>30 frames</span>
                </div>
              </div>

              <div className="control-group">
                <label>Alert Audio Volume: {Math.round(volume * 100)}%</label>
                <input 
                  type="range" min="0" max="1" step="0.1" 
                  value={volume} onChange={(e) => setVolume(parseFloat(e.target.value))} 
                />
                <div className="label-row">
                  <span>Mute</span>
                  <span>100%</span>
                </div>
              </div>

              <div className="control-group">
                <label>Active Model Path</label>
                <input 
                  type="text" className="model-path-input"
                  value={modelPath} onChange={(e) => setModelPath(e.target.value)} 
                />
              </div>

              <button 
                className="save-settings-btn" 
                onClick={handleSaveSettings}
                disabled={isSavingSettings}
              >
                {isSavingSettings ? "Saving Settings..." : "Save Configuration"}
              </button>
            </div>
          </div>

          {/* Logs History Card */}
          <div className="card logs-card">
            <div className="card-header">
              <h3>System Event Logs</h3>
              <button className="refresh-logs-btn" onClick={fetchLogs}>
                <RefreshCw size={12} />
              </button>
            </div>
            <div className="logs-viewport">
              {logs.length === 0 ? (
                <div className="empty-logs">No warning events logged yet.</div>
              ) : (
                logs.map((log, index) => {
                  const isDrowsy = log.Prediction === "Drowsy";
                  return (
                    <div key={index} className={`log-item ${isDrowsy ? 'drowsy' : ''}`}>
                      <div className="log-info">
                        <h4>{isDrowsy ? "DROWSY ALERT" : "OK STATUS"}</h4>
                        <span>{log.Date} at {log.Time}</span>
                      </div>
                      <span className={`log-score ${isDrowsy ? 'drowsy' : 'active'}`}>{log.Confidence}</span>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
