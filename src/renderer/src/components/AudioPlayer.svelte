<!-- src/renderer/src/components/AudioPlayer.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte'

  export let src: string
  export let title: string = ''

  let audio: HTMLAudioElement
  let duration = 0
  let currentTime = 0
  let volume = 0.8
  let waveformData: number[] = []
  let canvas: HTMLCanvasElement
  let canvasContext: CanvasRenderingContext2D | null = null
  let isPlaying = false

  onMount(() => {
    if (canvas) {
      canvasContext = canvas.getContext('2d')
      generateWaveform()
    }
  })

  onDestroy(() => {
    if (audio) {
      audio.pause()
      audio.src = ''
    }
  })

  function handleLoadedMetadata() {
    duration = audio.duration
  }

  function handleTimeUpdate() {
    currentTime = audio.currentTime
    drawWaveform()
  }

  function handleEnded() {
    currentTime = 0
  }

  function togglePlayPause() {
    if (audio.paused) {
      audio.play().catch((error) => console.error('Audio playback failed:', error))
    } else {
      audio.pause()
    }
  }

  function seek(event: MouseEvent) {
    if (!canvas || !duration) return
    
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const percentage = x / rect.width
    const newTime = percentage * duration
    
    audio.currentTime = newTime
    currentTime = newTime
  }

  function formatTime(time: number): string {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  function generateWaveform() {
    // Generate mock waveform data for visualization
    // In a real implementation, you'd analyze the audio file
    waveformData = Array.from({ length: 200 }, () => Math.random() * 0.8 + 0.2)
    drawWaveform()
  }

  function drawWaveform() {
    if (!canvasContext || !canvas) return

    const { width, height } = canvas
    canvasContext.clearRect(0, 0, width, height)

    // Background
    canvasContext.fillStyle = '#1e293b'
    canvasContext.fillRect(0, 0, width, height)

    // Waveform
    const barWidth = width / waveformData.length
    const progress = duration > 0 ? currentTime / duration : 0

    waveformData.forEach((amplitude, index) => {
      const x = index * barWidth
      const barHeight = amplitude * height * 0.8
      const y = (height - barHeight) / 2

      // Determine color based on progress
      const isPlayed = index / waveformData.length < progress
      canvasContext.fillStyle = isPlayed ? '#9333ea' : '#475569'
      
      canvasContext.fillRect(x, y, barWidth - 1, barHeight)
    })

    // Progress indicator
    const progressX = progress * width
    canvasContext.strokeStyle = '#f59e0b'
    canvasContext.lineWidth = 2
    canvasContext.beginPath()
    canvasContext.moveTo(progressX, 0)
    canvasContext.lineTo(progressX, height)
    canvasContext.stroke()
  }

  // React to src changes
  $: if (src && audio) {
    audio.src = src
    audio.load()
  }

  // React to volume changes
  $: if (audio) {
    audio.volume = volume
  }
</script>

<div class="audio-player">
  <div class="player-header">
    <h4>{title}</h4>
    <div class="time-display">
      {formatTime(currentTime)} / {formatTime(duration)}
    </div>
  </div>

  <div class="waveform-container">
    <canvas
      bind:this={canvas}
      width="400"
      height="100"
      on:click={seek}
      class="waveform"
    ></canvas>
  </div>

  <div class="player-controls">
    <button 
      class="play-button"
      on:click={togglePlayPause}
      title={isPlaying ? 'Pause' : 'Play'}
    >
      {#if isPlaying}
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
        </svg>
      {:else}
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5v14l11-7z"/>
        </svg>
      {/if}
    </button>

    <div class="volume-.control">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
      </svg>
      <input
        type="range"
        min="0"
        max="1"
        step="0.1"
        bind:value={volume}
        class="volume-slider"
      />
    </div>
  </div>

  <!-- Hidden audio element -->
  <audio
    bind:this={audio}
    on:loadedmetadata={handleLoadedMetadata}
    on:timeupdate={handleTimeUpdate}
    on:ended={handleEnded}
    on:play={() => (isPlaying = true)}
    on:pause={() => (isPlaying = false)}
    preload="metadata"
  ></audio>
</div>

<style>
  .audio-player {
    position: fixed;
    bottom: 1rem;
    left: 1rem;
    right: 1rem;
    z-index: 1000;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0.5rem;
    padding: 1rem;
    color: white;
    backdrop-filter: blur(10px);
  }

  .player-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .player-header h4 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    margin-right: 1rem;
  }

  .time-display {
    font-size: 0.875rem;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', monospace;
  }

  .waveform-container {
    margin-bottom: 1rem;
    border-radius: 0.375rem;
    overflow: hidden;
  }

  .waveform {
    width: 100%;
    height: 100px;
    cursor: pointer;
    display: block;
  }

  .player-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .play-button {
    background: #9333ea;
    border: none;
    color: white;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
  }

  .play-button:hover {
    background: #7c3aed;
    transform: scale(1.05);
  }

  .play-button:active {
    transform: scale(0.95);
  }

  .volume-control {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex: 1;
  }

  .volume-control svg {
    color: rgba(255, 255, 255, 0.7);
  }

  .volume-slider {
    flex: 1;
    height: 4px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
  }

  .volume-slider::-webkit-slider-thumb {
    width: 16px;
    height: 16px;
    background: #9333ea;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
  }

  .volume-slider::-moz-range-thumb {
    width: 16px;
    height: 16px;
    background: #9333ea;
    border-radius: 50%;
    border: none;
    cursor: pointer;
  }

  .volume-slider::-webkit-slider-track {
    background: rgba(255, 255, 255, 0.2);
    height: 4px;
    border-radius: 2px;
  }

  .volume-slider::-moz-range-track {
    background: rgba(255, 255, 255, 0.2);
    height: 4px;
    border-radius: 2px;
    border: none;
  }
</style>