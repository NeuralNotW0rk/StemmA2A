<!-- src/renderer/src/components/AudioPlayer.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte'

  interface Props {
    src: string
    title?: string
    onclose?: () => void
  }

  let { src, title = '', onclose }: Props = $props()

  let audio: HTMLAudioElement | undefined = $state()
  let duration = $state(0)
  let currentTime = $state(0)
  let volume = $state(0.8)
  let isPlaying = $state(false)
  let showVolume = $state(false)
  let volumeControlElement: HTMLElement | undefined = $state()
  let loadedSrc = $state('')

  function handleLoadedMetadata(): void {
    if (audio) {
      duration = audio.duration
    }
  }

  function handleTimeUpdate(): void {
    if (audio) {
      currentTime = audio.currentTime
    }
  }

  function handleEnded(): void {
    isPlaying = false
    currentTime = 0
  }

  function togglePlayPause(): void {
    if (!audio) return
    if (isPlaying) {
      audio.pause()
    } else {
      audio.play().catch((error) => console.error('Audio playback failed:', error))
    }
  }

  function formatTime(time: number): string {
    if (isNaN(time)) return '0:00'
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  function seek(event: MouseEvent): void {
    if (!audio || !duration) return
    const progressBar = event.currentTarget as HTMLElement
    const rect = progressBar.getBoundingClientRect()
    const x = event.clientX - rect.left
    const percentage = x / rect.width
    const newTime = percentage * duration
    audio.currentTime = newTime
  }

  function handleKeyDown(event: KeyboardEvent): void {
    if (!audio || !duration) return
    if (event.key === 'ArrowRight') {
      audio.currentTime = Math.min(duration, audio.currentTime + 5)
    } else if (event.key === 'ArrowLeft') {
      audio.currentTime = Math.max(0, audio.currentTime - 5)
    }
  }

  function toggleVolume(): void {
    showVolume = !showVolume
  }

  function handleClickOutside(event: MouseEvent): void {
    if (
      showVolume &&
      volumeControlElement &&
      !volumeControlElement.contains(event.target as Node)
    ) {
      showVolume = false
    }
  }

  onMount(() => {
    window.addEventListener('click', handleClickOutside)
  })

  onDestroy(() => {
    window.removeEventListener('click', handleClickOutside)
  })

  $effect(() => {
    if (audio && src && src !== loadedSrc) {
      audio.pause()
      audio.src = src
      audio.load()
      currentTime = 0
      duration = 0
      isPlaying = false
      loadedSrc = src
      // Play audio on selection
      audio.play().catch((error) => console.error('Audio playback failed:', error))
    }
  })

  $effect(() => {
    if (audio) {
      audio.volume = volume
    }
  })
</script>

<div class="audio-player">
  <button class="play-button" onclick={togglePlayPause} aria-label={isPlaying ? 'Pause' : 'Play'}>
    {#if isPlaying}
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
      </svg>
    {:else}
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M8 5v14l11-7z" />
      </svg>
    {/if}
  </button>
  <div class="player-content">
    <div class="title-bar">
      <h4 class="title">{title}</h4>
      <div class="time-display">{formatTime(currentTime)} / {formatTime(duration)}</div>
    </div>
    <div
      class="progress-bar"
      role="slider"
      aria-label="Audio progress"
      aria-valuemin={0}
      aria-valuemax={duration || 0}
      aria-valuenow={currentTime}
      tabindex="0"
      onclick={seek}
      onkeydown={handleKeyDown}
    >
      <div class="progress" style="width: {(currentTime / duration) * 100}%"></div>
    </div>
  </div>
  <div class="volume-control" bind:this={volumeControlElement}>
    <button
      class="volume-button"
      onclick={(e) => {
        e.stopPropagation()
        toggleVolume()
      }}
      aria-label="Volume control"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path
          d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"
        />
      </svg>
    </button>
    {#if showVolume}
      <div class="volume-slider-container">
        <input type="range" min="0" max="1" step="0.05" bind:value={volume} class="volume-slider" />
      </div>
    {/if}
  </div>

  <button class="close-button" onclick={() => onclose?.()} aria-label="Close audio player">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path
        d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"
      />
    </svg>
  </button>

  <audio
    bind:this={audio}
    onloadedmetadata={handleLoadedMetadata}
    ontimeupdate={handleTimeUpdate}
    onended={handleEnded}
    onplay={() => (isPlaying = true)}
    onpause={() => (isPlaying = false)}
    preload="auto"
  ></audio>
</div>

<style>
  .audio-player {
    position: absolute;
    z-index: 1001;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    width: 90%;
    max-width: 500px;
    background: var(--color-background-glass-1);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.5rem;
    padding: 0.5rem;
    color: var(--color-overlay-text);
    backdrop-filter: blur(10px);
  }

  .player-content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    overflow: hidden; /* Prevent content from overflowing */
  }

  .title-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .title {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: 1rem;
    min-width: 0; /* Allow title to shrink */
  }

  .time-display {
    font-size: 0.75rem;
    color: var(--color-text-overlay-primary);
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
  }

  .progress-bar {
    height: 4px;
    background: var(--color-overlay-border-primary);
    border-radius: 2px;
    cursor: pointer;
  }

  .progress {
    height: 100%;
    background: var(--color-primary);
    border-radius: 2px;
  }

  .play-button,
  .close-button,
  .volume-button {
    background: var(--color-primary);
    border: none;
    color: var(--color-overlay-text);
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
  }

  .play-button:hover,
  .close-button:hover,
  .volume-button:hover {
    background: var(--color-primary-hover);
    transform: scale(1.05);
  }

  .play-button:active,
  .close-button:active,
  .volume-button:active {
    transform: scale(0.95);
  }

  .close-button,
  .volume-button {
    background: var(--color-border-glass-1);
  }

  .close-button:hover,
  .volume-button:hover {
    background: var(--color-overlay-border-primary);
  }

  .volume-control {
    position: relative;
    display: flex;
    align-items: center;
  }

  .volume-slider-container {
    position: absolute;
    bottom: calc(100% + 0.5rem);
    left: 50%;
    transform: translateX(-50%);
    background: var(--color-background-glass-3);
    backdrop-filter: blur(10px);
    border-radius: 0.5rem;
    padding: 1rem 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .volume-slider {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    cursor: pointer;
    width: 100px; /* Length of the slider */
    transform: rotate(-90deg);
  }

  /* Webkit (Chrome, Safari) */
  .volume-slider::-webkit-slider-runnable-track {
    height: 4px;
    background: var(--color-overlay-border-primary);
    border-radius: 2px;
  }

  .volume-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    margin-top: -5px; /* (thumb height - track height) / 2 */
    width: 14px;
    height: 14px;
    background: var(--color-primary);
    border-radius: 50%;
    border: none;
  }

  /* Firefox */
  .volume-slider::-moz-range-track {
    height: 4px;
    background: var(--color-overlay-border-primary);
    border-radius: 2px;
  }

  .volume-slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    background: var(--color-primary);
    border-radius: 50%;
    border: none;
  }
</style>
