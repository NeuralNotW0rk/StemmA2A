<script lang="ts">
  import type { EngineConnection, EngineMode, EngineSettings, EngineState } from '../utils/types'

  // Each check goes through the backend to the engine service, often over a tunnel, so it is kept infrequent
  const CHECK_INTERVAL_MS = 20000

  interface Props {
    // Null until the backend has reported which engine is active
    engine?: EngineState | null
    onswitch?: (settings: EngineSettings) => Promise<void>
  }

  let { engine = null, onswitch }: Props = $props()

  let showMenu = $state(false)
  let menuElement: HTMLElement | undefined = $state()
  let urlInput: HTMLInputElement | undefined = $state()
  let remoteUrl = $state('')
  // The engine being switched to, while a switch is in progress
  let switchingTo: EngineMode | null = $state(null)
  let error: string | null = $state(null)

  let label = $derived(
    !engine ? 'Engine' : engine.mode === 'remote' ? 'Remote Engine' : 'Local Engine'
  )
  // Connect is only offered once the URL has been edited
  let urlEdited = $derived(!!remoteUrl.trim() && remoteUrl.trim() !== (engine?.remote_url ?? ''))

  // Result of the last check of the remote engine service; null before the first one finishes
  let connection: EngineConnection | null = $state(null)
  let checkedAt: Date | null = $state(null)
  let isChecking = $state(false)
  // Lets a check that finishes after a switch or a newer check be discarded
  let checkSeq = 0

  let connectionText = $derived.by(() => {
    const url = engine?.remote_url
    if (!connection) return `Checking connection to ${url}…`
    const time = checkedAt ? ` Checked at ${checkedAt.toLocaleTimeString()}.` : ''
    if (connection.reachable) {
      return `Connected to ${url}${connection.device ? ` (${connection.device})` : ''}.${time}`
    }
    return `Can't reach ${url}. ${connection.error ?? ''}${time}`
  })

  async function checkConnection(): Promise<void> {
    const seq = ++checkSeq
    isChecking = true
    let result: EngineConnection | null
    try {
      result = (await window.api.getEngine(true)).connection ?? null
    } catch (e: unknown) {
      result = {
        reachable: false,
        error: `The backend didn't answer: ${e instanceof Error ? e.message : String(e)}`
      }
    }
    if (seq !== checkSeq) return
    connection = result
    checkedAt = new Date()
    isChecking = false
  }

  // Checks right away whenever the remote engine becomes active or its URL changes, then on an interval
  $effect((): (() => void) | void => {
    checkSeq++
    connection = null
    isChecking = false
    if (engine?.mode !== 'remote' || !engine.remote_url) return
    checkConnection()
    const timer = window.setInterval(checkConnection, CHECK_INTERVAL_MS)
    return (): void => window.clearInterval(timer)
  })

  function openMenu(): void {
    showMenu = !showMenu
    if (showMenu && engine) {
      // Start from the backend's current state each time the menu opens
      remoteUrl = engine.remote_url ?? ''
      error = null
      if (engine.mode === 'remote' && !isChecking) checkConnection()
    }
  }

  async function switchTo(settings: EngineSettings): Promise<void> {
    switchingTo = settings.mode
    error = null
    try {
      await onswitch?.(settings)
      showMenu = false
    } catch (e: unknown) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      switchingTo = null
    }
  }

  function chooseLocal(): void {
    if (engine?.mode !== 'local') switchTo({ mode: 'local' })
  }

  function chooseRemote(): void {
    if (engine?.mode === 'remote' && !urlEdited) return
    if (!remoteUrl.trim()) {
      error = "Enter the engine service's URL first."
      urlInput?.focus()
      return
    }
    switchTo({ mode: 'remote', remote_url: remoteUrl.trim() })
  }

  // Choosing an option switches straight away
  const options: { mode: EngineMode; name: string; hint: string; choose: () => void }[] = [
    { mode: 'local', name: 'Local', hint: 'Runs models on this machine', choose: chooseLocal },
    {
      mode: 'remote',
      name: 'Remote',
      hint: 'Runs models on an engine service',
      choose: chooseRemote
    }
  ]
</script>

<svelte:window
  onclick={(event: MouseEvent): void => {
    if (menuElement && event.target instanceof Node && !menuElement.contains(event.target)) {
      showMenu = false
    }
  }}
  onkeydown={(e: KeyboardEvent): void => {
    if (e.key === 'Escape') showMenu = false
  }}
/>

<div class="engine-menu" bind:this={menuElement}>
  <button
    class="toolbar-button"
    onclick={openMenu}
    title={engine?.mode === 'remote' ? connectionText : 'Choose engine'}
    aria-label="Open engine menu"
  >
    {#if engine?.mode === 'remote'}
      <span
        class="connection-dot"
        class:reachable={connection?.reachable === true}
        class:unreachable={connection?.reachable === false}
      ></span>
    {/if}
    {label}
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="currentColor"
      class:rotate-180={showMenu}
      class="chevron"
    >
      <path d="M7 10l5 5 5-5z" />
    </svg>
  </button>

  {#if showMenu}
    <div class="engine-dropdown">
      <div class="dropdown-header">Engine</div>
      {#each options as option (option.mode)}
        <button
          class="mode-option"
          class:active={engine?.mode === option.mode}
          onclick={option.choose}
          disabled={switchingTo !== null}
        >
          <span class="mode-check">
            {#if switchingTo === option.mode}
              <span class="spinner small"></span>
            {:else if engine?.mode === option.mode}
              ✓
            {/if}
          </span>
          <span class="mode-name">{option.name}</span>
          <span class="mode-hint">{option.hint}</span>
        </button>
      {/each}
      <div class="url-row">
        <input
          class="url-input"
          type="text"
          placeholder="http://127.0.0.1:5001"
          aria-label="Remote engine URL"
          bind:this={urlInput}
          bind:value={remoteUrl}
          disabled={switchingTo !== null}
          onkeydown={(e) => {
            if (e.key === 'Enter') chooseRemote()
          }}
        />
        {#if urlEdited}
          <button class="primary connect" onclick={chooseRemote} disabled={switchingTo !== null}>
            Connect
          </button>
        {/if}
      </div>
      {#if engine?.mode === 'remote' && !urlEdited}
        <div class="connection-status">
          <span
            class="connection-dot"
            class:reachable={connection?.reachable === true}
            class:unreachable={connection?.reachable === false}
          ></span>
          <span class="connection-text">{connectionText}</span>
          {#if connection?.reachable === false}
            <button class="link-button" onclick={checkConnection} disabled={isChecking}>
              {isChecking ? 'Checking…' : 'Retry'}
            </button>
          {/if}
        </div>
      {/if}
      {#if error}
        <div class="engine-error">{error}</div>
      {/if}
      {#if switchingTo === 'local'}
        <div class="mode-hint switching-hint">
          The first switch to local loads its encoders, which can take a while.
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .engine-menu {
    position: relative;
  }

  .toolbar-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toolbar-button:hover {
    background: var(--color-background-glass-hover-1);
    transform: translateY(-1px);
  }

  .chevron {
    transition: transform 0.2s;
  }

  .rotate-180 {
    transform: rotate(180deg);
  }

  .engine-dropdown {
    position: absolute;
    top: 100%;
    right: 0;
    width: 280px;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding-bottom: 0.75rem;
    background: var(--color-background-glass-4);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.375rem;
    margin-top: 0.25rem;
    z-index: 1000;
    backdrop-filter: blur(10px);
  }

  .dropdown-header {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    border-bottom: 1px solid var(--color-border-glass-1);
    color: var(--color-text-overlay-primary);
  }

  /* Styled as menu items rather than buttons, since choosing one acts immediately */
  .mode-option {
    display: grid;
    grid-template-columns: 1rem 1fr;
    column-gap: 0.5rem;
    align-items: center;
    width: 100%;
    min-width: 0;
    min-height: 0;
    padding: 0.4rem 1rem;
    background: none;
    border: none;
    border-radius: 0;
    color: var(--color-overlay-text);
    text-align: left;
  }

  .mode-option:hover:not(:disabled):not(.active) {
    background: var(--color-border-glass-1);
  }

  .mode-option.active {
    cursor: default;
  }

  .mode-option.active .mode-name {
    font-weight: 600;
  }

  .mode-check {
    grid-row: span 2;
    display: flex;
    justify-content: center;
    color: var(--color-primary);
  }

  .mode-option .mode-hint {
    grid-column: 2;
    padding: 0;
  }

  .mode-hint {
    padding: 0 1rem;
    font-size: 0.75rem;
    color: var(--color-text-overlay-secondary);
  }

  .spinner.small {
    width: 12px;
    height: 12px;
    border-width: 2px;
  }

  .url-row {
    display: flex;
    gap: 0.5rem;
    /* Lines the field up with the option names, under the remote option */
    padding: 0 1rem 0 2.5rem;
  }

  .url-input {
    flex: 1;
    min-width: 0;
    padding: 0.4rem 0.6rem;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.375rem;
    color: var(--color-overlay-text);
  }

  .connect {
    min-width: 0;
    min-height: 0;
    padding: 0.4rem 0.75rem;
  }

  /* Gray while the first check is pending, then green or red */
  .connection-dot {
    flex-shrink: 0;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--color-text-overlay-secondary);
    animation: pulse 1.2s ease-in-out infinite;
  }

  .connection-dot.reachable {
    background: var(--color-success);
    animation: none;
  }

  .connection-dot.unreachable {
    background: var(--color-error);
    animation: none;
  }

  @keyframes pulse {
    50% {
      opacity: 0.3;
    }
  }

  .connection-status {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0 1rem 0 2.5rem;
    font-size: 0.75rem;
    color: var(--color-text-overlay-secondary);
    white-space: normal;
  }

  .connection-text {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .link-button {
    min-width: 0;
    min-height: 0;
    padding: 0;
    background: none;
    border: none;
    color: var(--color-primary);
    font-size: inherit;
    text-decoration: underline;
  }

  .engine-error {
    margin: 0 1rem;
    font-size: 0.8rem;
    color: var(--color-error, #f87171);
    white-space: normal;
  }
</style>
