<script lang="ts">
  import { onMount } from 'svelte'
  import type { FormConfig } from '../utils/forms'
  import ErrorDialog from './ErrorDialog.svelte'

  let { model_name, engine_name, onClose, onGenerate, onError } = $props<{
    model_name: string
    engine_name: string
    onClose: () => void
    onGenerate: (data: unknown) => void
    onError: (error: unknown) => void
  }>()

  let formConfig: FormConfig | null = $state(null)
  let formData: Record<string, string | number | boolean | null> = {}
  let isLoading = $state(true)
  let error: string | null = $state(null)
  let showErrorDialog = $state(false)
  let errorMessage = $state('')
  let inProgress = $state(false)

  onMount(async () => {
    try {
      isLoading = true
      const config = await window.api.getEngineConfig(engine_name)
      if (config && config.generate && Array.isArray(config.generate)) {
        formConfig = config.generate
        // Initialize formData with default values from the fetched config
        for (const field of formConfig) {
                      formData[field.name] = field.defaultValue as string | number | boolean | null        }
      } else {
        throw new Error('Invalid config format received from backend.')
      }
    } catch (e: unknown) {
      console.error('Failed to fetch form config:', e)
      error = e instanceof Error ? e.message : 'Failed to load configuration.'
      errorMessage = error
      showErrorDialog = true
    } finally {
      isLoading = false
    }
  })

  function handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      onClose()
    }
  }

  async function generate(): Promise<void> {
    const payload = {
      ...formData,
      model_name: model_name
    }
    inProgress = true
    try {
      const result = await window.api.generate(payload)
      onGenerate(result)
    } catch (e: unknown) {
      onError(e)
      if (e instanceof Error) {
        errorMessage = e.message
      } else {
        errorMessage = String(e)
      }
      showErrorDialog = true
    } finally {
      inProgress = false
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
  class="dialog-overlay"
  role="button"
  tabindex="0"
  aria-label="Close dialog"
  onclick={(e) => {
    if (e.target === e.currentTarget) onClose()
  }}
  onkeydown={(e) => {
    if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault()
      onClose()
    }
  }}
>
  <div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dialog-title" tabindex="-1">
    <div class="dialog-header">
      <h3 id="dialog-title">Generate</h3>
      <button onclick={onClose}>×</button>
    </div>

    <div class="dialog-content">
      {#if isLoading}
        <p>Loading configuration...</p>
      {:else if error}
        <div class="error-message">
          <p>Error: {error}</p>
        </div>
      {:else if formConfig}
        <form
          onsubmit={(e) => {
            e.preventDefault()
            generate()
          }}
        >
          {#each formConfig as field (field.name)}
            <div class="form-field">
              <label for={field.name}>{field.label}</label>
              {#if field.type === 'textarea'}
                <textarea
                  bind:value={formData[field.name]}
                  id={field.name}
                  placeholder={field.placeholder}
                ></textarea>
              {:else if field.type === 'number'}
                <input
                  type="number"
                  bind:value={formData[field.name]}
                  id={field.name}
                  placeholder={field.placeholder}
                />
              {:else if field.type === 'boolean'}
                <input
                  type="checkbox"
                  checked={formData[field.name] === 'true' || formData[field.name] === true}
                  onchange={(e) => (formData[field.name] = e.currentTarget.checked)}
                  id={field.name}
                />
              {:else}
                <!-- Default to text input for 'string' and others -->
                <input
                  type="text"
                  bind:value={formData[field.name]}
                  id={field.name}
                  placeholder={field.placeholder}
                />
              {/if}
            </div>
          {/each}
        </form>
      {/if}
    </div>

    <div class="dialog-actions">
      <button onclick={onClose}>Cancel</button>
      <button class="primary" onclick={generate} disabled={isLoading || !!error}>
        {#if inProgress}
          <div class="spinner"></div>
        {:else}
          Generate
        {/if}
      </button>
    </div>
  </div>
</div>

<ErrorDialog
  show={showErrorDialog}
  title="Generation Failed"
  message={errorMessage}
  onclose={() => (showErrorDialog = false)}
/>

<style>
  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--color-overlay-background-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    backdrop-filter: blur(4px);
  }

  .dialog {
    background: var(--color-background-medium);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.5rem;
    min-width: 400px;
    max-width: 500px;
  }

  .dialog-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--color-border-glass-1);
  }

  .dialog-header h3 {
    margin: 0;
    color: var(--color-overlay-text);
  }

  .dialog-header button {
    background: none;
    border: none;
    color: var(--color-overlay-text);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .dialog-content {
    padding: 1.5rem;
  }

  .dialog-content label {
    display: block;
    color: var(--color-overlay-text);
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .dialog-content input[type='text'],
  .dialog-content input[type='number'],
  .dialog-content textarea {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    box-sizing: border-box;
  }

  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem 1.5rem;
    border-top: 1px solid var(--color-border-glass-1);
  }

  .dialog-actions button {
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 80px;
    min-height: 38px;
  }

  .dialog-actions button:not(.primary) {
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
  }

  .dialog-actions button.primary {
    background: var(--color-primary);
    border: 1px solid var(--color-primary);
    color: var(--color-overlay-text);
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .dialog-actions button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinner {
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-left-color: #fff;
    border-radius: 50%;
    width: 18px;
    height: 18px;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .form-field {
    margin-bottom: 1rem;
  }

  .form-field:last-child {
    margin-bottom: 0;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
  }
</style>
