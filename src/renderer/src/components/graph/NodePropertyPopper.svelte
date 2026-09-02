<!-- src/renderer/src/components/graph/NodePropertyPopper.svelte -->
<script lang="ts">
  import type cytoscape from 'cytoscape'
  import { createPopper, type Instance } from '@popperjs/core'

  interface Props {
    targetNode: cytoscape.NodeSingular
    title: string
    inputType?: 'number' | 'text'
    initialValue?: string | number | null
    placeholder?: string
    submitLabel?: string
    allowClear?: boolean
    clearLabel?: string
    step?: string
    onsubmit: (value: string | number | null) => Promise<void> | void
    onclose: () => void
  }

  let {
    targetNode,
    title,
    inputType = 'text',
    initialValue = null,
    placeholder = '',
    submitLabel = 'Save',
    allowClear = false,
    clearLabel = 'Clear',
    step = 'any',
    onsubmit,
    onclose
  }: Props = $props()

  let inputVal = $state<number | string | null>(initialValue)
  let inputElement = $state<HTMLInputElement | null>(null)
  let popperElement = $state<HTMLElement | null>(null)
  let isSubmitting = $state<boolean>(false)
  let errorMessage = $state<string | null>(null)
  let popperInstance: Instance | null = null

  $effect(() => {
    inputVal = initialValue
    errorMessage = null
    setTimeout(() => {
      if (inputElement) {
        inputElement.focus()
        inputElement.select()
      }
    }, 50)
  })

  $effect(() => {
    if (targetNode && popperElement) {
      const cy = targetNode.cy()
      const ref = targetNode.popperRef()
      popperInstance = createPopper(ref, popperElement, {
        placement: 'right-start',
        strategy: 'fixed',
        modifiers: [
          {
            name: 'offset',
            options: {
              offset: [0, 10]
            }
          }
        ]
      })

      const update = (): void => {
        if (popperInstance) popperInstance.update()
      }

      targetNode.on('position', update)
      cy.on('pan zoom resize', update)

      return () => {
        if (targetNode) targetNode.off('position', update)
        if (cy) cy.off('pan zoom resize', update)
        if (popperInstance) {
          popperInstance.destroy()
          popperInstance = null
        }
      }
    }
    return undefined
  })

  function handleClickOutside(e: MouseEvent): void {
    if (popperElement && !popperElement.contains(e.target as Node)) {
      onclose()
    }
  }

  async function handleSubmit(): Promise<void> {
    let valToSubmit: string | number | null = null
    if (inputType === 'number') {
      const strVal = inputVal == null ? '' : String(inputVal).trim()
      if (strVal !== '') {
        const parsed = Number(strVal)
        if (isNaN(parsed)) {
          errorMessage = 'Please enter a valid number.'
          return
        }
        valToSubmit = parsed
      } else {
        valToSubmit = null
      }
    } else {
      valToSubmit = inputVal == null ? '' : String(inputVal)
    }

    isSubmitting = true
    errorMessage = null
    try {
      await onsubmit(valToSubmit)
      onclose()
    } catch (err: unknown) {
      errorMessage = err instanceof Error ? err.message : String(err)
    } finally {
      isSubmitting = false
    }
  }

  async function handleClear(): Promise<void> {
    inputVal = null
    await handleSubmit()
  }

  function handleKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleSubmit()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      onclose()
    }
  }
</script>

<svelte:window onmousedown={handleClickOutside} />

<div
  class="node-property-popper"
  bind:this={popperElement}
  onkeydown={handleKeyDown}
  role="dialog"
  aria-label={title}
  tabindex="-1"
>
  <div class="popper-header">
    <span class="popper-title">{title.toUpperCase()}</span>
    <span class="popper-node-id">{targetNode.data('name') || targetNode.id()}</span>
  </div>

  <div class="popper-body">
    <div class="popper-input-row">
      <input
        bind:this={inputElement}
        type={inputType}
        {step}
        bind:value={inputVal}
        {placeholder}
        disabled={isSubmitting}
      />
    </div>

    {#if errorMessage}
      <div class="popper-error">{errorMessage}</div>
    {/if}

    <div class="popper-actions">
      {#if allowClear}
        <button type="button" class="btn-clear" onclick={handleClear} disabled={isSubmitting}>
          {clearLabel}
        </button>
      {/if}
      <button type="button" class="btn-cancel" onclick={onclose} disabled={isSubmitting}>
        Cancel
      </button>
      <button type="button" class="btn-save" onclick={handleSubmit} disabled={isSubmitting}>
        {isSubmitting ? 'Saving...' : submitLabel}
      </button>
    </div>
  </div>
</div>

<style>
  .node-property-popper {
    position: fixed;
    width: 250px;
    background: var(--color-background-glass-2, rgba(20, 20, 25, 0.9));
    backdrop-filter: blur(14px);
    border: 1px solid var(--color-overlay-border-primary, rgba(255, 255, 255, 0.18));
    border-radius: 8px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    z-index: 10000;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    font-family: inherit;
  }

  .popper-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.04);
    gap: 0.5rem;
  }

  .popper-title {
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    color: var(--color-overlay-text, #fff);
  }

  .popper-node-id {
    font-size: 0.65rem;
    color: var(--color-text-muted, #aaa);
    max-width: 110px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    background: rgba(255, 255, 255, 0.08);
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
  }

  .popper-body {
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .popper-input-row input {
    width: 100%;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    color: #fff;
    font-size: 0.9rem;
    padding: 0.4rem 0.6rem;
    outline: none;
    transition:
      border-color 0.15s,
      box-shadow 0.15s;
    box-sizing: border-box;
  }

  .popper-input-row input:focus {
    border-color: var(--color-primary, #eb5e28);
    box-shadow: 0 0 0 1px var(--color-primary, #eb5e28);
  }

  .popper-error {
    font-size: 0.75rem;
    color: var(--color-error, #ef4444);
    line-height: 1.2;
  }

  .popper-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.4rem;
  }

  .popper-actions button {
    font-size: 0.75rem;
    padding: 0.35rem 0.65rem;
    border-radius: 4px;
    cursor: pointer;
    border: none;
    font-weight: 500;
    transition:
      background 0.15s,
      opacity 0.15s;
  }

  .popper-actions button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .popper-actions .btn-cancel {
    background: rgba(255, 255, 255, 0.08);
    color: var(--color-overlay-text, #ccc);
  }

  .popper-actions .btn-cancel:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.15);
    color: #fff;
  }

  .popper-actions .btn-clear {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .popper-actions .btn-clear:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.25);
  }

  .popper-actions .btn-save {
    background: var(--color-primary, #eb5e28);
    color: #fff;
  }

  .popper-actions .btn-save:hover:not(:disabled) {
    background: var(--color-primary-hover, #d44d18);
  }
</style>
