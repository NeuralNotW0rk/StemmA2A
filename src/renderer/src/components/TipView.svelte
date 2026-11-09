<script lang="ts">
  // Fixed contextual help panel. Any element with a data-tip attribute shows its
  // text here while hovered or focused, so hints never cover what they describe.
  // An optional data-tip-title on the same element is shown as a heading above it.
  // Elements without DOM nodes, such as those drawn on the graph canvas, set externalTip instead.
  import { externalTip, type Tip } from '../utils/tips'

  // Grace period before clearing, so the tip doesn't flicker across gaps between controls
  const CLEAR_DELAY_MS = 250

  // The DOM tip, from data-tip attributes
  let tip = $state<Tip | null>(null)
  let clearTimer: number | null = null

  // The external tip, cleared after the same grace period so moving between adjacent graph
  // elements doesn't flicker. It takes precedence, since the canvas itself carries no data-tip.
  let graphTip = $state<Tip | null>(null)
  let graphClearTimer: number | null = null
  let shown = $derived(graphTip ?? tip)

  function cancelGraphClear(): void {
    if (graphClearTimer !== null) {
      window.clearTimeout(graphClearTimer)
      graphClearTimer = null
    }
  }

  $effect((): (() => void) => {
    const unsubscribe = externalTip.subscribe((value): void => {
      if (value) {
        cancelGraphClear()
        graphTip = value
      } else if (graphClearTimer === null) {
        graphClearTimer = window.setTimeout((): void => {
          graphClearTimer = null
          graphTip = null
        }, CLEAR_DELAY_MS)
      }
    })
    return (): void => {
      unsubscribe()
      cancelGraphClear()
    }
  })

  function tippedElement(target: EventTarget | null): HTMLElement | SVGElement | null {
    if (!(target instanceof Element)) return null
    const tipped = target.closest<HTMLElement | SVGElement>('[data-tip]')
    return tipped?.dataset.tip ? tipped : null
  }

  function tipFrom(tipped: HTMLElement | SVGElement): Tip {
    return { title: tipped.dataset.tipTitle || null, body: tipped.dataset.tip ?? '' }
  }

  function cancelClear(): void {
    if (clearTimer !== null) {
      window.clearTimeout(clearTimer)
      clearTimer = null
    }
  }

  function scheduleClear(): void {
    if (clearTimer !== null) return
    clearTimer = window.setTimeout((): void => {
      clearTimer = null
      tip = null
    }, CLEAR_DELAY_MS)
  }

  function updateFrom(target: EventTarget | null): void {
    const tipped = tippedElement(target)
    if (tipped) {
      cancelClear()
      tip = tipFrom(tipped)
      // A control with its own tip takes over from a lingering graph tip straight away
      cancelGraphClear()
      graphTip = null
    } else {
      scheduleClear()
    }
  }

  // Shows the live value next to the title while a tipped slider or number field changes
  function showValue(target: EventTarget | null): void {
    if (!(target instanceof HTMLInputElement)) return
    if (target.type !== 'range' && target.type !== 'number') return
    const tipped = tippedElement(target)
    if (!tipped) return
    const base = tipFrom(tipped)
    const name = base.title ?? target.labels?.[0]?.textContent?.trim() ?? 'Value'
    cancelClear()
    tip = { title: `${name}: ${target.value}`, body: base.body }
  }

  $effect((): (() => void) => {
    const onPointerOver = (e: PointerEvent): void => updateFrom(e.target)
    const onFocusIn = (e: FocusEvent): void => updateFrom(e.target)
    const onInput = (e: Event): void => showValue(e.target)
    window.addEventListener('pointerover', onPointerOver)
    window.addEventListener('focusin', onFocusIn)
    window.addEventListener('input', onInput)
    return (): void => {
      window.removeEventListener('pointerover', onPointerOver)
      window.removeEventListener('focusin', onFocusIn)
      window.removeEventListener('input', onInput)
      cancelClear()
    }
  })
</script>

<div class="tip-view" role="status" aria-live="polite">
  {#if shown}
    {#if shown.title}
      <span class="tip-title">{shown.title}</span>
    {/if}
    <span class="tip-body">{shown.body}</span>
  {:else}
    <span class="tip-placeholder">Hover over a control or graph element to see what it does</span>
  {/if}
</div>

<style>
  .tip-view {
    position: absolute;
    left: 1rem;
    bottom: 1rem;
    z-index: 1000;
    box-sizing: border-box;
    width: 320px;
    height: var(--tip-view-height);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.625rem 0.75rem;
    overflow: hidden;
    background: var(--color-background-glass-1);
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.5rem;
    backdrop-filter: blur(10px);
    color: var(--color-overlay-text);
    font-size: 0.8rem;
    line-height: 1.4;
    pointer-events: none;
  }

  .tip-title {
    flex-shrink: 0;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tip-body {
    /* Line breaks separate a select's own tip from its selected option's */
    white-space: pre-line;
    display: -webkit-box;
    -webkit-line-clamp: 8;
    line-clamp: 8;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .tip-placeholder {
    color: var(--color-text-overlay-secondary);
  }
</style>
