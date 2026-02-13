<div class="graph-legend">
  <div class="legend-item">
    <div class="legend-color model"></div>
    <span>Models</span>
  </div>
  <div class="legend-item">
    <div class="legend-color audio"></div>
    <span>Audio</span>
  </div>
  <div class="legend-item">
    <div class="legend-color batch"></div>
    <span>Batches</span>
  </div>
  <div class="legend-item">
    <div class="legend-color external"></div>
    <span>External</span>
  </div>
</div>

<style>
  .graph-legend {
    position: absolute;
    bottom: 1rem;
    right: 1rem;
    background: var(--color-overlay-background-primary);
    border: 1px solid var(--color-overlay-border-primary);
    border-radius: 0.5rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    backdrop-filter: blur(10px);
    z-index: 100;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: var(--color-overlay-text);
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid var(--color-overlay-border-primary);
  }
  .model {
    background-color: var(--graph-gradient-3);
  }
  .audio {
    background-color: var(--graph-audio);
  }
  .batch {
    background-color: var(--graph-batch);
  }
  .external {
    background-color: var(--graph-gradient-2);
  }
</style>
