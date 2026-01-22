<script lang="ts">
  import { modelColor, audioColor, batchColor, externalColor } from './Style'
</script>

<div class="graph-legend">
  <div class="legend-item">
    <div class="legend-color" style="background-color: {modelColor};"></div>
    <span>Models</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background-color: {audioColor};"></div>
    <span>Audio</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background-color: {batchColor};"></div>
    <span>Batches</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background-color: {externalColor};"></div>
    <span>External</span>
  </div>
</div>

<style>
  .graph-legend {
    position: absolute;
    bottom: 1rem;
    right: 1rem;
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
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
    color: white;
  }

  .legend-color {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
</style>
