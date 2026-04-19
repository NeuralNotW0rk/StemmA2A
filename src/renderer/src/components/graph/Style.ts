import type { CssStyleDeclaration } from 'cytoscape'
import { getCssVar } from '../../utils/css'

const gradientColor1 = getCssVar('--graph-gradient-1')
const gradientColor2 = getCssVar('--graph-gradient-2')
const gradientColor3 = getCssVar('--graph-gradient-3')
const gradientColor4 = getCssVar('--graph-gradient-4')
const gradientColor5 = getCssVar('--graph-gradient-5')

const audioColor = getCssVar('--graph-audio')
const batchColor = getCssVar('--graph-batch')
const selectedColor = getCssVar('--graph-selected')
const boundColor = getCssVar('--graph-bound')
const modelColor = gradientColor3
const externalColor = gradientColor2

const defaultStyle: CssStyleDeclaration[] = [
  // General style configuration
  {
    selector: 'node',
    style: {
      color: 'white',
      'text-valign': 'center',
      'text-halign': 'center'
    }
  },
  {
    selector: 'edge',
    style: {
      color: 'white',
      'edge-text-rotation': 'autorotate',
      'curve-style': 'straight',
      'target-arrow-shape': 'triangle',
      'z-compound-depth': 'bottom'
    }
  },

  // Artifact-specific style configuration
  {
    selector: 'node[type="model"]',
    style: {
      label: 'data(name)',
      'background-color': modelColor,
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="audio"]',
    style: {
      label: 'data(name)',
      'background-color': audioColor,
      width: 30,
      height: 30
    }
  },
  {
    selector: 'node[type="audio"][rating="1"]',
    style: {
      'border-color': gradientColor1,
      'border-width': 2
    }
  },
  {
    selector: 'node[type="audio"][rating="2"]',
    style: {
      'border-color': gradientColor2,
      'border-width': 2
    }
  },
  {
    selector: 'node[type="audio"][rating="3"]',
    style: {
      'border-color': gradientColor3,
      'border-width': 2
    }
  },
  {
    selector: 'node[type="audio"][rating="4"]',
    style: {
      'border-color': gradientColor4,
      'border-width': 2
    }
  },
  {
    selector: 'node[type="audio"][rating="5"]',
    style: {
      'border-color': gradientColor5,
      'border-width': 2
    }
  },

  // Collection-specific style configuration
  {
    selector: 'node[type="batch"]',
    style: {
      label: 'data(id)',
      'text-valign': 'top',
      'background-color': batchColor,
      'background-opacity': 0.5,
      'border-width': 2,
      'shape': 'rectangle',
      'padding': '10px'
    }
  },
  {
    selector: 'node[type="batch"][member_type="audio"]',
    style: {
      'border-color': audioColor
    }
  },
  {
    selector: 'node[type="local_path"]',
    style: {
      label: 'data(name)',
      'background-color': externalColor,
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="directory"]',
    style: {
      label: 'data(id)',
      'text-valign': 'top',
      'background-color': batchColor,
      'background-opacity': 0.5,
      'border-color': externalColor,
      'border-width': 2,
      'shape': 'rectangle',
      'padding': '10px'
    }
  },

  // Edge-specific style configuration
  {
    selector: 'edge[type="spring"]',
    style: {
      'line-color': '#ffff00',
      'opacity': 'mapData(weight, 0, 1, 0, 1)',
      'curve-style': 'straight', // Changed from haystack to support edge labels
      'source-label': 'data(source_label)',
      'source-text-offset': 20,
      'source-text-rotation': 'autorotate',
      'target-arrow-shape': 'none',
      'font-size': 5,
      'text-background-color': '#111111',
      'text-background-opacity': 0.8,
      'text-background-padding': 4,
      'text-background-shape': 'roundrectangle',
    }
  },
  {
    selector: 'edge[type="spring"].hidden',
    style: {
      'display': 'none'
    }
  },
  {
    selector: 'edge[type="model"]',
    style: {
      label: 'data(seed)',
      'line-color': modelColor,
      'target-arrow-color': modelColor
    }
  },
  {
    selector: 'edge[type="audio"]',
    style: {
      label: 'data(strength)',
      'line-color': audioColor,
      'target-arrow-color': audioColor,
      'line-style': 'dashed'
    }
  },
  {
    selector: 'edge[type="local_path"]',
    style: {
      label: 'data(strength)',
      'line-color': externalColor,
      'target-arrow-color': externalColor
    }
  },

  // Overrides
  {
    selector: 'node.bound',
    style: {
      'border-color': boundColor,
      'border-width': '3px',
      'border-style': 'double'
    }
  },
  {
    selector: 'node.highlighted',
    style: {
      'border-color': 'yellow',
      'border-width': '3px'
    }
  },
  {
    selector: 'node.dimmed',
    style: {
      opacity: 0.4
    }
  },
  {
    selector: ':selected',
    style: {
      'border-color': selectedColor,
      'border-width': 4,
      'line-color': selectedColor,
      'target-arrow-color': selectedColor
    }
  }
]

export default defaultStyle
