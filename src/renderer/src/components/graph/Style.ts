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

  // Node-specific style configuration
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
    selector: 'node[type="external"]',
    style: {
      label: 'data(name)',
      'background-color': externalColor,
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="batch"], node[type="set"]',
    style: {
      label: 'data(alias)',
      'text-valign': 'top',
      'background-color': batchColor,
      'background-opacity': 0.5,
      'border-color': audioColor,
      'border-width': 2
    }
  },
  {
    selector: 'node[type="audio"]',
    style: {
      label: 'data(alias)',
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

  // Edge-specific style configuration
  {
    selector: 'edge[type="dd_generation"]',
    style: {
      label: 'data(seed)',
      'line-color': modelColor,
      'target-arrow-color': modelColor
    }
  },
  {
    selector: 'edge[type="dd_variation"]',
    style: {
      label: 'data(seed)',
      'line-color': modelColor,
      'target-arrow-color': modelColor,
      'line-style': 'dashed'
    }
  },
  {
    selector: 'edge[type="audio_source"]',
    style: {
      label: 'data(strength)',
      'line-color': audioColor,
      'target-arrow-color': audioColor,
      'line-style': 'dashed'
    }
  },
  {
    selector: 'edge[type="import"]',
    style: {
      'line-color': externalColor,
      'target-arrow-color': externalColor
    }
  },

  // Overrides
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
