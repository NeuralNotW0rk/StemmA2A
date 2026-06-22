import type { CssStyleDeclaration, NodeSingular } from 'cytoscape'
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
const latticeColor = gradientColor4
const latentColor = gradientColor5
const favoriteColor = 'rgb(0, 255, 255)'
const validColor = '#4CAF50'


const defaultStyle: CssStyleDeclaration[] = [
  // General style configuration
  {
    selector: 'node',
    style: {
      color: 'white',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap'
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
      label: (node: NodeSingular) => node.data('name') || node.data('id'),
      'background-color': modelColor,
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="lattice"]',
    style: {
      label: (node: NodeSingular) => node.data('name') || node.data('id'),
      'background-color': latticeColor,
      width: 50,
      height: 50
    }
  },
  {
    selector: 'node[type="audio"]',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name');
        const alias = node.data('alias');
        const promptTxt = node.data('context') && node.data('context')["prompt"];
        const secondary = alias || promptTxt;
        
        return secondary ? secondary : (name || node.data('id'));
      },
      'background-color': audioColor,
      width: 30,
      height: 30
    }
  },
  {
    selector: 'node[type="audio"].detailed',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name');
        const alias = node.data('alias');
        const promptTxt = node.data('context') && node.data('context')["prompt"];
        const secondary = alias || promptTxt;
        
        if (name && secondary && name !== secondary) {
          return `${name}\n[${secondary}]`;
        }
        return name || (secondary ? `[${secondary}]` : node.data('id'));
      }
    }
  },
  {
    selector: 'node[type="audio"][?favorite]',
    style: {
      'border-color': favoriteColor,
      'border-width': 4,
      'border-style': 'solid'
    }
  },
  {
    selector: 'node[type="latent"]',
    style: {
      label: (node: NodeSingular) => {
        const promptTxt = node.data('context') && node.data('context')["prompt"] || "[empty]";
        const strength = node.data('context') && node.data('context')["inversion_strength"];
        return `${promptTxt}\nx${strength}`;
      },
      'background-color': latentColor,
      shape: 'diamond',
      width: 30,
      height: 30
    }
  },

  // Collection-specific style configuration
  {
    selector: 'node[type="batch"]',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name');
        const alias = node.data('alias');
        const promptTxt = node.data('context') && node.data('context')["prompt"];
        const secondary = alias || promptTxt;
        
        return secondary ? secondary : (name || node.data('id'));
      },
      'text-valign': 'top',
      'background-color': batchColor,
      'background-opacity': 0.5,
      'border-width': 2,
      'shape': 'rectangle',
      'padding': '10px'
    }
  },
  {
    selector: 'node[type="batch"].detailed',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name');
        const alias = node.data('alias');
        const promptTxt = node.data('context') && node.data('context')["prompt"];
        const secondary = alias || promptTxt;
        
        if (name && secondary && name !== secondary) {
          return `${name}\n[${secondary}]`;
        }
        return name || (secondary ? `[${secondary}]` : node.data('id'));
      }
    }
  },
  {
    selector: 'node[type="batch"][member_type="audio"]',
    style: {
      'border-color': audioColor
    }
  },
  {
    selector: 'node[type="batch"][member_type="latent"]',
    style: {
      'border-color': latentColor
    }
  },
  {
    selector: 'node.compatible-drop-target',
    style: {
      'border-color': validColor,
      'border-width': 3,
      'border-style': 'dashed'
    }
  },
  {
    selector: 'node.active-drop-target',
    style: {
      'border-color': validColor,
      'border-width': 5,
      'border-style': 'solid',
      'background-opacity': 0.8
    }
  },
  {
    selector: 'node[type="local_path"]',
    style: {
      label: (node: NodeSingular) => node.data('name') || node.data('id'),
      'background-color': externalColor,
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="directory"]',
    style: {
      label: 'data(path)',
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
      'display': 'none',
      'curve-style': 'haystack'
    }
  },
  {
    selector: 'edge[type="spring"].visible',
    style: {
      'display': 'element',
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
    selector: 'edge[type="model"]',
    style: {
      'line-color': modelColor,
      'target-arrow-color': modelColor
    }
  },
  {
    selector: 'edge[type="lattice"]',
    style: {
      'line-color': latticeColor,
      'target-arrow-color': latticeColor
    }
  },
  {
    selector: 'edge[relation="binds_to"]',
    style: {
      'line-style': 'dashed'
    }
  },
  {
    selector: 'edge[type="audio"]',
    style: {
      'line-color': audioColor,
      'target-arrow-color': audioColor
    }
  },
  {
    selector: 'edge[type="latent"]',
    style: {
      'line-color': latentColor,
      'target-arrow-color': latentColor
    }
  },
  {
    selector: 'edge[type="local_path"]',
    style: {
      'line-color': externalColor,
      'target-arrow-color': externalColor
    }
  },


  // Overrides
  {
    selector: 'node.highlighted',
    style: {
      'border-color': validColor,
      'border-width': 3,
      'border-style': 'dashed'
    }
  },
  {
    selector: 'node.bound',
    style: {
      'border-color': 'yellow',
      'border-width': 4,
      'border-style': 'solid'
    }
  },
  {
    selector: 'node.bound-active',
    style: {
      'border-color': validColor,
      'border-width': 4,
      'border-style': 'solid'
    }
  },
  {
    selector: '.dimmed, node.dimmed, edge.dimmed, edge[type="spring"].dimmed',
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
