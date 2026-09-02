import type { CssStyleDeclaration, NodeSingular, EdgeSingular } from 'cytoscape'
import { getCssVar } from '../../utils/css'
import modelIcon from '../../assets/icons/model.svg'
import gratingIcon from '../../assets/icons/grating.svg'
import audioIcon from '../../assets/icons/audio.svg'
import latentIcon from '../../assets/icons/latent.svg'
import pathIcon from '../../assets/icons/local_path.svg'
import imageIcon from '../../assets/icons/image.svg'
import dnaIcon from '../../assets/icons/dna.svg'

import audioSvg from '../../assets/icons/audio.svg?raw'
import imageSvg from '../../assets/icons/image.svg?raw'
import latentSvg from '../../assets/icons/latent.svg?raw'
import gratingSvg from '../../assets/icons/grating.svg?raw'
import modelSvg from '../../assets/icons/model.svg?raw'
import pathSvg from '../../assets/icons/local_path.svg?raw'
import dnaSvg from '../../assets/icons/dna.svg?raw'
import bundleSvg from '../../assets/icons/bundle.svg?raw'

const gradientColor1 = getCssVar('--graph-gradient-1')
const gradientColor2 = getCssVar('--graph-gradient-2')
const gradientColor3 = getCssVar('--graph-gradient-3')
const gradientColor4 = getCssVar('--graph-gradient-4')
const gradientColor5 = getCssVar('--graph-gradient-5')

const mediaColor = getCssVar('--graph-media')
const groupColor = getCssVar('--graph-batch')
const selectedColor = getCssVar('--graph-selected')
const genomeColor = getCssVar('--graph-genome') || '#ff7a00'
const modelColor = gradientColor3
const externalColor = gradientColor2
const gratingColor = gradientColor4
const latentColor = gradientColor5
const favoriteColor = 'rgb(0, 255, 255)'
const validColor = '#4CAF50'

function extractSvgInner(svgContent: string): string {
  const match = svgContent.match(/<svg[^>]*>([\s\S]*?)<\/svg>/i)
  return match ? match[1].trim() : svgContent
}

function createStackedBundleSvg(rawSvg: string): string {
  const inner = extractSvgInner(rawSvg)
  const stackedSvg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">
  <defs>
    <g id="icon">
      ${inner}
    </g>
  </defs>
  <use href="#icon" xlink:href="#icon" transform="translate(7.5, 7.5) scale(0.65) translate(-12, -12)" opacity="0.45" />
  <use href="#icon" xlink:href="#icon" transform="translate(12, 12) scale(0.65) translate(-12, -12)" opacity="0.75" />
  <use href="#icon" xlink:href="#icon" transform="translate(16.5, 16.5) scale(0.65) translate(-12, -12)" opacity="1" />
</svg>`

  return `data:image/svg+xml;utf8,${encodeURIComponent(stackedSvg)}`
}

const BUNDLE_STACKED_ICONS: Record<string, string> = {
  individual: createStackedBundleSvg(dnaSvg),
  audio: createStackedBundleSvg(audioSvg),
  image: createStackedBundleSvg(imageSvg),
  latent: createStackedBundleSvg(latentSvg),
  grating: createStackedBundleSvg(gratingSvg),
  model: createStackedBundleSvg(modelSvg),
  local_path: createStackedBundleSvg(pathSvg),
  default: `data:image/svg+xml;utf8,${encodeURIComponent(bundleSvg)}`
}

function getBundleIcon(memberType?: string): string {
  if (memberType && BUNDLE_STACKED_ICONS[memberType]) {
    return BUNDLE_STACKED_ICONS[memberType]
  }
  return BUNDLE_STACKED_ICONS.default
}

function getBundleColor(memberType?: string): string {
  switch (memberType) {
    case 'individual':
      return genomeColor
    case 'audio':
    case 'image':
      return mediaColor
    case 'latent':
      return latentColor
    case 'grating':
      return gratingColor
    case 'model':
      return modelColor
    default:
      return externalColor
  }
}

const defaultStyle: CssStyleDeclaration[] = [
  // General style configuration
  {
    selector: 'node',
    style: {
      color: 'white',
      'text-valign': 'top',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-margin-y': -6
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
      'background-image': modelIcon,
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '75%',
      'background-height': '75%',
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="grating"]',
    style: {
      label: (node: NodeSingular) => node.data('name') || node.data('id'),
      'background-color': gratingColor,
      'background-image': gratingIcon,
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '75%',
      'background-height': '75%',
      width: 50,
      height: 50
    }
  },
  {
    selector: 'node[type="audio"]',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name')
        const alias = node.data('alias')
        const promptTxt = node.data('context') && node.data('context')['prompt']
        const secondary = alias || promptTxt

        return secondary ? secondary : name || node.data('id')
      },
      'background-color': mediaColor,
      'background-image': audioIcon,
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '75%',
      'background-height': '75%',
      width: 30,
      height: 30
    }
  },
  {
    selector: 'node[type="audio"].detailed',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name')
        const alias = node.data('alias')
        const promptTxt = node.data('context') && node.data('context')['prompt']
        const secondary = alias || promptTxt

        if (name && secondary && name !== secondary) {
          return `${name}\n[${secondary}]`
        }
        return name || (secondary ? `[${secondary}]` : node.data('id'))
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
    selector: 'node[type="image"]',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name')
        const alias = node.data('alias')
        const promptTxt = node.data('context') && node.data('context')['prompt']
        const secondary = alias || promptTxt

        return secondary ? secondary : name || node.data('id')
      },
      'background-color': mediaColor,
      'background-image': imageIcon,
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '75%',
      'background-height': '75%',
      width: 30,
      height: 30
    }
  },
  {
    selector: 'node[type="image"].detailed',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name')
        const alias = node.data('alias')
        const promptTxt = node.data('context') && node.data('context')['prompt']
        const secondary = alias || promptTxt

        if (name && secondary && name !== secondary) {
          return `${name}\n[${secondary}]`
        }
        return name || (secondary ? `[${secondary}]` : node.data('id'))
      }
    }
  },
  {
    selector: 'node[type="image"][?favorite]',
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
        const context = node.data('context')
        const isUnconditional =
          context &&
          (context['inversion_unconditional'] === true ||
            (context['inversion_metadata'] &&
              context['inversion_metadata']['inversion_unconditional'] === true))
        const promptTxt =
          (context && context['prompt']) || (isUnconditional ? '[unconditional]' : '[empty]')
        const strength = context && context['inversion_strength']
        return `${promptTxt}\nx${strength}`
      },
      'background-color': latentColor,
      'background-image': latentIcon,
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '75%',
      'background-height': '75%',
      width: 30,
      height: 30
    }
  },

  // Collection-specific style configuration
  {
    selector: 'node[type="group"]',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name')
        const alias = node.data('alias')
        const promptTxt = node.data('context') && node.data('context')['prompt']
        const secondary = alias || promptTxt

        return secondary ? secondary : name || node.data('id')
      },
      'text-valign': 'top',
      'text-margin-y': 0,
      'background-color': groupColor,
      'background-opacity': 0.5,
      'border-width': 2,
      shape: 'rectangle',
      padding: '10px'
    }
  },
  {
    selector: 'node[type="group"].detailed',
    style: {
      label: (node: NodeSingular) => {
        const name = node.data('name')
        const alias = node.data('alias')
        const promptTxt = node.data('context') && node.data('context')['prompt']
        const secondary = alias || promptTxt

        if (name && secondary && name !== secondary) {
          return `${name}\n[${secondary}]`
        }
        return name || (secondary ? `[${secondary}]` : node.data('id'))
      }
    }
  },
  {
    selector: 'node[type="group"][member_type="audio"]',
    style: {
      'border-color': mediaColor
    }
  },
  {
    selector: 'node[type="group"][member_type="image"]',
    style: {
      'border-color': mediaColor
    }
  },
  {
    selector: 'node[type="group"][member_type="latent"]',
    style: {
      'border-color': latentColor
    }
  },
  {
    selector: 'node[type="group"][member_type="individual"]',
    style: {
      'border-color': genomeColor
    }
  },
  {
    selector: 'node[type="individual"]',
    style: {
      label: (node: NodeSingular): string => {
        const name = (node.data('name') || node.data('id')) as string
        const fitness = node.data('fitness')
        if (fitness !== undefined && fitness !== null) {
          return `${name}\n[fit: ${fitness}]`
        }
        return name
      },
      'text-valign': 'top',
      'text-margin-y': -8,
      'border-width': 2,
      'border-color': genomeColor,
      'border-style': 'dashed',
      'background-color': genomeColor,
      'background-opacity': 0.08,
      'background-image': (node: NodeSingular): string =>
        (node.isParent && node.isParent()) || node.children().length > 0 ? 'none' : dnaIcon,
      'background-fit': 'contain',
      'background-clip': 'node',
      'background-width': '75%',
      'background-height': '75%',
      shape: 'rectangle',
      'compound-sizing-wrt-labels': 'include',
      padding: '12px',
      width: 30,
      height: 30
    }
  },
  {
    selector: 'node[type="bundle"]',
    style: {
      label: (node: NodeSingular): string => {
        const name = (node.data('name') || node.data('alias') || node.data('id')) as string
        const fitness = node.data('fitness')
        if (fitness !== undefined && fitness !== null) {
          return `${name}\n[fit: ${fitness}]`
        }
        return name
      },
      'background-color': (node: NodeSingular): string => getBundleColor(node.data('member_type')),
      'background-image': (node: NodeSingular): string => getBundleIcon(node.data('member_type')),
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '70%',
      'background-height': '70%',
      'border-width': 3,
      'border-color': (node: NodeSingular): string => getBundleColor(node.data('member_type')),
      width: 40,
      height: 40
    }
  },
  {
    selector: 'node[type="bundle"][member_type="individual"]',
    style: {
      'background-color': genomeColor,
      'background-image': getBundleIcon('individual'),
      'border-color': genomeColor
    }
  },
  {
    selector: 'node[type="bundle"][member_type="audio"]',
    style: {
      'background-color': mediaColor,
      'background-image': getBundleIcon('audio'),
      'border-color': mediaColor
    }
  },
  {
    selector: 'node[type="bundle"][member_type="image"]',
    style: {
      'background-color': mediaColor,
      'background-image': getBundleIcon('image'),
      'border-color': mediaColor
    }
  },
  {
    selector: 'node[type="bundle"][member_type="latent"]',
    style: {
      'background-color': latentColor,
      'background-image': getBundleIcon('latent'),
      'border-color': latentColor
    }
  },
  {
    selector: 'node[type="bundle"][member_type="grating"]',
    style: {
      'background-color': gratingColor,
      'background-image': getBundleIcon('grating'),
      'border-color': gratingColor
    }
  },
  {
    selector: 'node[type="bundle"][member_type="model"]',
    style: {
      'background-color': modelColor,
      'background-image': getBundleIcon('model'),
      'border-color': modelColor
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
      'background-image': pathIcon,
      'background-fit': 'none',
      'background-clip': 'node',
      'background-width': '60%',
      'background-height': '60%',
      width: 60,
      height: 60
    }
  },
  {
    selector: 'node[type="directory"]',
    style: {
      label: 'data(path)',
      'text-valign': 'top',
      'text-margin-y': 0,
      'background-color': groupColor,
      'background-opacity': 0.5,
      'border-color': externalColor,
      'border-width': 2,
      shape: 'rectangle',
      padding: '10px'
    }
  },

  // Edge-specific style configuration
  {
    selector: 'edge[type="spring"]',
    style: {
      display: 'none',
      'curve-style': 'haystack'
    }
  },
  {
    selector: 'edge[type="spring"].visible',
    style: {
      display: 'element',
      'line-color': '#ffff00',
      opacity: 'mapData(weight, 0, 1, 0, 1)',
      'curve-style': 'straight', // Changed from haystack to support edge labels
      'source-label': 'data(source_label)',
      'source-text-offset': 20,
      'source-text-rotation': 'autorotate',
      'target-arrow-shape': 'none',
      'font-size': 5,
      'text-background-color': '#111111',
      'text-background-opacity': 0.8,
      'text-background-padding': 4,
      'text-background-shape': 'roundrectangle'
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
    selector: 'edge[type="grating"]',
    style: {
      'line-color': gratingColor,
      'target-arrow-color': gratingColor
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
      'line-color': mediaColor,
      'target-arrow-color': mediaColor
    }
  },
  {
    selector: 'edge[type="image"]',
    style: {
      'line-color': mediaColor,
      'target-arrow-color': mediaColor
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
  {
    selector: 'edge[type="individual"]',
    style: {
      'line-color': genomeColor,
      'target-arrow-color': genomeColor
    }
  },
  {
    selector: 'edge[type="bundle"]',
    style: {
      'line-color': (edge: EdgeSingular): string => {
        const sourceNode = edge.source()
        return getBundleColor(sourceNode.data('member_type'))
      },
      'target-arrow-color': (edge: EdgeSingular): string => {
        const sourceNode = edge.source()
        return getBundleColor(sourceNode.data('member_type'))
      }
    }
  },
  {
    selector: 'edge[relation="member"]',
    style: {
      'line-color': genomeColor,
      'target-arrow-color': genomeColor,
      'line-style': 'dashed'
    }
  },
  {
    selector: 'edge[relation="expressed_to"]',
    style: {
      'line-color': genomeColor,
      'target-arrow-color': genomeColor,
      'line-style': 'dotted'
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
