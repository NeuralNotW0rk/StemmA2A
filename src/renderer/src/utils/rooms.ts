/**
 * Projects the full project graph onto one room's view.
 *
 * A node is in the room named by its `room` attribute, or wherever its top-level compound
 * ancestor is (exemplars follow their individual). A view shows:
 * - its own nodes, unchanged;
 * - each room inside it as a box holding "boundary" stand-ins: the top-level members connected
 *   to anything outside that room, with boundary individuals keeping their exemplars. Interior
 *   members and the edges among them are hidden;
 * - inside a room, faded "doorway" stand-ins for the outside nodes its members connect to.
 * Stand-ins keep the real node's ID (the real node is never in the same view), so selection and
 * operations act on the real element. They carry `_view_role` and must never save positions.
 */
import type { ElementDefinition, ElementsDefinition } from 'cytoscape'

export type ViewRole = 'boundary' | 'doorway'

export interface RoomCrumb {
  id: string
  name: string
}

interface Point {
  x: number
  y: number
}

type NodeData = Record<string, unknown> & { id: string }

const BOUNDARY_SPACING = 180
const EXEMPLAR_SPACING = 50
// Children kept with a boundary individual, so its sound can be auditioned from outside the room
const EXEMPLAR_TYPES = new Set(['audio', 'image'])

const isEdge = (ele: ElementDefinition): boolean =>
  ele.group === 'edges' || typeof ele.data?.source === 'string'

const isPoint = (value: unknown): value is Point =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as Point).x === 'number' &&
  typeof (value as Point).y === 'number'

class RoomIndex {
  readonly nodes = new Map<string, NodeData>()

  constructor(elements: ElementDefinition[]) {
    for (const ele of elements) {
      if (!isEdge(ele) && ele.data?.id) this.nodes.set(ele.data.id, ele.data as NodeData)
    }
  }

  isRoom(id: unknown): id is string {
    return typeof id === 'string' && this.nodes.get(id)?.type === 'room'
  }

  parentOf(id: string): string | null {
    const parent = this.nodes.get(id)?.parent
    return typeof parent === 'string' && this.nodes.has(parent) ? parent : null
  }

  topLevel(id: string): string {
    const seen = new Set([id])
    let parent = this.parentOf(id)
    while (parent && !seen.has(parent)) {
      id = parent
      seen.add(id)
      parent = this.parentOf(id)
    }
    return id
  }

  roomOf(id: string): string | null {
    const room = this.nodes.get(this.topLevel(id))?.room
    return this.isRoom(room) ? room : null
  }

  /** Rooms containing a node, innermost first. */
  enclosing(id: string): string[] {
    const rooms: string[] = []
    let room = this.roomOf(id)
    while (room && !rooms.includes(room)) {
      rooms.push(room)
      room = this.roomOf(room)
    }
    return rooms
  }

  inView(id: string, view: string | null): boolean {
    return view === null || this.enclosing(id).includes(view)
  }

  /** What stands for a node in a view it is somewhere inside: itself, or the room box it is in. */
  representative(id: string, view: string | null): string {
    if (this.roomOf(id) === view) return id
    return this.enclosing(id).find((room) => this.roomOf(room) === view) ?? id
  }
}

interface Endpoint {
  id: string
  role: ViewRole | null
  box?: string
}

export function flattenElements(
  graph: { elements?: ElementDefinition[] | ElementsDefinition } | null
): ElementDefinition[] {
  const elements = graph?.elements
  if (!elements) return []
  return Array.isArray(elements) ? elements : [...(elements.nodes ?? []), ...(elements.edges ?? [])]
}

export function roomPath(elements: ElementDefinition[], activeRoom: string | null): RoomCrumb[] {
  if (!activeRoom) return []
  const index = new RoomIndex(elements)
  if (!index.isRoom(activeRoom)) return []
  return [activeRoom, ...index.enclosing(activeRoom)].reverse().map((id) => ({
    id,
    name: String(index.nodes.get(id)?.name ?? id)
  }))
}

export function projectRoomView(
  elements: ElementDefinition[],
  activeRoom: string | null
): ElementDefinition[] {
  const index = new RoomIndex(elements)
  const view = index.isRoom(activeRoom) ? activeRoom : null
  const outerRooms = view ? index.enclosing(view) : []

  const mapEndpoint = (id: string): Endpoint => {
    if (index.inView(id, view)) {
      const rep = index.representative(id, view)
      if (rep === id) return { id, role: null }
      // Inside a room box: shown as the box member it belongs to, collapsed to its top level
      return { id: index.topLevel(index.representative(id, rep)), role: 'boundary', box: rep }
    }
    // Outside this room: shown at the level of the closest view containing both
    const shared = outerRooms.find((room) => index.inView(id, room)) ?? null
    return { id: index.topLevel(index.representative(id, shared)), role: 'doorway' }
  }

  const output: ElementDefinition[] = []
  const standIns = new Map<string, Endpoint>()
  const edgeIds = new Set<string>()

  for (const ele of elements) {
    if (!isEdge(ele)) continue
    const data = ele.data as NodeData & { source: string; target: string }
    if (!index.nodes.has(data.source) || !index.nodes.has(data.target)) continue
    let source = mapEndpoint(data.source)
    let target = mapEndpoint(data.target)
    // Nearly every evolved element binds to its model, so a binding would make every member a
    // boundary node. Bindings attach to the room box instead
    if (data.relation === 'binds_to') {
      if (source.box && source.box !== target.box) source = { id: source.box, role: null }
      if (target.box && target.box !== source.box) target = { id: target.box, role: null }
    }

    if (source.id === target.id) continue
    if (source.role === 'doorway' && target.role === 'doorway') continue
    // Edges among a room's own members are its interior
    if (source.box && source.box === target.box) continue
    // Similarity springs only make sense between nodes actually shown here
    if (data.type === 'spring' && (source.role || target.role)) continue

    const remapped = source.id !== data.source || target.id !== data.target
    const id = remapped ? `${source.id}->${target.id}` : data.id
    if (edgeIds.has(id)) continue
    edgeIds.add(id)
    output.push(
      remapped ? { ...ele, data: { ...data, id, source: source.id, target: target.id } } : ele
    )

    for (const endpoint of [source, target]) {
      if (endpoint.role) standIns.set(endpoint.id, endpoint)
    }
  }

  const memberCounts = new Map<string, number>()
  for (const id of index.nodes.keys()) {
    const room = index.roomOf(id)
    if (room && index.topLevel(id) === id) memberCounts.set(room, (memberCounts.get(room) ?? 0) + 1)
  }

  for (const ele of elements) {
    if (isEdge(ele) || index.roomOf(ele.data.id as string) !== view) continue
    const data = ele.data as NodeData
    output.push(
      index.isRoom(data.id)
        ? { ...ele, data: { ...data, _member_count: memberCounts.get(data.id) ?? 0 } }
        : ele
    )
  }

  // Boundary stand-ins are laid out in a grid around their box's saved position
  const boundaryByBox = new Map<string, string[]>()
  for (const [id, endpoint] of standIns) {
    if (endpoint.box)
      boundaryByBox.set(endpoint.box, [...(boundaryByBox.get(endpoint.box) ?? []), id])
  }
  const boundaryPositions = new Map<string, Point>()
  for (const [box, ids] of boundaryByBox) {
    const anchor = index.nodes.get(box)?.position
    const center = isPoint(anchor) ? anchor : { x: 0, y: 0 }
    ids.sort(
      (a, b) => Number(index.nodes.get(a)?.created ?? 0) - Number(index.nodes.get(b)?.created ?? 0)
    )
    const columns = Math.ceil(Math.sqrt(ids.length))
    const rows = Math.ceil(ids.length / columns)
    ids.forEach((id, i) => {
      boundaryPositions.set(id, {
        x: center.x + ((i % columns) - (columns - 1) / 2) * BOUNDARY_SPACING,
        y: center.y + (Math.floor(i / columns) - (rows - 1) / 2) * BOUNDARY_SPACING
      })
    })
  }

  for (const [id, endpoint] of standIns) {
    const data: NodeData = { ...(index.nodes.get(id) as NodeData), _view_role: endpoint.role }
    delete data.position
    if (endpoint.box) {
      data.parent = endpoint.box
      data.position = boundaryPositions.get(id)
    } else {
      delete data.parent
    }
    if (index.isRoom(id)) data._member_count = memberCounts.get(id) ?? 0
    // Boundary stand-ins move with their box but are arranged automatically
    output.push({
      group: 'nodes',
      data,
      classes: endpoint.role ?? undefined,
      grabbable: endpoint.role !== 'boundary'
    })

    // A boundary individual keeps its exemplars, spread in a row across its grid slot
    if (endpoint.role === 'boundary' && data.type === 'individual') {
      const slot = boundaryPositions.get(id) ?? { x: 0, y: 0 }
      const exemplars = [...index.nodes.values()].filter(
        (child) => child.parent === id && EXEMPLAR_TYPES.has(String(child.type))
      )
      exemplars.forEach((child, i) => {
        output.push({
          group: 'nodes',
          data: {
            ...child,
            _view_role: 'boundary',
            position: {
              x: slot.x + (i - (exemplars.length - 1) / 2) * EXEMPLAR_SPACING,
              y: slot.y
            }
          },
          classes: 'boundary',
          grabbable: false
        })
      })
    }
  }

  return output
}
