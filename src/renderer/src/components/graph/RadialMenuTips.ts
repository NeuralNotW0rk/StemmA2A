// Tips for the radial (context) menu's commands, shown while a command is highlighted. Keyed by the
// command's label; an entry for "scope:label" (scope is the node type, 'edge' or 'core' for the empty
// canvas) wins over the plain label where a command means something different there. Pinned
// operations carry their own tip, from the operation's description.

const COMMAND_TIPS: Record<string, string> = {
  // Shared by most nodes
  Rename: "Gives this element a new display name. Its identity and its files don't change.",
  'Break Out':
    'Moves this element, or the selection it is part of, into a new room: a view of its own, which keeps the main graph tidy. Its edges to the rest of the graph stay attached to the room.',
  'Fork to Project':
    'Copies this element, or the selection it is part of, into a new project, together with the models and ancestors it depends on. This project is left as it is.',
  Remove:
    'Removes this element, or the selection it is part of, from the graph, after a confirmation. Files it used stay in the project until Clean Up Project Files clears them.',
  'Operations...':
    'Opens the full list of operations that can run on this element, including those not pinned to this menu.',
  Replicate:
    'Reopens the operation that made this element, with its settings filled in, so you can make variations.',

  // Model
  Bend: "Opens the Bend panel to make a grating that transforms one of this model's layers, for example by eroding its feature maps or silencing channels.",
  'Import Grating':
    'Adds a grating saved elsewhere as a .safetensors file and binds it to this model.',

  // Individual
  Score:
    "Sets this individual's fitness score. Recombine's tournament selection favors higher scores; only their order matters, not the gaps between them.",

  // Audio and image
  Favorite: 'Marks this element as a favorite, with a cyan ring, so it stands out in the graph.',
  Unfavorite: 'Removes the favorite mark from this element.',
  Export:
    'Saves a copy of this audio file outside the project. Its settings travel inside the file, so importing it again restores its history.',

  // Sources
  Expand:
    "Loads this folder's audio and image files as nodes, inside a directory box. Subfolders appear as folders that can be expanded in turn.",
  Rescan: 'Checks this source again for files added or removed since it was last scanned.',
  'directory:Refresh':
    'Loads files added to the folder since it was expanded, and removes the nodes of files that are gone.',

  // Group
  'Update Group': 'Opens the grouping panel to choose which elements belong to this group.',
  'Export Group': "Saves copies of the group's members outside the project.",

  // Room
  'Enter Room': 'Opens this room in its own view. The breadcrumb in the toolbar leads back out.',
  'room:Rename': 'Gives this room a new name.',
  Rejoin: 'Dissolves this room, returning everything inside it to the surrounding graph.',

  // Edge
  'edge:Remove':
    'Removes this edge from the graph, after a confirmation. The elements at either end stay.',

  // Empty canvas
  'core:Tidy': 'Runs the layout again to tidy the graph.',
  'core:Fit': 'Zooms to fit the whole graph in view.',
  'core:Create Group':
    'Creates an empty group at this spot. Use Update Group, or Ctrl-drag nodes into it, to fill it.'
}

export function radialCommandTip(scope: string, label: string): string {
  const tip = COMMAND_TIPS[`${scope}:${label}`] ?? COMMAND_TIPS[label]
  if (!tip) {
    throw new Error(`No radial menu tip is defined for '${label}' on '${scope}'.`)
  }
  return tip
}
