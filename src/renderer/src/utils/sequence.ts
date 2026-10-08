// Parsing for "sequence mode" fields, whose values are expanded into a cartesian product of jobs.
//
// A sequence is a comma-separated list of items. Each item is one of:
//   - a number:            -1, 0.5, 1e-3
//   - a range:             start..end         (step 1, inclusive of end)
//   - a stepped range:     start..end:step    (step is a positive magnitude; direction follows start/end)
//   - anything else, kept as a string
//
// Examples: "-1..1:0.5" -> [-1, -0.5, 0, 0.5, 1], "3..1" -> [3, 2, 1]

const NUM = String.raw`[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?`
const RANGE = new RegExp(String.raw`^(${NUM})\s*\.\.\s*(${NUM})(?:\s*:\s*(${NUM}))?$`, 'i')
// The previous hyphen syntax ("1-5", "0-1:0.1"), rejected with a pointer to the new one
const LEGACY_RANGE = new RegExp(String.raw`^(${NUM})-(${NUM})(?::(${NUM}))?$`, 'i')

function expandRange(part: string, start: number, end: number, step: number): number[] {
  if (!(step > 0)) {
    throw new Error(`Invalid step in range "${part}": step must be positive.`)
  }
  const direction = end >= start ? 1 : -1
  // Step by index rather than accumulating, so float ranges like 0..1:0.1 land on clean values
  const count = Math.floor(Math.abs(end - start) / step + 1e-9)
  const values: number[] = []
  for (let k = 0; k <= count; k++) {
    values.push(Number((start + direction * k * step).toFixed(10)))
  }
  return values
}

export function parseSequence(str: string): (string | number)[] {
  const result: (string | number)[] = []

  for (const part of str.split(',').map((p) => p.trim())) {
    if (part === '') continue

    const range = part.match(RANGE)
    if (range) {
      const [, start, end, step] = range
      result.push(...expandRange(part, Number(start), Number(end), step ? Number(step) : 1))
      continue
    }

    const legacy = part.match(LEGACY_RANGE)
    if (legacy) {
      const [, start, end, step] = legacy
      const suggestion = `${start}..${end}${step ? `:${step}` : ''}`
      throw new Error(`Ranges are written with "..": use "${suggestion}" instead of "${part}".`)
    }

    const n = Number(part)
    result.push(isNaN(n) ? part : n)
  }

  return result
}

// Returns an error message if `str` is not a non-empty sequence of numbers, otherwise null
export function numericSequenceError(str: string): string | null {
  try {
    const values = parseSequence(str)
    if (values.length === 0) return 'Enter at least one value.'
    const bad = values.find((v) => typeof v !== 'number' || !isFinite(v))
    return bad === undefined ? null : `"${bad}" is not a number.`
  } catch (e) {
    return e instanceof Error ? e.message : String(e)
  }
}
