// src/renderer/src/utils/css.ts

/**
 * Reads the value of a CSS custom property from the root element.
 * @param propertyName The name of the custom property (e.g., '--color-primary').
 * @returns The computed value of the property.
 */
export function getCssVar(propertyName: string): string {
  if (typeof window === 'undefined') {
    return '' // Return a default or empty string in non-browser environments
  }
  return getComputedStyle(document.documentElement).getPropertyValue(propertyName).trim()
}
