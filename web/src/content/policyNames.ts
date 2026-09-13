const knownPolicyTokens = /CUSUM|EXP3|MOSS|UCB|KL|[A-Z][a-z0-9]*/g;

export function readablePolicyClassName(className: string): string {
  return className.match(knownPolicyTokens)?.join(" ") ?? className;
}
