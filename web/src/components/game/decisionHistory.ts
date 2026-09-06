import type { LessonSnapshot } from "../../engine/protocol";

export interface HistoryCue {
  name: string;
  value: string;
  symbol: string;
}

export interface HistoryReward {
  kind: "relic" | "empty" | "numeric";
  label: string;
  value: number;
}

export interface DecisionHistoryCell {
  arm: number;
  selected: boolean;
  primary: string | null;
  primaryLabel: string | null;
  secondary: string | null;
  state: "active" | "eliminated" | "recommended" | "unseen" | null;
  reward: HistoryReward | null;
}

export interface DecisionHistoryRow {
  round: number;
  cues: HistoryCue[];
  phase: number | null;
  reason: string | null;
  alarm: boolean;
  selectedArm: number;
  cells: DecisionHistoryCell[];
}

type HistoryEvent = LessonSnapshot["history"][number];
type UnknownRecord = Record<string, unknown>;

function record(value: unknown): UnknownRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : {};
}

function numberVector(value: unknown): Array<number | null> | null {
  if (!Array.isArray(value) || value.length < 1) return null;
  return value.map((item) => (typeof item === "number" && Number.isFinite(item) ? item : null));
}

function booleanVector(value: unknown): boolean[] | null {
  if (!Array.isArray(value) || value.length < 1) return null;
  return value.map(Boolean);
}

function numberList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is number => typeof item === "number");
}

function formatNumber(value: number): string {
  if (Math.abs(value) >= 100) return value.toPrecision(3);
  return value.toFixed(2);
}

function formatValue(label: string, value: number): string {
  if (/probability|chance/i.test(label)) return `${(value * 100).toFixed(0)}%`;
  return formatNumber(value);
}

function rewardFor(snapshot: LessonSnapshot, event: HistoryEvent): HistoryReward {
  const reward = event.reward;
  const outcome = record(event.diagnostic).outcomeLabel;
  if (snapshot.presentation.rewardPresentation === "binary") {
    return reward > 0
      ? {
          kind: "relic",
          label: typeof outcome === "string" ? outcome : snapshot.presentation.positiveOutcomeLabel,
          value: reward,
        }
      : {
          kind: "empty",
          label: typeof outcome === "string" ? outcome : snapshot.presentation.zeroOutcomeLabel,
          value: reward,
        };
  }
  const prefix = reward > 0 ? "+" : "";
  return {
    kind: "numeric",
    label:
      typeof outcome === "string"
        ? `${outcome}: ${prefix}${formatNumber(reward)}`
        : `Reward ${prefix}${formatNumber(reward)}`,
    value: reward,
  };
}

function cueValue(cue: HistoryEvent["visibleCues"][number]): string {
  const suffix = ` ${cue.name}`;
  return cue.label.endsWith(suffix) ? cue.label.slice(0, -suffix.length) : cue.label;
}

function cueSymbol(name: string): string {
  if (name === "light") return "◐";
  if (name === "echo") return "≋";
  if (name === "visitor") return "◎";
  if (name === "engagement") return "↗";
  if (name === "visit") return "◫";
  if (name === "device") return "▱";
  if (name === "account age") return "◷";
  if (name === "recent activity") return "↻";
  if (name === "price sensitivity") return "%";
  if (name === "session depth") return "≡";
  if (name === "traffic source") return "↗";
  if (name === "risk") return "!";
  if (name === "account") return "○";
  if (name === "endpoint") return "⌁";
  return "≈";
}

interface ExtractedDecision {
  label: string | null;
  values: Array<number | null> | null;
  secondaryLabel: string | null;
  secondaryValues: Array<number | null> | null;
  secondaryCombined: string[] | null;
  reason: string | null;
  phase: number | null;
  unseenArms: number[];
}

function extractDecision(event: HistoryEvent): ExtractedDecision {
  const diagnostic = record(event.diagnostic);
  const decision = record(diagnostic.decision);
  const before = record(diagnostic.before);

  if (diagnostic.kind === "epsilon") {
    return {
      label: "Estimate",
      values: numberVector(diagnostic.estimatesBefore),
      secondaryLabel: null,
      secondaryValues: null,
      secondaryCombined: null,
      reason:
        diagnostic.selectionBranch === "explore"
          ? "Explore"
          : diagnostic.selectionBranch === "exploit"
            ? "Exploit"
            : null,
      phase: null,
      unseenArms: [],
    };
  }

  if (diagnostic.kind === "linucb") {
    const predictions = numberVector(diagnostic.predictedMeans);
    const bonuses = numberVector(diagnostic.bonuses);
    const combined =
      predictions && bonuses
        ? predictions.map((prediction, index) => {
            const bonus = bonuses[index];
            return prediction === null || bonus === null || bonus === undefined
              ? "Prediction unavailable"
              : `Prediction ${formatNumber(prediction)} + bonus ${formatNumber(bonus)}`;
          })
        : null;
    return {
      label: "UCB score",
      values: numberVector(diagnostic.ucbScores),
      secondaryLabel: "Prediction and bonus",
      secondaryValues: null,
      secondaryCombined: combined,
      reason: null,
      phase: null,
      unseenArms: [],
    };
  }

  const label = typeof decision.label === "string" ? decision.label : null;
  const secondaryLabel =
    typeof decision.secondaryLabel === "string" ? decision.secondaryLabel : null;
  const branch = decision.selectionBranch;
  const fallbackValues =
    numberVector(before.actionProbabilities) ??
    numberVector(before.indices) ??
    numberVector(before.means) ??
    numberVector(before.estimates);
  return {
    label: label ?? (fallbackValues ? "Decision value" : null),
    values: numberVector(decision.values) ?? fallbackValues,
    secondaryLabel,
    secondaryValues: numberVector(decision.secondaryValues),
    secondaryCombined: null,
    reason: branch === "explore" ? "Explore" : branch === "exploit" ? "Exploit" : null,
    phase: typeof decision.environmentPhase === "number" ? decision.environmentPhase : null,
    unseenArms: numberList(decision.unseenArms),
  };
}

function detectedChange(diagnostic: UnknownRecord): boolean {
  const before = record(diagnostic.before);
  const after = record(diagnostic.after);
  const beforeCounts = numberVector(before.change_counts);
  const afterCounts = numberVector(after.change_counts);
  if (!beforeCounts || !afterCounts) return false;
  return afterCounts.some((value, index) => {
    const prior = beforeCounts[index];
    return value !== null && prior !== null && prior !== undefined && value > prior;
  });
}

export function buildDecisionHistory(snapshot: LessonSnapshot): DecisionHistoryRow[] {
  return snapshot.history.map((event, index) => {
    const extracted = extractDecision(event);
    const diagnostic = record(event.diagnostic);
    const after = record(diagnostic.after);
    const active = booleanVector(after.active);
    const recommendation =
      typeof diagnostic.recommendation === "number" ? diagnostic.recommendation : null;

    const cells = snapshot.presentation.arms.map((_, arm): DecisionHistoryCell => {
      const value = extracted.values?.[arm] ?? null;
      const secondaryValue = extracted.secondaryValues?.[arm] ?? null;
      const selected = event.selectedArm === arm;
      let state: DecisionHistoryCell["state"] = null;
      if (recommendation === arm) state = "recommended";
      else if (active) state = active[arm] ? "active" : "eliminated";
      else if (extracted.unseenArms.includes(arm)) state = "unseen";
      return {
        arm,
        selected,
        primary:
          state === "unseen"
            ? "Unseen"
            : value !== null && extracted.label
              ? formatValue(extracted.label, value)
              : null,
        primaryLabel: extracted.label,
        secondary:
          extracted.secondaryCombined?.[arm] ??
          (secondaryValue !== null && extracted.secondaryLabel
            ? `${extracted.secondaryLabel} ${formatValue(extracted.secondaryLabel, secondaryValue)}`
            : null),
        state,
        reward: selected ? rewardFor(snapshot, event) : null,
      };
    });

    return {
      round: index + 1,
      cues: event.visibleCues.map((cue) => ({
        name: cue.name,
        value: cueValue(cue),
        symbol: cueSymbol(cue.name),
      })),
      phase: extracted.phase,
      reason: extracted.reason,
      alarm: detectedChange(diagnostic),
      selectedArm: event.selectedArm,
      cells,
    };
  });
}

export function publicPathDetail(snapshot: LessonSnapshot, arm: number): string | null {
  if (snapshot.mode !== "freePlay" || !snapshot.environment) return null;
  const probabilities = snapshot.environment.probabilities;
  if (Array.isArray(probabilities) && typeof probabilities[arm] === "number") {
    const percentage = Number(probabilities[arm]) * 100;
    return `${percentage.toFixed(Number.isInteger(percentage) ? 0 : 1)}% reward chance`;
  }
  const means = snapshot.environment.means;
  if (Array.isArray(means) && typeof means[arm] === "number") {
    return `Mean ${formatNumber(Number(means[arm]))}`;
  }
  return null;
}
