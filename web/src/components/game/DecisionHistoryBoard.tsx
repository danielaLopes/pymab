import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

import type { LessonSnapshot } from "../../engine/protocol";
import { ArmSymbol } from "./ArmSymbol";
import {
  buildDecisionHistory,
  publicPathDetail,
  type DecisionHistoryCell,
  type DecisionHistoryRow,
} from "./decisionHistory";

const rowHeights = {
  regular: { past: 84, current: 96 },
  spacious: { past: 104, current: 120 },
};

const cueHelp: Record<string, string> = {
  light:
    "Light can be red or blue. The policy sees it before choosing a path. In the default environment, red strongly favors Moon and blue strongly favors Sun.",
  echo: "Echo can be low or high. The policy sees it before choosing a path. In the default environment, high slightly favors Moon and Sun, while low strongly favors Star.",
  tide: "Tide can be low or high. The policy sees it before choosing a path. In the default environment, low strongly favors Moon and high strongly favors Sun. Tide has a smaller effect on Star.",
  visitor: "Whether this visitor is new or returning.",
  engagement: "A score from 0 to 100 for activity during the current visit.",
  visit: "Whether this visit happens on a weekday or weekend.",
  device: "Whether the visitor is using a desktop or mobile device.",
  "account age": "The number of days since the account was created.",
  "recent activity": "The number of interactions in the visitor's recent history.",
  "price sensitivity": "A score from 0 for low sensitivity to 100 for high sensitivity.",
  "session depth": "The number of pages viewed in the current session.",
  "traffic source": "Whether the visit came from an organic or paid source.",
  risk: "A normalized score from an existing risk model. This lesson covers only the uncertain middle range.",
  account: "Whether the account is new or established.",
  endpoint: "Whether the requested endpoint is routine or sensitive.",
};

type Arm = LessonSnapshot["presentation"]["arms"][number];

function RewardMark({ cell }: { cell: DecisionHistoryCell }) {
  if (!cell.selected) {
    return (
      <span className="history-reward history-reward-unknown" aria-hidden="true">
        ?
      </span>
    );
  }
  if (!cell.reward) return null;
  if (cell.reward.kind === "relic") {
    return (
      <span className="history-reward history-reward-relic" title={cell.reward.label}>
        ✦
      </span>
    );
  }
  if (cell.reward.kind === "empty") {
    return (
      <span className="history-reward history-reward-empty" title={cell.reward.label}>
        ◇
      </span>
    );
  }
  return (
    <span
      className={`history-reward history-reward-numeric ${cell.reward.value < 0 ? "negative" : ""}`}
    >
      {cell.reward.label.replace("Reward ", "")}
    </span>
  );
}

function ContextRail({ row }: { row: DecisionHistoryRow }) {
  return (
    <div className="history-round-rail" role="rowheader">
      <strong>Round {row.round}</strong>
      {row.cues.length ? (
        <div className="history-cues" aria-label={`Round ${row.round} signals`}>
          {row.cues.map((cue) => (
            <Tooltip key={cue.name}>
              <TooltipTrigger asChild>
                <button
                  className="history-cue-button"
                  type="button"
                  aria-label={`About ${cue.name}`}
                >
                  <i aria-hidden="true">{cue.symbol}</i>
                  <small>{cue.name}</small>
                  <b>{cue.value}</b>
                </button>
              </TooltipTrigger>
              <TooltipContent>{cueHelp[cue.name]}</TooltipContent>
            </Tooltip>
          ))}
        </div>
      ) : (
        <span className="history-round-note">
          {row.phase ? `Environment phase ${row.phase}` : "Same information each round"}
        </span>
      )}
      <div className="history-row-badges">
        {row.reason && (
          <span className={`decision-reason ${row.reason.toLowerCase()}`}>{row.reason}</span>
        )}
        {row.alarm && <span className="change-alarm">Change detected</span>}
      </div>
    </div>
  );
}

function cellDescription(row: DecisionHistoryRow, cell: DecisionHistoryCell, arms: Arm[]): string {
  const path = arms[cell.arm]!.name;
  const parts = [`Round ${row.round}`, path];
  if (cell.selected) {
    parts.push("chosen", cell.reward?.label ?? "outcome unavailable");
  } else {
    parts.push("Reward not observed");
  }
  if (cell.primary && cell.primaryLabel) parts.push(`${cell.primaryLabel} ${cell.primary}`);
  if (cell.state) parts.push(cell.state);
  return parts.join(", ");
}

function DecisionCell({
  row,
  cell,
  arms,
}: {
  row: DecisionHistoryRow;
  cell: DecisionHistoryCell;
  arms: Arm[];
}) {
  return (
    <div
      className={`history-cell path-${cell.arm} ${cell.selected ? "selected" : "unselected"} ${cell.state ? `state-${cell.state}` : ""}`}
      role="cell"
      aria-label={cellDescription(row, cell, arms)}
    >
      <span className="history-mini-gate" aria-hidden="true">
        <ArmSymbol arm={arms[cell.arm]!} />
      </span>
      {cell.selected && <span className="history-explorer" aria-hidden="true" />}
      <RewardMark cell={cell} />
      <span className="history-cell-values">
        {cell.primary && (
          <strong>
            <small>{cell.primaryLabel}</small>
            {cell.primary}
          </strong>
        )}
        {cell.secondary && <small>{cell.secondary}</small>}
      </span>
      {cell.state && cell.state !== "active" && (
        <span className={`history-state history-state-${cell.state}`}>{cell.state}</span>
      )}
    </div>
  );
}

function HistoryRow({
  row,
  current,
  arms,
}: {
  row: DecisionHistoryRow;
  current: boolean;
  arms: Arm[];
}) {
  const chosen = row.cells[row.selectedArm]!;
  const cueSummary = row.cues.length
    ? ` Signals: ${row.cues.map((cue) => `${cue.name} ${cue.value}`).join(", ")}.`
    : "";
  return (
    <div
      className={`decision-history-row ${current ? "current" : ""}`}
      role="row"
      tabIndex={0}
      aria-label={`Round ${row.round}. ${arms[row.selectedArm]!.name} chosen. ${chosen.reward?.label ?? "Outcome unavailable"}.${cueSummary}`}
    >
      <ContextRail row={row} />
      {row.cells.map((cell) => (
        <DecisionCell key={cell.arm} row={row} cell={cell} arms={arms} />
      ))}
    </div>
  );
}

function AwaitingRow({ arms }: { arms: Arm[] }) {
  return (
    <div
      className="decision-history-row current awaiting"
      role="row"
      aria-label="Round 1 awaiting the first decision"
    >
      <div className="history-round-rail" role="rowheader">
        <strong>Round 1</strong>
        <span className="history-round-note">Awaiting the first decision</span>
      </div>
      {arms.map((path, arm) => (
        <div className={`history-cell path-${arm} unselected`} role="cell" key={path.name}>
          <span className="history-mini-gate" aria-hidden="true">
            <ArmSymbol arm={path} />
          </span>
          <span className="history-cell-values">
            <strong>Available</strong>
          </span>
        </div>
      ))}
    </div>
  );
}

function ChoiceTrail({
  rows,
  spacious,
  armCount,
}: {
  rows: DecisionHistoryRow[];
  spacious: boolean;
  armCount: number;
}) {
  if (rows.length < 2) return null;
  const rowHeight = spacious ? rowHeights.spacious : rowHeights.regular;
  const height = (rows.length - 1) * rowHeight.past + rowHeight.current;
  const points = rows
    .map((row, index) => {
      const rowCenter =
        index === rows.length - 1
          ? index * rowHeight.past + rowHeight.current / 2
          : index * rowHeight.past + rowHeight.past / 2;
      return `${(row.selectedArm + 0.5) * 100},${rowCenter}`;
    })
    .join(" ");
  return (
    <svg
      className="choice-trail"
      viewBox={`0 0 ${armCount * 100} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline points={points} />
    </svg>
  );
}

export function DecisionHistoryBoard({
  snapshot,
  pending = false,
  eyebrow = "Choice trail",
  title = "Every round, every observed outcome",
  spacious = false,
}: {
  snapshot: LessonSnapshot | null;
  pending?: boolean;
  eyebrow?: string;
  title?: string;
  spacious?: boolean;
}) {
  const rows = useMemo(() => (snapshot ? buildDecisionHistory(snapshot) : []), [snapshot]);
  const arms = snapshot?.presentation.arms ?? [
    { name: "Moon Path", shortName: "Moon", symbolKind: "moon" as const },
    { name: "Sun Path", shortName: "Sun", symbolKind: "sun" as const },
    { name: "Star Path", shortName: "Star", symbolKind: "star" as const },
  ];
  const scrollRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);

  useEffect(() => {
    if (!following) return;
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [following, rows.length]);

  const jumpToCurrent = () => {
    setFollowing(true);
    const container = scrollRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  };
  const historyGridStyle = {
    "--history-arms": arms.length,
    ...(spacious && arms.length > 4 ? { minWidth: `${15 + arms.length * 13}rem` } : {}),
  } as CSSProperties;

  return (
    <TooltipProvider>
      <section
        className={`decision-history-board world-${snapshot?.family ?? "foundations"} ${pending ? "deciding" : ""} ${spacious ? "spacious" : ""}`}
        aria-label="Decision history"
      >
        <div className="history-board-heading">
          <div>
            <p className="eyebrow">{eyebrow}</p>
            <h2>{title}</h2>
          </div>
          {!following && (
            <button type="button" className="jump-current" onClick={jumpToCurrent}>
              Jump to current round
            </button>
          )}
        </div>
        <div
          className="history-scroll"
          ref={scrollRef}
          onScroll={(event) => {
            const target = event.currentTarget;
            setFollowing(target.scrollHeight - target.scrollTop - target.clientHeight < 24);
          }}
        >
          <div
            className="history-grid"
            style={historyGridStyle}
            role="table"
            aria-rowcount={Math.max(rows.length, 1) + 1}
            aria-colcount={arms.length + 1}
          >
            <div className="history-column-headers" role="row">
              <div role="columnheader">Round</div>
              {arms.map((path, arm) => (
                <div role="columnheader" key={path.name}>
                  <span aria-hidden="true">
                    <ArmSymbol arm={path} />
                  </span>
                  <strong>{path.name}</strong>
                  {snapshot && publicPathDetail(snapshot, arm) && (
                    <small>{publicPathDetail(snapshot, arm)}</small>
                  )}
                </div>
              ))}
            </div>
            <div className="history-rows">
              <ChoiceTrail rows={rows} spacious={spacious} armCount={arms.length} />
              {rows.length ? (
                rows.map((row, index) => (
                  <HistoryRow
                    key={row.round}
                    row={row}
                    current={index === rows.length - 1}
                    arms={arms}
                  />
                ))
              ) : (
                <AwaitingRow arms={arms} />
              )}
            </div>
          </div>
        </div>
        <div className="history-legend" aria-label="Decision history legend">
          <span>
            <i className="legend-chosen" aria-hidden="true" /> Chosen
          </span>
          <span>
            <i aria-hidden="true">✦</i>{" "}
            {snapshot?.presentation.positiveOutcomeLabel ?? "Relic found"}
          </span>
          <span>
            <i aria-hidden="true">◇</i> {snapshot?.presentation.zeroOutcomeLabel ?? "No relic"}
          </span>
          <span>
            <i aria-hidden="true">?</i> Reward not observed
          </span>
          <span>
            <i className="legend-trail" aria-hidden="true" /> Choice trail
          </span>
        </div>
      </section>
    </TooltipProvider>
  );
}
