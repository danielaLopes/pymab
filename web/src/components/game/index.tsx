import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";

import {
  familyDefinitions,
  familyOrder,
  policiesByFamily,
  policyCatalog,
  type PolicyId,
} from "@/catalog/policies";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { recommendationContextFeatureHelp } from "@/state/scenarioCandidates";

import type { LessonId, LessonSnapshot, RuntimeProgress } from "../../engine/protocol";
import { loadPersistence, savePersistence } from "../../state/persistence";
import { ArmSymbol } from "./ArmSymbol";

export function AppShell({ children }: { children: ReactNode }) {
  const [motionOverride, setMotionOverride] = useState<boolean | null>(
    () => loadPersistence().preferences.reducedMotion,
  );
  useEffect(() => {
    const systemReduced =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.documentElement.dataset.reducedMotion = String(motionOverride ?? systemReduced);
  }, [motionOverride]);
  const cycleMotion = () => {
    const next = motionOverride === null ? true : motionOverride ? false : null;
    setMotionOverride(next);
    const persisted = loadPersistence();
    savePersistence({
      ...persisted,
      preferences: { ...persisted.preferences, reducedMotion: next },
    });
  };
  return (
    <div className="app-shell">
      <header className="site-header">
        <Link className="brand" to="/" aria-label="PyMAB Arcade home">
          <img
            className="brand-mark"
            src={`${import.meta.env.BASE_URL}pymab-mark.svg`}
            alt=""
            aria-hidden="true"
          />
          <span>
            PyMAB <b>Arcade</b>
          </span>
        </Link>
        <nav aria-label="Primary navigation">
          <Link to="/">Missions</Link>
          <Link to="/lab">Python Lab</Link>
          <a href="../docs/">Docs</a>
          <button className="motion-toggle" type="button" onClick={cycleMotion}>
            Motion: {motionOverride === null ? "system" : motionOverride ? "reduced" : "full"}
          </button>
          <a href="https://github.com/danielaLopes/pymab">GitHub</a>
        </nav>
      </header>
      {children}
      <footer>Runs locally in your browser · No account · No analytics</footer>
    </div>
  );
}

export function CampaignMap() {
  const [query, setQuery] = useState("");
  const normalized = query.trim().toLowerCase();
  const matches = Object.values(policyCatalog).filter((policy) =>
    `${policy.label} ${policy.className} ${policy.family}`.toLowerCase().includes(normalized),
  );
  return (
    <div className="mission-atlas">
      <div className="policy-search">
        <label htmlFor="policy-search">Find a policy or Python class</label>
        <input
          id="policy-search"
          type="search"
          placeholder="Try Thompson, UCB, or EXP3"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <span role="status">{matches.length} policies</span>
      </div>
      {normalized ? (
        <div className="policy-search-results">
          {matches.map((policy) => (
            <Link key={policy.id} className="policy-row" to={`/lesson/${policy.id}`}>
              <LessonBadge policyId={policy.id} />
              <span>
                <strong>{policy.label}</strong>
                <small>{policy.className}</small>
              </span>
              <span aria-hidden="true">→</span>
            </Link>
          ))}
        </div>
      ) : (
        <div className="family-atlas">
          {familyOrder.map((family) => {
            const definition = familyDefinitions[family];
            return (
              <section className={`family-card family-${family}`} key={family}>
                <div className="family-card-heading">
                  <span className="mission-number">{definition.number}</span>
                  <span className="tag">{definition.label.toUpperCase()}</span>
                  <h2>{definition.title}</h2>
                  <p>{definition.description}</p>
                </div>
                <div className="family-policy-list">
                  {policiesByFamily[family].map((policy) => (
                    <Link key={policy.id} className="policy-row" to={`/lesson/${policy.id}`}>
                      <LessonBadge policyId={policy.id} />
                      <span>
                        <strong>{policy.label}</strong>
                        <small>{policy.className}</small>
                      </span>
                      <span aria-hidden="true">→</span>
                    </Link>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function MissionHeader({
  eyebrow,
  title,
  intro,
}: {
  eyebrow: string;
  title: string;
  intro: string;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => heading.current?.focus(), [title]);
  return (
    <header className="mission-header">
      <Link to="/" className="back-link">
        ← Mission map
      </Link>
      <p className="eyebrow">{eyebrow}</p>
      <h1 ref={heading} tabIndex={-1}>
        {title}
      </h1>
      <p>{intro}</p>
    </header>
  );
}

export function OutcomeReveal({
  snapshot,
  explanation,
}: {
  snapshot: LessonSnapshot | null;
  explanation: string;
}) {
  const reward = snapshot?.reward;
  const binary = snapshot?.presentation.rewardPresentation === "binary";
  const outcomeLabel =
    snapshot?.diagnostic && typeof snapshot.diagnostic.outcomeLabel === "string"
      ? snapshot.diagnostic.outcomeLabel
      : null;
  const result =
    reward === null || reward === undefined
      ? ""
      : outcomeLabel
        ? snapshot?.presentation.rewardPresentation === "utility"
          ? `${outcomeLabel} (${reward > 0 ? "+" : ""}${reward.toFixed(2)} utility)`
          : outcomeLabel
        : binary
          ? reward > 0
            ? snapshot?.presentation.positiveOutcomeLabel
            : snapshot?.presentation.zeroOutcomeLabel
          : `Reward ${reward.toFixed(3)}`;
  return (
    <div className="outcome" aria-live="polite">
      <span className="outcome-icon" aria-hidden="true">
        {(reward ?? 0) > 0 ? "✦" : "◇"}
      </span>
      <div>
        <strong>
          {snapshot?.step ? `Round ${snapshot.step}: ${result}` : "Awaiting the first decision"}
        </strong>
        <p>{explanation}</p>
      </div>
    </div>
  );
}

export function ProgressTrail({ snapshot }: { snapshot: LessonSnapshot | null }) {
  const horizon = snapshot?.horizon ?? 12;
  const step = snapshot?.step ?? 0;
  return (
    <div className="progress-trail">
      <div>
        <span>Run</span>
        <strong>
          {step} / {horizon}
        </strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Run progress"
        aria-valuemin={0}
        aria-valuemax={horizon}
        aria-valuenow={step}
      >
        <span style={{ width: `${(step / horizon) * 100}%` }} />
      </div>
      <dl>
        <div>
          <dt>
            {snapshot?.objective === "best-arm"
              ? "Samples"
              : snapshot?.presentation.rewardPresentation === "utility"
                ? "Total utility"
                : "Total reward"}
          </dt>
          <dd>
            {snapshot?.objective === "best-arm" ? step : (snapshot?.totalReward ?? 0).toFixed(2)}
          </dd>
        </div>
        <div>
          <dt>Expected regret</dt>
          <dd>{(snapshot?.cumulativeExpectedRegret ?? 0).toFixed(2)}</dd>
        </div>
      </dl>
    </div>
  );
}

export function RunControls({
  pending,
  completed,
  autoRunning,
  onStep,
  onAutoRun,
  onPause,
  onReset,
}: {
  pending: boolean;
  completed: boolean;
  autoRunning: boolean;
  onStep: () => void;
  onAutoRun: () => void;
  onPause: () => void;
  onReset: () => void;
}) {
  return (
    <div className="run-controls">
      <button
        className="primary-button"
        type="button"
        disabled={pending || completed || autoRunning}
        onClick={onStep}
      >
        Advance one round
      </button>
      {autoRunning ? (
        <button type="button" onClick={onPause}>
          Pause run
        </button>
      ) : (
        <button type="button" disabled={pending || completed} onClick={onAutoRun}>
          Auto-run
        </button>
      )}
      <button type="button" disabled={pending} onClick={onReset}>
        Restart run
      </button>
    </div>
  );
}

export function Debrief({
  snapshot,
  onChallenge,
  onFreePlay,
}: {
  snapshot: LessonSnapshot;
  onChallenge: () => void;
  onFreePlay: () => void;
}) {
  const regretPath = snapshot.history.map((event, index) => ({
    step: index + 1,
    selectedArm: event.selectedArm,
    reward: event.reward,
    regret: event.instantaneousExpectedRegret,
    cumulativeRegret: snapshot.history
      .slice(0, index + 1)
      .reduce((total, item) => total + item.instantaneousExpectedRegret, 0),
  }));
  const hiddenTruth = snapshot.hiddenTruth;
  const probabilities = hiddenTruth?.probabilities;
  const optimalArms = hiddenTruth?.optimalArms;
  const environmentKind = policyCatalog[snapshot.policyId].environment;
  const numericRewards = snapshot.presentation.rewardPresentation !== "binary";
  const arms = snapshot.presentation.arms;
  const gateResults = arms.map((_, gateIndex) => {
    const selections = snapshot.history.filter((event) => event.selectedArm === gateIndex);
    return {
      selections: selections.length,
      rewards: selections.reduce((total, event) => total + event.reward, 0),
    };
  });
  return (
    <section className={`debrief ${snapshot.passed ? "passed" : "complete"}`}>
      <p className="eyebrow">Run complete</p>
      <h2>
        {snapshot.mode === "challenge"
          ? snapshot.passed
            ? "Challenge cleared"
            : "Challenge not cleared"
          : "Run complete"}
      </h2>
      <p>
        The run earned{" "}
        <strong>
          {snapshot.totalReward.toFixed(3)} total{" "}
          {snapshot.presentation.rewardPresentation === "utility" ? "utility" : "reward"}
        </strong>{" "}
        with <strong>{snapshot.cumulativeExpectedRegret.toFixed(2)} expected regret</strong>.
      </p>
      {snapshot.objective === "best-arm" && (
        <p className="recommendation-result">
          Recommended portal:{" "}
          <strong>
            {snapshot.recommendation === null
              ? "Not available"
              : arms[snapshot.recommendation]?.name}
          </strong>
          .
          {snapshot.passed
            ? " It matches the strongest portal in this run."
            : " It does not match the strongest portal in this run."}
        </p>
      )}
      <details className="debrief-details">
        <summary>Show environment values and regret by round</summary>
        {environmentKind !== "contextual-linear" &&
          environmentKind !== "contextual-logistic" &&
          Array.isArray(probabilities) &&
          probabilities.length === 3 &&
          probabilities.every((value) => typeof value === "number") && (
            <dl className="truth-grid" aria-label="Configured path reward probabilities">
              {probabilities.map((value, index) => {
                const result = gateResults[index]!;
                return (
                  <div key={arms[index]?.name}>
                    <dt>{arms[index]?.name}</dt>
                    <dd>{Math.round(value * 100)}% reward chance</dd>
                    <dd>
                      {result.selections
                        ? `${result.rewards} rewards from ${result.selections} selections`
                        : "Not selected in this run"}
                    </dd>
                  </div>
                );
              })}
            </dl>
          )}
        {(environmentKind === "contextual-linear" || environmentKind === "contextual-logistic") && (
          <>
            <p>
              The best {snapshot.scenarioId ? "action" : "path"} can change with the signals. The
              table below compares the optimal {snapshot.scenarioId ? "action" : "path"} with the
              policy's choice in each round.
            </p>
            <MatrixTable
              label={`Hidden environment coefficients by ${snapshot.scenarioId ? "action" : "path"}`}
              value={snapshot.hiddenTruth?.theta}
              rows={snapshot.presentation.arms.map((arm) => arm.name)}
              columns={snapshot.presentation.contextFeatures?.map((feature) => feature.name)}
            />
          </>
        )}
        {environmentKind === "stationary-gaussian" && (
          <MatrixTable label="Gaussian environment means" value={[snapshot.hiddenTruth?.means]} />
        )}
        {environmentKind === "changing-bernoulli" && (
          <pre className="truth-json">{JSON.stringify(snapshot.hiddenTruth?.rounds, null, 2)}</pre>
        )}
        <div className="debrief-table-wrap">
          <table className="debrief-table">
            <caption>Decision and expected-regret path</caption>
            <thead>
              <tr>
                <th>Round</th>
                <th>Chosen</th>
                {(snapshot.family === "contextual" ||
                  snapshot.family === "changing" ||
                  snapshot.family === "adversarial") && <th>Optimal</th>}
                <th>Reward</th>
                <th>Regret</th>
                <th>Total regret</th>
              </tr>
            </thead>
            <tbody>
              {regretPath.map((event) => (
                <tr key={event.step}>
                  <th scope="row">{event.step}</th>
                  <td>{arms[event.selectedArm]?.name}</td>
                  {(snapshot.family === "contextual" ||
                    snapshot.family === "changing" ||
                    snapshot.family === "adversarial") && (
                    <td>
                      {(() => {
                        const rounds = hiddenTruth?.rounds;
                        const fromRounds =
                          Array.isArray(rounds) &&
                          typeof rounds[event.step - 1] === "object" &&
                          rounds[event.step - 1] !== null
                            ? (rounds[event.step - 1] as Record<string, unknown>).optimalArm
                            : undefined;
                        const arm = Array.isArray(optimalArms)
                          ? (optimalArms as unknown[])[event.step - 1]
                          : fromRounds;
                        return typeof arm === "number" ? arms[arm]?.name : "Not available";
                      })()}
                    </td>
                  )}
                  <td>
                    {numericRewards
                      ? event.reward.toPrecision(4)
                      : event.reward
                        ? snapshot.presentation.positiveOutcomeLabel
                        : snapshot.presentation.zeroOutcomeLabel}
                  </td>
                  <td>{event.regret.toPrecision(4)}</td>
                  <td>{event.cumulativeRegret.toPrecision(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <div className="run-controls">
        <button className="primary-button" onClick={onChallenge}>
          Start challenge
        </button>
        <button onClick={onFreePlay}>Start free play</button>
      </div>
    </section>
  );
}

export function LoadingStages({ progress }: { progress: RuntimeProgress | null }) {
  return (
    <div className="loading-panel" role="status">
      <span className="loader" aria-hidden="true" />
      <div>
        <strong>Starting the Python runtime</strong>
        <p>{progress?.message ?? "Preparing the lesson..."}</p>
      </div>
    </div>
  );
}

export function ErrorRecovery({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="error-panel" role="alert">
      <h2>The Python runtime could not start</h2>
      <p>{message}</p>
      <button onClick={onRetry}>Retry</button>
    </div>
  );
}

export function UnsupportedBrowser({ reason }: { reason: string }) {
  return (
    <div className="error-panel">
      <h2>This browser cannot open the Arcade</h2>
      <p>{reason}</p>
      <p>
        You can still read about each policy in the <a href="../docs/">PyMAB documentation</a>.
      </p>
    </div>
  );
}

export function PolicyBars({ snapshot }: { snapshot: LessonSnapshot }) {
  const diagnostic = snapshot.diagnostic;
  if (!diagnostic) return null;
  const before =
    diagnostic.before && typeof diagnostic.before === "object"
      ? (diagnostic.before as Record<string, unknown>)
      : diagnostic;
  const after =
    diagnostic.after && typeof diagnostic.after === "object"
      ? (diagnostic.after as Record<string, unknown>)
      : diagnostic;
  const candidates: Array<[string, unknown]> =
    snapshot.policyId === "linucb"
      ? [["UCB score", diagnostic.ucbScores]]
      : [
          [
            typeof (diagnostic.decision as Record<string, unknown> | undefined)?.label === "string"
              ? String((diagnostic.decision as Record<string, unknown>).label)
              : "Decision value",
            (diagnostic.decision as Record<string, unknown> | undefined)?.values,
          ],
          ["Confidence index", before.indices ?? after.indices],
          [
            "Action probability",
            before.actionProbabilities ??
              after.actionProbabilities ??
              before.probabilities ??
              after.probabilities,
          ],
          ["Posterior mean", before.means ?? after.means],
          ["Preference", before.preferences ?? after.preferences],
          ["Weight", before.weights ?? after.weights ?? before.log_weights ?? after.log_weights],
          ["Estimate", diagnostic.estimatesAfter ?? after.estimates ?? before.estimates],
          ["Effective count", after.discounted_counts ?? before.discounted_counts],
        ];
  const selected = candidates.find(
    ([, value]) => Array.isArray(value) && value.length === snapshot.presentation.arms.length,
  );
  if (!selected)
    return (
      <p className="field-help">
        This policy's state is available in the validated snapshot below.
      </p>
    );
  const [label, values] = selected;
  const numeric = (values as unknown[]).map(Number);
  const scale = Math.max(...numeric.map(Math.abs), 1);
  if (snapshot.policyId === "linucb") {
    const means = Array.isArray(diagnostic.predictedMeans)
      ? diagnostic.predictedMeans.map(Number)
      : [];
    const bonuses = Array.isArray(diagnostic.bonuses) ? diagnostic.bonuses.map(Number) : [];
    return (
      <table className="score-table">
        <caption>LinUCB score decomposition</caption>
        <thead>
          <tr>
            <th>Path</th>
            <th>Prediction</th>
            <th>Bonus</th>
            <th>UCB</th>
          </tr>
        </thead>
        <tbody>
          {numeric.map((value, index) => (
            <tr key={index}>
              <th>{snapshot.presentation.arms[index]?.name}</th>
              <td>{means[index]?.toPrecision(4)}</td>
              <td>+ {bonuses[index]?.toPrecision(4)}</td>
              <td>
                <strong>{value.toPrecision(4)}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  return (
    <div
      className="policy-bars"
      role="img"
      aria-label={`${label}: ${numeric.map((value, i) => `${snapshot.presentation.arms[i]?.name} ${value.toPrecision(4)}`).join(", ")}`}
    >
      <small className="policy-bars-label">{label}</small>
      {numeric.map((value, index) => (
        <div key={index}>
          <ArmSymbol arm={snapshot.presentation.arms[index]!} />
          <i
            style={{ "--bar": `${Math.max(3, (Math.abs(value) / scale) * 100)}%` } as CSSProperties}
          />
          <code>{value.toPrecision(4)}</code>
        </div>
      ))}
    </div>
  );
}

function MatrixTable({
  label,
  value,
  rows,
  columns,
  help,
  columnHelp,
}: {
  label: string;
  value: unknown;
  rows?: string[];
  columns?: string[] | undefined;
  help?: string;
  columnHelp?: Array<string | undefined>;
}) {
  if (!Array.isArray(value) || !value.every((row) => Array.isArray(row))) return null;
  const matrix = value as unknown[][];
  return (
    <TooltipProvider>
      <table className="matrix-table">
        <caption>
          <span>{label}</span>
          {help && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="matrix-help-trigger" type="button" aria-label={`About ${label}`}>
                  i
                </button>
              </TooltipTrigger>
              <TooltipContent className="matrix-help-content">{help}</TooltipContent>
            </Tooltip>
          )}
        </caption>
        {columns && (
          <thead>
            <tr>
              <th scope="col">Action</th>
              {columns.map((column, columnIndex) => (
                <th scope="col" key={column}>
                  {columnHelp?.[columnIndex] ? (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          className="matrix-column-help"
                          type="button"
                          aria-label={`About ${column} values`}
                        >
                          {column}
                        </button>
                      </TooltipTrigger>
                      <TooltipContent className="matrix-help-content">
                        {columnHelp[columnIndex]}
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    column
                  )}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th scope="row">{rows?.[rowIndex] ?? `Path ${rowIndex + 1}`}</th>
              {row.map((cell, columnIndex) => (
                <td key={columnIndex}>{Number(cell).toPrecision(4)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </TooltipProvider>
  );
}

export function InspectPanel({
  snapshot,
  open,
  onToggle,
  onOpenLab,
}: {
  snapshot: LessonSnapshot | null;
  open: boolean;
  onToggle: () => void;
  onOpenLab: () => void;
}) {
  const [copyStatus, setCopyStatus] = useState("");
  const copyCode = () => {
    if (!snapshot) return;
    void navigator.clipboard.writeText(snapshot.generatedCode).then(
      () => setCopyStatus("Python copied."),
      () => setCopyStatus("Copy failed. Select the code manually."),
    );
  };
  return (
    <aside className="inspector">
      <button className="inspector-toggle" type="button" aria-expanded={open} onClick={onToggle}>
        <span>
          <small>Developer view</small>
          <strong>Inspect PyMAB</strong>
        </span>
        <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="inspector-body">
          {!snapshot ? (
            <p>Run one round to inspect the policy state.</p>
          ) : (
            <>
              <dl className="metadata">
                <div>
                  <dt>Class</dt>
                  <dd>{policyCatalog[snapshot.policyId].className}</dd>
                </div>
                <div>
                  <dt>PyMAB</dt>
                  <dd>{snapshot.packageVersion}</dd>
                </div>
                <div>
                  <dt>Seed</dt>
                  <dd>{snapshot.seed}</dd>
                </div>
                <div>
                  <dt>Constructor</dt>
                  <dd>
                    <code>
                      {`${policyCatalog[snapshot.policyId].className}(${Object.entries({
                        n_arms: snapshot.presentation.arms.length,
                        ...(snapshot.family === "contextual"
                          ? {
                              n_features:
                                snapshot.presentation.contextFeatures?.length ??
                                snapshot.publicContext?.[0]?.length ??
                                4,
                            }
                          : {}),
                        ...(snapshot.policyId === "moss" ? { horizon: snapshot.horizon } : {}),
                        ...snapshot.parameters,
                      })
                        .map(([key, value]) => `${key}=${String(value)}`)
                        .join(", ")})`}
                    </code>
                  </dd>
                </div>
                <div>
                  <dt>Commit</dt>
                  <dd>
                    <code>{snapshot.sourceCommit.slice(0, 8)}</code>
                  </dd>
                </div>
              </dl>
              <h2>Decision state</h2>
              <PolicyBars snapshot={snapshot} />
              {snapshot.family === "contextual" && snapshot.diagnostic && (
                <>
                  <MatrixTable
                    label="Current context matrix"
                    value={snapshot.diagnostic.contextMatrix}
                    rows={snapshot.presentation.arms.map((arm) => arm.name)}
                    columns={snapshot.presentation.contextFeatures?.map((feature) => feature.name)}
                    {...(snapshot.scenarioId === "recommendations"
                      ? {
                          help: "One round is one visitor visit. Every candidate is evaluated for the same visitor, so the rows repeat. Binary signals use -1 and +1. Numeric signals are scaled between them. The next round generates a new visitor context.",
                          ...(snapshot.presentation.contextFeatures
                            ? {
                                columnHelp: snapshot.presentation.contextFeatures.map((feature) =>
                                  recommendationContextFeatureHelp(feature.id),
                                ),
                              }
                            : {}),
                        }
                      : {})}
                  />
                  <MatrixTable
                    label="Learned coefficient estimates"
                    value={
                      snapshot.diagnostic.thetaBefore ??
                      (snapshot.diagnostic.after as Record<string, unknown> | undefined)?.theta
                    }
                    rows={snapshot.presentation.arms.map((arm) => arm.name)}
                    columns={snapshot.presentation.contextFeatures?.map((feature) => feature.name)}
                  />
                </>
              )}
              {snapshot.family === "best-arm" && snapshot.diagnostic?.recommendation !== null && (
                <p className="inspector-callout">
                  Current recommendation:{" "}
                  {snapshot.presentation.arms[Number(snapshot.diagnostic?.recommendation)]?.name}
                </p>
              )}
              <details>
                <summary className="validated-snapshot-summary">Full validated snapshot</summary>
                <pre>{JSON.stringify(snapshot.diagnostic, null, 2)}</pre>
              </details>
              {snapshot.hiddenTruth && (
                <details>
                  <summary>Environment values used by this run</summary>
                  <pre>{JSON.stringify(snapshot.hiddenTruth, null, 2)}</pre>
                </details>
              )}
              <h2>Equivalent Python</h2>
              <pre className="code-preview" tabIndex={0} aria-label="Equivalent Python code">
                <code>{snapshot.generatedCode}</code>
              </pre>
              <div className="run-controls">
                <button onClick={copyCode}>Copy code</button>
                <button className="primary-button" onClick={onOpenLab}>
                  Open in Python Lab
                </button>
              </div>
              <p className="copy-status" aria-live="polite">
                {copyStatus}
              </p>
            </>
          )}
        </div>
      )}
    </aside>
  );
}

export function LessonBadge({ policyId, lessonId }: { policyId?: PolicyId; lessonId?: LessonId }) {
  const id = policyId ?? lessonId ?? "epsilon-greedy";
  return (
    <span className="lesson-badge" title={policyCatalog[id].className}>
      {policyCatalog[id].badge}
    </span>
  );
}
