import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  familyDefinitions,
  familyOrder,
  policiesByFamily,
  policyCatalog,
  type PolicyId,
} from "@/catalog/policies";

import type { LessonId, LessonSnapshot, RuntimeProgress } from "../../engine/protocol";
import { loadPersistence, savePersistence } from "../../state/persistence";

const gateDetails = [
  { name: "Moon Gate", symbol: "☾", rune: "Memory" },
  { name: "Sun Gate", symbol: "☼", rune: "Promise" },
  { name: "Star Gate", symbol: "✦", rune: "Possibility" },
];

const cueHelp: Record<string, string> = {
  light: "Light can be red or blue. The policy sees it before choosing a portal.",
  echo: "Echo can be low or high. The policy sees it before choosing a portal.",
  tide: "Tide can be low or high. The policy sees it before choosing a portal.",
};

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

export function CueStrip({ snapshot }: { snapshot: LessonSnapshot | null }) {
  if (!snapshot?.visibleCues.length)
    return (
      <p className="cue-empty">
        This lesson has no context signals, so the available information is the same in every round.
      </p>
    );
  return (
    <TooltipProvider>
      <ul className="cue-strip" aria-label="Current round signals">
        {snapshot.visibleCues.map((cue) => (
          <li key={cue.name}>
            <span aria-hidden="true">
              {cue.name === "light" ? "◐" : cue.name === "echo" ? "≋" : "≈"}
            </span>
            <small>
              {cue.name}
              <Tooltip>
                <TooltipTrigger asChild>
                  <button className="cue-help" type="button" aria-label={`About ${cue.name}`}>
                    ?
                  </button>
                </TooltipTrigger>
                <TooltipContent>{cueHelp[cue.name]}</TooltipContent>
              </Tooltip>
            </small>
            <strong>{cue.label.replace(` ${cue.name}`, "")}</strong>
            <code>{cue.value > 0 ? "+1" : "−1"}</code>
          </li>
        ))}
      </ul>
    </TooltipProvider>
  );
}

export function Gate({
  index,
  selected,
  reward,
  insight,
  onInspect,
}: {
  index: number;
  selected: boolean;
  reward: number | null;
  insight?: string | null;
  onInspect: () => void;
}) {
  const gate = gateDetails[index]!;
  return (
    <button
      type="button"
      className={`gate gate-${index} ${selected ? "selected" : ""}`}
      aria-label={`${gate.name}, ${gate.rune}${insight ? `, ${insight}` : ""}${selected ? ", selected by PyMAB" : ""}`}
      onClick={onInspect}
    >
      <span className="gate-arch" aria-hidden="true">
        <span className="gate-symbol">{gate.symbol}</span>
        {selected && <span className="gate-glow" />}
      </span>
      <strong>{gate.name}</strong>
      <small>{gate.rune}</small>
      {insight && <span className="gate-insight">{insight}</span>}
      {selected && reward !== null && (
        <span className={`reward-token ${reward > 0 ? "won" : "empty"}`}>
          {reward === 1 ? "+1 RELIC" : reward === 0 ? "EMPTY" : reward.toFixed(2)}
        </span>
      )}
    </button>
  );
}

export function Chamber({
  snapshot,
  animationState = "idle",
}: {
  snapshot: LessonSnapshot | null;
  animationState?: string;
}) {
  const diagnosticAfter = snapshot?.diagnostic?.after;
  const after =
    diagnosticAfter && typeof diagnosticAfter === "object"
      ? (diagnosticAfter as Record<string, unknown>)
      : null;
  const predictedMeans =
    snapshot?.diagnostic?.predictedMeans ?? after?.predictedMeans ?? after?.estimates;
  const learnedEstimates =
    Array.isArray(predictedMeans) && predictedMeans.length === 3
      ? predictedMeans.map((value) => (typeof value === "number" ? value : null))
      : null;
  const publicEnvironment = snapshot?.environment;
  const configuredProbabilities =
    snapshot?.mode === "freePlay" && Array.isArray(publicEnvironment?.probabilities)
      ? (publicEnvironment.probabilities as unknown[])
      : null;
  const configuredMeans =
    snapshot?.mode === "freePlay" && Array.isArray(publicEnvironment?.means)
      ? (publicEnvironment.means as unknown[])
      : null;
  const configuredTheta =
    snapshot?.mode === "freePlay" && Array.isArray(publicEnvironment?.theta)
      ? (publicEnvironment.theta as unknown[])
      : null;
  const cueFeature =
    snapshot?.visibleCues.length === 3
      ? [1, ...snapshot.visibleCues.map((cue) => cue.value)]
      : null;
  const activeState = Array.isArray(after?.active) ? after.active : null;

  const insightFor = (index: number): string | null => {
    const probability = configuredProbabilities?.[index];
    if (typeof probability === "number") {
      return `Reward chance ${(probability * 100).toFixed(1)}%`;
    }
    const mean = configuredMeans?.[index];
    if (typeof mean === "number") return `Mean reward ${mean.toFixed(2)}`;
    const thetaRow = configuredTheta?.[index];
    if (Array.isArray(thetaRow) && cueFeature && thetaRow.length === 4) {
      const coefficients = thetaRow as unknown[];
      const linear = coefficients.reduce<number>(
        (total, coefficient, featureIndex) =>
          total + Number(coefficient) * cueFeature[featureIndex]!,
        0,
      );
      if (policyCatalog[snapshot!.policyId].environment === "contextual-logistic") {
        return `Current reward chance ${(100 / (1 + Math.exp(-linear))).toFixed(1)}%`;
      }
      return `Current expected reward ${linear.toFixed(2)}`;
    }
    if (snapshot?.family === "contextual") {
      const estimate = learnedEstimates?.[index];
      return typeof estimate === "number"
        ? `Learned estimate ${estimate.toFixed(2)}`
        : "No estimate yet";
    }
    if (snapshot?.family === "best-arm" && activeState) {
      return activeState[index] ? "Active candidate" : "Eliminated";
    }
    return null;
  };

  return (
    <section
      className={`chamber world-${snapshot?.family ?? "foundations"} ${animationState}`}
      aria-label="Independent decision round"
    >
      <div className="chamber-haze" aria-hidden="true" />
      <CueStrip snapshot={snapshot} />
      {snapshot?.family === "changing" && (
        <div className="phase-timeline" aria-label="Changing environment timeline">
          <span style={{ width: `${(snapshot.step / snapshot.horizon) * 100}%` }} />
          <strong>Round {snapshot.step || 1}: the environment may change over time</strong>
        </div>
      )}
      {snapshot?.family === "adversarial" && (
        <p className="world-note">
          The arena assigns rewards each round. Only the selected reward is revealed.
        </p>
      )}
      <div className="gates">
        {gateDetails.map((_, index) => (
          <Gate
            key={index}
            index={index}
            selected={snapshot?.selectedArm === index}
            reward={snapshot?.reward ?? null}
            insight={insightFor(index)}
            onInspect={() => undefined}
          />
        ))}
      </div>
    </section>
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
  const binary = snapshot
    ? ["stationary-bernoulli", "changing-bernoulli", "best-arm", "contextual-logistic"].includes(
        policyCatalog[snapshot.policyId].environment,
      )
    : true;
  const result =
    reward === null || reward === undefined
      ? ""
      : binary
        ? reward > 0
          ? "Relic found"
          : "No relic this time"
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
          <dt>{snapshot?.objective === "best-arm" ? "Samples" : "Total reward"}</dt>
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
  const numericRewards =
    environmentKind === "stationary-gaussian" ||
    environmentKind === "contextual-linear" ||
    environmentKind === "adversarial";
  const gateResults = gateDetails.map((_, gateIndex) => {
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
        The run earned <strong>{snapshot.totalReward.toFixed(3)} total reward</strong> with{" "}
        <strong>{snapshot.cumulativeExpectedRegret.toFixed(2)} expected regret</strong>.
      </p>
      {snapshot.objective === "best-arm" && (
        <p className="recommendation-result">
          Recommended portal:{" "}
          <strong>
            {snapshot.recommendation === null
              ? "Not available"
              : gateDetails[snapshot.recommendation]?.name}
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
            <dl className="truth-grid" aria-label="Configured gate reward probabilities">
              {probabilities.map((value, index) => {
                const result = gateResults[index]!;
                return (
                  <div key={gateDetails[index]?.name}>
                    <dt>{gateDetails[index]?.name}</dt>
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
              The best gate can change with the signals. The table below compares the optimal gate
              with the policy's choice in each round.
            </p>
            <MatrixTable
              label="Hidden environment coefficients by gate"
              value={snapshot.hiddenTruth?.theta}
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
                  <td>{gateDetails[event.selectedArm]?.name}</td>
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
                        return typeof arm === "number" ? gateDetails[arm]?.name : "Not available";
                      })()}
                    </td>
                  )}
                  <td>
                    {numericRewards
                      ? event.reward.toPrecision(4)
                      : event.reward
                        ? "Relic"
                        : "Empty"}
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
  const selected = candidates.find(([, value]) => Array.isArray(value) && value.length === 3);
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
            <th>Gate</th>
            <th>Prediction</th>
            <th>Bonus</th>
            <th>UCB</th>
          </tr>
        </thead>
        <tbody>
          {numeric.map((value, index) => (
            <tr key={index}>
              <th>{gateDetails[index]?.name}</th>
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
      aria-label={`${label}: ${numeric.map((value, i) => `${gateDetails[i]?.name} ${value.toPrecision(4)}`).join(", ")}`}
    >
      <small className="policy-bars-label">{label}</small>
      {numeric.map((value, index) => (
        <div key={index}>
          <span>{gateDetails[index]?.symbol}</span>
          <i
            style={{ "--bar": `${Math.max(3, (Math.abs(value) / scale) * 100)}%` } as CSSProperties}
          />
          <code>{value.toPrecision(4)}</code>
        </div>
      ))}
    </div>
  );
}

function MatrixTable({ label, value }: { label: string; value: unknown }) {
  if (!Array.isArray(value) || !value.every((row) => Array.isArray(row))) return null;
  const matrix = value as unknown[][];
  return (
    <table className="matrix-table">
      <caption>{label}</caption>
      <tbody>
        {matrix.map((row, rowIndex) => (
          <tr key={rowIndex}>
            <th scope="row">Gate {rowIndex + 1}</th>
            {row.map((cell, columnIndex) => (
              <td key={columnIndex}>{Number(cell).toPrecision(4)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
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
                        n_arms: 3,
                        ...(snapshot.family === "contextual" ? { n_features: 4 } : {}),
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
                  />
                  <MatrixTable
                    label="Learned coefficient estimates"
                    value={
                      snapshot.diagnostic.thetaBefore ??
                      (snapshot.diagnostic.after as Record<string, unknown> | undefined)?.theta
                    }
                  />
                </>
              )}
              {snapshot.family === "best-arm" && snapshot.diagnostic?.recommendation !== null && (
                <p className="inspector-callout">
                  Current recommendation:{" "}
                  {gateDetails[Number(snapshot.diagnostic?.recommendation)]?.name}
                </p>
              )}
              <details>
                <summary>Full validated snapshot</summary>
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
