import type { CSSProperties } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { scenarioCatalog } from "@/catalog/scenarios";
import type { LessonMode, ScenarioId } from "@/engine/protocol";
import {
  addRecommendationFeature,
  appendRecommendationCandidate,
  candidateNameErrors,
  defaultRecommendationFeatureIds,
  recommendationCandidateKinds,
  recommendationCandidateKindLabels,
  recommendationFeatureCatalog,
  recommendationFeatureIds,
  validRecommendationCandidates,
  validRecommendationFeatureIds,
  type RecommendationCandidate,
  type RecommendationCandidateKind,
  type RecommendationFeatureId,
  type RecommendationModelFeatureId,
} from "@/state/scenarioCandidates";
import { ArmSymbol } from "./ArmSymbol";

export interface ScenarioConfiguration {
  scenarioId: ScenarioId;
  mode: LessonMode;
  seed: number;
  parameters: Record<string, number>;
  environment: Record<string, unknown> | null;
  candidates: RecommendationCandidate[] | null;
  featureIds: RecommendationFeatureId[] | null;
  nextCandidateOrdinal: number;
}

const modes: LessonMode[] = ["guided", "challenge", "freePlay"];
const modeLabel: Record<LessonMode, string> = {
  guided: "Guided",
  challenge: "Challenge",
  freePlay: "Free play",
};

export function ScenarioSetupPanel({
  configuration,
  pending,
  expanded,
  onChange,
  onScenarioChange,
  onExpandedChange,
  onApply,
}: {
  configuration: ScenarioConfiguration;
  pending: boolean;
  expanded: boolean;
  onChange: (configuration: ScenarioConfiguration) => void;
  onScenarioChange: (scenarioId: ScenarioId) => void;
  onExpandedChange: (expanded: boolean) => void;
  onApply: () => void;
}) {
  const definition = scenarioCatalog[configuration.scenarioId];
  const featureIds = configuration.featureIds ?? defaultRecommendationFeatureIds;
  const candidatesValid =
    configuration.scenarioId !== "recommendations" ||
    (configuration.candidates !== null &&
      validRecommendationFeatureIds(featureIds) &&
      validRecommendationCandidates(configuration.candidates, featureIds));
  const setParameter = (key: string, value: number) =>
    onChange({
      ...configuration,
      parameters: { ...configuration.parameters, [key]: value },
    });

  return (
    <section
      className={`run-setup scenario-run-setup ${expanded ? "expanded" : "collapsed"}`}
      aria-labelledby="scenario-setup-title"
    >
      <button
        className="scenario-setup-toggle"
        type="button"
        aria-expanded={expanded}
        aria-controls="scenario-setup-controls"
        onClick={() => onExpandedChange(!expanded)}
      >
        <div>
          <p className="eyebrow">Run settings</p>
          <h2 id="scenario-setup-title">{definition.title}</h2>
        </div>
        <span className="scenario-setup-summary">
          <span>{modeLabel[configuration.mode]}</span>
          {configuration.candidates && <span>{configuration.candidates.length} candidates</span>}
          {configuration.featureIds && <span>{configuration.featureIds.length} signals</span>}
          {definition.parameterDefinitions.map((parameter) => (
            <span key={parameter.key}>
              {parameter.label} {configuration.parameters[parameter.key]}
            </span>
          ))}
          <strong>{expanded ? "Close settings" : "Edit settings"}</strong>
          <b aria-hidden="true">{expanded ? "−" : "+"}</b>
        </span>
      </button>
      {expanded && (
        <div id="scenario-setup-controls">
          <div className="scenario-setup-intro">
            <p>Changes apply when you start the configured run.</p>
          </div>
          <div className="run-setup-grid">
            <div className="run-setup-field run-setup-policy">
              <Label htmlFor="scenario-select">Scenario</Label>
              <Select
                value={configuration.scenarioId}
                disabled={pending}
                onValueChange={(value) => onScenarioChange(value as ScenarioId)}
              >
                <SelectTrigger id="scenario-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.values(scenarioCatalog).map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="field-help">{definition.className}</p>
            </div>
            <div className="run-setup-field run-setup-selector">
              <span className="run-setup-label" id="scenario-mode-label">
                Run mode
              </span>
              <ToggleGroup
                type="single"
                value={configuration.mode}
                disabled={pending}
                aria-labelledby="scenario-mode-label"
                onValueChange={(value) => {
                  if (!value) return;
                  const mode = value as LessonMode;
                  const seed =
                    mode === "challenge" ? definition.challengeSeed : definition.guidedSeed;
                  onChange({ ...configuration, mode, seed });
                }}
              >
                {modes.map((mode) => (
                  <ToggleGroupItem key={mode} value={mode}>
                    {modeLabel[mode]}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
              {configuration.mode === "challenge" && (
                <p className="mode-target">Keep cumulative expected regret below the target.</p>
              )}
              {configuration.mode === "freePlay" && (
                <p className="mode-target">
                  Free play unlocks the random seed and simulation coefficients. The current run
                  will not change until you start the configured run.
                </p>
              )}
            </div>
            {configuration.scenarioId === "recommendations" && configuration.candidates && (
              <>
                <RecommendationCandidateEditor
                  configuration={configuration}
                  pending={pending}
                  onChange={onChange}
                />
                <RecommendationFeatureEditor
                  configuration={configuration}
                  pending={pending}
                  onChange={onChange}
                />
              </>
            )}
            <div className="parameter-section">
              <div className="parameter-section-heading">
                <div>
                  <h3>Policy parameters</h3>
                  <p>These values go to the public PyMAB constructor.</p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    onChange({ ...configuration, parameters: { ...definition.parameters } })
                  }
                >
                  Restore defaults
                </Button>
              </div>
              <div className="dynamic-parameter-grid">
                {definition.parameterDefinitions.map((parameter) => {
                  const value = configuration.parameters[parameter.key] ?? parameter.minimum;
                  return (
                    <div className="run-setup-field parameter-card" key={parameter.key}>
                      <Label htmlFor={`scenario-${parameter.key}`}>{parameter.label}</Label>
                      <div className="parameter-inputs">
                        <Slider
                          aria-label={`${parameter.label} slider`}
                          min={parameter.minimum}
                          max={parameter.maximum}
                          step={parameter.step}
                          value={[value]}
                          disabled={pending}
                          onValueChange={(values) =>
                            setParameter(parameter.key, values[0] ?? value)
                          }
                        />
                        <Input
                          id={`scenario-${parameter.key}`}
                          className="parameter-number"
                          type="number"
                          min={parameter.minimum}
                          max={parameter.maximum}
                          step={parameter.step}
                          value={value}
                          disabled={pending}
                          onChange={(event) =>
                            setParameter(parameter.key, Number(event.target.value))
                          }
                        />
                      </div>
                      <p className="field-help">{parameter.help}</p>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="run-setup-field run-setup-seed">
              <Label htmlFor="scenario-seed">Random seed</Label>
              <Input
                id="scenario-seed"
                type="number"
                step={1}
                value={configuration.seed}
                readOnly={configuration.mode !== "freePlay"}
                disabled={pending}
                onChange={(event) =>
                  onChange({ ...configuration, seed: Number(event.target.value) })
                }
              />
              <p className="field-help">
                {configuration.mode === "freePlay"
                  ? "Use any safe whole number."
                  : "This mode uses a repeatable seed."}
              </p>
            </div>
            {configuration.mode === "freePlay" && (
              <ScenarioEnvironmentEditor
                configuration={configuration}
                pending={pending}
                onChange={onChange}
              />
            )}
          </div>
          <div className="run-setup-footer">
            <p className="field-help">
              The active run keeps its current values until you apply these changes.
            </p>
            <Button
              className="primary-button"
              disabled={pending || !Number.isSafeInteger(configuration.seed) || !candidatesValid}
              onClick={onApply}
            >
              {configuration.mode === "freePlay" ? "Start free play run" : "Start configured run"}
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}

function RecommendationCandidateEditor({
  configuration,
  pending,
  onChange,
}: {
  configuration: ScenarioConfiguration;
  pending: boolean;
  onChange: (configuration: ScenarioConfiguration) => void;
}) {
  const candidates = configuration.candidates ?? [];
  const nameErrors = candidateNameErrors(candidates);
  const updateCandidate = (id: string, update: Partial<RecommendationCandidate>) =>
    onChange({
      ...configuration,
      candidates: candidates.map((candidate) =>
        candidate.id === id ? { ...candidate, ...update } : candidate,
      ),
    });
  const moveCandidate = (index: number, offset: -1 | 1) => {
    const destination = index + offset;
    if (destination < 0 || destination >= candidates.length) return;
    const next = [...candidates];
    [next[index], next[destination]] = [next[destination]!, next[index]!];
    onChange({ ...configuration, candidates: next });
  };
  const addCandidate = (kind: RecommendationCandidateKind) => {
    onChange({
      ...configuration,
      candidates: appendRecommendationCandidate(
        candidates,
        kind,
        configuration.seed,
        configuration.nextCandidateOrdinal,
        configuration.featureIds ?? defaultRecommendationFeatureIds,
      ),
      nextCandidateOrdinal: configuration.nextCandidateOrdinal + 1,
    });
  };

  return (
    <section className="candidate-editor" aria-labelledby="candidate-editor-title">
      <div className="candidate-editor-heading">
        <div>
          <h3 id="candidate-editor-title">Candidates</h3>
          <p>Choose two to eight items for the recommendation slot.</p>
        </div>
        <span>{candidates.length} of 8</span>
      </div>
      <div className="candidate-list">
        {candidates.map((candidate, index) => (
          <div className="candidate-row" key={candidate.id}>
            <div className="candidate-name-field">
              <Label htmlFor={`candidate-name-${candidate.id}`}>Candidate {index + 1}</Label>
              <Input
                id={`candidate-name-${candidate.id}`}
                value={candidate.name}
                maxLength={32}
                disabled={pending}
                aria-invalid={Boolean(nameErrors[candidate.id])}
                aria-describedby={
                  nameErrors[candidate.id] ? `candidate-error-${candidate.id}` : undefined
                }
                onChange={(event) => updateCandidate(candidate.id, { name: event.target.value })}
              />
              {nameErrors[candidate.id] && (
                <p className="field-error" id={`candidate-error-${candidate.id}`}>
                  {nameErrors[candidate.id]}
                </p>
              )}
            </div>
            <div className="candidate-type-field">
              <Label htmlFor={`candidate-type-${candidate.id}`}>Visual type</Label>
              <Select
                value={candidate.symbolKind}
                disabled={pending}
                onValueChange={(value) =>
                  updateCandidate(candidate.id, {
                    symbolKind: value as RecommendationCandidateKind,
                  })
                }
              >
                <SelectTrigger id={`candidate-type-${candidate.id}`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {recommendationCandidateKinds.map((kind) => (
                    <SelectItem key={kind} value={kind}>
                      <span className="candidate-symbol-option">
                        <ArmSymbol
                          arm={{
                            name: recommendationCandidateKindLabels[kind],
                            shortName: recommendationCandidateKindLabels[kind],
                            symbolKind: kind,
                          }}
                        />
                        {recommendationCandidateKindLabels[kind]}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="candidate-row-actions" aria-label={`Reorder ${candidate.name}`}>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={pending || index === 0}
                aria-label={`Move ${candidate.name} up`}
                onClick={() => moveCandidate(index, -1)}
              >
                ↑
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={pending || index === candidates.length - 1}
                aria-label={`Move ${candidate.name} down`}
                onClick={() => moveCandidate(index, 1)}
              >
                ↓
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={pending || candidates.length <= 2}
                aria-label={`Remove ${candidate.name}`}
                onClick={() =>
                  onChange({
                    ...configuration,
                    candidates: candidates.filter((item) => item.id !== candidate.id),
                  })
                }
              >
                Remove
              </Button>
            </div>
          </div>
        ))}
      </div>
      <div className="candidate-add-actions" aria-label="Add candidate">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={pending || candidates.length >= 8}
          onClick={() => addCandidate("article")}
        >
          Add candidate
        </Button>
      </div>
    </section>
  );
}

function RecommendationFeatureEditor({
  configuration,
  pending,
  onChange,
}: {
  configuration: ScenarioConfiguration;
  pending: boolean;
  onChange: (configuration: ScenarioConfiguration) => void;
}) {
  const candidates = configuration.candidates ?? [];
  const active = configuration.featureIds ?? [...defaultRecommendationFeatureIds];
  const inactive = recommendationFeatureIds.filter((featureId) => !active.includes(featureId));
  const removeFeature = (featureId: RecommendationFeatureId) =>
    onChange({
      ...configuration,
      featureIds: active.filter((item) => item !== featureId),
    });
  const addFeature = (featureId: RecommendationFeatureId) =>
    onChange({
      ...configuration,
      featureIds: [...active, featureId],
      candidates: addRecommendationFeature(candidates, featureId, configuration.seed),
    });

  return (
    <section className="feature-editor" aria-labelledby="feature-editor-title">
      <div className="candidate-editor-heading">
        <div>
          <h3 id="feature-editor-title">Context signals</h3>
          <p>The base feature is always included. Add up to eight signals.</p>
        </div>
        <span>{active.length} of 8 signals</span>
      </div>
      {active.length ? (
        <div className="feature-list">
          {active.map((featureId) => {
            const feature = recommendationFeatureCatalog[featureId];
            return (
              <div className="feature-row" key={feature.id}>
                <span className="feature-type">{feature.type}</span>
                <span>
                  <strong>{feature.label}</strong>
                  <small>{feature.help}</small>
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={pending}
                  aria-label={`Remove ${feature.label}`}
                  onClick={() => removeFeature(featureId)}
                >
                  Remove
                </Button>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="field-help">Only the automatic base feature will be used.</p>
      )}
      <div className="feature-add-actions" aria-label="Add context signal">
        {inactive.map((featureId) => {
          const feature = recommendationFeatureCatalog[featureId];
          return (
            <Button
              key={feature.id}
              type="button"
              variant="outline"
              size="sm"
              disabled={pending || active.length >= 8}
              onClick={() => addFeature(feature.id)}
            >
              Add {feature.label}
            </Button>
          );
        })}
      </div>
    </section>
  );
}

function ScenarioEnvironmentEditor({
  configuration,
  pending,
  onChange,
}: {
  configuration: ScenarioConfiguration;
  pending: boolean;
  onChange: (configuration: ScenarioConfiguration) => void;
}) {
  const environment = configuration.environment ?? {};
  if (configuration.scenarioId === "recommendations") {
    const candidates = configuration.candidates ?? [];
    const activeFeatureIds = configuration.featureIds ?? defaultRecommendationFeatureIds;
    const features: Array<{ id: RecommendationModelFeatureId; label: string }> = [
      { id: "base", label: "Base" },
      ...activeFeatureIds.map((featureId) => ({
        id: featureId,
        label: recommendationFeatureCatalog[featureId].label,
      })),
    ];
    return (
      <details className="advanced-parameters scenario-environment">
        <summary>Simulation coefficients</summary>
        <p className="field-help">
          Positive values raise click probability when a signal is positive. Negative values lower
          it.
        </p>
        <div className="scenario-matrix-scroll">
          <div
            className="scenario-matrix-editor"
            style={
              {
                "--scenario-features": features.length,
                minWidth: `${Math.max(4, features.length) * 10}rem`,
              } as CSSProperties
            }
          >
            {candidates.map((candidate) =>
              features.map((feature) => {
                const value = candidate.coefficients[feature.id] ?? 0;
                return (
                  <div key={`${candidate.id}-${feature.id}`}>
                    <Label htmlFor={`theta-${candidate.id}-${feature.id}`}>
                      {candidate.name} · {feature.label}
                    </Label>
                    <Input
                      id={`theta-${candidate.id}-${feature.id}`}
                      type="number"
                      step="0.05"
                      value={value}
                      disabled={pending}
                      onChange={(event) => {
                        const coefficients = {
                          ...candidate.coefficients,
                          [feature.id]: Number(event.target.value),
                        };
                        onChange({
                          ...configuration,
                          candidates: candidates.map((item) =>
                            item.id === candidate.id ? { ...item, coefficients } : item,
                          ),
                        });
                      }}
                    />
                  </div>
                );
              }),
            )}
          </div>
        </div>
      </details>
    );
  }
  const fields = [
    ["baseRisk", "Base abuse probability", 0.32],
    ["riskWeight", "Risk-score influence", 0.16],
    ["newAccountWeight", "New-account influence", 0.07],
    ["sensitiveWeight", "Sensitive-endpoint influence", 0.06],
  ] as const;
  return (
    <details className="advanced-parameters scenario-environment">
      <summary>Simulation coefficients</summary>
      <p className="field-help">
        These values define the synthetic request stream. They do not replace the upstream risk
        model.
      </p>
      <div className="dynamic-parameter-grid">
        {fields.map(([key, label, fallback]) => (
          <div className="run-setup-field parameter-card" key={key}>
            <Label htmlFor={`defense-${key}`}>{label}</Label>
            <Input
              id={`defense-${key}`}
              type="number"
              min="0"
              max="0.5"
              step="0.01"
              value={Number(environment[key] ?? fallback)}
              disabled={pending}
              onChange={(event) =>
                onChange({
                  ...configuration,
                  environment: { ...environment, [key]: Number(event.target.value) },
                })
              }
            />
          </div>
        ))}
      </div>
    </details>
  );
}
