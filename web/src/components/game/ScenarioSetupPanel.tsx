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

export interface ScenarioConfiguration {
  scenarioId: ScenarioId;
  mode: LessonMode;
  seed: number;
  parameters: Record<string, number>;
  environment: Record<string, unknown> | null;
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
            </div>
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
              disabled={pending || !Number.isSafeInteger(configuration.seed)}
              onClick={onApply}
            >
              Start configured run
            </Button>
          </div>
        </div>
      )}
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
    const theta = (environment.theta as number[][] | undefined) ?? [
      [0.1, -0.65, 0.2, 0.35],
      [-0.35, 0.85, 0.7, -0.15],
      [0.05, -0.75, -0.55, 0.45],
    ];
    const names = ["Article", "Product", "Tutorial"];
    const features = ["Base", "Returning", "Engagement", "Weekend"];
    return (
      <details className="advanced-parameters scenario-environment">
        <summary>Simulation coefficients</summary>
        <p className="field-help">
          Positive values raise click probability when a signal is positive. Negative values lower
          it.
        </p>
        <div className="scenario-matrix-editor">
          {theta.map((row, rowIndex) =>
            row.map((value, columnIndex) => (
              <div key={`${rowIndex}-${columnIndex}`}>
                <Label htmlFor={`theta-${rowIndex}-${columnIndex}`}>
                  {names[rowIndex]} · {features[columnIndex]}
                </Label>
                <Input
                  id={`theta-${rowIndex}-${columnIndex}`}
                  type="number"
                  step="0.05"
                  value={value}
                  disabled={pending}
                  onChange={(event) => {
                    const next = theta.map((item) => [...item]);
                    next[rowIndex]![columnIndex] = Number(event.target.value);
                    onChange({ ...configuration, environment: { theta: next } });
                  }}
                />
              </div>
            )),
          )}
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
