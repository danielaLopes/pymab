import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { environmentLabel, type EnvironmentDraft, type Triple } from "@/state/environments";

const portalNames = ["Moon", "Sun", "Star"] as const;
const coefficientNames = ["Intercept", "Light", "Echo", "Tide"] as const;

interface EnvironmentEditorProps {
  draft: EnvironmentDraft;
  errors: Record<string, string>;
  source: "generated" | "custom";
  pending: boolean;
  onChange: (draft: EnvironmentDraft) => void;
  onRegenerate: () => void;
}

function SourceHeader({
  draft,
  source,
  pending,
  onRegenerate,
}: Pick<EnvironmentEditorProps, "draft" | "source" | "pending" | "onRegenerate">) {
  return (
    <div className="environment-heading">
      <div>
        <legend>{environmentLabel(draft.kind)}</legend>
        <p>These values define the reward environment, separately from the policy settings.</p>
      </div>
      <div className="probability-source">
        <span>{source === "generated" ? "Generated from seed" : "Custom"}</span>
        <Button type="button" variant="outline" size="sm" disabled={pending} onClick={onRegenerate}>
          Regenerate from seed
        </Button>
      </div>
    </div>
  );
}

function ProbabilityFields({
  values,
  path,
  errors,
  pending,
  onChange,
}: {
  values: Triple<string>;
  path: string;
  errors: Record<string, string>;
  pending: boolean;
  onChange: (values: Triple<string>) => void;
}) {
  return (
    <div className="portal-probability-grid">
      {portalNames.map((name, index) => {
        const numeric = Number(values[index]);
        const sliderValue = Number.isFinite(numeric) ? Math.max(0, Math.min(100, numeric)) : 50;
        const error = errors[`${path}.${index}`];
        return (
          <div className="portal-probability-card" key={name}>
            <div className="run-setup-label-row">
              <Label htmlFor={`${path}-${index}`}>{name}</Label>
              <span>{sliderValue.toFixed(1)}%</span>
            </div>
            <div className="parameter-inputs">
              <Slider
                aria-label={`${name} reward chance`}
                min={0}
                max={100}
                step={0.1}
                value={[sliderValue]}
                disabled={pending}
                onValueChange={([next]) => {
                  if (next === undefined) return;
                  const updated = [...values] as Triple<string>;
                  updated[index] = String(Number(next.toFixed(1)));
                  onChange(updated);
                }}
              />
              <div className="probability-number-wrap">
                <Input
                  id={`${path}-${index}`}
                  className="parameter-number"
                  type="number"
                  min={0}
                  max={100}
                  step={0.1}
                  value={values[index]}
                  disabled={pending}
                  aria-invalid={Boolean(error)}
                  onChange={(event) => {
                    const updated = [...values] as Triple<string>;
                    updated[index] = event.target.value;
                    onChange(updated);
                  }}
                />
                <span aria-hidden="true">%</span>
              </div>
            </div>
            {error && <p className="field-error">{error}</p>}
          </div>
        );
      })}
    </div>
  );
}

export function EnvironmentEditor(props: EnvironmentEditorProps) {
  const { draft, errors, pending, onChange } = props;
  return (
    <fieldset className={`environment-editor environment-${draft.kind}`}>
      <SourceHeader {...props} />

      {(draft.kind === "stationary-bernoulli" || draft.kind === "best-arm") && (
        <ProbabilityFields
          values={draft.probabilities}
          path="probabilities"
          errors={errors}
          pending={pending}
          onChange={(probabilities) => onChange({ ...draft, probabilities })}
        />
      )}

      {draft.kind === "stationary-gaussian" && (
        <div className="environment-number-grid">
          {portalNames.map((name, index) => (
            <div className="run-setup-field" key={name}>
              <Label htmlFor={`mean-${index}`}>{name} mean</Label>
              <Input
                id={`mean-${index}`}
                type="number"
                step={0.05}
                value={draft.means[index]}
                disabled={pending}
                aria-invalid={Boolean(errors[`means.${index}`])}
                onChange={(event) => {
                  const means = [...draft.means] as Triple<string>;
                  means[index] = event.target.value;
                  onChange({ ...draft, means });
                }}
              />
              {errors[`means.${index}`] && (
                <p className="field-error">{errors[`means.${index}`]}</p>
              )}
            </div>
          ))}
          <div className="run-setup-field">
            <Label htmlFor="reward-noise">Shared standard deviation</Label>
            <Input
              id="reward-noise"
              type="number"
              min={0.01}
              step={0.05}
              value={draft.standardDeviation}
              disabled={pending}
              aria-invalid={Boolean(errors.standardDeviation)}
              onChange={(event) => onChange({ ...draft, standardDeviation: event.target.value })}
            />
            {errors.standardDeviation && <p className="field-error">{errors.standardDeviation}</p>}
          </div>
        </div>
      )}

      {draft.kind === "changing-bernoulli" && (
        <div className="phase-editor">
          {draft.phases.map((phase, phaseIndex) => (
            <section className="phase-card" key={phaseIndex}>
              <div className="phase-title">
                <strong>Phase {phaseIndex + 1}</strong>
                <Label htmlFor={`phase-start-${phaseIndex}`}>Starts at round</Label>
                <Input
                  id={`phase-start-${phaseIndex}`}
                  type="number"
                  min={0}
                  step={1}
                  value={phase.start}
                  disabled={pending || phaseIndex === 0}
                  onChange={(event) => {
                    const phases = draft.phases.map((item, index) =>
                      index === phaseIndex ? { ...item, start: event.target.value } : item,
                    );
                    onChange({ ...draft, phases });
                  }}
                />
              </div>
              <ProbabilityFields
                values={phase.probabilities}
                path={`phases.${phaseIndex}.probabilities`}
                errors={errors}
                pending={pending}
                onChange={(probabilities) => {
                  const phases = draft.phases.map((item, index) =>
                    index === phaseIndex ? { ...item, probabilities } : item,
                  );
                  onChange({ ...draft, phases });
                }}
              />
            </section>
          ))}
          {errors.phases && <p className="field-error">{errors.phases}</p>}
        </div>
      )}

      {draft.kind === "adversarial" && (
        <div className="reward-table-wrap">
          <p className="field-help">
            Each row is a round. The policy observes only the reward for its selected portal.
          </p>
          <table className="environment-table">
            <caption>Rewards by round and portal</caption>
            <thead>
              <tr>
                <th>Round</th>
                {portalNames.map((name) => (
                  <th key={name}>{name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {draft.rewards.map((row, round) => (
                <tr key={round}>
                  <th scope="row">{round + 1}</th>
                  {row.map((value, arm) => (
                    <td key={arm}>
                      <Input
                        aria-label={`Round ${round + 1}, ${portalNames[arm]} reward`}
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={value}
                        disabled={pending}
                        aria-invalid={Boolean(errors[`rewards.${round}.${arm}`])}
                        onChange={(event) => {
                          const rewards = draft.rewards.map((item) => [...item] as Triple<string>);
                          rewards[round]![arm] = event.target.value;
                          onChange({ ...draft, rewards });
                        }}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {(draft.kind === "contextual-linear" || draft.kind === "contextual-logistic") && (
        <details className="coefficient-editor">
          <summary>Edit the 3 by 4 coefficient matrix</summary>
          <p className="field-help">
            The intercept and the three signal coefficients define each portal's conditional reward.
          </p>
          <div className="reward-table-wrap">
            <table className="environment-table">
              <caption>Signal coefficients by portal</caption>
              <thead>
                <tr>
                  <th>Portal</th>
                  {coefficientNames.map((name) => (
                    <th key={name}>{name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {draft.theta.map((row, arm) => (
                  <tr key={arm}>
                    <th scope="row">{portalNames[arm]}</th>
                    {row.map((value, coefficient) => (
                      <td key={coefficient}>
                        <Input
                          aria-label={`${portalNames[arm]} ${coefficientNames[coefficient]} coefficient`}
                          type="number"
                          step={0.05}
                          value={value}
                          disabled={pending}
                          aria-invalid={Boolean(errors[`theta.${arm}.${coefficient}`])}
                          onChange={(event) => {
                            const theta = draft.theta.map((item) => [
                              ...item,
                            ]) as typeof draft.theta;
                            theta[arm]![coefficient] = event.target.value;
                            onChange({ ...draft, theta });
                          }}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {draft.kind === "contextual-linear" && (
            <div className="run-setup-field contextual-noise">
              <Label htmlFor="contextual-noise">Reward standard deviation</Label>
              <Input
                id="contextual-noise"
                type="number"
                min={0.01}
                step={0.05}
                value={draft.standardDeviation}
                disabled={pending}
                onChange={(event) => onChange({ ...draft, standardDeviation: event.target.value })}
              />
            </div>
          )}
        </details>
      )}
    </fieldset>
  );
}
