import { planLabel } from "../chatScope";

export default function PlanSelector({ plans, value, onChange, disabled = false }) {
  return (
    <div className="plan-selector" role="group" aria-label="แผนการเรียน">
      {plans.map(plan => (
        <button key={plan.plan_key} type="button" aria-pressed={value === plan.plan_key}
          disabled={disabled} onClick={() => onChange(plan.plan_key)}>
          {planLabel(plan.plan_key)}
        </button>
      ))}
    </div>
  );
}
