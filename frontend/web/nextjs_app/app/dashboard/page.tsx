import { Activity, BrainCircuit, Gauge, Server, ShieldCheck } from "lucide-react";

import { MetricCard } from "../../components/MetricCard";
import { StatusPill } from "../../components/StatusPill";
import { deploymentChecks, modelBackends, platformSummary, traces } from "../../lib/platformData";

export default function DashboardPage() {
  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Internal Dashboard</p>
          <h1>Alignment operations cockpit</h1>
        </div>
        <span className="status">auth-ready</span>
      </section>

      <section className="metrics-grid">
        <MetricCard label="Active models" value={`${platformSummary.activeModels}`} detail="router backends" />
        <MetricCard label="Open safety findings" value={`${platformSummary.safetyFindingsOpen}`} detail="review queue" />
        <MetricCard label="Deployment stage" value={platformSummary.deploymentStage} detail="cloud manifests ready" />
      </section>

      <section className="two-column">
        <article className="surface">
          <div className="section-title">
            <BrainCircuit size={20} />
            <h2>Model routing</h2>
          </div>
          <div className="table">
            {modelBackends.map((backend) => (
              <div className="table-row" key={backend.name}>
                <strong>{backend.name}</strong>
                <span>{backend.quantization}</span>
                <span>{backend.latency}ms</span>
                <StatusPill status={backend.health} />
              </div>
            ))}
          </div>
        </article>

        <article className="surface">
          <div className="section-title">
            <Server size={20} />
            <h2>Deployment readiness</h2>
          </div>
          <div className="stack">
            {deploymentChecks.map((check) => (
              <div className="check-row" key={check.name}>
                <div>
                  <strong>{check.name}</strong>
                  <p>{check.detail}</p>
                </div>
                <StatusPill status={check.status} />
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="grid">
        <article className="panel">
          <Activity size={22} />
          <div>
            <h2>Trace stream</h2>
            <p>{traces.length} evaluation traces loaded from the operational fixture.</p>
          </div>
        </article>
        <article className="panel">
          <Gauge size={22} />
          <div>
            <h2>Latency budget</h2>
            <p>Router chooses backends under request-specific latency and memory budgets.</p>
          </div>
        </article>
        <article className="panel">
          <ShieldCheck size={22} />
          <div>
            <h2>Safety boundary</h2>
            <p>Prompt injection and PII signals are surfaced before model execution.</p>
          </div>
        </article>
      </section>
    </main>
  );
}
