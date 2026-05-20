import { Cloud, ServerCog } from "lucide-react";

import { StatusPill } from "../../components/StatusPill";
import { deploymentChecks } from "../../lib/platformData";

const targets = ["Vercel frontend", "Render API", "Railway API", "AWS ECS", "Google Cloud Run", "PostgreSQL", "Redis"];

export default function DeploymentPage() {
  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Deployment Health</p>
          <h1>Internet-facing SaaS deployment path</h1>
        </div>
        <span className="status">credentials required</span>
      </section>

      <section className="two-column">
        <article className="surface">
          <div className="section-title">
            <ServerCog size={20} />
            <h2>Readiness checks</h2>
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
        <article className="surface">
          <div className="section-title">
            <Cloud size={20} />
            <h2>Prepared targets</h2>
          </div>
          <div className="tag-list">
            {targets.map((target) => (
              <span key={target}>{target}</span>
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}
