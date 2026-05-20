import { Network } from "lucide-react";

import { traces } from "../../lib/platformData";

export default function TracesPage() {
  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Evaluation Traces</p>
          <h1>Request lifecycle evidence</h1>
        </div>
        <span className="status">stream-ready</span>
      </section>

      <section className="stack">
        {traces.map((trace) => (
          <article className="surface" key={trace.id}>
            <div className="section-title">
              <Network size={20} />
              <h2>{trace.id}</h2>
            </div>
            <p>
              Route: {trace.route} | Reward: {trace.reward} | Latency: {trace.latency}ms
            </p>
            <div className="trace-line">
              {trace.events.map((event) => (
                <span key={event}>{event}</span>
              ))}
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
