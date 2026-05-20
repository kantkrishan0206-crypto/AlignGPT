import { BarChart3 } from "lucide-react";

import { BarList } from "../../components/BarList";
import { MetricCard } from "../../components/MetricCard";
import { benchmarkSeries } from "../../lib/platformData";

export default function BenchmarksPage() {
  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Benchmark Viewer</p>
          <h1>Reproducible alignment benchmark summary</h1>
        </div>
        <span className="status">generated artifacts</span>
      </section>

      <section className="two-column">
        <article className="surface">
          <div className="section-title">
            <BarChart3 size={20} />
            <h2>Suite scores</h2>
          </div>
          <BarList rows={benchmarkSeries} />
        </article>
        <article className="surface">
          <h2>Scientific interpretation</h2>
          <p>
            The benchmark bundle combines hallucination, latency, throughput, robustness, bias,
            adversarial, and reproducibility measurements. Each run stores JSON, CSV, markdown, and
            tracker metadata so later runs can be compared without changing the evaluation contract.
          </p>
        </article>
      </section>

      <section className="metrics-grid">
        <MetricCard label="Best run" value="2026-05-20" detail="baseline operational proof bundle" />
        <MetricCard label="Regression threshold" value="2.5%" detail="suite-level alert gate" />
        <MetricCard label="Artifacts" value="JSON/CSV/MD" detail="committed sample outputs" />
      </section>
    </main>
  );
}
