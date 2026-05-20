import Link from "next/link";
import { Activity, ArrowRight, Database, Gauge, ShieldCheck } from "lucide-react";

import { BarList } from "../components/BarList";
import { MetricCard } from "../components/MetricCard";
import { benchmarkSeries, platformSummary } from "../lib/platformData";

const panels = [
  { label: "Experiment Runs", value: "Ready", icon: Activity },
  { label: "Benchmark Health", value: "Smoke passing", icon: Gauge },
  { label: "Safety Gates", value: "Enabled", icon: ShieldCheck },
  { label: "Dataset Registry", value: "Governed", icon: Database },
];

export default function Page() {
  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">AlignGPT Platform</p>
          <h1>Operational alignment evaluation for deployed AI systems</h1>
          <p className="hero-copy">
            A lab-grade framework and SaaS-ready control plane for routing inference, running
            reproducible benchmarks, tracking safety findings, and preparing AI systems for
            production deployment.
          </p>
          <div className="hero-actions">
            <Link href="/dashboard" className="primary-link">
              Open dashboard <ArrowRight size={16} />
            </Link>
            <Link href="/benchmarks" className="secondary-link">
              View benchmarks
            </Link>
          </div>
        </div>
        <div className="hero-panel">
          <BarList rows={benchmarkSeries.slice(0, 4)} />
        </div>
      </section>

      <section className="grid">
        {panels.map((panel) => {
          const Icon = panel.icon;
          return (
            <article className="panel" key={panel.label}>
              <Icon aria-hidden="true" size={22} />
              <div>
                <h2>{panel.label}</h2>
                <p>{panel.value}</p>
              </div>
            </article>
          );
        })}
      </section>

      <section className="metrics-grid">
        <MetricCard
          label="Benchmark pass rate"
          value={`${Math.round(platformSummary.benchmarkPassRate * 100)}%`}
          detail="latest reproducibility bundle"
        />
        <MetricCard
          label="p95 latency"
          value={`${platformSummary.p95LatencyMs}ms`}
          detail="API + routing estimate"
        />
        <MetricCard
          label="Throughput"
          value={`${platformSummary.throughputRps} rps`}
          detail="staging load profile"
        />
      </section>
    </main>
  );
}
