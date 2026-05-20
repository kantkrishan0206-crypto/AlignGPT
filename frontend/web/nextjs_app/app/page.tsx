import { Activity, Database, Gauge, ShieldCheck } from "lucide-react";

const panels = [
  { label: "Experiment Runs", value: "Ready", icon: Activity },
  { label: "Benchmark Health", value: "Smoke passing", icon: Gauge },
  { label: "Safety Gates", value: "Enabled", icon: ShieldCheck },
  { label: "Dataset Registry", value: "Governed", icon: Database },
];

export default function Page() {
  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">AlignGPT Platform</p>
          <h1>Alignment operations cockpit</h1>
        </div>
        <span className="status">development</span>
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
    </main>
  );
}
