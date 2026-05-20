export function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  return <span className={`pill pill-${normalized.replace(/[^a-z]/g, "-")}`}>{status}</span>;
}
