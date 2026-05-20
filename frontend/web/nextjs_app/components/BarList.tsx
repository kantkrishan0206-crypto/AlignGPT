export function BarList({ rows }: { rows: Array<{ label: string; value: number }> }) {
  return (
    <div className="bar-list">
      {rows.map((row) => (
        <div className="bar-row" key={row.label}>
          <div className="bar-label">
            <span>{row.label}</span>
            <strong>{Math.round(row.value * 100)}%</strong>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${row.value * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
