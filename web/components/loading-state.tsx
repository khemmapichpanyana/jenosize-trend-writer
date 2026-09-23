/** A skeleton placeholder with an accessible label (the shimmer lives in globals.css as .t-skeleton). */
export function LoadingState({ label = "Loading", rows = 3 }: { label?: string; rows?: number }) {
  const widths = ["w-full", "w-11/12", "w-2/3", "w-5/6", "w-3/4"];
  return (
    <div className="space-y-2.5 py-2" role="status" aria-live="polite">
      <span className="sr-only">{label}</span>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className={`t-skeleton h-3.5 rounded-md ${widths[i % widths.length]}`} aria-hidden />
      ))}
    </div>
  );
}
