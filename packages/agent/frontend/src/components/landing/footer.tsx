import { useMemo } from "react";

export function Footer() {
  const year = useMemo(() => new Date().getFullYear(), []);
  return (
    <footer className="mt-32 flex flex-col items-center justify-center" style={{ background: "#1A4840" }}>
      <hr className="from-border/0 to-border/0 m-0 h-px w-full border-none bg-linear-to-r via-white/10" />
      <div className="container flex h-20 flex-col items-center justify-center text-sm" style={{ color: "oklch(0.972 0.012 145)" }}>
        <p className="text-center font-serif text-lg md:text-xl">
          &quot;Originated from Open Source, give back to Open Source.&quot;
        </p>
      </div>
      <div className="container mb-8 flex flex-col items-center justify-center text-xs" style={{ color: "oklch(0.972 0.012 145 / 0.70)" }}>
        <p>Licensed under MIT License</p>
        <p>&copy; {year} Noldus</p>
      </div>
    </footer>
  );
}
