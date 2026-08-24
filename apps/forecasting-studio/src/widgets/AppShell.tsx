import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const links = [
  { to: "/", label: "Overview" },
  { to: "/data", label: "Data" },
  { to: "/features", label: "Features" },
  { to: "/studio", label: "ML Studio" },
  { to: "/compare", label: "Comparison" },
  { to: "/uncertainty", label: "Uncertainty" },
  { to: "/novelty", label: "Novelty" },
  { to: "/forecast", label: "Live forecast" },
];

export function AppShell() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen grid-fade">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-64 flex-col border-r border-line bg-ink-2/90 px-5 py-6 backdrop-blur-md lg:flex">
        <p className="text-[11px] uppercase tracking-[0.24em] text-mist">J26-DS-322 · IT23415836</p>
        <h1 className="font-display mt-2 text-2xl leading-tight">HTFE Studio</h1>
        <p className="mt-2 text-sm text-mist">Hybrid Temporal Forecasting Engine · Member 2 prototype</p>
        <nav className="mt-8 flex flex-col gap-1">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `rounded-xl px-3 py-2 text-sm transition ${
                  isActive ? "bg-harvest text-ink" : "text-cream/80 hover:bg-panel"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
        <p className="mt-auto text-xs leading-relaxed text-mist">
          No backend in this demo. Every number is scored on observed wholesale prices after a chronological split.
        </p>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-ink/75 px-4 py-3 backdrop-blur-md lg:hidden">
          <span className="font-display text-lg">HTFE Studio</span>
          <select
            className="rounded-lg border border-line bg-panel px-2 py-1 text-sm"
            onChange={(e) => navigate(e.target.value)}
          >
            {links.map((l) => (
              <option key={l.to} value={l.to}>
                {l.label}
              </option>
            ))}
          </select>
        </header>
        <motion.main
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="mx-auto max-w-6xl px-4 py-8 md:px-8"
        >
          <Outlet />
        </motion.main>
      </div>
    </div>
  );
}
