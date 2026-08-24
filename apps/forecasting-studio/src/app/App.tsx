import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { StudioContext, loadStudioData } from "@/entities/results/load";
import type { StudioData } from "@/entities/results/types";
import { AppShell } from "@/widgets/AppShell";
import { SystemPage } from "@/pages/SystemPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { DataPage } from "@/pages/DataPage";
import { FeaturesPage } from "@/pages/FeaturesPage";
import { StudioPage } from "@/pages/StudioPage";
import { ComparisonPage } from "@/pages/ComparisonPage";
import { UncertaintyPage } from "@/pages/UncertaintyPage";
import { NoveltyPage } from "@/pages/NoveltyPage";

export function App() {
  const [data, setData] = useState<StudioData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStudioData().then(setData).catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="grid min-h-screen place-items-center px-6 text-center">
        <div>
          <p className="font-display text-3xl">Could not load demo artifacts</p>
          <p className="mt-3 text-mist">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="grid min-h-screen place-items-center">
        <div className="text-center">
          <p className="text-[11px] uppercase tracking-[0.24em] text-harvest">Loading registry</p>
          <p className="font-display mt-2 text-3xl">Replaying experiment outputs</p>
        </div>
      </div>
    );
  }

  return (
    <StudioContext.Provider value={data}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<SystemPage />} />
            <Route path="evidence" element={<OverviewPage />} />
            <Route path="data" element={<DataPage />} />
            <Route path="features" element={<FeaturesPage />} />
            <Route path="studio" element={<StudioPage />} />
            <Route path="compare" element={<ComparisonPage />} />
            <Route path="uncertainty" element={<UncertaintyPage />} />
            <Route path="novelty" element={<NoveltyPage />} />
            <Route path="forecast" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </StudioContext.Provider>
  );
}
