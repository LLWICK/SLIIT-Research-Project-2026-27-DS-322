import { createContext, useContext } from "react";
import type { StudioData } from "./types";

async function j<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json() as Promise<T>;
}

export async function loadStudioData(): Promise<StudioData> {
  const [meta, coverage, split, dictionary, originMap, series, cobweb, comparison, ablation, weeks, importance, live, models] =
    await Promise.all([
      j("/data/meta.json"),
      j("/data/coverage.json"),
      j("/data/split.json"),
      j("/data/data_dictionary.json"),
      j("/data/origin_map.json"),
      j("/data/series.json"),
      j("/data/cobweb.json"),
      j("/data/comparison.json"),
      j("/data/ablation.json"),
      j("/data/commitment_weeks.json"),
      j("/data/importance.json"),
      j("/data/live_forecast.json"),
      j("/data/models.json"),
    ]);

  return {
    meta,
    coverage,
    split,
    dictionary,
    originMap,
    series,
    cobweb,
    comparison,
    ablation,
    weeks,
    importance,
    live,
    models,
  } as StudioData;
}

export const StudioContext = createContext<StudioData | null>(null);

export function useStudio() {
  const ctx = useContext(StudioContext);
  if (!ctx) throw new Error("Studio data is not ready");
  return ctx;
}
