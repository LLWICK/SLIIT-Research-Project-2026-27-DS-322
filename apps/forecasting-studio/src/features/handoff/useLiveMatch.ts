import { useMemo, useState } from "react";
import { useStudio } from "@/entities/results/load";
import type { LiveForecast } from "@/entities/results/types";

const ANCHOR = 0.85;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function interpolate(pool: LiveForecast[], intensity: number): LiveForecast {
  const sorted = [...pool].sort((a, b) => a.cultivation_intensity - b.cultivation_intensity);
  if (intensity <= sorted[0].cultivation_intensity) return sorted[0];
  const last = sorted[sorted.length - 1];
  if (intensity >= last.cultivation_intensity) return last;
  const right = sorted.findIndex((row) => row.cultivation_intensity >= intensity);
  const hi = sorted[right];
  const lo = sorted[right - 1];
  const t = (intensity - lo.cultivation_intensity) / (hi.cultivation_intensity - lo.cultivation_intensity || 1);
  return {
    ...lo,
    cultivation_intensity: intensity,
    predicted_price: lerp(lo.predicted_price, hi.predicted_price, t),
    lower_price: lerp(lo.lower_price, hi.lower_price, t),
    upper_price: lerp(lo.upper_price, hi.upper_price, t),
  };
}

function withPlantingPressure(base: LiveForecast, intensity: number): LiveForecast {
  // Cobweb: extra committed land means more harvest later, so the expected price eases.
  // Anchored at 85% so the default view matches the stored model output.
  const delta = intensity - ANCHOR;
  const factor = 1 - 0.18 * delta;
  const point = base.predicted_price * factor;
  const down = (base.predicted_price - base.lower_price) * (1 + 0.12 * Math.abs(delta));
  const up = (base.upper_price - base.predicted_price) * (1 + 0.12 * Math.abs(delta));
  return {
    ...base,
    cultivation_intensity: intensity,
    predicted_price: Math.round(point),
    lower_price: Math.round(Math.max(1, point - down)),
    upper_price: Math.round(point + up),
  };
}

export function useLiveMatch() {
  const { live } = useStudio();
  const crops = [...new Set(live.map((r) => r.crop))];
  const markets = [...new Set(live.map((r) => r.market))];
  const [crop, setCrop] = useState(crops.includes("carrot") ? "carrot" : (crops[0] ?? "carrot"));
  const [market, setMarket] = useState(markets.includes("colombo") ? "colombo" : (markets[0] ?? "colombo"));
  const [intensity, setIntensity] = useState(ANCHOR);

  const match = useMemo((): LiveForecast | null => {
    const pool = live.filter((r) => r.crop === crop && r.market === market);
    if (!pool.length) return null;
    const anchor = interpolate(pool, ANCHOR);
    return withPlantingPressure(anchor, intensity);
  }, [live, crop, market, intensity]);

  const request = {
    crop: crop.toUpperCase(),
    market: market.toUpperCase(),
    forecast_week: match?.forecast_week ?? "2025-12-22",
    current_commitment_hectares: Math.round(intensity * 1000),
  };

  const response = match
    ? {
        crop: request.crop,
        market: request.market,
        forecast_week: match.forecast_week,
        predicted_price: match.predicted_price,
        lower_price: match.lower_price,
        upper_price: match.upper_price,
        coverage_level: match.coverage_level,
        cultivation_intensity: match.cultivation_intensity,
        commitment_source: match.commitment_source,
        model_version: match.model_version,
      }
    : null;

  return { crops, markets, crop, setCrop, market, setMarket, intensity, setIntensity, match, request, response };
}
