"""
Farmer Recommendation UI Cards and Alternative Crop Routing Component.

Presents evidence-based decision support with strict separation between
data readiness (DRI), market saturation (MSRS), and crop viability (DCVS).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st

# Ensure dashboard root is on sys.path
_dashboard_root = str(Path(__file__).resolve().parent.parent)
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)

from engine.advisor import Recommendation


def render_advisory_card(rec: Recommendation) -> None:
    """
    Render transparent evidence-based farmer advisory card.

    Displays:
    1. Action and Headline.
    2. Selected Crop and Geographic Scope.
    3. Market Saturation (MSRS) and qualitative classification.
    4. Crop Viability (DCVS) and documented interpretation.
    5. Source Data Reliability (DRI) benchmark and scope.
    6. Recommendation Evidence and Confidence Limitation.
    7. Diagnostic breakdown and viable alternative crops.

    Args:
        rec: Recommendation object from FarmerAdvisoryEngine.
    """
    # 1. Main Action Banner
    st.markdown(
        f"""
        <div class="{rec.banner_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">{rec.headline}</span>
                <span class="badge-proxy">{rec.data_provenance}</span>
            </div>
            <div style="font-size: 1.35rem; font-weight: 800; color: {rec.action_color}; margin-bottom: 0.5rem; letter-spacing: -0.01em;">
                ACTION: {rec.action}
            </div>
            <div style="font-size: 0.86rem; color: #E8EFEA; line-height: 1.4;">
                Evaluated for <strong>{rec.crop}</strong> in <strong>{rec.district}</strong> ({rec.season_label}).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Four Core Evaluated Parameters (Structured Panel Grid)
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 12px 0 16px 0;">
            <div style="background: #132619; border: 1px solid #1E3A26; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #A3B899; font-weight: 700;">Market Saturation (MSRS)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">{rec.msrs_score:.2f}</div>
                <div style="font-size: 0.76rem; color: #C8D8CC;">Classification: <strong>{rec.msrs_tier}</strong></div>
            </div>
            <div style="background: #132619; border: 1px solid #1E3A26; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #A3B899; font-weight: 700;">Crop Viability (DCVS)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">{rec.dcvs_score:.1f}%</div>
                <div style="font-size: 0.76rem; color: #C8D8CC;">Viability Tier: <strong>{rec.dcvs_tier}</strong></div>
            </div>
            <div style="background: #132619; border: 1px solid #1E3A26; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #A3B899; font-weight: 700;">Source Data Reliability (DRI)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">{rec.dri_score:.2f}%</div>
                <div style="font-size: 0.76rem; color: #C8D8CC;">Source: <strong>{rec.dri_source} ({rec.dri_tier})</strong></div>
            </div>
            <div style="background: #132619; border: 1px solid #1E3A26; border-radius: 6px; padding: 10px 12px;">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #A3B899; font-weight: 700;">Reliability Scope</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #4ADE80; margin-top: 3px;">{rec.dri_scope}</div>
                <div style="font-size: 0.76rem; color: #C8D8CC;">Target: <strong>{rec.district}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Evidence Transparency & Confidence Limitation Callout Box
    st.markdown(
        f"""
        <div style="background: #0E1F14; border: 1px solid #204229; border-left: 4px solid #4ADE80; border-radius: 6px; padding: 12px 14px; margin-bottom: 16px;">
            <div style="font-size: 0.85rem; color: #E8EFEA; margin-bottom: 6px;">
                <strong>Recommendation Evidence:</strong> {rec.evidence_sufficiency}
            </div>
            <div style="font-size: 0.80rem; color: #A3B899; line-height: 1.45;">
                <strong>Confidence Limitation:</strong> {rec.confidence_limitation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Diagnostic & Evidence Breakdown
    st.markdown("#### Evidence & Diagnostic Breakdown")
    for pt in rec.rationale_points:
        st.markdown(f"- {pt}")

    # 5. Alternative Crop Recommendations
    if rec.suggested_alternatives:
        st.markdown("---")
        st.markdown(f"#### Recommended Alternative Crops for {rec.district}")
        st.caption("Evidence-based crop routing based on regional saturation (MSRS) and dynamic viability (DCVS):")

        cols = st.columns(len(rec.suggested_alternatives), gap="medium")
        for i, alt in enumerate(rec.suggested_alternatives):
            status_color = "#2E7D32" if alt["msrs"] < 0.30 else ("#D97706" if alt["msrs"] < 0.70 else "#C53030")
            expl_text = alt.get("explanation", "Alternative crop candidate.")
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #C8DEC9; border-radius: 8px; padding: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.20); height: 100%; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="font-weight: 800; font-size: 1.05rem; color: #1E5631; margin-bottom: 4px;">{alt['crop']}</div>
                            <div style="font-size: 0.82rem; color: {status_color}; font-weight: 700; margin-bottom: 2px;">MSRS: {alt['msrs']:.2f} — {alt['status']}</div>
                            <div style="font-size: 0.82rem; color: #1A2E1A; font-weight: 600; margin-bottom: 2px;">Viability (DCVS): <strong>{alt['dcvs']:.1f}%</strong></div>
                            <div style="font-size: 0.80rem; color: #2D4A33; margin-bottom: 6px;">Quoted Price: Rs. {alt['current_price']:.1f}/kg</div>
                        </div>
                        <div style="font-size: 0.76rem; color: #3D5743; border-top: 1px dashed #C8DEC9; padding-top: 6px; margin-top: 4px;">{expl_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
