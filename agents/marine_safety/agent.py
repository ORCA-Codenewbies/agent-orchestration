# agents/marine_safety/agent.py

from datetime import datetime, timezone
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation, AgentAudit
from .bsi import evaluate_conditions, SeaState, BoatProfile

class MarineSafetyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "marine_safety"

    def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        loc = plan.target_location or plan.location or plan.reference_location
        if not loc:
            return AgentResult(
                agent=self.name,
                status="ERROR",
                location=None,
                timestamp=datetime.now(timezone.utc),
                data={"error": "LOCATION_REQUIRED"},
                confidence=1.0,
                sources=["INCOIS_BSI_METHODOLOGY", "IMD_OFFICIAL_WARNING"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude

        weather_res = context.get("weather")
        ocean_res = context.get("ocean")

        weather_data = weather_res.data if weather_res else {}
        ocean_data = ocean_res.data if ocean_res else {}

        # Extract IMD warnings (if available)
        provider_data = weather_data.get("provider_data", {})
        fisherman_warning_sev = provider_data.get("official_marine_warning", "UNKNOWN")

        cyclone_context = weather_data.get("cyclone_context") or {}
        cyclone_status = cyclone_context.get("cyclone_data_status", "UNKNOWN")
        any_hazard = cyclone_context.get("any_within_hazard_radius", False)

        active_imd_warnings = []
        if fisherman_warning_sev and str(fisherman_warning_sev).upper() not in ("UNKNOWN", "NORMAL", "GREEN", "NONE", "NO WARNING"):
            active_imd_warnings.append(f"IMD Warning active: {fisherman_warning_sev}")
            
        # Catastrophic Hazard Detection
        from data_sources.hazards import detect_catastrophic_hazards
        hazard_eval = detect_catastrophic_hazards(lat, lon, plan.intent, cyclone_context)
        
        catastrophic_active = hazard_eval.get("catastrophic_active", False)
        catastrophic_status = hazard_eval.get("catastrophic_status", "NONE")
        for h in hazard_eval.get("hazards", []):
            active_imd_warnings.append(f"CATASTROPHIC HAZARD [{h['severity']}]: {h['description']}")

        if any_hazard:
            # Find the closest active hazard
            for cyc in cyclone_context.get("active_cyclones", []):
                if cyc.get("within_hazard_radius"):
                    active_imd_warnings.append(f"JTWC Cyclone Alert: {cyc['name']} (Cat {cyc['category']}) at {cyc['distance_km']}km")

        # Extract INCOIS ocean state
        incois_state = ocean_data.get("incois_ocean_state", {})

        sea_state = SeaState(
            hs_m=float(incois_state.get("wave_height", 1.2) or 1.2),
            tz_s=float(incois_state.get("wave_period", 6.5) or 6.5),
            wave_dir_deg=float(incois_state.get("wave_direction", 180.0) or 180.0),
            swell_dir_deg=180.0,
            wind_speed_ms=float(incois_state.get("wind_speed", 4.5) or 4.5)
        )

        boat = BoatProfile(max_hs_m=2.0) # Assume small fishing vessel
        bsi_report = evaluate_conditions(sea_state, boat)

        vessel_advisory = bsi_report.advisory
        if catastrophic_active is None or cyclone_status == "CYCLONE_DATA_UNAVAILABLE":
            vessel_advisory = "UNKNOWN"
        elif catastrophic_active is True or any_hazard or (fisherman_warning_sev and str(fisherman_warning_sev).upper() not in ("UNKNOWN", "NORMAL", "GREEN", "NONE", "NO WARNING")):
            vessel_advisory = "DANGER"

        payload = {
            "bsi_assessment": bsi_report.model_dump(),
            "imd_warnings": active_imd_warnings,
            "vessel_safety_advisory": vessel_advisory,
            "cyclone_data_status": cyclone_status,
            "catastrophic_active": catastrophic_active,
            "catastrophic_status": catastrophic_status
        }

        warnings = active_imd_warnings + bsi_report.hazards

        status = "SUCCESS"
        if cyclone_status == "CYCLONE_DATA_UNAVAILABLE" or catastrophic_status == "WARNING_SOURCE_UNAVAILABLE":
            status = "DEGRADED"

        return AgentResult(
            agent=self.name,
            status=status,
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.95,
            sources=["INCOIS_BSI_METHODOLOGY", "JTWC_CYCLONE", "IMD_OFFICIAL_WARNING"],
            warnings=warnings,
            audit=AgentAudit(
                inputs_used={"hs_m": sea_state.hs_m, "wind_speed_ms": sea_state.wind_speed_ms},
                outputs=payload,
                output_reason="Evaluated Boat Safety Index (BSI) and catastrophic hazards.",
                sources=["INCOIS_BSI_METHODOLOGY", "JTWC_CYCLONE", "IMD_OFFICIAL_WARNING"],
                score_source="RULE_BASED_WEIGHTED",
                score_reason="Vessel safety advisory is determined by evaluating IMD official warnings, cyclone hazards, and BSI thresholds."
            )
        )
