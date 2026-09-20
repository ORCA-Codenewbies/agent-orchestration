# agents/rules/safety_agent.py

from datetime import datetime, timezone
from typing import Dict
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation

class SafetyRuleAgent(BaseAgent):
    """
    Algorithmic Safety Agent (Rule Engine)

    Evaluates physical maritime safety thresholds and validates ML model outputs against
    deterministic meteorological & oceanographic rules using hierarchical arbitration.
    """
    @property
    def name(self) -> str:
        return "safety_rules"

    def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        loc = plan.target_location or plan.location or plan.reference_location
        if not loc:
            return AgentResult(
                agent=self.name,
                status="ERROR",
                location=None,
                timestamp=datetime.now(timezone.utc),
                data={"error": "LOCATION_REQUIRED", "safety_clearance": "UNKNOWN", "safety_status": "UNKNOWN", "rule_violations": [], "physical_metrics": {}},
                confidence=1.0,
                sources=["IMD_SAFETY_RULES_v1"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )
        lat = loc.latitude
        lon = loc.longitude

        weather_res = context.get("weather")
        ocean_res = context.get("ocean")
        geospatial_res = context.get("geospatial")
        marine_safety_res = context.get("marine_safety")

        weather_data = weather_res.data if weather_res else {}
        ocean_data = ocean_res.data if ocean_res else {}
        geo_data = geospatial_res.data if geospatial_res else {}
        marine_safety_data = marine_safety_res.data if marine_safety_res else {}

        # 1. Extract Official Warnings & BSI
        imd_official = weather_data.get("imd_official", {})
        fisherman_warning = imd_official.get("fisherman_warning", {})
        alert_level = fisherman_warning.get("alert_level", "NORMAL")

        bsi_assessment = marine_safety_data.get("bsi_assessment", {})
        bsi_advisory = bsi_assessment.get("advisory", "SAFE")

        # 2. Extract ML Models
        orca_weather = weather_data.get("orca_xgboost", {})
        ml_risk_level = orca_weather.get("risk_level", "NORMAL")

        # 3. Extract Geospatial Context
        in_domain = ocean_data.get("in_domain", True)
        allowed = geo_data.get("allowed", True)
        is_restricted = geo_data.get("origin", {}).get("is_restricted", False)

        warnings = []
        rule_violations = []

        if not in_domain:
            rule_violations.append("OUT_OF_DOMAIN")
            warnings.append(ocean_data.get("out_of_domain_warning", "UNSUPPORTED LOCATION"))
            
        if is_restricted:
            rule_violations.append("EEZ_RESTRICTED")
            warnings.append("ZONE RESTRICTION: Location is outside authorized fishing zone.")
        elif not allowed and in_domain:
            rule_violations.append("INLAND_LOCATION")
            warnings.append("UNSUPPORTED LOCATION: Target is an inland location without marine candidates.")

        # Extract ML risk from risk agent, NOT weather agent
        risk_agent_res = context.get("risk")
        risk_level = risk_agent_res.data.get("risk_level", "NORMAL") if risk_agent_res and risk_agent_res.data else "NORMAL"
        override_risk_level = risk_level

        # Priority 0: MARINE_SAFETY vessel advisory overrides everything
        vessel_advisory = marine_safety_data.get("vessel_safety_advisory", "SAFE")

        # Hierarchical Arbitration Logic

        # Check for Critical Domain Agent Failures
        critical_data_missing = False
        if not weather_res or weather_res.status == "error" or \
           not ocean_res or ocean_res.status == "error" or \
           not risk_agent_res or risk_agent_res.status == "error":
            critical_data_missing = True

        if not geospatial_res or geospatial_res.status == "error":
            critical_data_missing = True

        if not in_domain or not allowed:
            safety_clearance = "RESTRICTED"
            if not in_domain or (not allowed and not is_restricted):
                safety_status = "UNSUPPORTED_LOCATION"
            else:
                safety_status = "UNSAFE"
            override_risk_level = "SEVERE"
        elif vessel_advisory == "DANGER":
            safety_clearance = "RESTRICTED"
            safety_status = "UNSAFE"
            rule_violations.append("MARINE_SAFETY_DANGER")
            override_risk_level = "SEVERE"
        elif vessel_advisory == "WARNING":
            safety_clearance = "RESTRICTED"
            safety_status = "UNSAFE"
            rule_violations.append("MARINE_SAFETY_WARNING")
            override_risk_level = "DANGER"
        elif vessel_advisory == "UNKNOWN":
            safety_clearance = "UNKNOWN"
            safety_status = "UNVERIFIABLE"
            rule_violations.append("MARINE_SAFETY_UNKNOWN")
            override_risk_level = "UNKNOWN"
            warnings.append("Critical hazard data is unavailable. Safety cannot be verified.")
        elif vessel_advisory == "CAUTION":
            safety_clearance = "CAUTION"
            safety_status = "MODERATE_RISK"
            rule_violations.append("MARINE_SAFETY_CAUTION")
            override_risk_level = risk_level if risk_level in ("HIGH", "SEVERE", "DANGER", "DANGEROUS") else "MODERATE"
        elif alert_level == "RED" or alert_level == "SEVERE": # Priority 1: IMD RED
            safety_clearance = "RESTRICTED"
            safety_status = "UNSAFE"
            rule_violations.append("IMD_SEVERE_WARNING")
            override_risk_level = "SEVERE"
        elif bsi_advisory == "DANGER" or bsi_advisory == "WARNING": # Priority 2 & 3: BSI Danger/Warning
            safety_clearance = "RESTRICTED"
            safety_status = "UNSAFE"
            rule_violations.append("BSI_RESTRICTION")
            override_risk_level = "DANGER"
        elif critical_data_missing:
            safety_clearance = "UNKNOWN"
            safety_status = "UNVERIFIABLE"
            override_risk_level = "UNKNOWN"
            rule_violations.append("MISSING_CRITICAL_DATA")
            warnings.append("Safety cannot be verified due to missing or failed weather/ocean data.")
        elif alert_level == "ORANGE": # Priority 5: IMD Orange
            safety_clearance = "CAUTION"
            safety_status = "MODERATE_RISK"
            rule_violations.append("IMD_CAUTION_WARNING")
            override_risk_level = "MODERATE" if risk_level not in ("HIGH", "SEVERE", "DANGER", "DANGEROUS") else risk_level
        elif ml_risk_level in ("HIGH", "SEVERE", "DANGEROUS", "DANGER") or risk_level in ("HIGH", "SEVERE", "DANGEROUS", "DANGER"):
            safety_clearance = "RESTRICTED"
            safety_status = "UNSAFE"
            rule_violations.append("ML_SEVERE_RISK")
            override_risk_level = "DANGER" if "DANGER" in (ml_risk_level, risk_level) or "DANGEROUS" in (ml_risk_level, risk_level) else "SEVERE" if "SEVERE" in (ml_risk_level, risk_level) else "HIGH"
            warnings.append(f"ML models indicate dangerous conditions (Weather: {ml_risk_level}, Marine: {risk_level}).")
        elif ml_risk_level == "CAUTION" or risk_level == "CAUTION":
            safety_clearance = "CAUTION"
            safety_status = "MODERATE_RISK"
            rule_violations.append("ML_CAUTION_RISK")
            override_risk_level = "MODERATE"
            warnings.append(f"ML models indicate moderate risk (Weather: {ml_risk_level}, Marine: {risk_level}).")
        else: # Priority 8: All Clear. Official sources supersede ML.
            safety_clearance = "CLEARED"
            safety_status = "SAFE"
            override_risk_level = "NORMAL"

        # Handle CYCLONE_DATA_UNAVAILABLE explicitly to prevent defaulting to CLEARED
        cyclone_data = marine_safety_data.get("cyclone_data_status", "UNKNOWN")
        if cyclone_data == "CYCLONE_DATA_UNAVAILABLE" and safety_clearance != "RESTRICTED":
            warnings.append("CYCLONE_MONITORING_UNAVAILABLE: Live cyclone data could not be retrieved.")
            safety_clearance = "UNKNOWN"
            safety_status = "UNVERIFIABLE"

        payload = {
            "latitude": lat,
            "longitude": lon,
            "safety_clearance": safety_clearance,
            "safety_status": safety_status,
            "override_risk_level": override_risk_level,
            "raw_ml_risk": risk_level,
            "rule_violations": rule_violations,
            "bsi_advisory": bsi_advisory,
            "physical_metrics": {
                "geospatial_allowed": allowed
            }
        }

        from schemas.contracts import AgentAudit
        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=GeoLocation(latitude=lat, longitude=lon, name=loc.name if loc else None),
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.95,
            sources=["IMD_SAFETY_RULES_v1", "MARITIME_PHYSICAL_LIMITS", "INCOIS_BSI"],
            warnings=warnings + marine_safety_data.get("imd_warnings", []),
            audit=AgentAudit(
                inputs_used={"bsi_advisory": bsi_advisory, "ml_risk": ml_risk_level, "geospatial_allowed": allowed},
                outputs=payload,
                output_reason="Evaluated physical maritime safety thresholds and official advisories to determine final safety clearance.",
                sources=["IMD_SAFETY_RULES_v1", "MARITIME_PHYSICAL_LIMITS", "INCOIS_BSI"],
                score_source="RULE_BASED_WEIGHTED",
                score_reason="Safety clearance was determined through deterministic hierarchical arbitration of marine inputs, models, and official IMD/INCOIS warnings."
            )
        )
