# agents/rules/recommendation_agent.py

from datetime import datetime, timezone
from typing import Dict, Any
from enum import Enum
from schemas.contracts import BaseAgent, QueryPlan, AgentResult, GeoLocation, MarineCandidateIdentity, AgentAudit
from .fishing_search import rank_fishing_candidates
from agents.geospatial.astar import find_optimal_fishing_route_astar
from agents.pfz.model import predict as pfz_predict
from location.marine_identity import MarineIdentityResolver

class RecommendationAgent(BaseAgent):
    """
    Algorithmic Recommendation & Advisory Agent (Rule Engine)

    Synthesizes predictions from PFZ and Weather ML agents alongside Safety Rule checks
    and Geospatial Candidate Grid Search to formulate actionable, human-friendly maritime advisories.
    """
    @property
    def name(self) -> str:
        return "recommendation"

    def run(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        if plan.operation in ("COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "SELECT_BEST_FISHING_OPTION") and "comparison_meta" in context:
            return self._handle_comparison(plan, context)

        target_loc = plan.target_location or plan.location or plan.reference_location
        origin_loc = plan.reference_location or plan.location or plan.target_location
        if not target_loc and getattr(plan, "location_required", True) and plan.operation not in ("TEMPORAL_PFZ_GUIDANCE", "COMPARE_FISHING_REGIONS"):
            return AgentResult(
                agent=self.name,
                status="ERROR",
                location=None,
                timestamp=datetime.now(timezone.utc),
                data={
                    "decision": "NEEDS_CLARIFICATION",
                    "action_code": "NEEDS_CLARIFICATION",
                    "action_title": "LOCATION CLARIFICATION REQUIRED",
                    "recommendation_text": "Please specify a coastal location or port.",
                    "ranked_candidate_spots": []
                },
                confidence=1.0,
                sources=["ORCA_RECOMMENDATION_ENGINE_v1"],
                warnings=["LOCATION_REQUIRED: No valid spatial location provided in QueryPlan."]
            )

        lat = origin_loc.latitude if origin_loc else None
        lon = origin_loc.longitude if origin_loc else None

        pfz_res = context.get("pfz")
        weather_res = context.get("weather")
        safety_res = context.get("safety_rules")
        geospatial_res = context.get("geospatial")
        marine_safety_res = context.get("marine_safety")

        pfz_data = pfz_res.data if pfz_res else {}
        weather_data = weather_res.data if weather_res else {}
        safety_data = safety_res.data if safety_res else {}
        geospatial_data = geospatial_res.data if geospatial_res else {}
        marine_safety_data = marine_safety_res.data if marine_safety_res else {}

        ocean_res = context.get("ocean")
        ocean_data = ocean_res.data if ocean_res else {}

        pfz_prob = pfz_data.get("pfz_probability")
        pfz_present = pfz_data.get("pfz_present", False)
        pfz_qualified = pfz_data.get("pfz_qualified", pfz_present)
        pfz_signal_present = pfz_data.get("pfz_signal_present", pfz_present)
        safety_clearance = safety_data.get("safety_clearance")
        ml_weather_risk = weather_data.get("orca_xgboost", {}).get("risk_level", "NORMAL")

        warnings = safety_data.get("warnings", [])
        rule_violations = safety_data.get("rule_violations", [])
        safety_status = safety_data.get("safety_status", "SAFE")

        bsi_advisory = marine_safety_data.get("vessel_safety_advisory", "SAFE")
        imd_warnings = marine_safety_data.get("imd_warnings", [])
        bsi_assessment = marine_safety_data.get("bsi_assessment", {})
        catastrophic_active = marine_safety_data.get("catastrophic_active", False)
        catastrophic_status = marine_safety_data.get("catastrophic_status", "NONE")

        ranked_candidates = []
        rejected_candidates = []
        candidate_evals = []

        # Process Spatial Candidate Grid Search if Geospatial Agent generated candidate points
        candidate_points = geospatial_data.get("candidate_grid_points", [])
        spatial_intents = {"pfz_search", "marine_conditions", "marine_safety", "hazard_alert", "general_fishing_query", "fishing_search", "route_search", "marine_safety_forecast", "productivity_analysis", "fishing_zone_analysis"}

        # 1. Official PFZ Candidates
        official_candidates = pfz_data.get("candidates", [])
        if official_candidates:
            for cand in official_candidates:
                # `cand` is a PFZCandidate contract dict or object
                lat_c = getattr(cand, "latitude", cand.get("latitude")) if isinstance(cand, dict) else cand.latitude
                lon_c = getattr(cand, "longitude", cand.get("longitude")) if isinstance(cand, dict) else cand.longitude
                prob = getattr(cand, "confidence", cand.get("confidence", 0.0)) if isinstance(cand, dict) else getattr(cand, "confidence", 1.0)

                c_eval = {
                    "latitude": lat_c,
                    "longitude": lon_c,
                    "pfz_probability": prob,
                    "weather_risk": ml_weather_risk,
                    "safety_clearance": safety_clearance or "CLEARED"
                }
                candidate_evals.append(c_eval)

        # 2. Geospatial ML Grid points fallback
        elif candidate_points and (plan.intent in spatial_intents or not plan.intent):
            if pfz_prob is None:
                pass # No PFZ data available
            else:
                for pt in candidate_points:
                    try:
                        pt_pfz = pfz_predict(pt["latitude"], pt["longitude"], timestamp=None, env_data=ocean_data)
                        c_pfz = pt_pfz.get("pfz_probability", 0.0)
                    except Exception:
                        c_pfz = 0.0

                    c_eval = {
                        **pt,
                        "pfz_probability": c_pfz,
                        "weather_risk": ml_weather_risk,
                        "safety_clearance": safety_clearance or "CLEARED"
                    }
                    candidate_evals.append(c_eval)

        if candidate_evals:
            ranked_candidates = self._generate_ranked_candidates(
                lat, lon, candidate_evals, pfz_data, target_loc, ocean_data, timestamp=datetime.now(timezone.utc)
            )

        dist_summary = geospatial_data.get("distance_to_coast_summary", {})

        weather_source = weather_data.get("data_provenance", {}).get("primary_source", "OPEN_METEO")
        # Explicitly sourced evidence dictionary
        why_dict = {
            "primary_reason": "",
            "distance_km": dist_summary.get("distance_km", 0.0),
            "nearest_coast_name": dist_summary.get("nearest_coast_name", "Coastline"),
            "imd_weather": {
                "source": weather_source,
                "summary": "Active warnings: " + ", ".join(imd_warnings) if imd_warnings else "No active warnings"
            },
            "incois_bsi": {
                "source": "INCOIS_BSI_METHODOLOGY",
                "advisory": bsi_advisory,
                "active_hazards": bsi_assessment.get("hazards", [])
            },
            "orca_ml_supplement": {
                "source": "ORCA_XGBOOST",
                "weather_risk": ml_weather_risk,
                "ocean_suitability": ocean_data.get("orca_suitability", {}).get("ocean_suitability_score", 0.0)
            }
        }

        if pfz_prob is not None:
            if pfz_qualified:
                why_dict["fishing_reason"] = f"PFZ probability is {int(pfz_prob*100)}%"
            else:
                thresh_str = f"below configured threshold ({int(pfz_data.get('decision_threshold', 0.85)*100)}%)"
                if pfz_signal_present:
                    why_dict["fishing_reason"] = f"Model-estimated PFZ likelihood: {int(pfz_prob*100)}%, {thresh_str}. Not identified as a PFZ."
                else:
                    why_dict["fishing_reason"] = f"No PFZ signal detected. Model-estimated likelihood: {int(pfz_prob*100)}%, {thresh_str}."
        else:
            is_official = pfz_data.get("source") in ("INCOIS_LIVE", "INCOIS_OFFICIAL")
            if is_official and pfz_present:
                why_dict["fishing_reason"] = "Official INCOIS PFZ advisory present. Model-estimated probability: unavailable."
            else:
                why_dict["fishing_reason"] = "No PFZ data available."

        # Synthesis Matrix
        if plan.operation == "DISTANCE_TO_COAST":
            dist_km = dist_summary.get("distance_km", 0.0)
            n_coast = dist_summary.get("nearest_coast_name", "Coastline")
            loc_label = target_loc.name if target_loc and target_loc.name else "Target Hub"
            if dist_km == 0.0:
                action_code = "COASTAL_STATE_LOCATION"
                action_title = f"{loc_label} is a Coastal Region"
                recommendation_text = f"{loc_label} is a coastal region directly bordering the sea (0 km distance to coast)."
            else:
                action_code = "DISTANCE_TO_COAST_RESULT"
                action_title = f"NEAREST COASTLINE: {n_coast} ({dist_km:.1f} km)"
                recommendation_text = f"The nearest sea/coastline from {loc_label} is approximately {dist_km:.1f} km away at {n_coast}."

            why_dict["primary_reason"] = f"Geospatial distance calculation complete to {n_coast}."
            confidence = 0.99
            ranked_candidates = []
        elif plan.operation in ("SELECT_BEST_FISHING_OPTION", "NEAREST_PFZ_SEARCH", "FIND_FISHING_SPOTS"):
            if not candidate_evals:
                action_code = "UNAVAILABLE_PFZ"
                action_title = "PFZ DATA UNAVAILABLE"
                recommendation_text = "Cannot perform PFZ candidate search because PFZ data or coastal reachability is currently unavailable for this region."
                why_dict["primary_reason"] = "PFZ candidate search failed due to missing model data."
                confidence = 0.5
            else:
                action_code = "CANDIDATE_SEARCH_COMPLETE"
                action_title = "FISHING CANDIDATE SEARCH RESULTS"
                recommendation_text = f"Geospatial candidate search complete. Ranked {len(ranked_candidates)} safe reachability paths from origin."
                why_dict["primary_reason"] = "Candidate reachability calculated utilizing A* candidate ranker."
                confidence = 0.95
        elif plan.operation == "ROUTE_TO_FISHING_AREA" or plan.intent in ["route_search", "safe_route", "route"]:
            from agents.geospatial.astar import plan_safe_route_astar
            env_context = {"weather_data": weather_data, "cyclone_data": marine_safety_data}
            dest_lat = target_loc.latitude if target_loc else lat
            dest_lon = target_loc.longitude if target_loc else lon
            route_res = plan_safe_route_astar(lat, lon, dest_lat, dest_lon, env_context, resolution_km=10.0)
            
            # Determine language
            lang = getattr(plan, "language", "en")
            from enum import Enum
            if isinstance(lang, Enum):
                lang = lang.value
            
            if route_res["route_status"] == "SUCCESS":
                action_code = "ROUTE_GENERATED"
                action_title = "SAFE ROUTE GENERATED"
                dist = route_res['total_distance_km']
                eta = route_res['estimated_travel_time_h']
                
                if lang == "bn":
                    recommendation_text = f"সফলভাবে একটি নিরাপদ রুট তৈরি করা হয়েছে যা জমি এবং সীমাবদ্ধ অঞ্চল এড়িয়ে যায়। দূরত্ব: {dist} কিলোমিটার। পৌঁছানোর আনুমানিক সময় (ETA): {eta} ঘণ্টা।"
                elif lang in ["bn_en", "bn-Latn"]:
                    recommendation_text = f"Safolbhabe ekti nirapod route toiri kora hoyeche ja jomi o nishiddho onchol eriye chole. Durotto: {dist} km. ETA: {eta} ghonta."
                elif lang in ["hi", "hi-Latn", "hi_en"]:
                    recommendation_text = f"Safaltapurvak ek surakshit route banaya gaya hai jo zameen aur restricted zones se bachta hai. Doori: {dist} km. ETA: {eta} ghante."
                else:
                    recommendation_text = f"Successfully generated a safe route avoiding land and geofences. Distance: {dist} km. ETA: {eta} hours."
                
                why_dict["primary_reason"] = "A* Graph Traversal completed."
                why_dict["routing_metrics"] = route_res
                confidence = 0.98
            else:
                action_code = "NO_ROUTE_FOUND"
                action_title = "NO VALID ROUTE FOUND"
                if lang == "bn":
                    recommendation_text = "গন্তব্যে যাওয়ার কোনো বৈধ রুট পাওয়া যায়নি। এটি সম্ভবত জমি বা সীমাবদ্ধ অঞ্চল দ্বারা অবরুদ্ধ।"
                elif lang in ["bn_en", "bn-Latn"]:
                    recommendation_text = "Gontobye jaoar kono boidho route paoa jayni. Eti shombhoboto jomi ba restricted zone dara oboruddho."
                elif lang in ["hi", "hi-Latn", "hi_en"]:
                    recommendation_text = "Gantavya tak ka koi vaidh route nahi mila. Yeh shayad zameen ya restricted zones dwara block ho sakta hai."
                else:
                    recommendation_text = "Could not find a valid route to the destination. It may be blocked by land or restricted zones."
                    
                why_dict["primary_reason"] = "A* Graph Traversal failed to find a path."
                confidence = 0.90
        elif plan.operation == "TEMPORAL_PFZ_GUIDANCE":
            action_code = "TEMPORAL_PFZ_GUIDANCE"
            action_title = "TEMPORAL PFZ GUIDANCE"
            recommendation_text = (
                "For a fishing voyage planned for tomorrow, always follow the PFZ prediction and weather/marine-risk forecast "
                "valid for tomorrow, not today's PFZ. Check tomorrow's sea conditions before embarking."
            )
            why_dict["primary_reason"] = "Fishing operations must be guided by predictions valid for the targeted voyage date."
            confidence = 0.98
            ranked_candidates = []
        elif plan.operation == "COMPARE_FISHING_REGIONS":
            # Fallback if somehow comparison_meta is missing
            action_code = "COMPARE_FISHING_REGIONS"
            action_title = "MARINE REGIONS COMPARISON"
            recommendation_text = (
                "Evaluating regional marine conditions and historical productivity patterns across specified areas."
            )
            why_dict["primary_reason"] = "Regional comparison based on ocean productivity and seasonal weather risk."
            why_dict["fishing_reason"] = "Evaluating regional differences in fishing yield."
            why_dict["safety_reason"] = "Assessing differing hazard probabilities across zones."
            confidence = 0.8
            ranked_candidates = []
        elif safety_status == "UNSUPPORTED_LOCATION" or "OUT_OF_DOMAIN" in rule_violations:
            action_code = "UNSUPPORTED_LOCATION"
            action_title = "UNSUPPORTED LOCATION / OUT OF DOMAIN"
            recommendation_text = (
                f"Requested location ({lat:.2f}N, {lon:.2f}E) is outside the supported Indian Ocean operational domain (5°N–25°N, 65°E–95°E). "
                "Model predictions and safety clearances are unavailable for this area."
            )
            why_dict["primary_reason"] = "Location is outside the supported operational domain boundary."
            confidence = 0.99
        elif catastrophic_status == "ACTIVE" or catastrophic_active:
            action_code = "CANCEL_VOYAGE"
            action_title = "HARD SAFETY OVERRIDE: DO NOT EMBARK"
            recommendation_text = (
                "A catastrophic marine hazard (Earthquake / Tsunami / Cyclone) has been detected in this region. "
                "Fishing operations are strictly prohibited. Follow official emergency broadcasts."
            )
            why_dict["primary_reason"] = "Catastrophic hazard alert active in region."
            why_dict["safety_reason"] = "HARD SAFETY OVERRIDE triggered due to catastrophic event."
            confidence = 1.0
            ranked_candidates = [] # HARD SAFETY GATE
        elif plan.intent == "hazard_alert" and catastrophic_status == "UNKNOWN":
            action_code = "VERIFICATION_REQUIRED"
            action_title = "UNABLE TO VERIFY HAZARD CONDITIONS"
            recommendation_text = (
                "Live catastrophic hazard monitoring (Earthquake/Tsunami) is currently offline or unavailable. "
                "Cannot verify safety clearance. Exercise extreme caution and rely on local broadcasts before embarking."
            )
            why_dict["primary_reason"] = "Catastrophic hazard alert state cannot be established due to missing or failed data."
            confidence = 0.50
            ranked_candidates = [] # HARD SAFETY GATE
        elif plan.intent == "hazard_alert" and catastrophic_status == "NONE" and safety_clearance == "RESTRICTED":
            action_code = "CANCEL_VOYAGE"
            action_title = "NO CATASTROPHIC HAZARD, BUT WEATHER/OCEAN UNSAFE"
            recommendation_text = (
                "No live catastrophic hazard (Earthquake/Tsunami) has been detected, but severe weather or physical hazard alerts are active. "
                "Cancel planned fishing operations immediately and return to safe coastal harbor."
            )
            why_dict["primary_reason"] = "No catastrophic hazard, but active marine hazard / severe storm alert renders planned trip unsafe."
            why_dict["safety_reason"] = f"RESTRICTED safety clearance due to active hazards. See IMD/BSI reasoning."
            confidence = 0.95
            ranked_candidates = []
        elif safety_clearance == "RESTRICTED":
            action_code = "CANCEL_VOYAGE"
            action_title = "DO NOT EMBARK / RETURN TO HARBOR"
            recommendation_text = (
                "Severe weather or physical hazard alerts active. Cancel planned fishing operations immediately "
                "and return to safe coastal harbor."
            )
            why_dict["primary_reason"] = "Active marine hazard / severe storm alert renders planned trip unsafe."
            why_dict["safety_reason"] = f"RESTRICTED safety clearance due to active hazards. See IMD/BSI reasoning."
            confidence = 0.95
            ranked_candidates = [] # HARD SAFETY GATE: Do not recommend spots during RESTRICTED hazard!
        elif safety_clearance == "UNKNOWN":
            action_code = "VERIFICATION_REQUIRED"
            action_title = "UNABLE TO VERIFY SAFETY CONDITIONS"
            if pfz_qualified:
                recommendation_text = (
                    "An INCOIS PFZ advisory is available, but current safety status cannot be fully confirmed because hazard data is unavailable. "
                    "Do not rely on ORCA for a go/no-go decision; check official marine forecasts and safety advisories before departing."
                )
            else:
                recommendation_text = (
                    "Critical hazard data could not be verified. Do not rely on ORCA for a go/no-go decision; "
                    "check official marine forecasts and safety advisories before departing."
                )
            why_dict["primary_reason"] = "Safety state cannot be established due to missing or failed hazard data."
            confidence = 0.50
            ranked_candidates = [] # HARD SAFETY GATE: Do not recommend spots if safety cannot be verified!
        elif safety_clearance == "CAUTION" or bsi_advisory == "CAUTION" or ml_weather_risk == "CAUTION":
            if safety_status == "UNVERIFIED":
                action_code = "UNVERIFIED_SAFETY"
                action_title = "CYCLONE DATA UNAVAILABLE / PROCEED WITH CAUTION"
                recommendation_text = (
                    "Live cyclone monitoring is currently offline or unavailable. "
                    "Cannot verify full safety clearance. Exercise extreme caution and rely on local maritime radio broadcasts."
                )
                why_dict["primary_reason"] = "Missing real-time cyclone telemetry prevents full safety verification."
            elif pfz_qualified:
                action_code = "LIMITED_COASTAL_FISHING"
                action_title = "PROCEED WITH CAUTION (STAY NEAR SHORE)"
                recommendation_text = (
                    f"Potential Fishing Zone identified (prob: {pfz_prob:.2f}), but safety conditions require caution. "
                    "Limit operations to nearshore waters (<10 km) and maintain radio vigilance."
                )
                why_dict["primary_reason"] = "Promising fishing spot identified, but elevated weather/wave risk requires staying near shore."
            else:
                action_code = "EXERCISE_CAUTION"
                action_title = "MODERATE RISK / NO PFZ DETECTED"
                recommendation_text = (
                    "Weather conditions require caution. No significant fish aggregation detected at specified coordinates."
                )
                why_dict["primary_reason"] = "Elevated weather/wave risk and moderate fish aggregation require operational caution."
            confidence = 0.85
        else:
            if pfz_qualified:
                action_code = "OPTIMAL_FISHING_VOYAGE"
                action_title = "RECOMMENDED FISHING VOYAGE"
                recommendation_text = (
                    f"High Potential Fishing Zone identified (prob: {pfz_prob:.2f}) under safe weather conditions. "
                    "Favorable sea surface temperature and current patterns detected."
                )
                why_dict["primary_reason"] = "Optimal combination of high fish concentration and safe marine weather."
                confidence = 0.92
            else:
                action_code = "CLEAR_WEATHER_LOW_YIELD"
                action_title = "FAVORABLE WEATHER / LOW FISH CONCENTRATION"
                recommendation_text = (
                    "Weather and sea conditions are safe for navigation, but current thermal/current features do not indicate "
                    "dense fish school aggregation at this location."
                )
                why_dict["primary_reason"] = "Weather is safe for navigation, but fishing yield expectations are low at this specific hub."
                confidence = 0.88

        print(f"[DEBUG] plan.operation: {plan.operation}, action_code: {action_code}, safety_clearance: {safety_clearance}")
        
        payload = {
            "latitude": lat,
            "longitude": lon,
            "decision": action_code,
            "action_code": action_code,
            "action_title": action_title,
            "recommendation_text": recommendation_text,
            "distance_km": dist_summary.get("distance_km", 0.0) if (plan.operation == "DISTANCE_TO_COAST" or plan.intent in ("marine_geography", "nearest_coast")) else None,
            "nearest_coast_name": dist_summary.get("nearest_coast_name", "Coastline") if (plan.operation == "DISTANCE_TO_COAST" or plan.intent in ("marine_geography", "nearest_coast")) else None,
            "why": why_dict,
            "pfz_summary": {
                "pfz_present": pfz_present,
                "pfz_probability": pfz_prob
            },
            "weather_summary": {
                "risk_level": ml_weather_risk,
                "safety_clearance": safety_clearance
            },
            "ranked_candidate_spots": ranked_candidates,
            "rejected_candidate_spots": rejected_candidates,
            "active_warnings": warnings
        }

        loc_res_obj = GeoLocation(latitude=lat, longitude=lon, name=target_loc.name if target_loc else None) if (lat is not None and lon is not None) else None

        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=loc_res_obj,
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=confidence,
            sources=["ORCA_RECOMMENDATION_ENGINE_v1", "ASTAR_GEOSPATIAL_RANKER_v1"],
            warnings=warnings,
            audit=AgentAudit(
                inputs_used={"lat": lat, "lon": lon},
                outputs={"decision": payload["decision"], "ranked_candidate_spots_count": len(ranked_candidates)},
                output_reason="Synthesized fishing predictions and safety rules into a final advisory.",
                sources=["ORCA_RECOMMENDATION_ENGINE_v1", "ASTAR_GEOSPATIAL_RANKER_v1"],
                score_source="RULE_BASED_WEIGHTED",
                score_reason="Recommendation relies on composite rules across PFZ probability and geospatial safety checks."
            )
        )

    def _handle_comparison(self, plan: QueryPlan, context: Dict[str, AgentResult]) -> AgentResult:
        contexts = context["comparison_meta"].data.get("contexts", [])
        if len(contexts) < 2 or len(plan.compare_locations) < 2:
            return self._fallback_comparison()

        def risk_score(r):
            r_upper = r.upper() if isinstance(r, str) else "UNKNOWN"
            if r_upper in ["SAFE", "CLEARED"]: return 0
            if r_upper in ["CAUTION", "MODERATE"]: return 1
            if r_upper in ["RESTRICTED"]: return 2
            if r_upper in ["UNSAFE", "DANGER", "DANGEROUS", "SEVERE"]: return 3
            if r_upper in ["UNKNOWN / UNVERIFIABLE", "UNKNOWN"]: return 4
            return 3

        evaluations = []
        for i in range(min(len(plan.compare_locations), len(contexts))):
            loc = plan.compare_locations[i]
            ctx = contexts[i]

            pfz_res = ctx.get("pfz")
            prob = pfz_res.data.get("pfz_probability", 0.0) if pfz_res else 0.0

            risk_res = ctx.get("risk")
            safety_rules_res = ctx.get("safety_rules")
            
            if not risk_res or not safety_rules_res or risk_res.status != "success" or safety_rules_res.status != "success":
                risk = "UNKNOWN / UNVERIFIABLE"
            else:
                s_status = safety_rules_res.data.get("safety_status")
                s_clearance = safety_rules_res.data.get("safety_clearance")
                r_level = risk_res.data.get("risk_level")
                
                if s_status and s_status != "UNKNOWN":
                    risk = s_status
                elif s_clearance and s_clearance != "UNKNOWN":
                    risk = s_clearance
                elif r_level and r_level != "UNKNOWN":
                    risk = r_level
                else:
                    risk = "UNKNOWN / UNVERIFIABLE"

            evaluations.append({
                "location": loc,
                "name": loc.name,
                "prob": prob,
                "risk": risk,
                "score": risk_score(risk),
                "context": ctx
            })

        # Find the safest locations
        min_score = min(e["score"] for e in evaluations)
        safest_evals = [e for e in evaluations if e["score"] == min_score]

        # Among the safest, find the one with highest PFZ
        winner = max(safest_evals, key=lambda x: x["prob"])

        # Build comparison summary
        lines = [f"Based on analyzing {len(evaluations)} locations:"]
        for e in evaluations:
            lines.append(f"- **{e['name']}**: Safety Risk: {e['risk']} | PFZ Probability: {int((e['prob'] if e['prob'] is not None else 0.0)*100)}%")

        lang = getattr(plan, "language", "en")
        if isinstance(lang, Enum):
            lang = lang.value
            
        if winner["score"] >= 2:
            if lang == "bn":
                reason = f"তবে, {winner['name']} সহ কোনো স্থানই সম্পূর্ণ নিরাপদ নয় (ঝুঁকি: {winner['risk']})। দয়া করে সতর্কতা অবলম্বন করুন অথবা যাত্রা বাতিল করুন।"
                recommendation_line = f"\nসতর্কতা: কোনো স্থানই নিরাপদ নয়। মাছ ধরার জন্য যাওয়ার পরামর্শ দেওয়া হচ্ছে না।"
            elif lang in ["bn_en", "bn-Latn"]:
                reason = f"Tobe, {winner['name']} shoho kono jaygai ekhon safe noy (Risk: {winner['risk']}). Doya kore sabdhanota obolombo koroon ba jatra cancel korun."
                recommendation_line = f"\nWARNING: Kono location i safe noy. Fishing e jaoar poramorsho deoa hocche na."
            elif lang in ["hi", "hi-Latn", "hi_en"]:
                reason = f"Halaanki, {winner['name']} sahit koi bhi sthan abhi surakshit nahi hai (Risk: {winner['risk']}). Kripya savdhani bartein ya yatra radd karein."
                recommendation_line = f"\nWARNING: Koi bhi sthan surakshit nahi hai. Machhli pakadne ke liye jane ki salah nahi di jati."
            else:
                reason = f"However, neither option is fully safe (Risk: {winner['risk']}). Please exercise extreme caution or cancel the voyage."
                recommendation_line = f"\nWARNING: No safe options available. It is not recommended to embark on a fishing voyage."
            lines.append(recommendation_line + " " + reason)
        else:
            if lang == "bn":
                reason = f"{winner['name']} সবচেয়ে ভালো অপশন, এখানে {winner['risk']} নিরাপত্তা এবং {int((winner['prob'] if winner['prob'] is not None else 0.0)*100)}% PFZ সম্ভাবনা রয়েছে।"
            elif lang in ["bn_en", "bn-Latn"]:
                reason = f"{winner['name']} shobtheke bhalo option, ekhane {winner['risk']} safety condition aar {int((winner['prob'] if winner['prob'] is not None else 0.0)*100)}% PFZ probability ache."
            elif lang in ["hi", "hi-Latn", "hi_en"]:
                reason = f"{winner['name']} sabse behtar option hai, yahan {winner['risk']} safety condition aur {int((winner['prob'] if winner['prob'] is not None else 0.0)*100)}% PFZ probability hai."
            else:
                reason = f"{winner['name']} is the best option with {winner['risk']} safety conditions and a {int((winner['prob'] if winner['prob'] is not None else 0.0)*100)}% PFZ probability."
            lines.append(f"\nRecommendation: Go to **{winner['name']}**. {reason}")

        recommendation_text = "\n".join(lines)

        pfz_comp = ", ".join([f"{e['name']} ({e['prob']:.2f})" for e in evaluations])
        safe_comp = ", ".join([f"{e['name']} ({e['risk']})" for e in evaluations])

        win_ctx = winner.get("context", {})
        ranked_candidates = []

        if win_ctx:
            w_pfz_data = win_ctx.get("pfz").data if win_ctx.get("pfz") else {}
            w_ocean_data = win_ctx.get("ocean").data if win_ctx.get("ocean") else {}
            w_weather_data = win_ctx.get("weather").data if win_ctx.get("weather") else {}

            # Combine ocean and weather data for accurate PFZ prediction (matching pfz_agent.py)
            combined_env_data = {}
            combined_env_data.update(w_ocean_data)
            combined_env_data.update(w_weather_data)

            w_geo_data = win_ctx.get("geospatial").data if win_ctx.get("geospatial") else {}
            w_ml_weather = win_ctx.get("weather").data.get("orca_xgboost", {}).get("risk_level", "NORMAL") if win_ctx.get("weather") else "NORMAL"
            w_safety = win_ctx.get("safety_rules").data.get("safety_clearance", "CLEARED") if win_ctx.get("safety_rules") else "CLEARED"

            w_candidate_points = w_geo_data.get("candidate_grid_points", [])
            w_candidate_evals = []

            if w_pfz_data and "candidates" in w_pfz_data:
                for c in w_pfz_data["candidates"]:
                    c_lat = getattr(c, "latitude", None) if not isinstance(c, dict) else c.get("latitude")
                    c_lon = getattr(c, "longitude", None) if not isinstance(c, dict) else c.get("longitude")
                    c_conf = getattr(c, "confidence", None) if not isinstance(c, dict) else c.get("confidence")
                    c_src = getattr(c, "source", "INCOIS_OFFICIAL") if not isinstance(c, dict) else c.get("source", "INCOIS_OFFICIAL")
                    c_id = getattr(c, "id", f"{c_lat}_{c_lon}") if not isinstance(c, dict) else c.get("id", f"{c_lat}_{c_lon}")
                    w_candidate_evals.append({
                        "id": c_id,
                        "latitude": c_lat,
                        "longitude": c_lon,
                        "pfz_probability": c_conf,
                        "weather_risk": w_ml_weather,
                        "safety_clearance": w_safety,
                        "source": c_src
                    })

            if w_candidate_points:
                for pt in w_candidate_points:
                    try:
                        pt_pfz = pfz_predict(pt["latitude"], pt["longitude"], timestamp=None, env_data=combined_env_data)
                        c_pfz = pt_pfz.get("pfz_probability", 0.0)
                    except Exception:
                        c_pfz = 0.0

                    c_eval = {
                        **pt,
                        "pfz_probability": c_pfz,
                        "weather_risk": w_ml_weather,
                        "safety_clearance": w_safety
                    }
                    w_candidate_evals.append(c_eval)

            if w_candidate_evals:
                ranked_candidates = self._generate_ranked_candidates(
                    winner["location"].latitude, winner["location"].longitude,
                    w_candidate_evals, w_pfz_data, winner["location"],
                    w_ocean_data, timestamp=datetime.now(timezone.utc)
                )

        payload = {
            "latitude": winner["location"].latitude,
            "longitude": winner["location"].longitude,
            "decision": "COMPARE_FISHING_REGIONS",
            "action_code": "COMPARE_FISHING_REGIONS",
            "action_title": f"COMPARISON: {winner['name'].upper()} RECOMMENDED",
            "recommendation_text": recommendation_text,
            "why": {
                "primary_reason": reason,
                "fishing_reason": f"Compared PFZ: {pfz_comp}",
                "safety_reason": f"Compared Safety: {safe_comp}"
            },
            "ranked_candidate_spots": ranked_candidates,
            "rejected_candidate_spots": [],
            "active_warnings": []
        }

        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=None,
            timestamp=datetime.now(timezone.utc),
            data=payload,
            confidence=0.9,
            sources=["ORCA_RECOMMENDATION_ENGINE_v1"],
            warnings=[],
            audit=AgentAudit(
                inputs_used={"contexts": len(contexts)},
                outputs={"decision": payload["decision"]},
                output_reason="Computed regional comparison of fishing regions.",
                sources=["ORCA_RECOMMENDATION_ENGINE_v1"],
                score_source="RULE_BASED_WEIGHTED",
                score_reason="Recommendation scores based on comparative regional marine contexts."
            )
        )

    def _generate_ranked_candidates(self, lat: float, lon: float, candidate_evals: list, pfz_data: dict, target_loc, ocean_data: dict, timestamp) -> list:
        ranked_candidates, rejected_candidates = rank_fishing_candidates(lat, lon, candidate_evals, top_k=3)
        formatted_candidates = []
        if ranked_candidates:
            # Run A* routing from origin to the valid candidates
            ranked_route = find_optimal_fishing_route_astar(lat, lon, ranked_candidates)

            for spot in ranked_route:
                # Ensure spot is resolved to a human-readable identity
                spot_lat = spot["latitude"]
                spot_lon = spot["longitude"]

                identity = MarineIdentityResolver.resolve(
                    lat=spot_lat,
                    lon=spot_lon,
                    official_landmark=None, # Incois PFZ might provide this in future
                    region=target_loc.name if target_loc else "Marine Region"
                )

                cand_pfz_prob = spot.get("pfz_probability")
                cand_source = spot.get("source") or pfz_data.get("source", "ORCA_PFZ_MODEL")
                is_official = cand_source in ("INCOIS_LIVE", "INCOIS_OFFICIAL")

                if cand_pfz_prob is not None:
                    cand_pfz_present = bool(cand_pfz_prob >= pfz_data.get("decision_threshold", 0.85))
                    if cand_pfz_present:
                        pfz_desc = f"Model-estimated PFZ likelihood: {int(cand_pfz_prob*100)}%"
                    else:
                        pfz_desc = f"Model-estimated PFZ likelihood: {int(cand_pfz_prob*100)}%, below threshold ({int(pfz_data.get('decision_threshold', 0.85)*100)}%)"
                else:
                    cand_pfz_present = True if is_official else False
                    if is_official:
                        pfz_desc = "INCOIS advisory present\nModel-estimated probability: unavailable"
                    else:
                        pfz_desc = "No PFZ data available"

                formatted_cand = MarineCandidateIdentity(
                    candidate_id=str(spot.get("id", f"{spot_lat}_{spot_lon}")),
                    latitude=spot_lat,
                    longitude=spot_lon,
                    display_name=identity.display_name,
                    region=identity.region,
                    reference_landmark=identity.reference_landmark,
                    distance_from_landmark_km=identity.distance_from_landmark_km,
                    bearing_from_landmark=identity.compass_direction,
                    depth_m=spot.get("incois_depth_m_range", str(spot.get("depth_m"))) if spot.get("incois_depth_m_range") or spot.get("depth_m") else None,
                    query_distance_km=spot.get("distance_km", 0.0),
                    pfz_probability=cand_pfz_prob,
                    pfz_present=cand_pfz_present,
                    pfz_signal_present=bool(cand_pfz_prob > 0) if cand_pfz_prob is not None else cand_pfz_present,
                    pfz_qualified=cand_pfz_present,
                    pfz_source=cand_source,
                    pfz_threshold=pfz_data.get("decision_threshold", 0.85),
                    pfz_description=pfz_desc,
                    safety_status=spot.get("safety_clearance", "UNKNOWN"),
                    weather_status=spot.get("weather_risk", "UNKNOWN"),
                    ocean_status=spot.get("ocean_status", "SUCCESS" if ocean_data else "UNKNOWN"),
                    reachability_km=round(spot.get("distance_km", 0.0), 1),
                    composite_score=spot.get("composite_score", 0.0),
                    source=cand_source,
                    data_quality="MODEL_ESTIMATE",
                    eligible=spot.get("eligible", True),
                    rejection_reason=spot.get("rejection_reason"),
                    selection_reason=spot.get("selection_reason", ""),
                    rank=spot.get("rank")
                )
                formatted_candidates.append(formatted_cand.model_dump())
        return formatted_candidates

    def _fallback_comparison(self) -> AgentResult:
        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            location=None,
            timestamp=datetime.now(timezone.utc),
            data={
                "decision": "COMPARE_FISHING_REGIONS",
                "action_code": "COMPARE_FISHING_REGIONS",
                "action_title": "MARINE REGIONS COMPARISON",
                "recommendation_text": "Evaluating regional marine conditions and historical productivity patterns.",
                "why": {"primary_reason": "Regional comparison based on ocean productivity and seasonal weather risk."},
                "ranked_candidate_spots": [],
                "rejected_candidate_spots": [],
                "active_warnings": []
            },
            confidence=0.8,
            sources=["ORCA_RECOMMENDATION_ENGINE_v1"],
            warnings=[],
            audit=AgentAudit(
                inputs_used={},
                outputs={"decision": "COMPARE_FISHING_REGIONS"},
                output_reason="Generated fallback regional marine comparison.",
                sources=["ORCA_RECOMMENDATION_ENGINE_v1"],
                score_source="DEFAULT_FALLBACK",
                score_reason="Comparison defaulted due to insufficient regions."
            )
        )
