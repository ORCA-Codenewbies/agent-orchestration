import re
import time
import logging
from schemas.extraction import ExtractionResult
from schemas.contracts import QueryPlan, QueryTime
from conversation.model import ConversationModel
from conversation.prompts import EXTRACTION_SYSTEM_PROMPT
from conversation.state import ConversationState, _clean
from conversation.normalize import normalize_extraction

logger = logging.getLogger(__name__)

# Orchestrator-owned mapping. The LLM never names an agent directly —
# it only proposes an intent, and this dict is the sole authority on
# which agents that intent triggers. This is the actual safety boundary.
INTENT_AGENTS = {
  "marine_safety": ["risk", "pfz", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "pfz_search": ["risk", "pfz", "weather", "ocean", "geospatial", "productivity", "safety_rules", "recommendation"],
  "marine_conditions": ["risk", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "hazard_alert": ["risk", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "nearest_coast": ["geospatial", "recommendation"],
  "marine_geography": ["geospatial", "recommendation"],
  "nearest_pfz": ["risk", "pfz", "weather", "ocean", "geospatial", "productivity", "safety_rules", "recommendation"],
  "marine_safety_forecast": ["risk", "pfz", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "fishing_zone_analysis": ["risk", "pfz", "weather", "ocean", "geospatial", "productivity", "safety_rules", "recommendation"],
  "safe_route": ["risk", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "productivity_analysis": ["risk", "pfz", "weather", "ocean", "geospatial", "productivity", "safety_rules", "recommendation"],
  "hazardous_zone_filter": ["risk", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "unknown": [],
}

OPERATION_AGENTS = {
  "DISTANCE_TO_COAST": ["geospatial", "recommendation"],
  "FIND_FISHING_SPOTS": ["risk", "pfz", "weather", "ocean", "geospatial", "productivity", "safety_rules", "recommendation"],
  "SAFE_ALTERNATIVE_ZONE": ["risk", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "TEMPORAL_PFZ_GUIDANCE": ["recommendation"],
  "COMPARE_FISHING_REGIONS": ["risk", "pfz", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "COMPARE_LOCATIONS": ["risk", "pfz", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "SELECT_BEST_FISHING_OPTION": ["risk", "pfz", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "NEAREST_PFZ_SEARCH": ["risk", "pfz", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
  "FISHING_SAFETY_TRADEOFF": ["risk", "pfz", "weather", "ocean", "safety_rules", "recommendation"],
  "ROUTE_TO_FISHING_AREA": ["risk", "weather", "ocean", "geospatial", "safety_rules", "recommendation"],
}

INTENT_RESULT_TYPES = {
  "marine_safety": "SAFETY_ASSESSMENT",
  "pfz_search": "PFZ_RESULT",
  "marine_conditions": "CONDITIONS_RESULT",
  "hazard_alert": "HAZARD_RESULT",
  "nearest_coast": "NEAREST_COAST_RESULT",
  "marine_geography": "GEOGRAPHY_RESULT",
  "nearest_pfz": "PFZ_RESULT",
  "marine_safety_forecast": "SAFETY_ASSESSMENT",
  "fishing_zone_analysis": "COMPARE_RESULT",
  "safe_route": "ROUTE_RESULT",
  "productivity_analysis": "PRODUCTIVITY_RESULT",
  "hazardous_zone_filter": "SAFETY_ASSESSMENT",
  "unknown": "SAFETY_ASSESSMENT",
}

from schemas.contracts import SpatialConstraint, GeoLocation
from location.resolver import normalize_direction, calculate_destination_point

def detect_special_operations(raw_query: str, extraction=None) -> str | None:
    q_low = raw_query.lower()

    # 1. Temporal PFZ Guidance check
    temporal_phrases = ["ajker pfz", "kaler pfz", "today pfz", "tomorrow pfz", "ajke...kal", "today or tomorrow", "ajker naki kaler", "naki kaler"]
    if any(p in q_low for p in temporal_phrases):
        return "TEMPORAL_PFZ_GUIDANCE"

    # 2. Conceptual Tradeoff Check
    has_pfz = "pfz" in q_low
    has_weather_safety = any(w in q_low for w in ["weather", "safe", "unsafe", "kharap"])
    has_but = any(w in q_low for w in ["but", "kintu", "lekin", "ar", "aar"])
    if has_but and has_pfz and has_weather_safety:
        return "FISHING_SAFETY_TRADEOFF"

    # 3. Action-Type Override
    if extraction and hasattr(extraction, "action_type"):
        act_val = getattr(extraction.action_type, "value", extraction.action_type)
        if str(act_val).upper() == "COMPARE":
            return "COMPARE_FISHING_REGIONS"

    # 3. Candidate Search vs Region Comparison check
    regions_found = [r for r in ["bay of bengal", "arabian sea", "indian ocean"] if r in q_low]

    has_origin_kw = any(re.search(rf"\b{re.escape(kw)}\b", q_low) for kw in ["mp", "madhya pradesh", "kolkata", "digha", "kerala", "goa", "mumbai", "theke", "from", "dariye achi"])

    if len(regions_found) >= 1 and has_origin_kw:
        return "SELECT_BEST_FISHING_OPTION"

    route_kws = [
        "safest route", "safe route", "route for a vessel", "navigate", "safe navigation", "route",
        "raasta", "marg", "रास्ता", "मार्ग", "रूट", "सुरक्षित रास्ता", # Hindi/Hinglish
        "rasta", "poth", "রাস্তা", "পথ", "নিরাপদ রুট" # Bengali/Benglish
    ]
    if any(kw in q_low for kw in route_kws):
        return "ROUTE_TO_FISHING_AREA"

    if "nearest" in q_low and "pfz" in q_low:
        return "NEAREST_PFZ_SEARCH"

    locs_count = len(extraction.locations) if extraction and hasattr(extraction, "locations") else 0
    compare_kws = ["choose", "konta", "better", "best", "kon ", "vs", "versus", "option", "kothay"]
    has_comp_kw = any(kw in q_low for kw in compare_kws)

    if len(regions_found) >= 2 or (has_comp_kw and len(regions_found) >= 1) or (locs_count >= 2 and has_comp_kw):
        return "COMPARE_FISHING_REGIONS"

    search_spot_kws = ["which area", "which spot", "which location", "where to fish", "choose area", "best area", "fishing spot", "which zone", "best spot", "where should", "which fishing area"]
    if any(kw in q_low for kw in search_spot_kws) or (has_comp_kw and locs_count <= 1):
        return "FIND_FISHING_SPOTS"

    has_env_kw = any(kw in q_low for kw in ["chlorophyll", "sst", "temperature", "tapmatra", "tapman"])
    print(f"DEBUG: q_low={q_low}, has_env_kw={has_env_kw}")
    if has_env_kw and not "pfz" in q_low:
        print("DEBUG: returning PRODUCTIVITY_ANALYSIS")
        return "PRODUCTIVITY_ANALYSIS"

    weather_only_kws = ["weather", "mausam", "aabohawa", "abohawa", "temperature", "rain", "wind", "cloud"]
    ocean_kws = ["tide", "wave", "joar", "bhata", "jowar", "dhau", "sea", "ocean", "samundar", "somudro", "current", "sst", "surface"]
    if any(kw in q_low for kw in weather_only_kws) and not any(kw in q_low for kw in ocean_kws):
        if extraction.intent.value in ("marine_conditions", "unknown"):
            return "ASSESS_WEATHER"

    return None

def parse_relative_spatial_constraint(raw_query: str, ref_loc: GeoLocation | None, extraction: ExtractionResult | None = None) -> tuple[SpatialConstraint | None, GeoLocation | None]:
    q_low = raw_query.lower()
    dist_km = None
    raw_dir = None

    if extraction and extraction.spatial_distance_km:
        dist_km = float(extraction.spatial_distance_km)
    else:
        dist_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometre)s?\b", q_low)
        if dist_match:
            dist_km = float(dist_match.group(1))

    if extraction and extraction.spatial_direction:
        raw_dir = extraction.spatial_direction
    else:
        dir_match = re.search(r"\b(north|south|east|west|northeast|northwest|southeast|southwest|purbo|poshchim|uttor|dokkhin|dokkhhin|ne|se|nw|sw)\b", q_low)
        if dir_match:
            raw_dir = dir_match.group(1)

    if dist_km and ref_loc:
        if raw_dir:
            brg_deg, dir_label = normalize_direction(raw_dir)
            if brg_deg is not None and dir_label is not None:
                spatial_c = SpatialConstraint(
                    distance_km=dist_km,
                    direction=dir_label,
                    bearing_deg=brg_deg,
                    radius_km=30.0
                )
                tgt_lat, tgt_lon = calculate_destination_point(ref_loc.latitude, ref_loc.longitude, dist_km, brg_deg)
                target_loc = GeoLocation(
                    latitude=tgt_lat,
                    longitude=tgt_lon,
                )
                return spatial_c, target_loc

        spatial_c = SpatialConstraint(
            distance_km=dist_km,
            radius_km=30.0
        )
        return spatial_c, None

    return None, None

def sanitize_temporal(raw_query: str, extraction_relative: str | None, extraction_offset: int | None = None) -> tuple[str | None, int | None]:
  q_lower = raw_query.lower()
  match = re.search(r"\b([2-9]|\d{2,})\s*(?:din|days?)\b", q_lower)
  if match:
    return "custom", int(match.group(1))

  if extraction_relative in ("today", "tomorrow", "day_after_tomorrow"):
    return extraction_relative, None

  if extraction_relative == "custom" and extraction_offset:
    return "custom", extraction_offset

  return None, None


def extract_time_details(raw_query: str, extraction_relative: str | None = None, extraction_offset: int | None = None, extraction_period: str | None = None) -> tuple[str | None, int | None, int | None, int | None, str | None]:
  q_lower = raw_query.lower()
  offset_days = None
  if any(k in q_lower for k in ["kal", "tomorrow"]):
    rel_time = "tomorrow"
  elif any(k in q_lower for k in ["aaj", "ajke", "today"]):
    rel_time = "today"
  else:
    rel_time, offset_days = sanitize_temporal(raw_query, extraction_relative, extraction_offset)

  hour = None
  minute = None
  period = None

  if extraction_period and extraction_period != "none":
      period = extraction_period

  if not period:
    if any(p in q_lower for p in ["shokal", "subah", "morning", "am"]):
      period = "morning"
    elif any(p in q_lower for p in ["dupur", "dopahar", "noon", "afternoon"]):
      period = "afternoon"
    elif any(p in q_lower for p in ["bikel", "sandhya", "evening", "pm"]):
      period = "evening"
    elif any(p in q_lower for p in ["raat", "night"]):
      period = "night"

  match_colon = re.search(r"\b(\d{1,2}):(\d{2})\b", q_lower)
  match_specific = re.search(r"\b(\d{1,2})\s*(?:tay|baje|ter|am|pm|o'clock)\b", q_lower)

  if match_colon:
    hour = int(match_colon.group(1))
    minute = int(match_colon.group(2))
  elif match_specific:
    hour = int(match_specific.group(1))
    minute = 0
    if (period in ["evening", "afternoon"] or "pm" in q_lower or "bikel" in q_lower) and hour < 12:
      hour += 12

  return rel_time, offset_days, hour, minute, period


from conversation.language import detect_language

def build_query_plan(raw_query: str, extraction: ExtractionResult, extract_location_fn) -> QueryPlan:
  loc = None
  loc_type = "unknown"
  inland_name = None
  ref_loc = None
  tgt_loc = None

  target_text = None
  ref_text = None
  compare_locations = []

  locs_list = getattr(extraction, "locations", [])
  if not locs_list and getattr(extraction, "location_text", None):
      from schemas.extraction import LocationRole, LocationItem
      role_str = getattr(extraction, "location_role", None)
      locs_list = [LocationItem(text=extraction.location_text, role=LocationRole(role_str) if role_str else LocationRole.TARGET)]

  for loc_item in locs_list:
      loc_item_role = loc_item.get("role") if isinstance(loc_item, dict) else getattr(loc_item, "role", None)
      role = str(getattr(loc_item_role, "value", loc_item_role)).upper()
      loc_item_text = loc_item.get("text") if isinstance(loc_item, dict) else getattr(loc_item, "text", None)

      if role == "REFERENCE":
          if not ref_text:
              ref_text = loc_item_text
      else:
          if not target_text:
              target_text = loc_item_text

      if loc_item_text:
          res_comp = extract_location_fn(loc_item_text.lower())
          if hasattr(res_comp, "geo_location") and res_comp.geo_location:
              # Skip inland locations if this is a comparison
              if res_comp.location_type != "inland" or role == "REFERENCE":
                  compare_locations.append(res_comp.geo_location)
              elif role != "REFERENCE":
                  # If it's inland and not explicitly a reference, it might just be the user's origin
                  if not ref_text:
                      ref_text = loc_item_text
                      ref_loc = res_comp.geo_location

  res = None

  if target_text:
      res = extract_location_fn(target_text.lower())
      if hasattr(res, "geo_location"):
          loc = res.geo_location
          tgt_loc = loc
          loc_type = res.location_type
          inland_name = res.inland_name
          
          # If inland location extracted for coastal operations, reject it.
          if loc_type == "inland" and extraction.intent.value != "safe_route":
              # We clear it so it gets picked up as MISSING_LOCATION and triggers a clarification
              loc = None
              tgt_loc = None

  if ref_text:
      res_ref = extract_location_fn(ref_text.lower())
      if hasattr(res_ref, "geo_location") and res_ref.geo_location:
          ref_loc = res_ref.geo_location
          if not loc:
              loc = ref_loc
              loc_type = res_ref.location_type
              inland_name = res_ref.inland_name
              res = res_ref
              
          if loc_type == "inland" and extraction.intent.value != "safe_route":
              loc = None
              ref_loc = None

  candidate_locs = getattr(res, "candidate_names", [])
  ext_period = getattr(extraction, "time_period", None)
  rel_time, offset_days, q_hour, q_min, q_period = extract_time_details(raw_query, extraction.time_relative, extraction.time_offset_days, ext_period)
  qwen_lang = extraction.language.value if getattr(extraction, "language", None) else None
  det_lang = detect_language(raw_query)
  # Prefer deterministic detector when Qwen returns the default 'en' but the query is clearly non-English
  trusted_lang = det_lang if (not qwen_lang or qwen_lang == "en") and det_lang != "en" else (qwen_lang or det_lang)
  res_type = INTENT_RESULT_TYPES.get(extraction.intent.value, "SAFETY_ASSESSMENT")

  from schemas.extraction import LocationRole, ActionType, LocationItem
  locs = getattr(extraction, "locations", [])
  loc_role = locs[0].role if locs and hasattr(locs[0], "role") else LocationRole.TARGET
  act_type = getattr(extraction, "action_type", ActionType.ASSESS)
  if act_type is None:
    act_type = ActionType.ASSESS
  user_const = getattr(extraction, "user_constraint", None)

  ext_count = getattr(extraction, "count", 1)
  if ext_count <= 1:
      c_match = re.search(r"\b(?:top|best|show|find)?\s*(\d+)\s*(?:fishing\s+)?(?:spots?|places?|locations?|options?)\b", raw_query.lower())
      if c_match:
          ext_count = int(c_match.group(1))
      else:
          c_match2 = re.search(r"\b(?:top|best)\s+(\d+)\b", raw_query.lower())
          if c_match2:
              ext_count = int(c_match2.group(1))

  locations_list = getattr(extraction, "locations", [])
  current_intent = extraction.intent.value
  
  if current_intent == "safe_route":
      op = "ROUTE_TO_FISHING_AREA"
  elif current_intent == "nearest_pfz":
      op = "NEAREST_PFZ_SEARCH"
  else:
      op = detect_special_operations(raw_query, extraction)
      if not op:
        if current_intent in ("marine_geography", "nearest_coast"):
          op = "DISTANCE_TO_COAST"
        elif current_intent in ("nearest_pfz", "pfz_search"):
          op = "NEAREST_PFZ_SEARCH" if current_intent == "nearest_pfz" else "FIND_FISHING_SPOTS"
        elif current_intent == "fishing_zone_analysis":
          if len(compare_locations) >= 2 or (target_text and " vs " in raw_query.lower()):
              op = "COMPARE_FISHING_REGIONS"
          else:
              op = "FIND_FISHING_SPOTS"
        elif current_intent == "productivity_analysis":
          op = "FIND_FISHING_SPOTS"
        elif current_intent == "hazardous_zone_filter":
          op = "SAFE_ALTERNATIVE_ZONE"
        elif current_intent == "hazard_alert":
          op = "ASSESS_HAZARD"
        else:
          op = "ASSESS"

  loc_req = True
  if op in ("TEMPORAL_PFZ_GUIDANCE", "COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "FISHING_SAFETY_TRADEOFF", "GENERAL_MARINE_QUERY"):
    loc_req = False

  spatial_c, derived_target_loc = parse_relative_spatial_constraint(raw_query, ref_loc or loc, extraction)
  if derived_target_loc:
    tgt_loc = derived_target_loc
    loc = derived_target_loc
    if not op or op == "DISTANCE_TO_COAST":
      op = "FIND_FISHING_SPOTS"

  # Determine Candidate Regions from extractions
  candidate_regions = [r.title() for r in ["bay of bengal", "arabian sea", "indian ocean"] if r in raw_query.lower()]
  if candidate_regions and not any(l.role == LocationRole.REGION for l in locations_list):
      for r in candidate_regions:
          locations_list.append(LocationItem(text=r, role=LocationRole.REGION))
          res_comp = extract_location_fn(r.lower())
          if hasattr(res_comp, "geo_location") and res_comp.geo_location:
              compare_locations.append(res_comp.geo_location)

  # Region Invariant: If operation is COMPARE_FISHING_REGIONS, force plan.location = None
  if op in ("COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "SELECT_BEST_FISHING_OPTION"):
    loc = None
    loc_role = LocationRole.REGION
    tgt_loc = None

  if op in ("SELECT_BEST_FISHING_OPTION", "NEAREST_PFZ_SEARCH", "ROUTE_TO_FISHING_AREA"):
      if not ref_loc:
          # Try extracting reference from known keywords
          for origin_kw in ["mp", "madhya pradesh", "kolkata", "digha", "kerala", "goa", "mumbai"]:
              if re.search(rf"\b{re.escape(origin_kw)}\b", raw_query.lower()):
                  res_origin = extract_location_fn(origin_kw)
                  if getattr(res_origin, "geo_location", None):
                      ref_loc = res_origin.geo_location
                      loc = ref_loc # Base operations from origin
                      break

  semantic_target_obj = None
  if target_text and not tgt_loc:
      from conversation.semantic_target import parse_semantic_target
      semantic_target_obj = parse_semantic_target(target_text, spatial_c)

  return QueryPlan(
      query=raw_query,
      intent=extraction.intent.value,
      operation=op,
      action_type=act_type,
      result_type=res_type,
      language=trusted_lang,
      location=loc,
      reference_location=ref_loc,
      target_location=tgt_loc,
      semantic_target=semantic_target_obj,
      spatial_constraint=spatial_c,
      location_required=loc_req,
      location_type=loc_type,
      location_role=loc_role,
      locations=locations_list,
      inland_name=inland_name,
      candidate_locations=candidate_locs,
      time=QueryTime(
        relative=rel_time,
        offset_days=offset_days,
        hour=q_hour,
        minute=q_min,
        period=q_period or (None if extraction.time_period == "none" else extraction.time_period),
      ),
      activity=None if extraction.activity == "none" else extraction.activity,
      user_constraint=user_const,
      agents=OPERATION_AGENTS.get(op, INTENT_AGENTS.get(extraction.intent.value, [])),
      dependencies=[],
      constraints=[{"type": "max_risk", "value": "MEDIUM"}],
      count=ext_count,
      compare_locations=compare_locations,
      explanation_target=getattr(extraction, "explanation_target", None)
    )


def resolve_action(result: ExtractionResult, plan: QueryPlan) -> str:
  from schemas.extraction import Intent, Action, ActionType

  MARINE_OPS = ["SELECT_BEST_FISHING_OPTION", "COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "DISTANCE_TO_COAST", "FIND_FISHING_SPOTS", "NEAREST_PFZ_SEARCH", "ROUTE_TO_FISHING_AREA", "FISHING_SAFETY_TRADEOFF", "TEMPORAL_PFZ_GUIDANCE", "SAFE_ALTERNATIVE_ZONE"]
  if plan.operation in MARINE_OPS and plan.intent == "unknown":
      if "FISHING" in plan.operation or "PFZ" in plan.operation:
          plan.intent = "pfz_search"
      else:
          plan.intent = "marine_safety"
      result.intent = Intent(plan.intent)

  if plan.operation in ("COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "SELECT_BEST_FISHING_OPTION") and len(plan.compare_locations) < 2:
      plan.clarification_reason = "MISSING_COMPARISON_CANDIDATES"
      plan.readiness_status = "NEED_COMPARISON_CANDIDATES"
      return "CLARIFY"

  if plan.operation in ("FISHING_SAFETY_TRADEOFF", "GENERAL_MARINE_QUERY"):
      if plan.location or plan.reference_location:
          plan.readiness_status = "CAN_EXECUTE"
      else:
          plan.readiness_status = "CONCEPTUAL_QUERY"
      return "ORCA_QUERY"

  if plan.operation in ("TEMPORAL_PFZ_GUIDANCE", "COMPARE_FISHING_REGIONS", "COMPARE_LOCATIONS", "DISTANCE_TO_COAST"):
    plan.readiness_status = "CAN_EXECUTE"
    return "ORCA_QUERY"

  if plan.operation == "ROUTE_TO_FISHING_AREA":
      if not plan.location and not plan.reference_location:
          plan.clarification_reason = "MISSING_ORIGIN"
          plan.readiness_status = "NEED_ORIGIN"
          return "CLARIFY"
          
      if not plan.target_location and not plan.semantic_target and not plan.spatial_constraint:
          plan.clarification_reason = "MISSING_DESTINATION"
          plan.readiness_status = "NEED_DESTINATION"
          return "CLARIFY"
          
      plan.readiness_status = "CAN_EXECUTE"
      return "ORCA_QUERY"

  if plan.spatial_constraint and (plan.target_location or plan.reference_location):
    plan.readiness_status = "CAN_EXECUTE"
    return "ORCA_QUERY"
  if result.intent in (Intent.marine_geography, Intent.nearest_coast) or result.action_type == ActionType.EXPLAIN:
    plan.readiness_status = "CAN_EXECUTE"
    return "ORCA_QUERY"

  if result.intent != Intent.unknown:
    if plan.location is not None or plan.location_type == "inland" or plan.reference_location is not None:
      plan.readiness_status = "CAN_EXECUTE"
      return "ORCA_QUERY"
    plan.clarification_reason = "MISSING_LOCATION"
    plan.readiness_status = "NEED_LOCATION"
    return "CLARIFY"

  # Only truly unknown intent with no location becomes CHAT
  if not plan.location and plan.location_type != "inland":
    return "CHAT"

  if plan.location is not None or plan.location_type == "inland" or plan.reference_location is not None:
    plan.readiness_status = "CAN_EXECUTE"
    return "ORCA_QUERY"
  plan.clarification_reason = "MISSING_LOCATION"
  plan.readiness_status = "NEED_LOCATION"
  return "CLARIFY"

def build_clarification_context(plan: QueryPlan, state: ConversationState) -> dict:
  return {
    "language": plan.language,
    "intent": plan.intent,
    "operation": plan.operation,
    "known_locations": {
        "reference": plan.reference_location.name if plan.reference_location else None,
        "target": plan.target_location.name if plan.target_location else None,
        "comparison_candidates": [loc.name for loc in plan.compare_locations] if plan.compare_locations else []
    },
    "region": plan.region or (state.region_locations[0] if state.region_locations else None),
    "clarification_reason": plan.clarification_reason,
    "previous_clarifications": state.previous_clarifications
  }


def llm_route(
  query: str,
  conv_model: ConversationModel,
  extract_location_fn,
) -> tuple[str, QueryPlan | None, ExtractionResult]:
  """Returns (action, plan_or_none, raw_extraction).
  raw_extraction is returned too so callers can read chat_reply /
  clarify_question without a second model call."""
  conv_model.reset_turn_stats()
  trusted_lang = detect_language(query)
  result = conv_model.extract(EXTRACTION_SYSTEM_PROMPT, query)
  from schemas.extraction import Language, Action
  
  # Authoritative language propagation: do not let Qwen overwrite detected language
  result.language = Language(trusted_lang)
  
  result = result.ensure_non_null()
  result = normalize_extraction(result)

  plan = build_query_plan(query, result, extract_location_fn)
  action_str = resolve_action(result, plan)
  result.action = Action(action_str)

  return action_str, plan, result

CONTINUATION_TEMPLATE = """CONTEXT: This is a follow-up message in an ongoing conversation.
Already known: {known}
The system previously asked: "{question}"
The user's reply is: "{reply}"

Extract fields from this reply. The reply may be short (e.g. just a place name) — that is
expected, it is answering the question above, not starting a new topic. Do not treat a short
answer as CHAT or as an unrelated message."""

def llm_route_stateful(
  query: str,
  conv_model,
  extract_location_fn,
  state: ConversationState,
  fallback_location: str | None = None,
):
  """Stateful version of llm_route. Mutates `state` in place.
  Returns (action, plan_or_none, extraction, timings)."""
  t0_route = time.perf_counter()
  conv_model.reset_turn_stats()

  q_low = query.lower()
  
  # Remove aggressive state clearing logic so we naturally retain context across queries
  # We will let the LLM see the previous context and decide if it's a new query.
  
  if fallback_location and fallback_location.lower() == "string":
    fallback_location = None

  # 1. Language Detection timing
  t_lang_start = time.perf_counter()
  trusted_lang = detect_language(query)
  state.language = trusted_lang
  t_lang_ms = (time.perf_counter() - t_lang_start) * 1000.0

  # 2. Multilingual Fast Capability Router
  t_fast_start = time.perf_counter()
  from conversation.fast_router import detect_specialized_capability
  fast_route = detect_specialized_capability(query, trusted_lang)
  
  # 3. Location Pre-Check (reusing existing resolver)
  t_loc_start = time.perf_counter()
  loc_res_query = extract_location_fn(query.lower())
  t_loc_ms = (time.perf_counter() - t_loc_start) * 1000.0
  
  if fast_route.matched:
      from orchestrator.handlers import SPECIALIZED_HANDLERS
      if fast_route.intent not in SPECIALIZED_HANDLERS:
          logger.info(f"Router fast-path: matched {fast_route.intent} but it lacks a specialized handler. Falling back to generic pipeline.")
          fast_route.matched = False

  if fast_route.matched:
      from schemas.extraction import Intent, Action, LocationRole, ActionType, LocationItem
      
      # Use the resolved location or fallback to session state
      loc_text = None
      if getattr(loc_res_query, "geo_location", None):
          loc_text = loc_res_query.geo_location.name
      elif state.target_location_text:
          loc_text = state.target_location_text
      elif state.reference_location_text:
          loc_text = state.reference_location_text
          
      # Let the inland fast path trigger if the location is inland
      if getattr(loc_res_query, "location_type", "unknown") != "inland" or loc_text is None:
          # Build fast plan
          fast_result = ExtractionResult(
              action=Action.ORCA_QUERY,
              action_type=ActionType.ASSESS,
              intent=Intent(fast_route.intent) if hasattr(Intent, fast_route.intent) else Intent.marine_safety,
              locations=[LocationItem(text=loc_text, role=LocationRole.TARGET)] if loc_text else [],
              activity="fishing",
              time_relative="today",
          ).ensure_non_null()
          
          # Force the intent enum just in case
          fast_result.intent = Intent(fast_route.intent) if fast_route.intent in [e.value for e in Intent] else Intent.unknown
          
          state.merge(fast_result)
          plan = _plan_from_state(query, state, extract_location_fn, current_extraction=fast_result, fallback_location=fallback_location)
          
          # Overwrite operations to match the fast intent, but preserve specialized operations
          if plan.operation in (None, "ASSESS"):
              if fast_route.intent == "hazard_alert":
                  plan.operation = "ASSESS_HAZARD"
              elif fast_route.intent == "nearest_pfz":
                  plan.operation = "NEAREST_PFZ_SEARCH"
              elif fast_route.intent == "marine_safety_forecast":
                  plan.operation = "ASSESS"
              elif fast_route.intent == "marine_conditions":
                  plan.operation = "ASSESS"
              elif fast_route.intent == "productivity_analysis":
                  plan.operation = "ASSESS"
              
          t_total_ms = (time.perf_counter() - t0_route) * 1000.0
          timings = {
              "lang_detect_ms": t_lang_ms,
              "qwen_extract_ms": 0.0,
              "location_res_ms": t_loc_ms,
              "planning_ms": max(0.0, t_total_ms - t_lang_ms - t_loc_ms),
              "total_route_ms": t_total_ms
          }
          conv_model.last_route_timings = timings
          logger.info(f"Router fast-path: capability matched | intent={fast_route.intent} | confidence={fast_route.confidence:.2f}")
          
          # Proceed to check if location is missing
          if plan.location is None and plan.reference_location is None and plan.location_type != "inland":
              plan.clarification_reason = "MISSING_LOCATION"
              plan.readiness_status = "NEED_LOCATION"
              return "CLARIFY", plan, fast_result, timings
              
          return "ORCA_QUERY", plan, fast_result, timings

  q_low = query.lower()

  # Deterministic Geography Fast Path (0ms Qwen extraction time for geography queries)
  GEOGRAPHY_PATTERNS = [
    r"\bnearest sea\b", r"\bnearest ocean\b", r"\bclosest sea\b", r"\bclosest ocean\b",
    r"\bkitna door\b", r"\bhow far\b", r"\bdistance\b", r"\bnearest coast\b",
    r"\bkon sea\b", r"\bkothay ocean\b", r"\bsea kitna door\b"
  ]
  is_geo_fast_path = any(re.search(p, q_low) for p in GEOGRAPHY_PATTERNS) and not detect_special_operations(query)
  if is_geo_fast_path and (getattr(loc_res_query, "geo_location", None) or getattr(loc_res_query, "inland_name", None)):
    from schemas.extraction import Intent, Action, LocationRole, ActionType, LocationItem
    loc_text = loc_res_query.geo_location.name if loc_res_query.geo_location else loc_res_query.inland_name

    fast_result = ExtractionResult(
      action=Action.ORCA_QUERY,
      action_type=ActionType.EXPLAIN,
      intent=Intent.marine_geography,
      locations=[LocationItem(text=loc_text, role=LocationRole.REFERENCE)],
      activity="none",
      time_relative="today",
    ).ensure_non_null()

    state.merge(fast_result)
    plan = _plan_from_state(query, state, extract_location_fn, current_extraction=fast_result, fallback_location=fallback_location)
    plan.operation = "DISTANCE_TO_COAST"

    t_total_ms = (time.perf_counter() - t0_route) * 1000.0
    timings = {
      "lang_detect_ms": t_lang_ms,
      "qwen_extract_ms": 0.0,
      "location_res_ms": t_loc_ms,
      "planning_ms": max(0.0, t_total_ms - t_lang_ms - t_loc_ms),
      "total_route_ms": t_total_ms
    }
    conv_model.last_route_timings = timings
    logger.info(f"Router fast-path: geography | intent=marine_geography | location={loc_text}")
    return "ORCA_QUERY", plan, fast_result, timings

  # Only trigger inland fast-path clarify if NOT asking a reference origin query or geography
  is_locate_or_geo = any(k in q_low for k in ["sabse paas", "nearest", "part", "coast", "which sea", "kahan", "kothay", "door"])
  if getattr(loc_res_query, "location_type", "unknown") == "inland" and not is_locate_or_geo:
    inland_name = getattr(loc_res_query, "inland_name", "Inland region")
    from schemas.extraction import Intent, Action, LocationRole, ActionType, LocationItem

    if any(k in q_low for k in ["mach", "machli", "fish", "spot"]):
      intent_val = Intent.nearest_pfz
    elif any(k in q_low for k in ["safe", "weather", "hoga"]):
      intent_val = Intent.marine_safety_forecast
    else:
      intent_val = Intent.marine_conditions

    fast_result = ExtractionResult(
      action=Action.NON_COASTAL_ERROR,
      action_type=ActionType.EXPLAIN,
      intent=intent_val,
      locations=[LocationItem(text=inland_name, role=LocationRole.REFERENCE)],
      activity="fishing",
      time_relative="today",
    ).ensure_non_null()

    state.merge(fast_result)
    plan = _plan_from_state(query, state, extract_location_fn, current_extraction=fast_result, fallback_location=fallback_location)
    t_total_ms = (time.perf_counter() - t0_route) * 1000.0
    timings = {
      "lang_detect_ms": t_lang_ms,
      "qwen_extract_ms": 0.0,
      "location_res_ms": t_loc_ms,
      "planning_ms": max(0.0, t_total_ms - t_lang_ms - t_loc_ms),
      "total_route_ms": t_total_ms
    }
    conv_model.last_route_timings = timings
    logger.info(f"Router fast-path: inland intercept | location={inland_name}")
    return "NON_COASTAL_ERROR", plan, fast_result, timings

  if state.pending_clarification:
    prompt_input = CONTINUATION_TEMPLATE.format(
      known=state.known_summary(),
      question=state.pending_clarification,
      reply=query,
    )
  else:
    known_summary = state.known_summary()
    if known_summary != "nothing yet":
      prompt_input = f"[PREVIOUS CONTEXT: {known_summary}]\n\nUser Query: {query}"
    else:
      prompt_input = query

  # 3. Extraction timing
  t_extract_start = time.perf_counter()
  
  result = conv_model.extract(EXTRACTION_SYSTEM_PROMPT, prompt_input)
      
  t_extract_ms = (time.perf_counter() - t_extract_start) * 1000.0
  
  # Authoritative language propagation: do not let Qwen overwrite detected language
  from schemas.extraction import Language
  result.language = Language(trusted_lang)
  
  result = result.ensure_non_null()
  result = normalize_extraction(result)
  loc_names = [l.text for l in getattr(result, "locations", []) if l.text]
  loc_str = ", ".join(loc_names) if loc_names else "None"
  logger.info(f"Router extraction: intent={result.intent.value} | action={getattr(result.action, 'value', result.action)} | location={loc_str} | extract_ms={t_extract_ms:.1f}")

  # 4. State Merge & Planning timing
  t_plan_start = time.perf_counter()
  state.merge(result)

  if result.action.value == "CLARIFY":
    state.clarify_rounds += 1
  else:
    state.clarify_rounds = 0

  def _make_timings(action_res, plan_res):
    t_plan_ms = (time.perf_counter() - t_plan_start) * 1000.0
    return {
      "lang_detect_ms": t_lang_ms,
      "qwen_extract_ms": t_extract_ms,
      "location_res_ms": t_loc_ms,
      "planning_ms": t_plan_ms,
      "total_route_ms": (time.perf_counter() - t0_route) * 1000.0
    }

  if result.action.value == "CLARIFY" and state.clarify_rounds >= ConversationState.MAX_CLARIFY_ROUNDS:
    if state.location_text:
      plan = _plan_from_state(query, state, extract_location_fn, current_extraction=result, fallback_location=fallback_location)
      state.pending_clarification = None
      state.clarify_rounds = 0
      timings = _make_timings("ORCA_QUERY", plan)
      conv_model.last_route_timings = timings
      return "ORCA_QUERY", plan, result, timings
    else:
      state.clear()
      timings = _make_timings("GIVE_UP", None)
      conv_model.last_route_timings = timings
      logger.warning(f"Router: GIVE_UP after {ConversationState.MAX_CLARIFY_ROUNDS} clarification rounds")
      return "GIVE_UP", None, result, timings

  plan = _plan_from_state(query, state, extract_location_fn, current_extraction=result, fallback_location=fallback_location)
  action_str = resolve_action(result, plan)
  from schemas.extraction import Action
  result.action = Action(action_str)

  # Fix A: Synchronize QueryPlan -> State
  state.update_from_plan(plan)

  timings = _make_timings(action_str, plan)
  conv_model.last_route_timings = timings

  if action_str == "ORCA_QUERY":
    state.pending_clarification = None
    state.clarify_rounds = 0
    return "ORCA_QUERY", plan, result, timings

  if action_str != "ORCA_QUERY":
    state.pending_clarification = result.clarify_question
    return action_str, plan, result, timings

  state.pending_clarification = None
  state.clarify_rounds = 0
  return "ORCA_QUERY", plan, result, timings


def _plan_from_state(raw_query: str, state: ConversationState, extract_location_fn, current_extraction=None, fallback_location=None) -> QueryPlan:
  """Build the QueryPlan from the MERGED state, delegating entirely to build_query_plan."""
  from schemas.extraction import ExtractionResult, LocationItem, Action, Intent, LocationRole, ActionType, Language
  locations_list = []
  seen_locs = set()

  loc_source = None

  if current_extraction and getattr(current_extraction, "locations", []):
    for loc_item in current_extraction.locations:
      if loc_item.text and loc_item.text not in seen_locs:
        locations_list.append(loc_item)
        seen_locs.add(loc_item.text)
    if locations_list:
      loc_source = "query"

  if not locations_list and fallback_location:
    locations_list.append(LocationItem(text=fallback_location, role=LocationRole.TARGET))
    seen_locs.add(fallback_location)
    loc_source = "structured_request"

  if not locations_list:
    if state.target_location_text and state.target_location_text not in seen_locs:
      locations_list.append(LocationItem(text=state.target_location_text, role=LocationRole.TARGET))
      seen_locs.add(state.target_location_text)
    elif state.location_text and state.location_text not in seen_locs:
      locations_list.append(LocationItem(text=state.location_text, role=LocationRole(state.location_role) if state.location_role else LocationRole.TARGET))
      seen_locs.add(state.location_text)

    if state.reference_location_text and state.reference_location_text not in seen_locs:
      locations_list.append(LocationItem(text=state.reference_location_text, role=LocationRole.REFERENCE))
      seen_locs.add(state.reference_location_text)

    for r in state.region_locations:
      if r not in seen_locs:
        locations_list.append(LocationItem(text=r, role=LocationRole.REGION))
        seen_locs.add(r)

    for c in state.comparison_location_texts:
      if c not in seen_locs:
        locations_list.append(LocationItem(text=c, role=LocationRole.COMPARISON))
        seen_locs.add(c)
        
    if locations_list:
        loc_source = "conversation"

  mock_extraction = ExtractionResult(
    language=Language(state.language) if state.language else Language.en,
    intent=Intent(state.intent) if state.intent else Intent.unknown,
    action=Action.ORCA_QUERY,
    action_type=ActionType(state.action_type) if state.action_type else ActionType.ASSESS,
    locations=locations_list,
    activity=state.activity or "none",
    time_relative=state.time_relative or "none",
    time_offset_days=state.time_offset_days,
    time_period=state.time_period or "none",
    user_constraint=state.user_constraint,
    count=state.count,
    explanation_target=state.explanation_target
  )
  
  plan = build_query_plan(raw_query, mock_extraction, extract_location_fn)
  plan.location_source = loc_source
  return plan
