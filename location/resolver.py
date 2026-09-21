# location/resolver.py

import re
import math
import difflib
import requests
from typing import Optional, List
from schemas.contracts import GeoLocation
from .gazetteer import GAZETTEER
from .location_metadata import INLAND_LOCATIONS, LOCATION_METADATA
from .models import ResolvedLocation

DIRECTION_BEARINGS = {
    "northeast": 45.0, "north-east": 45.0, "uttor-purbo": 45.0, "ne": 45.0,
    "southeast": 135.0, "south-east": 135.0, "dokkhin-purbo": 135.0, "se": 135.0,
    "southwest": 225.0, "south-west": 225.0, "dokkhin-poshchim": 225.0, "sw": 225.0,
    "northwest": 315.0, "north-west": 315.0, "uttor-poshchim": 315.0, "nw": 315.0,
    "north": 0.0, "uttor": 0.0, "north side": 0.0, "north-e": 0.0, "n": 0.0,
    "east": 90.0, "purbo": 90.0, "east side": 90.0, "east-e": 90.0, "east e": 90.0, "e": 90.0,
    "south": 180.0, "dokkhhin": 180.0, "dokkhin": 180.0, "south side": 180.0, "south-e": 180.0, "s": 180.0,
    "west": 270.0, "poshchim": 270.0, "west side": 270.0, "west-e": 270.0, "w": 270.0,
}

def normalize_direction(raw_dir: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    if not raw_dir:
        return None, None
    d_clean = raw_dir.lower().strip()
    sorted_keys = sorted(DIRECTION_BEARINGS.keys(), key=lambda k: len(k), reverse=True)
    for key in sorted_keys:
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, d_clean) or key == d_clean:
            brg = DIRECTION_BEARINGS[key]
            label = "N" if brg == 0 else "NE" if brg == 45 else "E" if brg == 90 else "SE" if brg == 135 else "S" if brg == 180 else "SW" if brg == 225 else "W" if brg == 270 else "NW"
            return brg, label
    return None, None

def calculate_destination_point(lat: float, lon: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
    """
    Calculates destination (latitude, longitude) given an origin point,
    distance in km, and bearing angle in degrees (Haversine formula).
    """
    R = 6371.0 # Earth's radius in km
    d_rad = distance_km / R
    brg_rad = math.radians(bearing_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_rad) +
        math.cos(lat1) * math.sin(d_rad) * math.cos(brg_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brg_rad) * math.sin(d_rad) * math.cos(lat1),
        math.cos(d_rad) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)

class LocationResolution(ResolvedLocation):
    """
    Backwards-compatible wrapper inheriting from ResolvedLocation.
    Exposes geo_location, location_type, inland_name, candidate_names, coastal_access, metadata.
    """
    def __init__(
        self,
        geo_location: Optional[GeoLocation] = None,
        location_type: str = "unknown",
        inland_name: Optional[str] = None,
        candidate_names: Optional[List[str]] = None,
        coastal_access: bool = True,
        metadata: Optional[dict] = None,
        canonical_name: str = "Unknown",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_class: str = "coastal_point",
        marine_access: str = "direct",
        region: Optional[str] = None,
        role: str = "TARGET",
        confidence: float = 1.0,
        source: str = "GAZETTEER",
        resolution_status: str = "EXACT",
        matched_text: Optional[str] = None
    ):
        super().__init__(
            canonical_name=canonical_name or (geo_location.name if geo_location else inland_name or "Unknown"),
            latitude=latitude if latitude is not None else (geo_location.latitude if geo_location else None),
            longitude=longitude if longitude is not None else (geo_location.longitude if geo_location else None),
            location_class=location_class,
            marine_access=marine_access,
            coastal_access=coastal_access,
            region=region,
            role=role,
            confidence=confidence,
            source=source,
            resolution_status=resolution_status,
            matched_text=matched_text,
            candidate_names=candidate_names or []
        )
        self.inland_name = inland_name
        self.metadata = metadata or {}

_ONLINE_GEOCODE_CACHE = {}

def _online_geocode(query: str):
    """Bounded online fallback using Nominatim."""
    if not query or len(query) < 3:
        return None, None, None
        
    cache_key = query.lower().strip()
    if cache_key in _ONLINE_GEOCODE_CACHE:
        return _ONLINE_GEOCODE_CACHE[cache_key]

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "ORCA-Marine-Agent/1.0"},
            timeout=3
        )
        if resp.status_code == 200 and resp.json():
            data = resp.json()[0]
            res = (float(data["lat"]), float(data["lon"]), data.get("display_name", query).split(",")[0].strip())
            _ONLINE_GEOCODE_CACHE[cache_key] = res
            return res
    except Exception:
        pass
    
    _ONLINE_GEOCODE_CACHE[cache_key] = (None, None, None)
    return None, None, None

class LocationResolver:
    """
    Unified Location Resolution Service for ORCA.
    
    Provides deterministic exact matching and transliteration fuzzy matching
    against GAZETTEER (coordinates) and LOCATION_METADATA (coastal access & region types).
    """

    def resolve(self, query_lower: str) -> LocationResolution:
        if not query_lower:
            return LocationResolution(
                geo_location=None,
                location_type="unknown",
                inland_name=None,
                candidate_names=[],
                coastal_access=True,
                canonical_name="Unknown",
                resolution_status="NOT_FOUND",
                confidence=0.0
            )

        clean_text = query_lower.lower().strip()

        # Strip common Banglish / Hinglish postpositions
        for suffix in ["-r kache", "-r", "-er", " ke paas", " ke", " kache", " kachhe", " r"]:
            if clean_text.endswith(suffix):
                clean_text = clean_text[:-len(suffix)].strip()

        # Tokenize input words for word-boundary matching
        words = re.findall(r"\b[a-z]{2,}\b", clean_text)

        # 1. Exact word-boundary search against GAZETTEER
        coastal_matches = []
        seen_names = set()
        for key, (lat, lon, name) in GAZETTEER.items():
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, clean_text):
                if name not in seen_names:
                    coastal_matches.append((key, lat, lon, name, 1.0))
                    seen_names.add(name)

        # 2. Exact word-boundary search against INLAND_LOCATIONS
        inland_matches = []
        for key, name in INLAND_LOCATIONS.items():
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, clean_text):
                if name not in seen_names:
                    inland_matches.append((key, name, 1.0))
                    seen_names.add(name)

        # 3. High-Confidence Fuzzy Matching for Transliterations (Threshold >= 0.82)
        fuzzy_matched = False
        if not coastal_matches and not inland_matches:
            best_coastal_score = 0.0
            best_coastal_candidate = None
            for word in words:
                if len(word) < 4:
                    continue
                for key, (lat, lon, name) in GAZETTEER.items():
                    ratio = difflib.SequenceMatcher(None, word, key).ratio()
                    if ratio >= 0.82 and ratio > best_coastal_score:
                        best_coastal_score = ratio
                        best_coastal_candidate = (key, lat, lon, name, ratio)

            best_inland_score = 0.0
            best_inland_candidate = None
            for word in words:
                if len(word) < 4:
                    continue
                for key, name in INLAND_LOCATIONS.items():
                    ratio = difflib.SequenceMatcher(None, word, key).ratio()
                    if ratio >= 0.82 and ratio > best_inland_score:
                        best_inland_score = ratio
                        best_inland_candidate = (key, name, ratio)

            if best_coastal_candidate and best_coastal_score >= max(0.82, best_inland_score + 0.05):
                _, lat, lon, name, score = best_coastal_candidate
                coastal_matches.append((best_coastal_candidate[0], lat, lon, name, score))
                fuzzy_matched = True
            elif best_inland_candidate and best_inland_score >= 0.82:
                inland_matches.append((best_inland_candidate[0], best_inland_candidate[1], best_inland_score))
                fuzzy_matched = True

        status = "FUZZY_HIGH_CONFIDENCE" if fuzzy_matched else "EXACT"

        # Evaluate matches and construct output LocationResolution
        if coastal_matches:
            match_key, lat, lon, name = coastal_matches[0][0], coastal_matches[0][1], coastal_matches[0][2], coastal_matches[0][3]
            conf = coastal_matches[0][4] if len(coastal_matches[0]) > 4 else 1.0
            meta = LOCATION_METADATA.get(match_key, {"name": name, "type": "coastal_town", "coastal_access": True})
            loc_obj = GeoLocation(latitude=lat, longitude=lon, name=name)
            loc_type = "coastal" if meta.get("coastal_access", True) else "inland"
            c_names = [m[3] for m in coastal_matches]
            loc_cls = meta.get("type", "coastal_point")
            m_access = "direct" if meta.get("coastal_access", True) else "nearby"
            return LocationResolution(
                geo_location=loc_obj,
                location_type=loc_type,
                inland_name=meta.get("name") if not meta.get("coastal_access") else None,
                candidate_names=c_names,
                coastal_access=meta.get("coastal_access", True),
                metadata=meta,
                canonical_name=name,
                latitude=lat,
                longitude=lon,
                location_class=loc_cls,
                marine_access=m_access,
                confidence=conf,
                source="GAZETTEER",
                resolution_status=status,
                matched_text=match_key
            )

        if inland_matches:
            match_key, name = inland_matches[0][0], inland_matches[0][1]
            conf = inland_matches[0][2] if len(inland_matches[0]) > 2 else 1.0
            meta = LOCATION_METADATA.get(match_key, {"name": name, "type": "state", "coastal_access": False})
            c_names = [m[1] for m in inland_matches]
            loc_cls = meta.get("type", "inland_region")
            
            geo_loc = None
            if match_key in GAZETTEER:
                g_lat, g_lon, g_name = GAZETTEER[match_key]
                geo_loc = GeoLocation(latitude=g_lat, longitude=g_lon, name=g_name)
                
            return LocationResolution(
                geo_location=geo_loc,
                location_type="inland",
                inland_name=name,
                candidate_names=c_names,
                coastal_access=False,
                metadata=meta,
                canonical_name=name,
                location_class=loc_cls,
                marine_access="none",
                confidence=conf,
                source="METADATA",
                resolution_status=status,
                matched_text=match_key
            )

        # 4. Online Geocoding Fallback
        lat, lon, name = _online_geocode(clean_text)
        if lat is not None and lon is not None:
            # Validate marine relevance
            from agents.geospatial.coastline import distance_to_coast
            dist = distance_to_coast(lat, lon)
            is_coastal = dist is not None and dist < 25.0
            
            loc_type = "coastal" if is_coastal else "inland"
            geo_loc = GeoLocation(latitude=lat, longitude=lon, name=name)
            return LocationResolution(
                geo_location=geo_loc,
                location_type=loc_type,
                inland_name=name if not is_coastal else None,
                candidate_names=[name],
                coastal_access=is_coastal,
                canonical_name=name,
                latitude=lat,
                longitude=lon,
                location_class="coastal_point" if is_coastal else "inland_region",
                marine_access="direct" if is_coastal else "none",
                confidence=0.9,
                source="NOMINATIM",
                resolution_status="EXACT" if is_coastal else "ONLINE_RESOLVED",
                matched_text=clean_text
            )

        return LocationResolution(
            geo_location=None,
            location_type="unknown",
            inland_name=None,
            candidate_names=[],
            coastal_access=True,
            canonical_name="Unknown",
            resolution_status="NOT_FOUND",
            confidence=0.0
        )


# Singleton Resolver instance
_RESOLVER = LocationResolver()

def extract_location(query_lower: str) -> LocationResolution:
    """Standard entry point for location extraction across ORCA."""
    return _RESOLVER.resolve(query_lower)
