import requests
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)

def check_live_earthquakes(lat: float, lon: float, radius_km: float = 200.0) -> dict:
    """
    Query the USGS live earthquake API for recent significant earthquakes near the location.
    Looks back 48 hours for earthquakes magnitude 4.5 or greater.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=2)
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.isoformat(),
        "endtime": end_time.isoformat(),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": 4.5
    }
    
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        
        events = data.get("features", [])
        if events:
            # Get the most significant/closest one
            # Features are already sorted by time (newest first)
            event = events[0]
            props = event.get("properties", {})
            return {
                "hazard_detected": True,
                "status": "SUCCESS",
                "type": "EARTHQUAKE",
                "magnitude": props.get("mag"),
                "place": props.get("place"),
                "time": props.get("time"),
                "tsunami_warning": bool(props.get("tsunami")),
                "url": props.get("url")
            }
        
        return {"hazard_detected": False, "status": "SUCCESS"}
    except Exception as e:
        logger.warning(f"Failed to query USGS earthquake API: {e}")
        
    return {"hazard_detected": False, "status": "ERROR"}

def detect_catastrophic_hazards(lat: float, lon: float, intent: str, cyclone_data: dict) -> dict:
    """
    Aggregates catastrophic hazard detection (Earthquakes, Tsunamis, Cyclones).
    """
    hazards = []
    
    # 1. Check Earthquakes
    eq = check_live_earthquakes(lat, lon)
    if eq.get("hazard_detected"):
        hazards.append({
            "type": "EARTHQUAKE",
            "severity": "CRITICAL" if eq.get("tsunami_warning") else "HIGH",
            "description": f"Magnitude {eq.get('magnitude')} earthquake near {eq.get('place')}. Tsunami warning: {eq.get('tsunami_warning')}"
        })
        if eq.get("tsunami_warning"):
            hazards.append({
                "type": "TSUNAMI_WARNING",
                "severity": "CRITICAL",
                "description": "Active Tsunami Warning from recent seismic event."
            })
            
    # 2. Check Cyclones
    if cyclone_data and cyclone_data.get("any_within_hazard_radius"):
        for cyc in cyclone_data.get("active_cyclones", []):
            if cyc.get("within_hazard_radius"):
                hazards.append({
                    "type": "CYCLONE",
                    "severity": "CRITICAL",
                    "description": f"Cyclone {cyc['name']} (Cat {cyc['category']}) is dangerously close ({cyc['distance_km']}km)."
                })
                
    is_error = eq.get("status") == "ERROR" or (cyclone_data and cyclone_data.get("cyclone_data_status") == "CYCLONE_DATA_UNAVAILABLE")
    
    if len(hazards) > 0:
        catastrophic_status = "ACTIVE"
        catastrophic_active = True
    elif is_error:
        catastrophic_status = "WARNING_SOURCE_UNAVAILABLE"
        catastrophic_active = None
    else:
        catastrophic_status = "WARNING_ABSENT"
        catastrophic_active = False

    return {
        "catastrophic_active": catastrophic_active,
        "catastrophic_status": catastrophic_status,
        "hazards": hazards
    }
