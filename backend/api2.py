import requests
import json
from requests.auth import HTTPBasicAuth
import os
from pathlib import Path
from dotenv import load_dotenv

# ======================
# CONFIGURATION
# ======================

# Always load .env from project root
ROOT_DIR = Path(__file__).resolve().parents[1]   # go up one level from backend/
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

SPACETRACK_USER = os.getenv("SPACETRACK_USER")
SPACETRACK_PASS = os.getenv("SPACETRACK_PASS")
# Public DONKI dataset base URL
DONKI_BASE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get"

# Optional API logging
LOG_API = os.getenv("ODIN_API_LOG", "1") == "1"

def _log_api(name, url, params=None):
    if LOG_API:
        try:
            print(f"[API] {name}: {url} params={params or {}}")
        except Exception:
            pass

if LOG_API:
    try:
        print(f"[API] ENV CHECK: .env_loaded_from={globals().get('_found', None)} SPACETRACK_USER set={bool(SPACETRACK_USER)} SPACETRACK_PASS set={bool(SPACETRACK_PASS)}")
    except Exception:
        pass

# ======================
# NASA DONKI DATA (Public Dataset URLs)
# ======================
def fetch_donki_cme(start_dt=None, end_dt=None):
    params = {}
    if start_dt and end_dt:
        params["startDate"] = start_dt.strftime("%Y-%m-%d")
        params["endDate"] = end_dt.strftime("%Y-%m-%d")
    url = f"{DONKI_BASE_URL}/CME"
    _log_api("DONKI_CME", url, params)
    data = requests.get(url, params=params).json()
    cme_list = []
    for cme in data:
        entry = {
            "startTime": cme.get("startTime"),
            "sourceLocation": cme.get("sourceLocation"),
            "speed": None,
            "width": None,
            "type": None
        }
        if "cmeAnalyses" in cme and cme["cmeAnalyses"]:
            analysis = cme["cmeAnalyses"][0]
            entry["speed"] = analysis.get("speed")
            entry["width"] = analysis.get("halfAngle") or analysis.get("width")
            entry["type"] = analysis.get("type")
        cme_list.append(entry)
    return cme_list

def fetch_donki_flare(start_dt=None, end_dt=None):
    params = {}
    if start_dt and end_dt:
        params["startDate"] = start_dt.strftime("%Y-%m-%d")
        params["endDate"] = end_dt.strftime("%Y-%m-%d")
    url = f"{DONKI_BASE_URL}/FLR"
    _log_api("DONKI_FLR", url, params)
    data = requests.get(url, params=params).json()
    flare_list = []
    for flr in data:
        entry = {
            "beginTime": flr.get("beginTime"),
            "peakTime": flr.get("peakTime"),
            "endTime": flr.get("endTime"),
            "classType": flr.get("classType"),
            "sourceLocation": flr.get("sourceLocation")
        }
        flare_list.append(entry)
    return flare_list

def fetch_donki_gst(start_dt=None, end_dt=None):
    params = {}
    if start_dt and end_dt:
        params["startDate"] = start_dt.strftime("%Y-%m-%d")
        params["endDate"] = end_dt.strftime("%Y-%m-%d")
    url = f"{DONKI_BASE_URL}/GST"
    _log_api("DONKI_GST", url, params)
    data = requests.get(url, params=params).json()
    gst_list = []
    for gst in data:
        entry = {
            "startTime": gst.get("startTime"),
            "endTime": gst.get("endTime"),
            "allKpIndex": gst.get("allKpIndex")
        }
        gst_list.append(entry)
    return gst_list

def fetch_donki_sep(start_dt=None, end_dt=None):
    params = {}
    if start_dt and end_dt:
        params["startDate"] = start_dt.strftime("%Y-%m-%d")
        params["endDate"] = end_dt.strftime("%Y-%m-%d")
    url = f"{DONKI_BASE_URL}/SEP"
    _log_api("DONKI_SEP", url, params)
    data = requests.get(url, params=params).json()
    sep_list = []
    for sep in data:
        entry = {
            "eventTime": sep.get("eventTime"),
            "linkedEvents": sep.get("linkedEvents"),
            "protonFlux": sep.get("protonFlux")
        }
        sep_list.append(entry)
    return sep_list

# ======================
# SPACE-TRACK DATA
# ======================
def fetch_spacetrack_tle(start_dt=None, end_dt=None, limit=200):
    """
    Fetch TLEs from Space-Track. If start/end provided, query EPOCH between dates.
    Requires SPACETRACK_USER/PASS env vars. Returns [] if unavailable or on error.
    """
    if not SPACETRACK_USER or not SPACETRACK_PASS:
        if LOG_API:
            print("[API] SPACETRACK_TLE skipped: missing SPACETRACK_USER/PASS env vars")
        return []
    session = requests.Session()

    try:
        # Perform login (Space-Track requires cookie-based auth)
        login_url = "https://www.space-track.org/ajaxauth/login"
        login_payload = {"identity": SPACETRACK_USER, "password": SPACETRACK_PASS}
        _log_api("SPACETRACK_LOGIN", login_url, {"identity": "***", "password": "***"})
        lr = session.post(login_url, data=login_payload, timeout=20)
        lr.raise_for_status()

        if start_dt and end_dt:
            start_str = start_dt.strftime("%Y-%m-%d")
            end_str = end_dt.strftime("%Y-%m-%d")
            tle_url = (
                "https://www.space-track.org/basicspacedata/query/"
                f"class/tle/EPOCH/{start_str}--{end_str}/orderby/EPOCH%20asc/limit/{int(limit)}/format/json"
            )
        else:
            tle_url = "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/format/json"
        _log_api("SPACETRACK_TLE", tle_url, None)
        resp = session.get(tle_url, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        if LOG_API:
            print("[API] Space-Track call failed or credentials missing")
        return []

    debris_list = []
    for obj in data:
        entry = {
            "NORAD_CAT_ID": obj.get("NORAD_CAT_ID"),
            "OBJECT_NAME": obj.get("OBJECT_NAME"),
            "EPOCH": obj.get("EPOCH"),
            "INCLINATION": obj.get("INCLINATION"),
            "RA_OF_ASC_NODE": obj.get("RA_OF_ASC_NODE"),
            "ECCENTRICITY": obj.get("ECCENTRICITY"),
            "ARG_OF_PERICENTER": obj.get("ARG_OF_PERICENTER"),
            "MEAN_ANOMALY": obj.get("MEAN_ANOMALY"),
            "MEAN_MOTION": obj.get("MEAN_MOTION"),
            "OBJECT_TYPE": obj.get("OBJECT_TYPE"),
            "APOGEE": obj.get("APOGEE"),
            "PERIGEE": obj.get("PERIGEE")
        }
        debris_list.append(entry)
    return debris_list

def fetch_donki_cme_analysis(start_dt, end_dt, most_accurate_only=True, min_speed=500, min_half_angle=30, catalog="ALL"):
    """
    Fetch CMEAnalysis between start_dt and end_dt.
    start_dt, end_dt: naive UTC datetimes
    Returns list of dicts from DONKI.
    """
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    params = {
        "startDate": start_str,
        "endDate": end_str,
        "mostAccurateOnly": str(most_accurate_only).lower(),
        "speed": str(min_speed),
        "halfAngle": str(min_half_angle),
        "catalog": catalog,
    }
    url = f"{DONKI_BASE_URL}/CMEAnalysis"
    try:
        _log_api("DONKI_CMEAnalysis", url, params)
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        if LOG_API:
            print("[API] DONKI_CMEAnalysis call failed")
        return []

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    output = {
        "CME_Data": fetch_donki_cme(),
        "Solar_Flares": fetch_donki_flare(),
        "Geomagnetic_Storms": fetch_donki_gst(),
        "SEP_Events": fetch_donki_sep(),
        "Orbital_Debris": fetch_spacetrack_tle()
    }

    with open("odin_space_data.json", "w") as f:
        json.dump(output, f, indent=4)

    print("✅ Unified JSON file created: odin_space_data.json (using public DONKI datasets)")
