from flask import Flask, render_template, jsonify, send_file, request, Response
import requests
import re
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import json
import time
import threading
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
import matplotlib.patheffects
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Set font to system sans-serif with fallbacks
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.facecolor'] = '#1e293b'
matplotlib.rcParams['figure.facecolor'] = '#0f172a'
matplotlib.rcParams['text.color'] = '#f8fafc'
matplotlib.rcParams['axes.labelcolor'] = '#94a3b8'
matplotlib.rcParams['xtick.color'] = '#64748b'
matplotlib.rcParams['ytick.color'] = '#64748b'
matplotlib.rcParams['grid.color'] = '#334155'
matplotlib.rcParams['axes.edgecolor'] = '#334155'

# Attempt to register local Metropolis fonts for server-side plotting.
# Place font files (TTF/WOFF/WOFF2) into static/fonts/metropolis/.
METROPOLIS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'fonts', 'metropolis')
if os.path.isdir(METROPOLIS_DIR):
    try:
        import matplotlib.font_manager as font_manager
        for fpath in glob.glob(os.path.join(METROPOLIS_DIR, '*')):
            lower = fpath.lower()
            if lower.endswith(('.ttf', '.otf', '.woff', '.woff2')):
                try:
                    font_manager.fontManager.addfont(fpath)
                    print(f"Added font to matplotlib: {fpath}")
                except Exception as e:
                    print(f"Could not add font {fpath}: {e}")
        # Prepend Metropolis to sans-serif fallback list if available
        sans = matplotlib.rcParams.get('font.sans-serif', [])
        if 'Metropolis' not in sans:
            matplotlib.rcParams['font.sans-serif'] = ['Metropolis'] + list(sans)
    except Exception as e:
        print('Error registering local fonts:', e)

app = Flask(__name__)

# External data endpoints
PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
PLASMA_2HR_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
MAG_2HR_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
SCALES_URL = "https://services.swpc.noaa.gov/products/noaa-scales.json"
GOES_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
GOES_XRAY_1DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
GOES_PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
SOLAR_REGIONS_URL = "https://services.swpc.noaa.gov/json/solar_regions.json"
FLARE_EVENT_FEED_URL = "https://services.swpc.noaa.gov/json/solar_events_last_30_days.json"
SOLAR_CYCLE_INDICES_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
SUNSPOTS_SMOOTHED_URL = "https://services.swpc.noaa.gov/json/solar-cycle/sunspots-smoothed.json"
SOLAR_REGION_SUMMARY_URLS = [
    "https://services.swpc.noaa.gov/text/srs.txt",
    "https://services.swpc.noaa.gov/text/solar-region-summary.txt",
    "https://services.swpc.noaa.gov/text/solar-regions.txt",
]
SWPC_REPORT_TEXTS = {
    'discussion': {
        'title': 'Forecast Discussion',
        'url': 'https://services.swpc.noaa.gov/text/discussion.txt'
    },
    'three_day': {
        'title': '3-Day Forecast',
        'url': 'https://services.swpc.noaa.gov/text/3-day-forecast.txt'
    },
    'solar_geomag': {
        'title': '3-Day Solar-Geophysical Predictions',
        'url': 'https://services.swpc.noaa.gov/text/3-day-solar-geomag-predictions.txt'
    },
    'solar_regions': {
        'title': 'Solar Region Summary',
        'url': 'https://services.swpc.noaa.gov/text/srs.txt'
    },
}
OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
HEMI_POWER_URL = "https://services.swpc.noaa.gov/text/aurora-nowcast-hemi-power.txt"
GOES_MAG_PRIMARY_URL = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-6-hour.json"
GOES_MAG_SECONDARY_URL = "https://services.swpc.noaa.gov/json/goes/secondary/magnetometers-6-hour.json"
NASA_API_KEY = os.getenv('NASA_API_KEY', 'bgduJ4idKoFqHnlU7nUkToH4QJtrg7F44xhiuAwm')
FLARE_ALERT_MAX_AGE_HOURS = 24

# Bounded in-memory cache for expensive operations. Keep values directly in
# _cache so the stale aurora-map fallback can still read old bytes.
def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


_CACHE_MAX_ENTRIES = max(8, _env_int('AURORA_CACHE_MAX_ENTRIES', 80))
_CACHE_MAX_BYTES = max(4, _env_int('AURORA_CACHE_MAX_MB', 24)) * 1024 * 1024
_CACHE_ENTRY_MAX_BYTES = max(1, _env_int('AURORA_CACHE_ENTRY_MAX_MB', 8)) * 1024 * 1024
_STALE_FALLBACK_KEYS = {'aurora_map'}
_cache = OrderedDict()
_cache_timestamps = {}
_cache_expirations = {}
_cache_sizes = {}
_cache_lock = threading.RLock()

# Backwards-compatible cache helper used by newer endpoints
class _SimpleCache:
    def get(self, key, default=None):
        v = get_cached(key, max_age_seconds=None)
        return v if v is not None else default
    def set(self, key, value, timeout=None):
        set_cached(key, value, timeout=timeout)

# expose `cache` for code that expects a Flask-like cache object
cache = _SimpleCache()

# Thread lock for expensive operations to prevent concurrent generation
_map_generation_lock = threading.Lock()
_map_generating = False  # Flag to indicate generation in progress

def _estimate_cache_value_size(value):
    """Best-effort deep size estimate without allocating serialized copies."""
    seen = set()
    objects_seen = 0

    def walk(obj):
        nonlocal objects_seen
        if objects_seen > 5000:
            return 0
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        objects_seen += 1

        size = sys.getsizeof(obj, 0)
        if isinstance(obj, dict):
            for k, v in obj.items():
                size += walk(k) + walk(v)
                if size > _CACHE_ENTRY_MAX_BYTES:
                    break
        elif isinstance(obj, (list, tuple, set, frozenset, OrderedDict)):
            for item in obj:
                size += walk(item)
                if size > _CACHE_ENTRY_MAX_BYTES:
                    break
        return size

    return walk(value)


def _delete_cached_unlocked(key):
    _cache.pop(key, None)
    _cache_timestamps.pop(key, None)
    _cache_expirations.pop(key, None)
    _cache_sizes.pop(key, None)


def _purge_expired_unlocked(now=None):
    now = now or time.time()
    for key, expires_at in list(_cache_expirations.items()):
        if expires_at is not None and expires_at <= now and key not in _STALE_FALLBACK_KEYS:
            _delete_cached_unlocked(key)


def _enforce_cache_limits_unlocked():
    total_size = sum(_cache_sizes.values())
    while _cache and (len(_cache) > _CACHE_MAX_ENTRIES or total_size > _CACHE_MAX_BYTES):
        key, _ = _cache.popitem(last=False)
        total_size -= _cache_sizes.pop(key, 0)
        _cache_timestamps.pop(key, None)
        _cache_expirations.pop(key, None)


def get_cached(key, max_age_seconds=30):
    """Get cached value if not expired"""
    now = time.time()
    with _cache_lock:
        if key not in _cache:
            return None

        expires_at = _cache_expirations.get(key)
        if expires_at is not None and expires_at <= now:
            if key not in _STALE_FALLBACK_KEYS:
                _delete_cached_unlocked(key)
            return None

        timestamp = _cache_timestamps.get(key)
        if max_age_seconds is not None and (timestamp is None or now - timestamp >= max_age_seconds):
            if key not in _STALE_FALLBACK_KEYS:
                _delete_cached_unlocked(key)
            return None

        _cache.move_to_end(key)
        return _cache[key]


def get_stale_cached(key):
    """Return a cached value without age checks for explicit fallback paths."""
    with _cache_lock:
        value = _cache.get(key)
        if value is not None:
            _cache.move_to_end(key)
        return value


def _validate_date_range(start_str, end_str, max_days=11):
    """Validate that start/end are YYYY-MM-DD and within max_days inclusive."""
    try:
        s = datetime.strptime(start_str, '%Y-%m-%d')
        e = datetime.strptime(end_str, '%Y-%m-%d')
    except Exception:
        return False, 'Invalid date format, expected YYYY-MM-DD'
    delta = (e - s).days
    if delta < 0:
        return False, 'End date must be on or after start date'
    if delta > max_days:
        return False, f'Date range too large (max {max_days} days)'
    return True, ''

def set_cached(key, value, timeout=None):
    """Set cached value with current timestamp"""
    now = time.time()
    value_size = _estimate_cache_value_size(value)
    if value_size > _CACHE_ENTRY_MAX_BYTES:
        with _cache_lock:
            _delete_cached_unlocked(key)
        print(f"[CACHE] Skipped oversized cache entry {key}: {value_size / (1024 * 1024):.1f} MB")
        return

    with _cache_lock:
        _purge_expired_unlocked(now)
        _cache[key] = value
        _cache.move_to_end(key)
        _cache_timestamps[key] = now
        _cache_expirations[key] = now + timeout if timeout else None
        _cache_sizes[key] = value_size
        _enforce_cache_limits_unlocked()


def _figure_to_png_buffer(fig, **savefig_kwargs):
    """Serialize and close a matplotlib figure promptly."""
    buf = BytesIO()
    try:
        fig.savefig(buf, format='png', **savefig_kwargs)
        buf.seek(0)
        return buf
    finally:
        plt.close(fig)



def interpolate_gaps(values, max_gap_minutes=10, time_interval_minutes=1):
    """Linearly fill short None/NaN gaps while leaving extended outages untouched."""
    if not values:
        return values

    arr = np.array([
        np.nan if value is None else float(value)
        for value in values
    ], dtype=float)

    valid_indices = np.where(~np.isnan(arr))[0]
    if len(valid_indices) < 2:
        return values

    max_gap_points = max(1, int(max_gap_minutes / max(time_interval_minutes, 1)))
    result = arr.copy()

    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        gap_size = end_idx - start_idx - 1

        if gap_size > 0 and gap_size <= max_gap_points:
            start_val = arr[start_idx]
            end_val = arr[end_idx]
            for j in range(1, gap_size + 1):
                alpha = j / (gap_size + 1)
                result[start_idx + j] = start_val + alpha * (end_val - start_val)

    return [None if np.isnan(value) else float(value) for value in result]


def _swpc_table_to_rows(table_rows):
    """Convert SWPC JSON table payloads into header-keyed dictionaries."""
    if not isinstance(table_rows, list) or len(table_rows) < 2:
        return []

    header = [str(column).strip() for column in table_rows[0]]
    parsed_rows = []
    for row in table_rows[1:]:
        if not isinstance(row, list):
            continue
        parsed_rows.append({
            header[index]: row[index] if index < len(row) else None
            for index in range(len(header))
        })
    return parsed_rows


def _swpc_numeric(value):
    if value in (None, ''):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return None if numeric <= -999 else numeric


def _fallback_solar_wind_history(sw_current=None):
    if not sw_current:
        return None

    current_time = parse_swpc_datetime(sw_current.get('time')) or datetime.now(timezone.utc)
    return {
        'times': [current_time],
        'speeds': [sw_current.get('speed')],
        'densities': [sw_current.get('density')],
        'bzs': [sw_current.get('bz')],
        'bts': [sw_current.get('bt')],
    }


def fetch_solar_wind_data():
    """Fetch current solar wind plasma and magnetic field data"""
    try:
        # Fetch plasma data (speed, density)
        plasma_response = requests.get(PLASMA_URL, timeout=10)
        plasma_response.raise_for_status()
        plasma_rows = _swpc_table_to_rows(plasma_response.json())
        
        # Fetch magnetic field data (Bt, Bz)
        mag_response = requests.get(MAG_URL, timeout=10)
        mag_response.raise_for_status()
        mag_rows = _swpc_table_to_rows(mag_response.json())
        
        latest_plasma = None
        latest_plasma_time = None
        for row in reversed(plasma_rows):
            row_time = parse_swpc_datetime(row.get('time_tag'))
            if row_time:
                latest_plasma = row
                latest_plasma_time = row_time
                break

        latest_mag = None
        for row in reversed(mag_rows):
            row_time = parse_swpc_datetime(row.get('time_tag'))
            if not row_time:
                continue
            if latest_plasma_time and row_time == latest_plasma_time:
                latest_mag = row
                break
            if latest_mag is None:
                latest_mag = row

        if latest_plasma and latest_mag:
            return {
                'time': latest_plasma.get('time_tag') or latest_mag.get('time_tag'),
                'speed': _swpc_numeric(latest_plasma.get('speed')),
                'density': _swpc_numeric(latest_plasma.get('density')),
                'temperature': _swpc_numeric(latest_plasma.get('temperature')),
                'bt': _swpc_numeric(latest_mag.get('bt')),
                'bz': _swpc_numeric(latest_mag.get('bz_gsm')),
                'bx': _swpc_numeric(latest_mag.get('bx_gsm')),
                'by': _swpc_numeric(latest_mag.get('by_gsm')),
            }
    except Exception as e:
        print(f"Error fetching solar wind data: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_solar_wind_history():
    """Fetch 2-hour history of solar wind data for plotting"""
    try:
        plasma_response = requests.get(PLASMA_2HR_URL, timeout=10)
        mag_response = requests.get(MAG_2HR_URL, timeout=10)
        
        # Check for valid responses
        if plasma_response.status_code != 200 or mag_response.status_code != 200:
            print(f"Error: Got status {plasma_response.status_code}/{mag_response.status_code} from SWPC")
            return None
        
        try:
            plasma_rows = _swpc_table_to_rows(plasma_response.json())
            mag_rows = _swpc_table_to_rows(mag_response.json())
        except json.JSONDecodeError as e:
            print(f"JSON decode error from SWPC (data provider issue): {e}")
            sw_current = fetch_solar_wind_data()
            return _fallback_solar_wind_history(sw_current)
        
        plasma_by_time = {}
        for row in plasma_rows:
            dt = parse_swpc_datetime(row.get('time_tag'))
            if not dt:
                continue
            plasma_by_time[dt] = {
                'speed': _swpc_numeric(row.get('speed')),
                'density': _swpc_numeric(row.get('density')),
            }

        mag_by_time = {}
        for row in mag_rows:
            dt = parse_swpc_datetime(row.get('time_tag'))
            if not dt:
                continue
            mag_by_time[dt] = {
                'bz': _swpc_numeric(row.get('bz_gsm')),
                'bt': _swpc_numeric(row.get('bt')),
            }

        common_times = sorted(set(plasma_by_time) & set(mag_by_time))
        if len(common_times) == 0:
            print("Warning: No overlapping plasma/magnetic solar wind timestamps")
            return _fallback_solar_wind_history(fetch_solar_wind_data())

        times = []
        speeds = []
        densities = []
        bzs = []
        bts = []
        
        for dt in common_times:
            try:
                times.append(dt)
                speeds.append(plasma_by_time[dt]['speed'])
                densities.append(plasma_by_time[dt]['density'])
                bzs.append(mag_by_time[dt]['bz'])
                bts.append(mag_by_time[dt]['bt'])
            except Exception as row_error:
                print(f"Error parsing row {dt.isoformat()}: {row_error}")
                continue
        
        if len(times) == 0:
            print("Warning: No valid historical data parsed")
            return _fallback_solar_wind_history(fetch_solar_wind_data())

        time_interval_minutes = 1
        if len(times) >= 2:
            time_deltas = [
                (times[index + 1] - times[index]).total_seconds() / 60.0
                for index in range(len(times) - 1)
                if times[index + 1] > times[index]
            ]
            if time_deltas:
                time_interval_minutes = max(1, int(round(float(np.median(time_deltas)))))
        
        # Interpolate gaps in the data to smooth over temporary dropouts
        # Use 10-minute max gap to avoid interpolating over extended outages
        speeds = interpolate_gaps(speeds, max_gap_minutes=10, time_interval_minutes=time_interval_minutes)
        densities = interpolate_gaps(densities, max_gap_minutes=10, time_interval_minutes=time_interval_minutes)
        bzs = interpolate_gaps(bzs, max_gap_minutes=10, time_interval_minutes=time_interval_minutes)
        bts = interpolate_gaps(bts, max_gap_minutes=10, time_interval_minutes=time_interval_minutes)
        
        return {
            'times': times,
            'speeds': speeds,
            'densities': densities,
            'bzs': bzs,
            'bts': bts
        }
    except Exception as e:
        print(f"Error fetching solar wind history: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_solar_regions(data):
    """Parse Solar Regions JSON data"""
    if not data:
        return []
        
    # Find the latest observed date
    latest_date = None
    for entry in data:
        obs_date = entry.get('observed_date')
        if obs_date:
            if latest_date is None or obs_date > latest_date:
                latest_date = obs_date
    
    if not latest_date:
        return []
        
    regions = []
    # Map mag class abbreviations to full names
    mag_map = {
        'A': 'Alpha',
        'B': 'Beta',
        'G': 'Gamma',
        'BG': 'Beta-Gamma',
        'BD': 'Beta-Delta',
        'GD': 'Gamma-Delta',
        'BGD': 'Beta-Gamma-Delta'
    }
    
    for entry in data:
        if entry.get('observed_date') == latest_date:
            mag_class = entry.get('mag_class')
            mag_type = mag_map.get(mag_class, mag_class) if mag_class else 'Alpha' # Default to Alpha if missing? Or just empty.
            
            # If mag_class is None, it might be a plage or decaying region.
            # The frontend expects a mag_type.
            if not mag_type:
                mag_type = 'Alpha'
            
            # Filter out regions with no spot class (plage/decayed)
            if not entry.get('spot_class'):
                continue

            regions.append({
                'number': str(entry.get('region', '')),
                'observed_date': entry.get('observed_date') or latest_date,
                'location': entry.get('location') or '',
                'lo': str(entry.get('longitude') or ''),
                'area': str(entry.get('area') or ''),
                'z': entry.get('spot_class') or '',
                'll': str(entry.get('latitude') or ''),
                'nn': '', 
                'mag_class': mag_class or '',
                'mag_type': mag_type,
                'c_flares': entry.get('c_xray_events', 0),
                'm_flares': entry.get('m_xray_events', 0),
                'x_flares': entry.get('x_xray_events', 0),
                'c_flare_probability': entry.get('c_flare_probability'),
                'm_flare_probability': entry.get('m_flare_probability'),
                'x_flare_probability': entry.get('x_flare_probability')
            })
            
    # Sort by region number
    regions.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 0, reverse=True)
    
    return regions


def build_active_region_watchlist(regions):
    """Build a factual watchlist from SWPC region fields only."""
    if not regions:
        return []

    complexity_rank = {
        'Alpha': 0,
        'Beta': 1,
        'Gamma': 2,
        'Beta-Gamma': 3,
        'Beta-Delta': 4,
        'Gamma-Delta': 5,
        'Beta-Gamma-Delta': 6,
    }

    watchlist = []
    for region in regions:
        mag_type = region.get('mag_type') or ''
        area = _safe_int(region.get('area'), 0)
        m_prob = _safe_int(region.get('m_flare_probability'), 0)
        x_prob = _safe_int(region.get('x_flare_probability'), 0)
        m_flares = _safe_int(region.get('m_flares'), 0)
        x_flares = _safe_int(region.get('x_flares'), 0)
        complex_region = 'Gamma' in mag_type or 'Delta' in mag_type

        high_probability = x_prob >= 5 or m_prob >= 10
        complex_with_probability = complex_region and (m_prob > 0 or x_prob > 0)
        recent_major_flare = m_flares > 0 or x_flares > 0
        include = (
            high_probability or
            complex_with_probability or
            recent_major_flare
        )
        if not include:
            continue

        reasons = []
        if x_prob >= 5:
            reasons.append(f'SWPC X {x_prob}%')
        if m_prob >= 10:
            reasons.append(f'SWPC M {m_prob}%')
        elif m_prob > 0 and complex_region:
            reasons.append(f'SWPC M {m_prob}%')
        if complex_region:
            reasons.append(f'{mag_type} ')
        if x_flares > 0:
            reasons.append(f'{x_flares} recent X flare{"s" if x_flares != 1 else ""}')
        if m_flares > 0:
            reasons.append(f'{m_flares} recent M flare{"s" if m_flares != 1 else ""}')
        if area >= 650:
            reasons.append(f'{area} MSH area')

        item = dict(region)
        item['watch_reasons'] = reasons
        item['watch_level'] = (
            'Primary watch'
            if x_prob >= 5 or m_prob >= 20 or x_flares > 0
            else 'Elevated watch'
        )
        item['watch_basis'] = 'SWPC probabilities, magnetic class, and observed flare counts'
        item['_sort'] = (
            x_prob,
            m_prob,
            complexity_rank.get(mag_type, 0),
            area,
            x_flares,
            m_flares,
        )
        watchlist.append(item)

    watchlist.sort(key=lambda item: item['_sort'], reverse=True)
    watchlist = watchlist[:5]
    for item in watchlist:
        item.pop('_sort', None)
    return watchlist


def parse_returning_regions(data):
    """Parse Regions Due to Return from SWPC solar_regions JSON if present.
    This function tries multiple heuristics to locate returning-region information
    and returns a list of dicts with at least 'number' and optional 'expected_return_date'.
    """
    if not data:
        return []

    # If payload is a dict with explicit key
    if isinstance(data, dict):
        # Common possible keys
        for key in ('regions_due_to_return', 'returning_regions', 'regions_to_return'):
            if key in data and isinstance(data[key], list):
                out = []
                for entry in data[key]:
                    out.append({
                        'number': str(entry.get('region') or entry.get('number') or ''),
                        'expected_return_date': entry.get('expected_return_date') or entry.get('return_date') or ''
                    })
                return out

    # If payload is list (regular solar regions file), look for entries flagged as returning
    if isinstance(data, list):
        candidates = []
        for entry in data:
            # Some feeds may mark entries with flags/notes
            note = (entry.get('note') or '')
            status = (entry.get('status') or '')
            if isinstance(status, str) and 'return' in status.lower():
                candidates.append(entry)
                continue
            if isinstance(note, str) and 'return' in note.lower():
                candidates.append(entry)
                continue
            # Explicit boolean flag
            if entry.get('due_to_return') or entry.get('returning'):
                candidates.append(entry)

        # Map candidates to minimal structure
        out = []
        for e in candidates:
            out.append({
                'number': str(e.get('region') or e.get('number') or ''),
                'expected_return_date': e.get('expected_return_date') or e.get('return_date') or ''
            })
        return out

    return []


def fetch_solar_region_summary_text(timeout=10):
    """Attempt to fetch the SWPC Solar Region Summary text product from known endpoints.
    Returns the raw text if successful, otherwise None.
    """
    for url in SOLAR_REGION_SUMMARY_URLS:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text and 'Regions Due to Return' in r.text:
                return r.text
        except requests.RequestException:
            continue
    return None


def parse_returning_regions_from_summary_text(text):
    """Parse the 'II. Regions Due to Return' section from the SWPC Solar Region Summary text.
    Returns a list of {'number': str, 'expected_return_date': str (may be blank)}
    """
    if not text:
        return []

    lines = text.splitlines()
    start_idx = None
    header_line = ''
    for i, line in enumerate(lines):
        if re.search(r'II\.|II\s*\.', line) and 'Regions' in line and 'Return' in line:
            start_idx = i
            header_line = line.strip()
            break

    if start_idx is None:
        # Try simpler match
        for i, line in enumerate(lines):
            if 'Regions Due to Return' in line:
                start_idx = i
                header_line = line.strip()
                break

    if start_idx is None:
        return []

    # The block usually has a header row like: Nmbr Lat    Lo
    # and then rows like: 4343 S09    113
    out = []
    # scan forward from start_idx to find rows that look like region entries
    for j in range(start_idx + 1, min(len(lines), start_idx + 50)):
        l = lines[j].strip()
        # stop if next Roman numeral section begins (e.g., 'III.')
        if re.match(r'^[IVX]+\.', l):
            break
        # skip empty and header-like lines
        if not l or l.lower().startswith('nmbr') or l.lower().startswith('number'):
            continue
        # Accept lines that start with a number (region number)
        m = re.match(r'^(\d{3,5})\s+([NS]\d{1,2})\s+(-?\d{1,3})', l)
        if m:
            number = m.group(1)
            lat = m.group(2)
            lo = m.group(3)
            out.append({'number': str(number), 'expected_return_date': header_line})
            continue
        # Some lines may be tab/space separated with 3 columns
        parts = re.split(r'\s+', l)
        if len(parts) >= 2 and parts[0].isdigit():
            number = parts[0]
            out.append({'number': str(number), 'expected_return_date': header_line})

    return out

def fetch_json_with_retry(url, retries=3, timeout=30):
    """Fetch JSON from URL with retry logic"""
    for i in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if i == retries - 1:
                print(f"Failed to fetch {url} after {retries} attempts: {e}")
                return None
            time.sleep(1)
    return None

def fetch_solar_data():
    """Fetch solar activity data (X-rays, Protons, Sunspots)"""
    def safe_get_json(url):
        r = requests.get(url, timeout=10)
        try:
            return r.json()
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as e:
            if "Extra data" in str(e) and hasattr(e, 'pos'):
                print(f"JSON Extra data detected for {url}. Attempting recovery.")
                return json.loads(r.text[:e.pos])
            raise e

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                'xray': executor.submit(safe_get_json, GOES_XRAY_URL),
                'xray_1day': executor.submit(safe_get_json, GOES_XRAY_1DAY_URL),
                'proton': executor.submit(safe_get_json, GOES_PROTON_URL),
                'regions': executor.submit(fetch_json_with_retry, SOLAR_REGIONS_URL, 3, 30),
                'summary': executor.submit(fetch_solar_region_summary_text)
            }

            results = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"Error fetching {key} for solar data: {e}")
                    results[key] = None

        xray_data = results['xray']
        xray_1day_data = results['xray_1day']
        proton_data = results['proton']
        regions_data = results['regions']
        sunspots = parse_solar_regions(regions_data)
        watchlist_regions = build_active_region_watchlist(sunspots)
        returning = parse_returning_regions(regions_data)
        region_updated = None
        for region in sunspots:
            observed_date = region.get('observed_date')
            if observed_date and (region_updated is None or observed_date > region_updated):
                region_updated = observed_date

        # If JSON didn't include explicit returning list, try parsing SWPC Solar Region Summary text
        if not returning:
            try:
                summary_text = results.get('summary')
                if summary_text:
                    parsed = parse_returning_regions_from_summary_text(summary_text)
                    if parsed:
                        returning = parsed
            except Exception as _:
                # ignore text parsing failures
                returning = returning
        
        return {
            'xray': xray_data,
            'xray_1day': xray_1day_data,
            'proton': proton_data,
            'sunspots': sunspots,
            'watchlist_regions': watchlist_regions,
            'solar_regions_updated': region_updated,
            'returning_regions': returning
        }
    except Exception as e:
        print(f"Error fetching solar data: {e}")
        return None

def fetch_kp_index():
    """Fetch current Kp index from forecast endpoint (has most recent observed data)"""
    try:
        # Use forecast endpoint which has more up-to-date observed values
        KP_FORECAST_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        response = requests.get(KP_FORECAST_URL, timeout=10)
        kp_data = response.json()
        
        # Find the most recent OBSERVED Kp value (not forecast)
        if len(kp_data) > 1:
            latest_observed = None
            latest_time = None
            
            # Iterate backwards to find most recent observed value
            for row in reversed(kp_data[1:]):  # Skip header
                if len(row) >= 3 and row[2] == 'observed':
                    try:
                        kp_value = float(row[1])
                        kp_time = row[0]
                        latest_observed = kp_value
                        latest_time = kp_time
                        break
                    except (ValueError, TypeError):
                        continue
            
            if latest_observed is not None:
                print(f"[KP_INDEX] Retrieved latest observed Kp: {latest_observed} at {latest_time}")
                return {
                    'time': latest_time,
                    'kp': latest_observed
                }
            
        print("[KP_INDEX] No observed Kp data available")
        return None
    except Exception as e:
        print(f"[KP_INDEX] Error fetching Kp index: {e}")
        return None

def fetch_noaa_scales():
    """Fetch NOAA space weather scales"""
    try:
        response = requests.get(SCALES_URL, timeout=10)
        scales_data = response.json()
        
        # Get current conditions (index "0")
        current = scales_data.get("0", {})
        
        return {
            'time': f"{current.get('DateStamp', '')} {current.get('TimeStamp', '')}",
            'g_scale': current.get('G', {}).get('Scale', '0'),
            'g_text': current.get('G', {}).get('Text', 'none'),
            'r_scale': current.get('R', {}).get('Scale', '0'),
            's_scale': current.get('S', {}).get('Scale', '0')
        }
    except Exception as e:
        print(f"Error fetching NOAA scales: {e}")
        return None


def _safe_int(value, default=0):
    try:
        return int(float(str(value).strip().rstrip('%')))
    except (TypeError, ValueError, AttributeError):
        return default


def parse_swpc_datetime(value):
    """Parse SWPC/DONKI timestamps into timezone-aware UTC datetimes."""
    if not value or value == 'N/A':
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    normalized = normalized.replace(' ', 'T')
    if normalized.endswith('Z'):
        normalized = normalized[:-1] + '+00:00'

    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def format_utc_display(dt_value):
    if not dt_value:
        return 'Unknown'
    return dt_value.astimezone(timezone.utc).strftime('%b %d, %Y %H:%M UTC')


def normalize_flare_class(raw_value):
    normalized = re.sub(r'\s+', '', str(raw_value or '').upper())
    match = re.match(r'^([ABCMX])(\d+(?:\.\d+)?)', normalized)
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}"


def flare_class_to_flux(flare_class):
    normalized = normalize_flare_class(flare_class)
    if not normalized:
        return None

    scale = {
        'A': 1e-8,
        'B': 1e-7,
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4,
    }

    try:
        return float(normalized[1:]) * scale[normalized[0]]
    except (KeyError, ValueError):
        return None


def flux_to_flare_class(flux):
    if flux is None:
        return 'Unknown'

    try:
        flux_value = float(flux)
    except (TypeError, ValueError):
        return 'Unknown'

    if flux_value < 1e-7:
        return f"A{flux_value * 1e8:.2f}"
    if flux_value < 1e-6:
        return f"B{flux_value * 1e7:.2f}"
    if flux_value < 1e-5:
        return f"C{flux_value * 1e6:.2f}"
    if flux_value < 1e-4:
        return f"M{flux_value * 1e5:.2f}"
    return f"X{flux_value * 1e4:.2f}"


def flare_flux_to_r_scale(flux):
    if flux is None:
        return 0
    if flux >= 2e-3:
        return 5
    if flux >= 1e-3:
        return 4
    if flux >= 1e-4:
        return 3
    if flux >= 5e-5:
        return 2
    if flux >= 1e-5:
        return 1
    return 0


def proton_flux_to_s_scale(flux):
    if flux is None:
        return 0
    if flux >= 100000:
        return 5
    if flux >= 10000:
        return 4
    if flux >= 1000:
        return 3
    if flux >= 100:
        return 2
    if flux >= 10:
        return 1
    return 0


def get_flare_palette(flare_class):
    class_letter = (normalize_flare_class(flare_class) or 'C1.0')[0]
    if class_letter == 'X':
        return {
            'base': '#b91c1c',
            'accent': '#ef4444',
            'soft': 'rgba(239, 68, 68, 0.18)',
            'text': '#fff5f5',
        }
    if class_letter == 'M':
        return {
            'base': '#c2410c',
            'accent': '#f97316',
            'soft': 'rgba(249, 115, 22, 0.18)',
            'text': '#fff7ed',
        }
    return {
        'base': '#ca8a04',
        'accent': '#facc15',
        'soft': 'rgba(250, 204, 21, 0.18)',
        'text': '#fefce8',
    }


def get_flare_impact_copy(flare_class):
    class_letter = (normalize_flare_class(flare_class) or 'C1.0')[0]
    if class_letter == 'X':
        return [
            'A solar flare is an eruption of solar particles. X-level flares are uncommon to rare.',
            'A flare of this magnitude can result in a radiation storm and notable radio blackouts.',
            'It is unknown if this flare produced a CME, but coronagraph imagery can help determine if any direct impacts to Earth are possible.',
        ]
    if class_letter == 'M':
        return [
            'A solar flare is an eruption of solar particles. M-level flares are relatively common.',
            'A flare of this magnitude can result in a radiation storm and radio blackouts.',
            'It is unknown if this flare produced a CME, but coronagraph imagery can help determine if any direct impacts to Earth are possible.',
        ]
    return [
        'A solar flare is an eruption of solar particles. C-level flares are very common.',
        'Little to no impacts are expected regarding radiation or radio blackouts.',
        'A CME is unlikely from a flare of this magnitude, but if one is launched it will likely be slow with minor to no effects here on Earth.',
    ]


def parse_solar_region_location(location):
    if not location:
        return None

    match = re.search(r'([NS])(\d{1,2})([EW])(\d{1,2})', str(location).upper())
    if not match:
        return None

    latitude = int(match.group(2)) * (1 if match.group(1) == 'N' else -1)
    longitude = int(match.group(4)) * (1 if match.group(3) == 'W' else -1)
    return latitude, longitude


def solar_region_to_disk_xy(location):
    coords = parse_solar_region_location(location)
    if not coords:
        return None

    latitude, longitude = coords
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    x = np.sin(lon_rad) * np.cos(lat_rad)
    y = -np.sin(lat_rad)
    return x, y


def build_flare_event_id(event_dt, flare_class, region_number=None):
    timestamp = event_dt.astimezone(timezone.utc).strftime('%Y%m%d%H%M') if event_dt else 'unknown'
    region_part = re.sub(r'[^0-9A-Za-z]+', '-', str(region_number or 'na'))
    class_part = re.sub(r'[^0-9A-Za-z]+', '-', normalize_flare_class(flare_class) or 'flare')
    return f"{timestamp}-{class_part}-{region_part}"


def fetch_remote_image(urls, timeout=10):
    for url in urls:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as exc:
            print(f"[FLARE_ALERT] Failed to load image {url}: {exc}")
    return None


def detect_recent_xray_flares(xray_rows, min_flux=1e-6, window=15):
    flare_candidates = []
    if not xray_rows:
        return flare_candidates

    long_channel = []
    for row in xray_rows:
        if row.get('energy') != '0.1-0.8nm':
            continue
        try:
            flux_value = float(row.get('flux'))
            time_value = parse_swpc_datetime(row.get('time_tag'))
            if not time_value or flux_value <= 0:
                continue
            long_channel.append({'time': time_value, 'flux': flux_value})
        except (TypeError, ValueError):
            continue

    for index in range(len(long_channel)):
        candidate = long_channel[index]
        if candidate['flux'] < min_flux:
            continue

        start_index = max(0, index - window)
        end_index = min(len(long_channel) - 1, index + window)
        if any(long_channel[compare_index]['flux'] > candidate['flux'] for compare_index in range(start_index, end_index + 1) if compare_index != index):
            continue

        onset_index = index
        end_event_index = index
        threshold = max(min_flux, candidate['flux'] * 0.35)

        while onset_index > 0 and long_channel[onset_index - 1]['flux'] >= threshold:
            onset_index -= 1
        while end_event_index < len(long_channel) - 1 and long_channel[end_event_index + 1]['flux'] >= threshold:
            end_event_index += 1

        flare_candidates.append({
            'time': candidate['time'],
            'peak_time': candidate['time'],
            'start_time': long_channel[onset_index]['time'],
            'end_time': long_channel[end_event_index]['time'],
            'flux': candidate['flux'],
            'event_class': flux_to_flare_class(candidate['flux']),
        })

    flare_candidates.sort(key=lambda item: item['peak_time'], reverse=True)
    return flare_candidates


def fetch_recent_donki_flares(days_back=5):
    cache_key = f'donki_flares_{days_back}'
    cached = get_cached(cache_key, max_age_seconds=600)
    if cached is not None:
        return cached

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    try:
        response = requests.get(
            'https://api.nasa.gov/DONKI/FLR',
            params={
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'api_key': NASA_API_KEY,
            },
            timeout=12,
        )
        if response.status_code != 200:
            print(f"[FLARE_ALERT] DONKI returned status {response.status_code}")
            set_cached(cache_key, [])
            return []

        donki_rows = response.json()
    except Exception as exc:
        print(f"[FLARE_ALERT] Failed to fetch DONKI flare events: {exc}")
        set_cached(cache_key, [])
        return []

    flare_rows = []
    for row in donki_rows:
        flare_class = normalize_flare_class(row.get('classType'))
        if not flare_class or flare_class[0] not in {'C', 'M', 'X'}:
            continue
        peak_dt = parse_swpc_datetime(row.get('peakTime'))
        if not peak_dt:
            continue
        flare_rows.append({
            'event_class': flare_class,
            'time': parse_swpc_datetime(row.get('beginTime')),
            'peak_time': peak_dt,
            'end_time': parse_swpc_datetime(row.get('endTime')),
            'region_number': str(row.get('activeRegionNum') or '').strip() or None,
            'source_location': row.get('sourceLocation'),
        })

    flare_rows.sort(key=lambda item: item['peak_time'], reverse=True)
    set_cached(cache_key, flare_rows)
    return flare_rows


def infer_source_region_metadata(regions, flare_class, preferred_region_number=None):
    if not regions:
        return None

    if preferred_region_number:
        for region in regions:
            if str(region.get('number')) == str(preferred_region_number):
                return region

    class_letter = (normalize_flare_class(flare_class) or 'C1.0')[0]
    target_key = {'X': 'x_flares', 'M': 'm_flares'}.get(class_letter, 'c_flares')
    complexity_rank = {
        'Alpha': 1,
        'Beta': 2,
        'Beta-Gamma': 3,
        'Beta-Delta': 4,
        'Gamma-Delta': 5,
        'Beta-Gamma-Delta': 6,
    }

    def region_rank(region):
        area_value = _safe_int(region.get('area'), 0)
        return (
            _safe_int(region.get(target_key), 0),
            _safe_int(region.get('x_flares'), 0),
            _safe_int(region.get('m_flares'), 0),
            _safe_int(region.get('c_flares'), 0),
            complexity_rank.get(region.get('mag_type'), 0),
            area_value,
        )

    return sorted(regions, key=region_rank, reverse=True)[0]


def fetch_latest_flare_alert(max_age_hours=FLARE_ALERT_MAX_AGE_HOURS):
    cache_key = 'latest_flare_alert'
    cached = get_cached(cache_key, max_age_seconds=180)
    if cached:
        return cached

    solar_data = fetch_solar_data() or {}
    flare_candidates = detect_recent_xray_flares(solar_data.get('xray_1day') or [])
    latest = flare_candidates[0] if flare_candidates else None
    if not latest:
        payload = {'active': False, 'flare': None}
        set_cached(cache_key, payload)
        return payload

    now = datetime.now(timezone.utc)
    scales = fetch_noaa_scales() or {}
    current_r_scale = _safe_int(scales.get('r_scale'), 0)
    flare_class = latest['event_class']
    donki_rows = fetch_recent_donki_flares()
    matched_donki = None
    for row in donki_rows:
        if row['event_class'][0] != flare_class[0]:
            continue
        time_delta = abs((row['peak_time'] - latest['peak_time']).total_seconds()) / 60.0
        if time_delta <= 120:
            matched_donki = row
            break

    region_number = matched_donki.get('region_number') if matched_donki else None
    region_meta = infer_source_region_metadata(solar_data.get('sunspots') or [], flare_class, region_number)
    if not region_number and region_meta and region_meta.get('number'):
        region_number = str(region_meta.get('number'))

    source_region = f"AR {region_number}" if region_number else 'Source region unavailable'
    location = None
    if matched_donki and matched_donki.get('source_location'):
        location = matched_donki['source_location']
    elif region_meta and region_meta.get('location'):
        location = region_meta.get('location')

    event_dt = latest['peak_time']
    peak_r_scale = max(current_r_scale, flare_flux_to_r_scale(latest['flux']))
    age_hours = max(0.0, (now - event_dt).total_seconds() / 3600.0)
    palette = get_flare_palette(flare_class)

    latest = {
        'id': build_flare_event_id(event_dt, flare_class, region_number),
        'event_class': flare_class,
        'class_letter': flare_class[0],
        'magnitude': flare_class[1:],
        'time': (matched_donki.get('time') if matched_donki and matched_donki.get('time') else latest['start_time']).isoformat() if (matched_donki and matched_donki.get('time')) or latest.get('start_time') else None,
        'peak_time': (matched_donki.get('peak_time') if matched_donki and matched_donki.get('peak_time') else latest['peak_time']).isoformat(),
        'end_time': (matched_donki.get('end_time') if matched_donki and matched_donki.get('end_time') else latest.get('end_time')).isoformat() if (matched_donki and matched_donki.get('end_time')) or latest.get('end_time') else None,
        'event_time_display': format_utc_display(event_dt),
        'peak_time_display': format_utc_display(event_dt),
        'source_region': source_region,
        'region_number': region_number,
        'location': location,
        'current_r_scale': current_r_scale,
        'peak_r_scale': peak_r_scale,
        'r_scale_label': f"R{peak_r_scale}",
        'flux': latest['flux'],
        'color': palette['accent'],
        'base_color': palette['base'],
        'age_minutes': int(round(age_hours * 60)),
        'age_hours': round(age_hours, 2),
        'summary_text': f"{flare_class} flare detected{f' from {source_region}' if region_number else ''}",
        'what_to_know': get_flare_impact_copy(flare_class),
    }

    payload = {
        'active': bool(latest and latest['age_hours'] <= max_age_hours),
        'flare': latest,
    }
    set_cached(cache_key, payload)
    return payload


def draw_flare_alert_graphic(flare_info):
    import matplotlib.dates as mdates
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.ticker import LogLocator, NullFormatter
    import textwrap

    flare_class = flare_info.get('event_class') or 'C1.0'
    palette = get_flare_palette(flare_class)
    figure_bg = '#080b10'
    panel_bg = '#05070a'
    panel_edge = '#d9e2ec'
    card_bg = '#0b1017'
    card_edge = (1, 1, 1, 0.08)
    bright_text = '#f8fafc'
    muted_text = '#94a3b8'
    soft_text = '#cbd5e1'
    fig = plt.figure(figsize=(14.8, 8.4), facecolor=figure_bg)
    gs = GridSpec(22, 24, figure=fig, left=0.028, right=0.985, top=0.905, bottom=0.055, wspace=0.34, hspace=0.54)

    peak_time_display = flare_info.get('peak_time_display') or flare_info.get('event_time_display') or format_utc_display(datetime.now(timezone.utc))
    source_region = flare_info.get('source_region') or 'Source region unavailable'
    location_label = flare_info.get('location') or 'Location unavailable'
    age_hours = flare_info.get('age_hours')
    if isinstance(age_hours, (int, float)):
        age_label = f"Detected {age_hours:.1f}h ago" if age_hours >= 1 else f"Detected {int(round(age_hours * 60))}m ago"
    else:
        age_label = 'Recent event'

    fig.text(
        0.035, 0.955, 'NEW FLARE EVENT', ha='left', va='center', fontsize=28,
        color=palette['text'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.08', facecolor=palette['base'], edgecolor=palette['accent'], linewidth=1.5)
    )
    fig.text(0.56, 0.958, peak_time_display, ha='left', va='center', fontsize=19, color=bright_text, fontweight='bold')
    fig.text(0.56, 0.927, f"{source_region} | {age_label}", ha='left', va='center', fontsize=10.5, color=muted_text, fontweight='bold')
    fig.text(
        0.965, 0.955, flare_class, ha='right', va='center', fontsize=28,
        color=palette['text'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.08', facecolor=palette['base'], edgecolor=palette['accent'], linewidth=1.5)
    )

    ax_goes = fig.add_subplot(gs[1:11, 0:8])
    ax_source = fig.add_subplot(gs[1:11, 9:17])
    ax_info = fig.add_subplot(gs[1:19, 18:24])
    ax_xray = fig.add_subplot(gs[11:18, 0:9])
    ax_proton = fig.add_subplot(gs[11:18, 9:18])

    content_axes = [ax_goes, ax_source, ax_info, ax_xray, ax_proton]
    for axis in content_axes:
        axis.set_facecolor(panel_bg)
        for spine in axis.spines.values():
            spine.set_color(panel_edge)
            spine.set_linewidth(1.15)

    def style_chart_axis(axis):
        axis.set_facecolor(panel_bg)
        axis.tick_params(axis='x', labelsize=10, colors=soft_text, pad=4)
        axis.tick_params(axis='y', labelsize=10, colors=soft_text, pad=6)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color(panel_edge)
            spine.set_linewidth(1.05)

    def add_chart_header(axis, title, value_text=None, value_color=bright_text):
        axis.text(0.015, 0.972, title, transform=axis.transAxes, ha='left', va='top',
                  fontsize=16, color=palette['accent'], fontweight='bold')
        if value_text:
            axis.text(
                0.985, 0.972, value_text, transform=axis.transAxes, ha='right', va='top',
                fontsize=10, color=value_color, fontweight='bold', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.08', facecolor=card_bg, edgecolor=card_edge, linewidth=0.9)
            )

    def add_xray_gridlines(axis):
        major_lines = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        minor_lines = []
        for base in major_lines:
            for multiplier in range(2, 10):
                minor_lines.append(base * multiplier)

        for value in minor_lines:
            axis.axhline(value, color=(1, 1, 1, 0.05), linewidth=0.7, zorder=0)
        for value in major_lines:
            axis.axhline(value, color=(1, 1, 1, 0.14), linewidth=0.9, zorder=0)

    def add_info_bullet(axis, top_y, text_value):
        wrapped = textwrap.fill(text_value, width=28)
        line_count = wrapped.count('\n') + 1
        bullet_center_y = top_y - 0.017
        axis.add_patch(
            mpatches.Circle((0.078, bullet_center_y), 0.009, transform=axis.transAxes,
                            facecolor=palette['accent'], edgecolor='none', zorder=3)
        )
        axis.text(0.108, top_y, wrapped, ha='left', va='top', fontsize=9.8,
                  color=bright_text, linespacing=1.28, transform=ax_info.transAxes)
        return top_y - (0.031 * line_count) - 0.028

    def add_info_card(axis, x, y, width, height, label, value, secondary=None, value_color=bright_text, value_size=16, mono=False):
        card = mpatches.FancyBboxPatch(
            (x, y - height), width, height,
            boxstyle='round,pad=0.014,rounding_size=0.03',
            transform=axis.transAxes,
            facecolor=card_bg,
            edgecolor=card_edge,
            linewidth=0.9,
        )
        axis.add_patch(card)
        axis.text(x + 0.025, y - 0.018, label, ha='left', va='top', fontsize=8.8,
                  color=muted_text, fontweight='bold', transform=axis.transAxes)
        axis.text(x + 0.025, y - 0.063, value, ha='left', va='top', fontsize=value_size,
                  color=value_color, fontweight='bold', fontfamily='monospace' if mono else None, transform=axis.transAxes)
        if secondary:
            axis.text(x + 0.025, y - height + 0.03, secondary, ha='left', va='bottom', fontsize=10.2,
                      color=soft_text, fontweight='bold', transform=axis.transAxes)

    def annotate_source_region(axis, image_width, image_height, region_x, region_y, label):
        label_dx = 18 if region_x < image_width * 0.28 else (-18 if region_x > image_width * 0.72 else 0)
        label_dy = -18 if region_y < image_height * 0.28 else 18
        label_ha = 'left' if label_dx > 0 else 'right' if label_dx < 0 else 'center'
        label_va = 'top' if label_dy < 0 else 'bottom'

        axis.annotate(
            label,
            xy=(region_x, region_y),
            xytext=(label_dx, label_dy),
            textcoords='offset points',
            ha=label_ha,
            va=label_va,
            fontsize=11.2,
            color=bright_text,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18,rounding_size=0.05', facecolor=(8 / 255, 11 / 255, 16 / 255, 0.88), edgecolor='none'),
            arrowprops=dict(arrowstyle='-', color=palette['accent'], linewidth=1.2, shrinkA=4, shrinkB=10),
            annotation_clip=True,
        )

    def draw_image_placeholder(axis, headline, detail):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.add_patch(mpatches.Circle((0.5, 0.55), 0.295, transform=axis.transAxes,
                                       facecolor='#121922', edgecolor=(1, 1, 1, 0.08), linewidth=1.2))
        axis.add_patch(mpatches.Circle((0.5, 0.55), 0.255, transform=axis.transAxes,
                                       facecolor=(250 / 255, 204 / 255, 21 / 255, 0.09), edgecolor=palette['accent'], linewidth=1.0))
        axis.add_patch(mpatches.Circle((0.5, 0.55), 0.05, transform=axis.transAxes,
                                       facecolor=(250 / 255, 204 / 255, 21 / 255, 0.26), edgecolor='none'))
        axis.text(0.5, 0.17, headline, ha='center', va='center', fontsize=13.5, color=bright_text, fontweight='bold', transform=axis.transAxes)
        axis.text(0.5, 0.11, detail, ha='center', va='center', fontsize=10.2, color=muted_text, transform=axis.transAxes)

    goes_image = fetch_remote_image([
        'https://services.swpc.noaa.gov/images/animations/suvi/primary/131/latest.png',
        'https://services.swpc.noaa.gov/images/animations/suvi/secondary/131/latest.png',
    ])
    if goes_image:
        ax_goes.imshow(goes_image)
        ax_goes.text(0.02, 0.035, 'GOES-19 SUVI composite 131A', ha='left', va='center', fontsize=9.2,
                     color=bright_text, transform=ax_goes.transAxes,
                     bbox=dict(boxstyle='round,pad=0.18,rounding_size=0.06', facecolor=(5 / 255, 7 / 255, 10 / 255, 0.84), edgecolor='none'))
    else:
        draw_image_placeholder(ax_goes, 'GOES imagery unavailable', 'Waiting for the latest SWPC frame')
    ax_goes.set_title('GOES Imagery', color=bright_text, fontsize=20, fontweight='bold', pad=8)
    ax_goes.set_xticks([])
    ax_goes.set_yticks([])

    source_image = fetch_remote_image([
        'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMII.jpg',
        'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIBC.jpg',
        'https://soho.nascom.nasa.gov/data/realtime/hmi_igr/1024/latest.jpg',
    ])
    if source_image:
        ax_source.imshow(source_image)
        disk_xy = solar_region_to_disk_xy(flare_info.get('location'))
        if disk_xy:
            height, width = source_image.height, source_image.width
            radius = min(width, height) * 0.44
            center_x = width / 2
            center_y = height / 2
            region_x = center_x + disk_xy[0] * radius
            region_y = center_y + disk_xy[1] * radius
            box_size = max(70, width * 0.09)
            rect = mpatches.Rectangle(
                (region_x - box_size / 2, region_y - box_size / 2),
                box_size, box_size, fill=False, edgecolor=palette['accent'], linewidth=2.1
            )
            ax_source.add_patch(rect)
            ax_source.scatter([region_x], [region_y], s=24, color=palette['accent'], edgecolors=bright_text, linewidths=0.7, zorder=4)
            annotate_source_region(ax_source, width, height, region_x, region_y, source_region)
        ax_source.text(0.02, 0.035, 'Latest continuum / intensity view', ha='left', va='center', fontsize=9.2,
                       color=bright_text, transform=ax_source.transAxes,
                       bbox=dict(boxstyle='round,pad=0.18,rounding_size=0.06', facecolor=(5 / 255, 7 / 255, 10 / 255, 0.84), edgecolor='none'))
    else:
        draw_image_placeholder(ax_source, source_region, location_label)
        if flare_info.get('location'):
            ax_source.add_patch(mpatches.Rectangle((0.43, 0.48), 0.14, 0.14, transform=ax_source.transAxes,
                                                   fill=False, edgecolor=palette['accent'], linewidth=1.8))
    ax_source.set_title('Source Region', color=bright_text, fontsize=20, fontweight='bold', pad=8)
    ax_source.set_xticks([])
    ax_source.set_yticks([])

    r_scale_label = flare_info.get('r_scale_label') or 'R0'
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    ax_info.add_patch(mpatches.Rectangle((0, 0.905), 1, 0.095, transform=ax_info.transAxes,
                                         facecolor=(250 / 255, 204 / 255, 21 / 255, 0.06), edgecolor='none'))
    ax_info.text(0.05, 0.95, 'WHAT TO KNOW:', ha='left', va='top', fontsize=20, color=bright_text, fontweight='bold', transform=ax_info.transAxes)
    source_region_display = textwrap.shorten(source_region, width=18, placeholder='...')
    add_info_card(ax_info, 0.05, 0.865, 0.27, 0.105, 'FLARE', flare_class, value_color=palette['accent'], value_size=18)
    add_info_card(ax_info, 0.35, 0.865, 0.43, 0.105, 'SOURCE', source_region_display, value_color=bright_text, value_size=14.2)
    add_info_card(ax_info, 0.81, 0.865, 0.14, 0.105, 'R-SCALE', r_scale_label, value_color=bright_text, value_size=16.5, mono=True)
    add_info_card(ax_info, 0.05, 0.72, 0.90, 0.105, 'PEAK TIME', peak_time_display, value_color=bright_text, value_size=13.1)
    ax_info.text(0.05, 0.585, 'LOCATION', ha='left', va='top', fontsize=9.3, color=muted_text, fontweight='bold', transform=ax_info.transAxes)
    ax_info.text(0.95, 0.585, age_label, ha='right', va='top', fontsize=9.2, color=muted_text, fontweight='bold', transform=ax_info.transAxes)
    location_text = textwrap.fill(location_label, width=28)
    location_line_count = location_text.count('\n') + 1
    ax_info.text(0.05, 0.548, location_text, ha='left', va='top', fontsize=10.4,
                 color=soft_text, fontweight='bold', transform=ax_info.transAxes)

    bullet_y = 0.50 - (0.03 * max(0, location_line_count - 1))
    for bullet in get_flare_impact_copy(flare_class):
        bullet_y = add_info_bullet(ax_info, bullet_y, bullet)

    ax_info.text(0.05, 0.02, textwrap.fill('R scale is estimated from the flare peak flux and the latest NOAA radio-blackout scale.', width=34), ha='left', va='bottom', fontsize=7.4, color=muted_text, transform=ax_info.transAxes)

    xray_data = []
    try:
        response = requests.get(GOES_XRAY_URL, timeout=12)
        response.raise_for_status()
        xray_data = response.json()
    except Exception as exc:
        print(f"[FLARE_ALERT] Failed to fetch X-ray chart data: {exc}")

    long_xray = []
    for entry in xray_data:
        if entry.get('energy') != '0.1-0.8nm':
            continue
        try:
            flux_value = float(entry.get('flux'))
            if flux_value <= 0:
                continue
            time_value = parse_swpc_datetime(entry.get('time_tag'))
            if not time_value:
                continue
            long_xray.append((time_value, flux_value))
        except (TypeError, ValueError):
            continue

    style_chart_axis(ax_xray)
    if long_xray:
        times = [item[0] for item in long_xray]
        values = [item[1] for item in long_xray]
        ax_xray.axhspan(1e-8, 1e-6, facecolor=(16 / 255, 185 / 255, 129 / 255, 0.12), zorder=0)
        ax_xray.axhspan(1e-6, 1e-5, facecolor=(234 / 255, 179 / 255, 8 / 255, 0.12), zorder=0)
        ax_xray.axhspan(1e-5, 1e-4, facecolor=(249 / 255, 115 / 255, 22 / 255, 0.12), zorder=0)
        ax_xray.axhspan(1e-4, 1e-3, facecolor=(220 / 255, 38 / 255, 38 / 255, 0.14), zorder=0)
        ax_xray.axhspan(1e-3, 1e-2, facecolor=(168 / 255, 85 / 255, 247 / 255, 0.14), zorder=0)
        add_xray_gridlines(ax_xray)
        ax_xray.grid(True, which='major', axis='x', color='#334155', alpha=0.28, linewidth=0.7)
        ax_xray.plot(times, values, color='#34d399', linewidth=2.35, solid_capstyle='round', zorder=4)
        ax_xray.fill_between(times, values, 1e-8, color=(52 / 255, 211 / 255, 153 / 255, 0.10))
        peak_time = parse_swpc_datetime(flare_info.get('peak_time') or flare_info.get('time'))
        peak_flux = flare_info.get('flux')
        if peak_time and times[0] <= peak_time <= times[-1]:
            ax_xray.axvline(peak_time, color=palette['accent'], linewidth=1.5, linestyle='--', alpha=0.92)
            if isinstance(peak_flux, (int, float)) and peak_flux > 0:
                ax_xray.scatter([peak_time], [peak_flux], s=34, color=palette['accent'], edgecolors=bright_text, linewidths=0.8, zorder=5)
                label_offset_x = -34 if peak_time >= times[-1] - timedelta(minutes=45) else 8
                label_align = 'right' if label_offset_x < 0 else 'left'
                ax_xray.annotate('Peak', xy=(peak_time, peak_flux), xytext=(label_offset_x, 10), textcoords='offset points',
                                 ha=label_align,
                                 fontsize=9.4, color=palette['accent'], fontweight='bold')
        ax_xray.set_yscale('log')
        ax_xray.set_ylim(1e-8, 1e-2)
        ax_xray.set_xlim(times[0], times[-1])
        ax_xray.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
        ax_xray.set_yticklabels(['A', 'B', 'C', 'M', 'X', 'X10'])
        ax_xray.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_xray.yaxis.set_minor_formatter(NullFormatter())
        ax_xray.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_xray.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        add_chart_header(ax_xray, 'GOES X-ray Flux', f"Current {flux_to_flare_class(values[-1])}")
    else:
        ax_xray.text(0.5, 0.5, 'X-ray chart unavailable', ha='center', va='center', color=muted_text, fontsize=13, transform=ax_xray.transAxes)
        ax_xray.set_xticks([])
        ax_xray.set_yticks([])
        add_chart_header(ax_xray, 'GOES X-ray Flux')

    proton_data = []
    try:
        response = requests.get(GOES_PROTON_URL, timeout=12)
        response.raise_for_status()
        proton_data = response.json()
    except Exception as exc:
        print(f"[FLARE_ALERT] Failed to fetch proton chart data: {exc}")

    proton_series = {'>=10 MeV': [], '>=50 MeV': [], '>=100 MeV': []}
    for entry in proton_data:
        energy = entry.get('energy')
        if energy not in proton_series:
            continue
        try:
            time_value = parse_swpc_datetime(entry.get('time_tag'))
            flux_value = float(entry.get('flux'))
            if not time_value or flux_value <= 0:
                continue
            proton_series[energy].append((time_value, flux_value))
        except (TypeError, ValueError):
            continue

    style_chart_axis(ax_proton)
    ax_proton.yaxis.tick_right()
    ax_proton.tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False, pad=4)
    has_proton_data = any(proton_series.values())
    if has_proton_data:
        proton_colors = {
            '>=10 MeV': '#facc15',
            '>=50 MeV': '#38bdf8',
            '>=100 MeV': '#a78bfa',
        }
        ax_proton.axhspan(10, 100, facecolor=(250 / 255, 204 / 255, 21 / 255, 0.055), zorder=0)
        ax_proton.axhspan(100, 1000, facecolor=(249 / 255, 115 / 255, 22 / 255, 0.05), zorder=0)
        ax_proton.axhspan(1000, 1e5, facecolor=(239 / 255, 68 / 255, 68 / 255, 0.05), zorder=0)
        for energy, points in proton_series.items():
            if not points:
                continue
            ax_proton.plot([point[0] for point in points], [point[1] for point in points], color=proton_colors[energy], linewidth=1.95, label=energy)
        peak_time = parse_swpc_datetime(flare_info.get('peak_time') or flare_info.get('time'))
        sample_series = proton_series['>=10 MeV'] or proton_series['>=50 MeV'] or proton_series['>=100 MeV']
        if peak_time and sample_series and sample_series[0][0] <= peak_time <= sample_series[-1][0]:
            ax_proton.axvline(peak_time, color=palette['accent'], linewidth=1.5, linestyle='--', alpha=0.92)
        ax_proton.axhline(10, color='#facc15', linewidth=1.0, linestyle='--', alpha=0.45)
        ax_proton.axhline(100, color='#f97316', linewidth=1.0, linestyle='--', alpha=0.45)
        ax_proton.axhline(1000, color='#ef4444', linewidth=1.0, linestyle='--', alpha=0.45)
        ax_proton.set_yscale('log')
        ax_proton.set_ylim(1e-2, 1e5)
        sample_series_times = [point[0] for point in sample_series]
        ax_proton.set_xlim(sample_series_times[0], sample_series_times[-1])
        ax_proton.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax_proton.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax_proton.grid(True, which='major', axis='x', color='#334155', alpha=0.28, linewidth=0.7)
        ax_proton.grid(True, which='major', axis='y', color='#334155', alpha=0.22, linewidth=0.7)
        ax_proton.legend(loc='upper left', bbox_to_anchor=(0.015, 0.84), fontsize=9, frameon=False, labelcolor=bright_text)
        label_transform = blended_transform_factory(ax_proton.transAxes, ax_proton.transData)
        ax_proton.text(0.985, 10, 'S1', transform=label_transform, ha='right', va='bottom', fontsize=8.8, color='#facc15', fontweight='bold')
        ax_proton.text(0.985, 100, 'S2', transform=label_transform, ha='right', va='bottom', fontsize=8.8, color='#f97316', fontweight='bold')
        ax_proton.text(0.985, 1000, 'S3+', transform=label_transform, ha='right', va='bottom', fontsize=8.8, color='#ef4444', fontweight='bold')
        latest_p10 = proton_series['>=10 MeV'][-1][1] if proton_series['>=10 MeV'] else None
        add_chart_header(ax_proton, 'GOES Proton Flux', f"Current S{proton_flux_to_s_scale(latest_p10)}" if latest_p10 is not None else 'Current S0')
    else:
        ax_proton.text(0.5, 0.5, 'Proton chart unavailable', ha='center', va='center', color=muted_text, fontsize=13, transform=ax_proton.transAxes)
        ax_proton.set_xticks([])
        ax_proton.set_yticks([])
        add_chart_header(ax_proton, 'GOES Proton Flux')

    fig.text(0.03, 0.03, 'NOAA SWPC live data | GOES and SDO imagery', ha='left', va='bottom', fontsize=9.5, color=muted_text, fontweight='bold')
    fig.text(0.97, 0.03, 'spacewx.weathertrackus.com', ha='right', va='bottom', fontsize=11, color='#e2e8f0', fontweight='bold')

    return _figure_to_png_buffer(
        fig,
        dpi=160,
        facecolor=figure_bg,
        edgecolor='none',
        bbox_inches='tight',
        pad_inches=0.22
    )

def fetch_ovation_data():
    """Fetch the latest OVATION auroral probability data from SWPC"""
    try:
        snapshot_payload = fetch_ovation_latest_snapshot()
        lons_north, lats_north, aurora_north, frame_label, _ = extract_ovation_north_frame(snapshot_payload)
        return lons_north, lats_north, aurora_north, frame_label
    except Exception as e:
        print(f"Error fetching OVATION data: {e}")
        return None, None, None, None


def parse_ovation_snapshot_key(timestamp_key):
    """Parse OVATION snapshot keys like YYYY-MM-DD_HHMM."""
    return datetime.strptime(timestamp_key, '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)


def parse_ovation_timestamp(timestamp_value):
    """Parse SWPC OVATION timestamps into UTC datetimes."""
    if isinstance(timestamp_value, datetime):
        return timestamp_value.astimezone(timezone.utc) if timestamp_value.tzinfo else timestamp_value.replace(tzinfo=timezone.utc)

    normalized = str(timestamp_value or '').strip()
    if not normalized:
        raise ValueError('Missing OVATION timestamp')

    normalized = normalized.replace(' ', 'T')
    if normalized.endswith('Z'):
        normalized = normalized[:-1] + '+00:00'

    parsed = datetime.fromisoformat(normalized)
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def format_ovation_snapshot_key(snapshot_time):
    """Format UTC datetimes into OVATION snapshot keys."""
    return snapshot_time.astimezone(timezone.utc).strftime('%Y-%m-%d_%H%M')


def extract_ovation_north_frame(snapshot_payload):
    """Extract northern OVATION grid arrays and metadata from a raw payload."""
    if not snapshot_payload or 'coordinates' not in snapshot_payload:
        raise ValueError('OVATION payload is missing coordinates')

    forecast_time = parse_ovation_timestamp(
        snapshot_payload.get('Forecast Time')
        or snapshot_payload.get('Observation Time')
        or snapshot_payload.get('generated_at')
    )

    coords = np.asarray(snapshot_payload['coordinates'], dtype=np.float32)
    lons = coords[:, 0]
    lats = coords[:, 1]
    aurora_vals = coords[:, 2]

    north_mask = lats > 0
    lons_north = lons[north_mask]
    lats_north = lats[north_mask]
    aurora_north = aurora_vals[north_mask]

    snapshot_key = format_ovation_snapshot_key(forecast_time)
    frame_label = forecast_time.strftime('%Y-%m-%d %H:%M UTC')
    return lons_north, lats_north, aurora_north, frame_label, snapshot_key


def fetch_ovation_latest_snapshot():
    """Fetch the latest OVATION payload from SWPC."""
    response = requests.get(OVATION_URL, timeout=10)
    response.raise_for_status()
    return response.json()

def fetch_hemispheric_power():
    """Fetch hemispheric power data from SWPC"""
    try:
        response = requests.get(HEMI_POWER_URL, timeout=10)
        lines = response.text.strip().split('\n')
        
        times = []
        powers = []
        
        for line in lines:
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Format: obs_time forecast_time north_power south_power
                        time_str = parts[0].replace('_', ' ')
                        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                        north_power = float(parts[2])
                        
                        times.append(time_obj)
                        powers.append(north_power)
                    except:
                        continue
        
        print(f"Hemispheric Power: Fetched {len(powers)} total data points from SWPC")
        return {'times': times, 'powers': powers}
    except Exception as e:
        print(f"Error fetching hemispheric power: {e}")
        return None

def fetch_goes_magnetometer():
    """Fetch GOES magnetometer data from SWPC for both satellites"""
    try:
        # Separate data by satellite
        goes18_times = []
        goes18_hp = []
        goes19_times = []
        goes19_hp = []

        def append_entry(entry):
            try:
                time_str = entry.get('time_tag')
                satellite = entry.get('satellite')
                hp = entry.get('Hp')

                if not time_str or hp is None:
                    return

                time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                hp_value = float(hp)

                if satellite == 18:
                    goes18_times.append(time_obj)
                    goes18_hp.append(hp_value)
                elif satellite == 19:
                    goes19_times.append(time_obj)
                    goes19_hp.append(hp_value)
            except Exception:
                return
        
        # SWPC's primary/secondary feeds can swap satellite assignment, so
        # always trust the record's `satellite` field instead of the URL.
        try:
            response_primary = requests.get(GOES_MAG_PRIMARY_URL, timeout=10)
            if response_primary.status_code == 200:
                try:
                    data_primary = response_primary.json()
                except json.JSONDecodeError:
                    print("Warning: GOES primary magnetometer JSON decode error, skipping")
                    data_primary = []
                
                for entry in data_primary:
                    append_entry(entry)
        except Exception as e:
            print(f"Warning: Could not fetch GOES primary magnetometer: {e}")
        
        try:
            response_secondary = requests.get(GOES_MAG_SECONDARY_URL, timeout=10)
            if response_secondary.status_code == 200:
                try:
                    data_secondary = response_secondary.json()
                except json.JSONDecodeError:
                    print("Warning: GOES secondary magnetometer JSON decode error, skipping")
                    data_secondary = []
                
                for entry in data_secondary:
                    append_entry(entry)
        except Exception as e:
            print(f"Warning: Could not fetch GOES secondary magnetometer: {e}")
        
        # Return last 2 hours of data (120 points at 1-minute cadence)
        if len(goes18_times) > 120:
            goes18_times = goes18_times[-120:]
            goes18_hp = goes18_hp[-120:]
        if len(goes19_times) > 120:
            goes19_times = goes19_times[-120:]
            goes19_hp = goes19_hp[-120:]
        
        print(f"GOES Magnetometer - GOES-18: {len(goes18_times)} points, GOES-19: {len(goes19_times)} points")
        
        return {
            'goes18_times': goes18_times,
            'goes18_hp': goes18_hp,
            'goes19_times': goes19_times,
            'goes19_hp': goes19_hp
        }
    except Exception as e:
        print(f"Error fetching GOES magnetometer data: {e}")
        return None

def get_aurora_likelihood(kp, g_scale):
    """Determine aurora viewing likelihood based on Kp and G-scale"""
    if kp is None:
        return "Unknown"
    
    if kp >= 7:
        return "Very High - Visible at mid-latitudes"
    elif kp >= 6:
        return "High - Visible at higher mid-latitudes"
    elif kp >= 5:
        return "Moderate - Visible at high latitudes"
    elif kp >= 4:
        return "Low - Visible near polar regions"
    else:
        return "Very Low - Minimal activity"

def get_condition_status(kp, speed, bz, g_scale):
    """Get overall aurora conditions status"""
    score = 0
    
    # Kp scoring
    if kp and kp >= 6:
        score += 3
    elif kp and kp >= 4:
        score += 2
    elif kp and kp >= 3:
        score += 1
    
    # Solar wind speed scoring
    if speed and speed >= 600:
        score += 2
    elif speed and speed >= 500:
        score += 1
    
    # Bz scoring (negative Bz is good for auroras)
    if bz and bz < -5:
        score += 3
    elif bz and bz < -2:
        score += 2
    elif bz and bz < 0:
        score += 1
    
    # G-scale scoring
    if g_scale:
        try:
            g_val = int(g_scale)
            if g_val >= 3:
                score += 2
            elif g_val >= 1:
                score += 1
        except:
            pass
    
    if score >= 7:
        return "Excellent"
    elif score >= 5:
        return "Good"
    elif score >= 3:
        return "Fair"
    else:
        return "Poor"

def generate_aurora_image():
    """Generate the complete aurora monitoring dashboard image"""
    # Fetch all data
    sw_current = fetch_solar_wind_data()
    sw_history = fetch_solar_wind_history()
    kp_data = fetch_kp_index()
    scales = fetch_noaa_scales()
    ovation_lons, ovation_lats, ovation_aurora, ovation_time = fetch_ovation_data()
    goes_mag = fetch_goes_magnetometer()
    
    kp_value = kp_data.get('kp') if kp_data else 0
    g_scale = scales.get('g_scale') if scales else '0'
    condition = get_condition_status(
        kp_value,
        sw_current.get('speed') if sw_current else None,
        sw_current.get('bz') if sw_current else None,
        g_scale
    )
    likelihood = get_aurora_likelihood(kp_value, g_scale)
    
    # Create figure with modern dark theme - enhanced with panel backgrounds
    fig = plt.figure(figsize=(24, 14), facecolor='#0f172a')
    # 5 rows: header, auroral oval (spans 1-3), speed, density, bt, bz (row 4)
    # Adjusted height ratios to be more uniform so charts fill their boxes better
    gs = GridSpec(5, 3, figure=fig, width_ratios=[1.5, 1, 1], height_ratios=[0.3, 1.1, 1.1, 1.1, 1.1],
                  hspace=0.3, wspace=0.3, left=0.04, right=0.97, top=0.94, bottom=0.05)
    
    # === HEADER SECTION ===
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    # Main title with subtle glow effect
    fig.text(0.02, 0.965, 'AURORA DASHBOARD', fontsize=34, color='#f8fafc', 
             fontweight='bold', va='top',
             path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground='#38bdf8', alpha=0.15)])
    
    # Add WTrUS watermark logo below title on left edge
    try:
        logo_img = Image.open('wtusredlogotransparentx.png')
        # Create axes for logo (x, y, width, height in figure coordinates)
        # Moved to left edge (x=0.000) and slightly lower (y=0.89)
        ax_logo = fig.add_axes([-0.008, 0.90, 0.12, 0.04])
        ax_logo.imshow(logo_img)
        ax_logo.axis('off')
    except Exception as e:
        print(f"Could not load watermark logo: {e}")
    
    # Timestamp
    fig.text(0.98, 0.965, f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}',
             fontsize=14, color='#94a3b8', ha='right', va='top', fontfamily='monospace')
    
    # Status badges - Use actual NOAA G-scale as primary
    # Align with the right-side graphs (columns 1 and 2 of GridSpec)
    status_y = 0.915
    
    # Get actual G-scale from NOAA (not calculated from Kp)
    g_val = int(g_scale) if g_scale and g_scale != '0' else 0
    
    # Create main status badge based on actual NOAA G-scale
    if g_val >= 1:
        storm_text = f'G{g_val} STORM'
        badge_colors = ['#88ff00', '#ffaa00', '#ff8800', '#ff4444', '#ff0000']
        badge_color = badge_colors[min(g_val-1, 4)]
    else:
        storm_text = f'Kp {kp_value:.1f}'
        badge_color = '#10b981' if kp_value < 3 else '#f59e0b' if kp_value < 4 else '#ef4444'
    
    # Position badges to align with graph columns - modern boxy design
    # Create gradient background effect for badges
    badge_style = dict(boxstyle='round,pad=0.8', facecolor='#1e293b', 
                      edgecolor=badge_color, linewidth=2, 
                      path_effects=[matplotlib.patheffects.withSimplePatchShadow(offset=(2, -2), 
                                                                                  shadow_rgbFace='#000000', 
                                                                                  alpha=0.3)])
    
    fig.text(0.585, status_y, storm_text, 
             fontsize=24, color=badge_color, fontweight='bold',
             bbox=badge_style, ha='center')
    
    # === HEMISPHERIC POWER GRAPH (Top right, replaces badge location) ===
    # Create small subplot for hemispheric power bar graph
    ax_hemi = fig.add_axes([0.69, 0.88, 0.20, 0.055])  # [x, y, width, height] - moved left and wider
    ax_hemi.set_facecolor('#1e293b')
    ax_hemi.patch.set_edgecolor('#334155')
    ax_hemi.patch.set_linewidth(1)
    
    # Fetch and display hemispheric power with color coding
    hemi_power = fetch_hemispheric_power()
    if hemi_power and len(hemi_power['powers']) > 0:
        powers = hemi_power['powers']
        times = hemi_power['times']
        
        # Filter to last 2 hours to match other graphs
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=2)
        
        # Filter data points within last 2 hours
        recent_indices = [i for i, t in enumerate(times) if t.replace(tzinfo=timezone.utc) >= cutoff_time]
        
        if recent_indices and len(recent_indices) >= 5:
            display_powers = [powers[i] for i in recent_indices]
            display_times = [times[i] for i in recent_indices]
        else:
            # If very sparse data in last 2 hours, expand to 6 hours or use last 60 points
            print(f"Warning: Only {len(recent_indices)} points in last 2 hours, expanding time window...")
            cutoff_time_extended = current_time - timedelta(hours=6)
            recent_indices = [i for i, t in enumerate(times) if t.replace(tzinfo=timezone.utc) >= cutoff_time_extended]
            
            if recent_indices:
                display_powers = [powers[i] for i in recent_indices]
                display_times = [times[i] for i in recent_indices]
            else:
                # Last resort: take last 60 points
                display_powers = powers[-60:] if len(powers) > 60 else powers
                display_times = times[-60:] if len(times) > 60 else times
        
        current_power = powers[-1]
        
        print(f"Hemispheric Power: Displaying {len(display_powers)} data points")
        
        # Calculate dynamic y-axis range based on actual data
        max_power = max(display_powers) if display_powers else 200
        
        # Determine y-axis limits dynamically
        y_min = 0  # Keep minimum at 0 GW
        
        # Add 25-50 GW overhead based on max value
        if max_power > 150:
            overhead = 50  # Use 50 GW overhead for very high activity
        else:
            overhead = 35  # Use 35 GW overhead for normal activity
        
        y_max = max_power + overhead
        # Round up to nearest 25 for cleaner display
        y_max = int(np.ceil(y_max / 25.0)) * 25
        
        # Ensure minimum scale of 100 GW for readability
        if y_max < 100:
            y_max = 100
        
        print(f"Hemispheric Power: Y-axis range 0-{y_max} GW (max data: {max_power:.0f} GW)")
        
        # Color coding based on power level for each bar
        colors = []
        for p in display_powers:
            if p < 30:
                colors.append('#94a3b8')  # gray
            elif p < 50:
                colors.append('#4ade80')  # green
            elif p < 80:
                colors.append('#facc15')  # yellow
            elif p < 100:
                colors.append('#fb923c')  # orange
            elif p < 150:
                colors.append('#f87171')  # red
            else:
                colors.append('#c084fc')  # purple
        
        # Plot bars
        x_positions = range(len(display_powers))
        ax_hemi.bar(x_positions, display_powers, color=colors, width=1.0, edgecolor='none')
        
        # Styling with dynamic y-axis
        ax_hemi.set_ylim([y_min, y_max])
        ax_hemi.set_xlim([0, len(display_powers)])
        ax_hemi.set_ylabel('GW', fontsize=9, color='#94a3b8', fontweight='bold')
        ax_hemi.set_title('HEMISPHERIC POWER', fontsize=10, color='#f8fafc', 
                         fontweight='bold', pad=4, loc='left')
        ax_hemi.tick_params(colors='#64748b', labelsize=7, length=0)
        ax_hemi.set_xticks([])
        ax_hemi.grid(True, alpha=0.1, color='#94a3b8', linestyle='-', linewidth=0.5, axis='y')
        ax_hemi.spines['top'].set_visible(False)
        ax_hemi.spines['right'].set_visible(False)
        ax_hemi.spines['left'].set_color('#334155')
        ax_hemi.spines['left'].set_linewidth(1)
        ax_hemi.spines['bottom'].set_color('#334155')
        ax_hemi.spines['bottom'].set_linewidth(1)
        
        # Determine color for current value badge
        if current_power < 30:
            power_color = '#94a3b8'
        elif current_power < 50:
            power_color = '#4ade80'
        elif current_power < 80:
            power_color = '#facc15'
        elif current_power < 100:
            power_color = '#fb923c'
        elif current_power < 150:
            power_color = '#f87171'
        else:
            power_color = '#c084fc'
        
        # Badge to the right of the graph
        power_text = f'{current_power:.0f} GW'
        
        # Modern boxy badge with gradient and shadow - positioned to the right
        power_badge_style = dict(boxstyle='round,pad=0.6', facecolor='#1e293b', 
                                edgecolor=power_color, linewidth=2,
                                path_effects=[matplotlib.patheffects.withSimplePatchShadow(offset=(2, -2), 
                                                                                           shadow_rgbFace='#000000', 
                                                                                           alpha=0.3)])
        
        fig.text(0.945, 0.915, power_text, 
                 fontsize=20, color=power_color, fontweight='bold',
                 bbox=power_badge_style, ha='center')
    
    # === AURORAL OVAL MAP (Left side, spans rows 1-3 - much larger) ===
    # Adjusted projection to focus on North America
    ax_map = fig.add_subplot(gs[1:4, 0], projection=ccrs.Orthographic(-95, 55))
    # Taller extent to fill the box vertically (25N to 85N)
    ax_map.set_extent([-135, -55, 25, 85], crs=ccrs.PlateCarree())
    ax_map.set_facecolor('#1e293b')
    # Add subtle border to map panel
    for spine in ax_map.spines.values():
        spine.set_edgecolor('#334155')
        spine.set_linewidth(1)
    
    # Add map features with cleaner look
    # Land: Dark slate blue/grey, Ocean: Very dark blue/black
    ax_map.add_feature(cfeature.LAND, facecolor='#1e293b', edgecolor='none', zorder=0)
    ax_map.add_feature(cfeature.OCEAN, facecolor='#020617', edgecolor='none', zorder=0)
    ax_map.add_feature(cfeature.LAKES, facecolor='#020617', edgecolor='none', zorder=0)
    
    # Lines: Thinner and sharper for a cleaner look
    ax_map.add_feature(cfeature.COASTLINE, edgecolor='#38bdf8', linewidth=0.8, alpha=0.8, zorder=1)
    ax_map.add_feature(cfeature.BORDERS, edgecolor="#8C99AA", linewidth=0.6, alpha=0.6, zorder=1)
    ax_map.add_feature(cfeature.STATES, edgecolor="#E1E2E2", linewidth=0.8, linestyle=':', alpha=0.9, zorder=1)
    
    # Add lat/lon grid
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.3, color='#475569', 
                          alpha=0.3, linestyle='--', zorder=2)
    
    # Plot OVATION auroral probability data if available
    if ovation_lons is not None and len(ovation_lons) > 0:
        from scipy.interpolate import griddata
        
        # OVATION data: aurora values range 0-100 (percentage), but typically 0-40
        # Filter to show only significant aurora activity
        threshold = 5
        mask = ovation_aurora >= threshold
        filtered_lons = ovation_lons[mask]
        filtered_lats = ovation_lats[mask]
        filtered_aurora = ovation_aurora[mask]
        
        if len(filtered_lons) > 0:
            # Convert longitude from 0-360 to -180 to 180 BEFORE interpolation
            adjusted_lons = np.where(filtered_lons > 180, filtered_lons - 360, filtered_lons)
            
            # Fix 180° gap by wrapping boundary data
            # Duplicate points near -180 to +180 and vice versa
            wrap_mask_left = adjusted_lons < -170
            wrap_mask_right = adjusted_lons > 170
            
            wrapped_lons = np.concatenate([
                adjusted_lons,
                adjusted_lons[wrap_mask_left] + 360,  # Duplicate left edge to right
                adjusted_lons[wrap_mask_right] - 360  # Duplicate right edge to left
            ])
            wrapped_lats = np.concatenate([
                filtered_lats,
                filtered_lats[wrap_mask_left],
                filtered_lats[wrap_mask_right]
            ])
            wrapped_aurora = np.concatenate([
                filtered_aurora,
                filtered_aurora[wrap_mask_left],
                filtered_aurora[wrap_mask_right]
            ])
            
            # Create high-resolution grid for smoother interpolation
            lon_grid = np.linspace(-180, 180, 1440)  # Higher resolution for smoother oval
            lat_grid = np.linspace(35, 85, 600)  # Higher resolution for smoother oval
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Interpolate using cubic method for smoother results
            aurora_grid = griddata(
                (wrapped_lons, wrapped_lats), 
                wrapped_aurora,
                (lon_mesh, lat_mesh),
                method='cubic',
                fill_value=np.nan
            )
            
            # Clamp values to valid range (cubic can overshoot)
            aurora_grid = np.clip(aurora_grid, 0, 100)
            
            # Apply multi-pass smoothing for natural-looking contours
            from scipy.ndimage import gaussian_filter
            valid = ~np.isnan(aurora_grid)
            if np.any(valid):
                # Stronger smoothing with multiple passes
                smoothed = np.nan_to_num(aurora_grid, 0)
                smoothed = gaussian_filter(smoothed, sigma=7.5)
                smoothed = gaussian_filter(smoothed, sigma=5.5)
                aurora_grid = np.where(valid, smoothed, np.nan)
            
            # Strict masking: only show aurora above threshold and with valid data
            aurora_grid = np.ma.masked_where((aurora_grid < threshold + 1) | np.isnan(aurora_grid), aurora_grid)
            
            lon_mesh_adjusted = lon_mesh
            
            # Create custom colormap for aurora matching SWPC reference
            # Green (low) -> Yellow (50%) -> Orange (75%) -> Red (90%+)
            colors = ["#477e47", '#88ff00', '#ffff00', '#ffaa00', '#ff0000']
            positions = [0.0, 0.25, 0.5, 0.75, 1.0]
            cmap = LinearSegmentedColormap.from_list('aurora', list(zip(positions, colors)), N=256)
            
            # Plot using pcolormesh for smooth blending
            # vmax=100 to show full probability range, alpha=0.8 for map visibility
            mesh = ax_map.pcolormesh(lon_mesh_adjusted, lat_mesh, aurora_grid,
                                    cmap=cmap, alpha=0.6, shading='gouraud',
                                    transform=ccrs.PlateCarree(), zorder=3,
                                    vmin=threshold, vmax=100)
            
            # Add colorbar legend with specific ticks matching reference
            cbar = plt.colorbar(mesh, ax=ax_map, orientation='horizontal', 
                               pad=0.05, shrink=0.8, aspect=20, 
                               ticks=[10, 50, 90])
            cbar.set_label('Aurora Probability (%)', fontsize=12, color='#00ff88', fontweight='bold')
            cbar.ax.set_xticklabels(['10%', '50%', '90%'])
            cbar.ax.tick_params(labelsize=10, colors='#8899aa')
            cbar.outline.set_edgecolor('#4a5a6a')
    
    # City markers handled later (avoid duplicate drawing)
    
    # Title with enhanced styling
    title_map = ax_map.set_title('REAL-TIME AURORAL OVAL',
                    fontsize=15, color='#0ea5e9', fontweight='bold', pad=10)
    title_map.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='#0f172a', alpha=0.8)])
    
    # === GOES MAGNETOMETER (Bottom left, aligned with Bz) ===
    ax_goes = fig.add_subplot(gs[4, 0])
    ax_goes.set_facecolor('#1e293b')
    # Add panel border
    ax_goes.patch.set_edgecolor('#334155')
    ax_goes.patch.set_linewidth(1)
    
    if goes_mag:
        # Plot GOES-18 (blue) and GOES-19 (red) like the reference
        if len(goes_mag.get('goes18_times', [])) > 0:
            goes18_times = goes_mag['goes18_times']
            goes18_hp = goes_mag['goes18_hp']
            ax_goes.plot(goes18_times, goes18_hp, color='#38bdf8', linewidth=2, 
                        label='GOES-18', zorder=3, alpha=0.9)
        
        if len(goes_mag.get('goes19_times', [])) > 0:
            goes19_times = goes_mag['goes19_times']
            goes19_hp = goes_mag['goes19_hp']
            ax_goes.plot(goes19_times, goes19_hp, color='#f87171', linewidth=2, 
                        label='GOES-19', zorder=3, alpha=0.9)
        
        # Add reference line at 100 nT (typical baseline)
        ax_goes.axhline(100, color='#94a3b8', linestyle='--', linewidth=1, alpha=0.3, zorder=1)
        
        # Set x-axis limits based on available data
        all_times = []
        if len(goes_mag.get('goes18_times', [])) > 0:
            all_times.extend(goes_mag['goes18_times'])
        if len(goes_mag.get('goes19_times', [])) > 0:
            all_times.extend(goes_mag['goes19_times'])
        
        if all_times:
            ax_goes.set_xlim([min(all_times), max(all_times)])
            
        # Add legend with white text
        legend = ax_goes.legend(loc='upper left', fontsize=8, framealpha=0.7, 
                      facecolor='#0f172a', edgecolor='#334155')
        for text in legend.get_texts():
            text.set_color('#f1f5f9')
    
    # Substorm detection logic (outside the if all_times block so it always displays)
    # Active: Rapid increase (40+ nT spike) within last 15-20 minutes AND still elevated
    # Inactive: No recent spike OR spike occurred but has been declining for 15+ minutes
    substorm_active = False
    detection_window = 20  # minutes to look back
    threshold_change = 40  # nT spike threshold
    recent_window = 15  # minutes - must have spike within this window to be "active"
    
    if goes_mag:
        # Check GOES-18
        if len(goes_mag.get('goes18_hp', [])) >= 20:
            recent_hp = goes_mag['goes18_hp'][-20:]  # Last 20 minutes
            recent_15min = goes_mag['goes18_hp'][-15:]  # Last 15 minutes
            
            # Find the maximum spike in the last 20 minutes
            hp_max = max(recent_hp)
            hp_min = min(recent_hp)
            hp_change = hp_max - hp_min
            
            # Check if spike occurred recently (within last 15 min) and is still elevated
            if hp_change >= threshold_change:
                # Find when the max occurred
                max_idx = recent_hp.index(hp_max)
                # If max is in the last 15 data points (15 minutes), it's recent
                if max_idx >= 5:  # Occurred within last 15 minutes
                    # Check if still elevated (not declining significantly)
                    current_hp = recent_hp[-1]
                    if current_hp >= hp_min + (threshold_change * 0.5):  # Still at least 50% elevated
                        substorm_active = True
        
        # Check GOES-19
        if len(goes_mag.get('goes19_hp', [])) >= 20:
            recent_hp = goes_mag['goes19_hp'][-20:]  # Last 20 minutes
            recent_15min = goes_mag['goes19_hp'][-15:]  # Last 15 minutes
            
            # Find the maximum spike in the last 20 minutes
            hp_max = max(recent_hp)
            hp_min = min(recent_hp)
            hp_change = hp_max - hp_min
            
            # Check if spike occurred recently (within last 15 min) and is still elevated
            if hp_change >= threshold_change:
                # Find when the max occurred
                max_idx = recent_hp.index(hp_max)
                # If max is in the last 15 data points (15 minutes), it's recent
                if max_idx >= 5:  # Occurred within last 15 minutes
                    # Check if still elevated (not declining significantly)
                    current_hp = recent_hp[-1]
                    if current_hp >= hp_min + (threshold_change * 0.5):  # Still at least 50% elevated
                        substorm_active = True
    
    # Display substorm status above the magnetometer graph with modern styling
    if substorm_active:
        substorm_text = "SUBSTORM: ACTIVE"
        substorm_color = '#ef4444'
    else:
        substorm_text = "SUBSTORM: INACTIVE"
        substorm_color = '#10b981'
    
    ax_goes.set_ylabel('Hp (nT)', fontsize=11, color='#94a3b8', fontweight='bold')
    title_goes = ax_goes.set_title('GOES MAGNETOMETER (Hp)', fontsize=12, color='#38bdf8',
                     fontweight='bold', pad=8, loc='left')
    title_goes.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='#0f172a', alpha=0.7)])
    
    # Add substorm status badge to the right of the title using axes coordinates
    substorm_badge_style = dict(boxstyle='round,pad=0.4', facecolor='#1e293b', 
                               edgecolor=substorm_color, linewidth=2)
    ax_goes.text(0.99, 1.05, substorm_text, transform=ax_goes.transAxes,
                fontsize=10, color=substorm_color, fontweight='bold',
                bbox=substorm_badge_style, ha='right', va='bottom')
    
    ax_goes.tick_params(colors='#64748b', labelsize=8, length=0)
    ax_goes.grid(True, alpha=0.1, color='#94a3b8', linestyle='-', linewidth=0.5)
    ax_goes.set_xticklabels([])
    ax_goes.spines['top'].set_visible(False)
    ax_goes.spines['right'].set_visible(False)
    ax_goes.spines['left'].set_color('#334155')
    ax_goes.spines['left'].set_linewidth(1)
    ax_goes.spines['bottom'].set_color('#334155')
    
    # === SOLAR WIND SPEED GRAPH ===
    ax_speed = fig.add_subplot(gs[1, 1:])
    ax_speed.set_facecolor('#1e293b')
    # Add panel border
    ax_speed.patch.set_edgecolor('#334155')
    ax_speed.patch.set_linewidth(1)
    
    if sw_history and len(sw_history['times']) > 0:
        times = sw_history['times']
        speeds = [s if s else np.nan for s in sw_history['speeds']]
        
        # Create smooth gradient fill based on speed values
        # Use gradient effect similar to SpaceWeatherLive
        import matplotlib.dates as mdates
        
        # Calculate dynamic y-axis range based on actual data
        valid_speeds = [s for s in speeds if not np.isnan(s)]
        max_speed = max(valid_speeds) if valid_speeds else 800
        min_speed = min(valid_speeds) if valid_speeds else 200
        
        # Determine y-axis limits dynamically
        y_min = 200  # Keep minimum at 200 km/s
        
        # If max speed exceeds 800, add 100-150 km/s overhead
        if max_speed > 800:
            y_max = max_speed + 125  # Add 125 km/s overhead
            # Round up to nearest 50 for cleaner display
            y_max = int(np.ceil(y_max / 50.0)) * 50
        else:
            y_max = 800  # Default maximum
        
        # Create custom colormap: green (low) -> yellow -> orange -> red (high)
        # Adjust color positions based on dynamic range
        speed_colors = ['#10b981', '#84cc16', '#facc15', '#fb923c', '#f87171', '#ef4444']
        # Yellow should start around 400 km/s regardless of range
        yellow_pos = (400 - y_min) / (y_max - y_min) if y_max > y_min else 0.33
        yellow_pos = max(0.1, min(0.5, yellow_pos))  # Clamp between 0.1 and 0.5
        
        speed_positions = [0.0, yellow_pos * 0.6, yellow_pos, 0.55, 0.75, 1.0]
        from matplotlib.colors import LinearSegmentedColormap as LSC
        speed_cmap = LSC.from_list('speed_gradient', 
                                   list(zip(speed_positions, speed_colors)), N=256)
        
        # Set axis limits FIRST to ensure proper clipping
        ax_speed.set_xlim([times[0], times[-1]])
        ax_speed.set_ylim([y_min, y_max])
        
        # Create vertical gradient fill using imshow
        # The gradient goes from bottom (green) to top (red) based on y-axis values
        gradient_height = np.linspace(0, 1, 100).reshape(-1, 1)
        gradient_array = np.repeat(gradient_height, 100, axis=1)
        
        # Set extent slightly INSIDE the data range to prevent gradient bleed at edges
        # Calculate a small offset to pull the gradient away from the right edge
        time_start = mdates.date2num(times[0])
        time_end = mdates.date2num(times[-1])
        time_range = time_end - time_start
        time_offset = time_range * 0.002  # Pull gradient 0.2% away from edges
        extent = [time_start + time_offset, time_end - time_offset, y_min, y_max]
        
        im = ax_speed.imshow(gradient_array, aspect='auto', extent=extent,
                       origin='lower', cmap=speed_cmap, zorder=1, alpha=0.8, 
                       interpolation='nearest')
        
        # Mask the area above the data line with panel background color
        ax_speed.fill_between(times, speeds, y_max, color='#1e293b', edgecolor='none', zorder=2, interpolate=True, antialiased=False)
        
        # Plot line on top
        ax_speed.plot(times, speeds, color='#f1f5f9', linewidth=2.5, zorder=5, alpha=0.9)
        
        # Current value annotation - enhanced with badge style
        if speeds[-1] and not np.isnan(speeds[-1]):
            speed_text = f'{speeds[-1]:.0f} km/s'
            # Main text with strong shadow
            ax_speed.text(0.98, 0.95, speed_text,
                         transform=ax_speed.transAxes,
                         ha='right', va='top', fontsize=26, color='#38bdf8', 
                         fontweight='bold', alpha=1.0,
                         path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground='#0f172a', alpha=0.8)])
    
    ax_speed.set_ylabel('Speed (km/s)', fontsize=11, color='#94a3b8', fontweight='bold')
    title_speed = ax_speed.set_title('SOLAR WIND SPEED', fontsize=13, color='#38bdf8', 
                      fontweight='bold', pad=10, loc='left')
    title_speed.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='#0f172a', alpha=0.7)])
    ax_speed.tick_params(colors='#64748b', labelsize=9, length=0)
    ax_speed.set_xticklabels([])
    ax_speed.spines['top'].set_visible(False)
    ax_speed.spines['right'].set_visible(False)
    ax_speed.spines['left'].set_color('#334155')
    ax_speed.spines['left'].set_linewidth(1)
    ax_speed.spines['bottom'].set_color('#334155')
    ax_speed.spines['bottom'].set_linewidth(1)
    # Draw grid on top of everything (after all plotting)
    ax_speed.grid(True, alpha=0.1, color='#94a3b8', linestyle='-', linewidth=0.8, zorder=10)
    ax_speed.set_axisbelow(False)
    
    # === PROTON DENSITY GRAPH ===
    ax_density = fig.add_subplot(gs[2, 1:])
    ax_density.set_facecolor('#1e293b')
    # Add panel border
    ax_density.patch.set_edgecolor('#334155')
    ax_density.patch.set_linewidth(1)
    
    if sw_history and len(sw_history['times']) > 0:
        densities = [d if d else np.nan for d in sw_history['densities']]
        
        # Create gradient fill
        ax_density.fill_between(times, 0, densities, alpha=0.6, 
                               color='#fb923c', edgecolor='none')
        ax_density.plot(times, densities, color='#fb923c', linewidth=2, zorder=3)
        
        # Current value annotation - large text on right with drop shadow
        if densities[-1] and not np.isnan(densities[-1]):
            # Drop shadow
            ax_density.text(0.98, 0.95, f'{densities[-1]:.1f}',
                           transform=ax_density.transAxes,
                           ha='right', va='top', fontsize=24, color='#000000',
                           fontweight='bold', alpha=0.5,
                           path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='#000000')])
            # Main text
            ax_density.text(0.98, 0.95, f'{densities[-1]:.1f}',
                           transform=ax_density.transAxes,
                           ha='right', va='top', fontsize=24, color='#fb923c',
                           fontweight='bold', alpha=1.0)
        
        # Set x-axis limits to match data range
        ax_density.set_xlim([times[0], times[-1]])
    
    ax_density.set_ylabel('Density (p/cm³)', fontsize=11, color='#94a3b8', fontweight='bold')
    title_density = ax_density.set_title('PROTON DENSITY', fontsize=13, color='#fb923c',
                        fontweight='bold', pad=10, loc='left')
    title_density.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='#0f172a', alpha=0.7)])
    ax_density.tick_params(colors='#64748b', labelsize=9, length=0)
    ax_density.grid(True, alpha=0.1, color='#94a3b8', linestyle='-', linewidth=0.8)
    ax_density.set_xticklabels([])
    ax_density.spines['top'].set_visible(False)
    ax_density.spines['right'].set_visible(False)
    ax_density.spines['left'].set_color('#334155')
    ax_density.spines['left'].set_linewidth(1)
    ax_density.spines['bottom'].set_color('#334155')
    
    # === IMF Bt (TOTAL) GRAPH ===
    ax_bt = fig.add_subplot(gs[3, 1:])
    ax_bt.set_facecolor('#1e293b')
    # Add panel border
    ax_bt.patch.set_edgecolor('#334155')
    ax_bt.patch.set_linewidth(1)
    
    if sw_history and len(sw_history['times']) > 0:
        bts = [b if b else np.nan for b in sw_history['bts']]
        
        # Create gradient fill
        ax_bt.fill_between(times, 0, bts, alpha=0.6, 
                          color='#c084fc', edgecolor='none')
        ax_bt.plot(times, bts, color='#c084fc', linewidth=2, zorder=3)
        
        # Current value annotation - large text on right with drop shadow
        if bts[-1] and not np.isnan(bts[-1]):
            # Drop shadow
            ax_bt.text(0.98, 0.95, f'{bts[-1]:.1f}',
                      transform=ax_bt.transAxes,
                      ha='right', va='top', fontsize=24, color='#000000',
                      fontweight='bold', alpha=0.5,
                      path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='#000000')])
            # Main text
            ax_bt.text(0.98, 0.95, f'{bts[-1]:.1f}',
                      transform=ax_bt.transAxes,
                      ha='right', va='top', fontsize=24, color='#c084fc',
                      fontweight='bold', alpha=1.0)
        
        # Set x-axis limits to match data range
        ax_bt.set_xlim([times[0], times[-1]])
    
    ax_bt.set_ylabel('Bt (nT)', fontsize=11, color='#94a3b8', fontweight='bold')
    title_bt = ax_bt.set_title('IMF Bt', fontsize=13, color='#c084fc',
                   fontweight='bold', pad=10, loc='left')
    title_bt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='#0f172a', alpha=0.7)])
    ax_bt.tick_params(colors='#64748b', labelsize=9, length=0)
    ax_bt.grid(True, alpha=0.1, color='#94a3b8', linestyle='-', linewidth=0.8)
    ax_bt.set_xticklabels([])
    ax_bt.spines['top'].set_visible(False)
    ax_bt.spines['right'].set_visible(False)
    ax_bt.spines['left'].set_color('#334155')
    ax_bt.spines['left'].set_linewidth(1)
    ax_bt.spines['bottom'].set_color('#334155')
    
    # === IMF Bz (NORTH-SOUTH) GRAPH ===
    ax_bz = fig.add_subplot(gs[4, 1:])
    ax_bz.set_facecolor('#1e293b')
    # Add panel border
    ax_bz.patch.set_edgecolor('#334155')
    ax_bz.patch.set_linewidth(1)
    
    if sw_history and len(sw_history['times']) > 0:
        bzs = [b if b else np.nan for b in sw_history['bzs']]
        
        # Fill positive and negative regions differently
        ax_bz.fill_between(times, 0, bzs, where=np.array(bzs)>=0, 
                          alpha=0.4, color='#10b981', edgecolor='none', interpolate=True)
        ax_bz.fill_between(times, 0, bzs, where=np.array(bzs)<0,
                          alpha=0.4, color='#ef4444', edgecolor='none', interpolate=True)
        
        # Plot Bz line in white on top of fills
        ax_bz.plot(times, bzs, color='#f1f5f9', linewidth=2, zorder=3)
        
        # Reference line (zero line only)
        ax_bz.axhline(0, color='#94a3b8', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
        
        # Current value annotation - large text on right with drop shadow
        if bzs[-1] and not np.isnan(bzs[-1]):
            bz_color = '#10b981' if bzs[-1] >= 0 else '#ef4444'
            direction = 'NORTH' if bzs[-1] >= 0 else 'SOUTH'
            bz_text = f'{bzs[-1]:.1f} {direction}'
            # Drop shadow
            ax_bz.text(0.98, 0.95, bz_text,
                      transform=ax_bz.transAxes,
                      ha='right', va='top', fontsize=24, color='#000000',
                      fontweight='bold', alpha=0.5,
                      path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='#000000')])
            # Main text
            ax_bz.text(0.98, 0.95, bz_text,
                      transform=ax_bz.transAxes,
                      ha='right', va='top', fontsize=24, color=bz_color,
                      fontweight='bold', alpha=1.0)
        
        # Set x-axis limits to match data range
        ax_bz.set_xlim([times[0], times[-1]])
    
    ax_bz.set_ylabel('Bz (nT)', fontsize=11, color='#94a3b8', fontweight='bold')
    ax_bz.set_xlabel('Time (UTC)', fontsize=11, color='#94a3b8', fontweight='bold')
    title_bz = ax_bz.set_title('IMF Bz', fontsize=13, 
                   color='#f1f5f9', fontweight='bold', pad=10, loc='left')
    title_bz.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='#0f172a', alpha=0.7)])
    ax_bz.tick_params(colors='#64748b', labelsize=9, length=0)
    ax_bz.grid(True, alpha=0.1, color='#94a3b8', linestyle='-', linewidth=0.8)
    ax_bz.spines['top'].set_visible(False)
    ax_bz.spines['right'].set_visible(False)
    ax_bz.spines['left'].set_color('#334155')
    ax_bz.spines['left'].set_linewidth(1)
    ax_bz.spines['bottom'].set_color('#334155')
    ax_bz.spines['bottom'].set_linewidth(1)
    
    # Format x-axis time labels
    import matplotlib.dates as mdates
    ax_bz.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax_bz.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # === FOOTER with data source and timestamp ===
    kp_val = kp_data.get('kp') if kp_data else None
    g_scale = scales.get('g_scale') if scales else None
    likelihood = get_aurora_likelihood(kp_val, g_scale)
    
    # Determine footer color based on conditions
    if likelihood in ['Excellent', 'High']:
        footer_color = '#10b981'
    elif likelihood in ['Good', 'Moderate']:
        footer_color = '#f59e0b'
    else:
        footer_color = '#94a3b8'
    
    footer_left = f" Aurora Dashboard    Data: NOAA Space Weather Prediction Center"
    footer_right = f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    
    fig.text(0.02, 0.01, footer_left, ha='left', va='bottom', fontsize=9, 
             color='#64748b', fontweight='normal')
    fig.text(0.98, 0.01, footer_right, ha='right', va='bottom', fontsize=9,
             color=footer_color, fontweight='bold')
    
    return _figure_to_png_buffer(
        fig,
        dpi=150,
        facecolor='#0f172a',
        edgecolor='none',
        bbox_inches='tight'
    )

@app.route('/')
def index():
    """Render the dashboard"""
    return render_template('index.html')

@app.route('/wtusredlogotransparentx.png')
def logo():
    """Serve the logo file"""
    return send_file('wtusredlogotransparentx.png', mimetype='image/png')

@app.route('/solar')
def solar_page():
    """Render the solar activity page"""
    return render_template('solar.html')

@app.route('/historical')
def historical_page():
    """Render the historical data archive page"""
    return render_template('historical.html')

# ── Historical Data API Endpoints ────────────────────────────────────────

@app.route('/api/historical/donki-flares')
def historical_donki_flares():
    """Proxy NASA DONKI solar flare data for a date range with caching"""
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    if not start or not end:
        return jsonify([])
    ok, msg = _validate_date_range(start, end)
    if not ok:
        return jsonify({'error': msg}), 400
    
    cache_key = f'donki_flares_{start}_{end}'
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)
    
    try:
        url = "https://api.nasa.gov/DONKI/FLR"
        params = {'startDate': start, 'endDate': end, 'api_key': NASA_API_KEY}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            print("[HISTORICAL] DONKI flares: rate limit hit")
            return jsonify({'_error': 'Rate limit - try again shortly'})
        if resp.status_code == 200:
            data = resp.json()
            result = data if isinstance(data, list) else []
            cache.set(cache_key, result, timeout=3600)
            print(f"[HISTORICAL] Fetched {len(result)} flares")
            return jsonify(result)
        return jsonify([])
    except Exception as e:
        print(f"[HISTORICAL] DONKI flares error: {e}")
        return jsonify([])

@app.route('/api/historical/donki-cmes')
def historical_donki_cmes():
    """Proxy NASA DONKI CME data for a date range with caching"""
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    if not start or not end:
        return jsonify([])
    ok, msg = _validate_date_range(start, end)
    if not ok:
        return jsonify({'error': msg}), 400
    
    cache_key = f'donki_cmes_{start}_{end}'
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)
    
    try:
        url = "https://api.nasa.gov/DONKI/CME"
        params = {'startDate': start, 'endDate': end, 'api_key': NASA_API_KEY}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            print("[HISTORICAL] DONKI CMEs: rate limit hit")
            return jsonify({'_error': 'Rate limit - try again shortly'})
        if resp.status_code == 200:
            data = resp.json()
            result = data if isinstance(data, list) else []
            cache.set(cache_key, result, timeout=3600)
            print(f"[HISTORICAL] Fetched {len(result)} CMEs")
            return jsonify(result)
        return jsonify([])
    except Exception as e:
        print(f"[HISTORICAL] DONKI CMEs error: {e}")
        return jsonify([])

@app.route('/api/historical/donki-storms')
def historical_donki_storms():
    """Proxy NASA DONKI geomagnetic storm data for a date range with caching"""
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    if not start or not end:
        return jsonify([])
    ok, msg = _validate_date_range(start, end)
    if not ok:
        return jsonify({'error': msg}), 400
    
    cache_key = f'donki_storms_{start}_{end}'
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)
    
    try:
        url = "https://api.nasa.gov/DONKI/GST"
        params = {'startDate': start, 'endDate': end, 'api_key': NASA_API_KEY}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            print("[HISTORICAL] DONKI storms: rate limit hit")
            return jsonify({'_error': 'Rate limit - try again shortly'})
        if resp.status_code == 200:
            data = resp.json()
            result = data if isinstance(data, list) else []
            cache.set(cache_key, result, timeout=3600)
            print(f"[HISTORICAL] Fetched {len(result)} storms")
            return jsonify(result)
        return jsonify([])
    except Exception as e:
        print(f"[HISTORICAL] DONKI storms error: {e}")
        return jsonify([])

@app.route('/api/historical/donki-seps')
def historical_donki_seps():
    """Proxy NASA DONKI solar energetic particle data for a date range with caching"""
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    if not start or not end:
        return jsonify([])
    ok, msg = _validate_date_range(start, end)
    if not ok:
        return jsonify({'error': msg}), 400
    
    cache_key = f'donki_seps_{start}_{end}'
    cached = cache.get(cache_key)
    if cached is not None:
        return jsonify(cached)
    
    try:
        url = "https://api.nasa.gov/DONKI/SEP"
        params = {'startDate': start, 'endDate': end, 'api_key': NASA_API_KEY}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            print("[HISTORICAL] DONKI SEPs: rate limit hit")
            return jsonify({'_error': 'Rate limit - try again shortly'})
        if resp.status_code == 200:
            data = resp.json()
            result = data if isinstance(data, list) else []
            cache.set(cache_key, result, timeout=3600)
            print(f"[HISTORICAL] Fetched {len(result)} SEP events")
            return jsonify(result)
        return jsonify([])
    except Exception as e:
        print(f"[HISTORICAL] DONKI SEPs error: {e}")
        return jsonify([])

@app.route('/api/historical/solar-wind')
def historical_solar_wind():
    """Fetch historical solar wind at 1-minute resolution from CDAWeb HAPI OMNI.
    Combines 1-minute solar wind data with hourly Kp/Dst indices."""
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    if not start or not end:
        return jsonify({'data': []})
    ok, msg = _validate_date_range(start, end)
    if not ok:
        return jsonify({'data': [], 'error': msg}), 400
    ok, msg = _validate_date_range(start, end)
    if not ok:
        return jsonify({'data': [], 'error': msg}), 400
    try:
        hapi_url = "https://cdaweb.gsfc.nasa.gov/hapi/data"
        
        # Fetch 1-minute solar wind data (Bt, Bz, speed, density)
        # Parameters MUST be in schema order: F, BZ_GSM, flow_speed, proton_density
        params_1min = {
            'id': 'OMNI_HRO_1MIN',
            'parameters': 'F,BZ_GSM,flow_speed,proton_density',
            'time.min': f'{start}T00:00:00Z',
            'time.max': f'{end}T23:59:59Z',
            'format': 'json'
        }
        resp_1min = requests.get(hapi_url, params=params_1min, timeout=60)
        
        # Fetch hourly Kp/Dst data
        params_hourly = {
            'id': 'OMNI2_H0_MRG1HR',
            'parameters': 'KP1800,DST1800',
            'time.min': f'{start}T00:00:00Z',
            'time.max': f'{end}T23:59:59Z',
            'format': 'json'
        }
        resp_hourly = requests.get(hapi_url, params=params_hourly, timeout=30)
        
        if resp_1min.status_code != 200:
            print(f"[HISTORICAL] 1-min HAPI error: {resp_1min.status_code}")
            return jsonify({'data': []})
        
        raw_1min = resp_1min.json()
        rows_1min = raw_1min.get('data', [])
        
        # Build hourly Kp/Dst lookup
        kp_dst_map = {}
        if resp_hourly.status_code == 200:
            raw_hourly = resp_hourly.json()
            for row in raw_hourly.get('data', []):
                if not row or len(row) < 3:
                    continue
                try:
                    time_str = str(row[0])[:13]  # Hour precision: '2024-05-10T12'
                    kp_val = float(row[1]) if row[1] is not None else None
                    dst_val = float(row[2]) if row[2] is not None else None
                    kp = kp_val if kp_val and kp_val < 90 else None
                    dst = dst_val if dst_val and abs(dst_val) < 9999 else None
                    kp_dst_map[time_str] = {'kp': kp, 'dst': dst}
                except (ValueError, IndexError, TypeError):
                    continue
        
        # Merge 1-minute data with hourly Kp/Dst
        # Row format: [time, F(bt), BZ_GSM(bz), flow_speed, proton_density]
        result = []
        for row in rows_1min:
            if not row or len(row) < 5:
                continue
            try:
                time_str = str(row[0])
                hour_key = time_str[:13]
                bt_val = float(row[1]) if row[1] is not None else None
                bz_val = float(row[2]) if row[2] is not None else None
                speed_val = float(row[3]) if row[3] is not None else None
                density_val = float(row[4]) if row[4] is not None else None
                
                # Filter out fill values
                bt = bt_val if bt_val and bt_val < 999 else None
                bz = bz_val if bz_val and abs(bz_val) < 999 else None
                speed = speed_val if speed_val and speed_val < 9999 else None
                density = density_val if density_val and density_val < 999 else None
            except (ValueError, IndexError, TypeError):
                continue
            
            # Get Kp/Dst from hourly data for this hour
            hourly_data = kp_dst_map.get(hour_key, {})
            result.append({
                't': time_str,
                'bt': bt,
                'bz': bz,
                'n': density,
                'v': speed,
                'kp': hourly_data.get('kp'),
                'dst': hourly_data.get('dst')
            })
        
        print(f"[HISTORICAL] Fetched {len(result)} 1-minute records")
        return jsonify({'data': result})
    except Exception as e:
        print(f"[HISTORICAL] OMNI solar wind error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'data': []})

@app.route('/api/historical/sdo-images')
def historical_sdo_images():
    """Generate SDO/LASCO solar imagery URLs for a date range and channel via Helioviewer API.
    Returns images at specified intervals (15, 30, 45, 60, 120 minutes), spanning up to 11 days."""
    start_str = request.args.get('start', '')
    end_str = request.args.get('end', '')
    channel = request.args.get('channel', '0193')
    # Optional scale parameter (float). Smaller ~ shows less zoom depending on Helioviewer semantics.
    try:
        scale = float(request.args.get('scale', '4.0'))
    except Exception:
        scale = 4.0
    # Frame interval in minutes (15, 30, 45, 60, 120)
    try:
        interval = int(request.args.get('interval', '30'))
        if interval not in [15, 30, 45, 60, 120]:
            interval = 30
    except Exception:
        interval = 30
    
    if not start_str:
        return jsonify({'images': []})
    try:
        start_dt = datetime.strptime(start_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_str, '%Y-%m-%d') if end_str else start_dt

        # Cap at 11 days
        max_end = start_dt + timedelta(days=11)
        if end_dt > max_end:
            end_dt = max_end

        total_days = (end_dt - start_dt).days + 1

        source_map = {
            '0094': ('SDO', 'AIA', 'AIA', '94'),
            '0131': ('SDO', 'AIA', 'AIA', '131'),
            '0171': ('SDO', 'AIA', 'AIA', '171'),
            '0193': ('SDO', 'AIA', 'AIA', '193'),
            '0211': ('SDO', 'AIA', 'AIA', '211'),
            '0304': ('SDO', 'AIA', 'AIA', '304'),
            '0335': ('SDO', 'AIA', 'AIA', '335'),
            '1600': ('SDO', 'AIA', 'AIA', '1600'),
            '1700': ('SDO', 'AIA', 'AIA', '1700'),
            'HMIB': ('SDO', 'HMI', 'HMI', 'magnetogram'),
            'HMIIC': ('SDO', 'HMI', 'HMI', 'continuum'),
            'LASCO-C2': ('SOHO', 'LASCO', 'LASCO', 'white-light'),
            'LASCO-C3': ('SOHO', 'LASCO', 'LASCO', 'white-light'),
        }

        info = source_map.get(channel, source_map['0193'])
        obs, inst, det, measurement = info
        
        # For LASCO, we need to specify which coronagraph (C2 or C3) in the detector field
        if channel == 'LASCO-C2':
            det = 'C2'
        elif channel == 'LASCO-C3':
            det = 'C3'
        
        layers_str = f"[{obs},{inst},{det},{measurement},1,100]"

        from urllib.parse import quote
        layers_encoded = quote(layers_str)

        # LASCO imagery requires smaller scale values to show the full coronagraph
        # Also LASCO has different cadence, so we'll request closest available image
        if channel.startswith('LASCO'):
            if channel == 'LASCO-C2':
                effective_scale = 2.5  # C2: 2-6 solar radii
            else:  # LASCO-C3
                effective_scale = 1.0  # C3: 3.7-30 solar radii
        else:
            effective_scale = scale

        images = []
        current = start_dt
        while current <= end_dt:
            img_dt = current.replace(hour=0, minute=0, second=0)
            # Generate frames based on interval
            while img_dt.date() == current.date():
                date_iso = img_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

                # Use takeScreenshot for SDO/AIA/HMI (regular cadence)
                # For LASCO, we need closest image matching due to irregular cadence
                base_params = f"date={date_iso}&imageScale={effective_scale}&layers={layers_encoded}&x0=0&y0=0&watermark=false"
                
                if channel.startswith('LASCO'):
                    # LASCO has irregular cadence, so build URL that works with their data
                    url_thumb = f"https://api.helioviewer.org/v2/takeScreenshot/?{base_params}&width=512&height=512&display=true"
                    url_full = f"https://api.helioviewer.org/v2/takeScreenshot/?{base_params}&width=1024&height=1024&display=true"
                else:
                    # SDO/AIA/HMI have regular cadence
                    url_thumb = f"https://api.helioviewer.org/v2/takeScreenshot/?{base_params}&width=512&height=512&display=true"
                    url_full = f"https://api.helioviewer.org/v2/takeScreenshot/?{base_params}&width=1024&height=1024&display=true"

                time_label = img_dt.strftime('%Y-%m-%d %H:%M UTC')
                images.append({
                    'url_thumb': url_thumb,
                    'url_full': url_full,
                    'label': f'{inst} {det if channel.startswith("LASCO") else measurement}',
                    'time': time_label,
                    'timestamp': date_iso
                })
                
                img_dt += timedelta(minutes=interval)
            current += timedelta(days=1)

        # If too many images (long date range), sample down to ~528
        if len(images) > 528:
            step = len(images) // 528
            images = images[::step]

        return jsonify({'images': images, 'total_days': total_days})
    except Exception as e:
        print(f"[HISTORICAL] SDO/LASCO images error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'images': [], 'error': str(e)})


@app.route('/api/historical/goes-xray')
def historical_goes_xray():
    """Fetch historical GOES X-ray flux (1-minute resolution) for a date range from CDAWeb HAPI.
    Returns records: {'t': ISO, 'xrs_long': val, 'xrs_short': val}
    Uses GOES-17 or GOES-16 XRS 1-minute data from CDAWeb.
    """
    start = request.args.get('start', '')
    end = request.args.get('end', '')
    if not start or not end:
        return jsonify({'data': []})
    
    try:
        hapi_url = "https://cdaweb.gsfc.nasa.gov/hapi/data"
        
        # Try GOES-16 and GOES-17 XRS 1-minute averaged datasets
        # Parameters: xrsa_flux (0.5-4 Angstrom), xrsb_flux (1-8 Angstrom)
        candidates = [
            ('GOES16_XRS_1M', 'xrsa_flux,xrsb_flux'),  # GOES-16 1-min
            ('GOES17_XRS_1M', 'xrsa_flux,xrsb_flux'),  # GOES-17 1-min
            ('G16_XRSF_1M', 'xrsa_flux,xrsb_flux'),
            ('G17_XRSF_1M', 'xrsa_flux,xrsb_flux')
        ]
        
        for dataset_id, params in candidates:
            try:
                params_req = {
                    'id': dataset_id,
                    'parameters': params,
                    'time.min': f'{start}T00:00:00Z',
                    'time.max': f'{end}T23:59:59Z',
                    'format': 'json'
                }
                resp = requests.get(hapi_url, params=params_req, timeout=60)
                
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                rows = data.get('data', [])
                if not rows:
                    continue
                
                # Row format: [time, xrsa_flux, xrsb_flux]
                # xrsa = 0.5-4 Angstrom (short channel)
                # xrsb = 1-8 Angstrom (long channel)
                result = []
                for row in rows:
                    try:
                        time_str = row[0]
                        xrsa = row[1] if row[1] is not None and row[1] > 0 and row[1] < 1 else None  # Filter bad data
                        xrsb = row[2] if row[2] is not None and row[2] > 0 and row[2] < 1 else None
                        result.append({
                            't': time_str,
                            'xrs_short': xrsa,  # 0.5-4 Angstrom
                            'xrs_long': xrsb    # 1-8 Angstrom
                        })
                    except Exception:
                        continue
                
                if result:
                    print(f"[HISTORICAL] GOES X-ray: fetched {len(result)} records from {dataset_id}")
                    return jsonify({'data': result})
                    
            except Exception as e:
                print(f"[HISTORICAL] GOES X-ray attempt error for {dataset_id}: {e}")
                continue
        
        # If CDAWeb fails, try SWPC rolling products as fallback (only for recent dates)
        swpc_urls = [
            'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',
            'https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json'
        ]
        
        # Parse start/end datetimes for filtering
        try:
            start_dt = datetime.fromisoformat(f'{start}T00:00:00+00:00')
            end_dt = datetime.fromisoformat(f'{end}T23:59:59+00:00')
        except Exception as e:
            print(f"[HISTORICAL] GOES X-ray: invalid date format {start}/{end}: {e}")
            print(f"[HISTORICAL] GOES X-ray: no data found for {start} to {end}")
            return jsonify({'data': []})
        
        for url in swpc_urls:
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                
                swpc_data = resp.json()
                if not swpc_data or not isinstance(swpc_data, list):
                    continue
                
                # SWPC JSON format: list of dict with time_tag, energy, flux
                # energy: "0.1-0.8nm" (long/1-8Å) or "0.05-0.4nm" (short/0.5-4Å)
                by_time = {}
                filtered_count = 0
                for item in swpc_data:
                    if not isinstance(item, dict):
                        continue
                    t_str = item.get('time_tag', '')
                    if not t_str:
                        continue
                    
                    # Filter by date range
                    try:
                        item_dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
                        if not (start_dt <= item_dt <= end_dt):
                            continue
                        filtered_count += 1
                    except Exception as e:
                        # Skip items with unparseable timestamps
                        continue
                    
                    if t_str not in by_time:
                        by_time[t_str] = {}
                    
                    energy = item.get('energy', '')
                    flux = item.get('flux')
                    if flux is not None and flux > 0 and flux < 1:
                        if '0.1-0.8' in energy:  # 0.1-0.8nm = 1-8 Angstrom (long)
                            by_time[t_str]['xrs_long'] = flux
                        elif '0.05-0.4' in energy:  # 0.05-0.4nm = 0.5-4 Angstrom (short)
                            by_time[t_str]['xrs_short'] = flux
                
                # Build result array combining both channels
                result = []
                for t in sorted(by_time.keys()):
                    vals = by_time[t]
                    if 'xrs_long' in vals and 'xrs_short' in vals:
                        result.append({
                            't': t,
                            'xrs_long': vals['xrs_long'],
                            'xrs_short': vals['xrs_short']
                        })
                
                if result:
                    print(f"[HISTORICAL] GOES X-ray: fetched {len(result)} records from SWPC {url} (filtered {filtered_count} items in range)")
                    return jsonify({'data': result})
                    
            except Exception as e:
                print(f"[HISTORICAL] GOES X-ray SWPC error for {url}: {e}")
                continue
        
        print(f"[HISTORICAL] GOES X-ray: no data found for {start} to {end}")
        return jsonify({'data': []})
        
    except Exception as e:
        print(f"[HISTORICAL] GOES X-ray error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'data': []})

@app.route('/api/solar-data')
def get_solar_data():
    """Get solar activity data"""
    cache_key = 'solar_data'
    cached = get_cached(cache_key, max_age_seconds=45)
    if cached:
        return jsonify(cached)

    data = fetch_solar_data()
    if data:
        set_cached(cache_key, data)
        return jsonify(data)
    return jsonify({'error': 'Failed to fetch data'}), 500


@app.route('/api/active-region-watchlist')
def get_active_region_watchlist():
    """Return SWPC-backed active regions worth watching."""
    cache_key = 'solar_data'
    data = get_cached(cache_key, max_age_seconds=45)
    if not data:
        data = fetch_solar_data()
        if data:
            set_cached(cache_key, data)

    if not data:
        return jsonify({'error': 'Failed to fetch solar region data', 'watchlist_regions': []}), 500

    return jsonify({
        'watchlist_regions': data.get('watchlist_regions', []),
        'solar_regions_updated': data.get('solar_regions_updated'),
        'source': 'NOAA SWPC solar_regions.json'
    })


@app.route('/api/flare-alert')
def get_flare_alert():
    """Return the latest flare alert summary for popup notifications."""
    payload = fetch_latest_flare_alert()
    flare = payload.get('flare')
    if flare:
        graphic_params = {
            'event_class': flare.get('event_class') or '',
            'start_time': flare.get('time') or '',
            'peak_time': flare.get('peak_time') or '',
            'end_time': flare.get('end_time') or '',
            'region': flare.get('region_number') or '',
            'location': flare.get('location') or '',
            'source_region': flare.get('source_region') or '',
            'r_scale': flare.get('r_scale_label') or '',
        }
        flare['graphic_url'] = '/api/flare-alert-graphic?' + '&'.join(
            f"{key}={requests.utils.quote(str(value))}" for key, value in graphic_params.items() if value
        )

    response = jsonify(payload)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


@app.route('/api/flare-alert-graphic')
def get_flare_alert_graphic():
    """Render a shareable flare alert graphic using live imagery and charts."""
    flare_payload = fetch_latest_flare_alert()
    flare = dict(flare_payload.get('flare') or {})

    event_class = normalize_flare_class(request.args.get('event_class'))
    if event_class:
        flare.update({
            'event_class': event_class,
            'source_region': request.args.get('source_region') or flare.get('source_region'),
            'region_number': request.args.get('region') or flare.get('region_number'),
            'location': request.args.get('location') or flare.get('location'),
            'time': request.args.get('start_time') or flare.get('time'),
            'peak_time': request.args.get('peak_time') or flare.get('peak_time'),
            'end_time': request.args.get('end_time') or flare.get('end_time'),
        })

    if not flare or not flare.get('event_class'):
        return jsonify({'error': 'No recent flare alert available'}), 404

    peak_dt = parse_swpc_datetime(flare.get('peak_time')) or parse_swpc_datetime(flare.get('time')) or datetime.now(timezone.utc)
    flare['peak_time_display'] = format_utc_display(peak_dt)
    flare['event_time_display'] = format_utc_display(parse_swpc_datetime(flare.get('time')) or peak_dt)
    flare['r_scale_label'] = request.args.get('r_scale') or flare.get('r_scale_label') or f"R{flare_flux_to_r_scale(flare_class_to_flux(flare.get('event_class')))}"

    image_buffer = draw_flare_alert_graphic(flare)
    response = send_file(image_buffer, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/api/xray-data')
def get_xray_data():
    """Get X-ray flux data with specified time range"""
    time_range = request.args.get('range', '6h')
    
    # Check cache first (60 second cache for X-ray data)
    cache_key = f'xray_data_{time_range}'
    cached = get_cached(cache_key, max_age_seconds=60)
    if cached:
        return jsonify(cached)
    
    print(f"Fetching X-ray data for range: {time_range}")
    
    # Map time ranges to NOAA API endpoints
    range_urls = {
        '2h': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json',  # Filter later
        '6h': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json',
        '12h': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',  # Filter later
        '24h': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',
        '2d': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json',
        '5d': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json',  # Filter later
        '7d': 'https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json'
    }
    
    url = range_urls.get(time_range, range_urls['6h'])
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        try:
            data = response.json()
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError) as e:
            if "Extra data" in str(e) and hasattr(e, 'pos'):
                print(f"JSON Extra data detected at pos {e.pos}. Attempting recovery for {time_range}.")
                data = json.loads(response.text[:e.pos])
            else:
                raise e
        
        # Filter data based on time range
        if data and len(data) > 0:
            now = datetime.now(timezone.utc)
            hours_map = {
                '2h': 2, '6h': 6, '12h': 12, '24h': 24,
                '2d': 48, '5d': 120, '7d': 168
            }
            hours = hours_map.get(time_range, 6)
            cutoff = now - timedelta(hours=hours)
            
            filtered_data = [
                d for d in data 
                if datetime.strptime(d['time_tag'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc) >= cutoff
            ]
            
            print(f"Returning {len(filtered_data)} X-ray data points for range {time_range}")
            set_cached(cache_key, filtered_data)
            return jsonify(filtered_data)
        return jsonify([])
    except Exception as e:
        print(f"Error fetching X-ray data for range {time_range}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/proton-data')
def get_proton_data():
    """Get proton flux data with specified time range"""
    time_range = request.args.get('range', '24h')

    cache_key = f'proton_data_{time_range}'
    cached = get_cached(cache_key, max_age_seconds=60)
    if cached:
        return jsonify(cached)
    
    print(f"Fetching proton data for range: {time_range}")
    
    # Map time ranges to NOAA API endpoints
    range_urls = {
        '2h': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json',  # Filter later
        '6h': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json',  # Filter later
        '12h': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json',  # Filter later
        '24h': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json',
        '2d': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json',
        '5d': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json',  # Filter later
        '7d': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json'
    }
    
    url = range_urls.get(time_range, range_urls['24h'])
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Filter data based on time range
        if data and len(data) > 0:
            now = datetime.now(timezone.utc)
            hours_map = {
                '2h': 2, '6h': 6, '12h': 12, '24h': 24,
                '2d': 48, '5d': 120, '7d': 168
            }
            hours = hours_map.get(time_range, 24)
            cutoff = now - timedelta(hours=hours)
            
            filtered_data = [
                d for d in data 
                if datetime.strptime(d['time_tag'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc) >= cutoff
            ]
            
            print(f"Returning {len(filtered_data)} proton data points for range {time_range}")
            set_cached(cache_key, filtered_data)
            return jsonify(filtered_data)
        return jsonify([])
    except Exception as e:
        print(f"Error fetching proton data for range {time_range}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/aurora-image.png')
def get_aurora_image():
    """Generate and serve the aurora monitoring image"""
    try:
        layout = request.args.get('layout', 'new').lower()

        if layout == 'legacy':
            img_buffer = generate_aurora_image()
        else:
            img = generate_full_dashboard_image()
            img_buffer = BytesIO()
            try:
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
            finally:
                img.close()

        force_download = request.args.get('download', '').lower() in {'1', 'true', 'yes'}
        return send_file(img_buffer, mimetype='image/png', as_attachment=force_download, 
                        download_name=f'aurora_dashboard_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")}.png')
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating image: {e}", 500

@app.route('/api/aurora-data')
def get_aurora_data():
    """API endpoint to get all aurora-related data"""
    cache_key = 'aurora_data'
    cached_response = get_cached(cache_key, max_age_seconds=20)
    if cached_response:
        return jsonify(cached_response)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            'solar_wind': executor.submit(fetch_solar_wind_data),
            'kp_data': executor.submit(fetch_kp_index),
            'noaa_scales': executor.submit(fetch_noaa_scales),
            'sw_history': executor.submit(fetch_solar_wind_history),
            'hemi_power': executor.submit(fetch_hemispheric_power),
            'goes_mag': executor.submit(fetch_goes_magnetometer),
        }

        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"Error fetching {key} for /api/aurora-data: {e}")
                results[key] = None

    solar_wind = results['solar_wind']
    kp_data = results['kp_data']
    noaa_scales = results['noaa_scales']
    sw_history = results['sw_history']
    hemi_power = results['hemi_power']
    goes_mag = results['goes_mag']
    
    kp_value = kp_data.get('kp') if kp_data else None
    g_scale = noaa_scales.get('g_scale') if noaa_scales else None
    
    # Helper to convert datetime lists to ISO strings
    def convert_times(data_dict, time_keys):
        if not data_dict: return None
        new_dict = data_dict.copy()
        for key in time_keys:
            if key in new_dict and isinstance(new_dict[key], list):
                new_dict[key] = [t.isoformat() if isinstance(t, datetime) else t for t in new_dict[key]]
        return new_dict

    # Convert history data
    sw_history_clean = convert_times(sw_history, ['times'])
    hemi_power_clean = convert_times(hemi_power, ['times'])
    goes_mag_clean = convert_times(goes_mag, ['goes18_times', 'goes19_times'])
    
    response = {
        'solar_wind': solar_wind,
        'solar_wind_history': sw_history_clean,
        'hemispheric_power': hemi_power_clean,
        'goes_magnetometer': goes_mag_clean,
        'kp_index': kp_data,
        'noaa_scales': noaa_scales,
        'aurora_likelihood': get_aurora_likelihood(kp_value, g_scale),
        'condition_status': get_condition_status(
            kp_value,
            solar_wind.get('speed') if solar_wind else None,
            solar_wind.get('bz') if solar_wind else None,
            g_scale
        ),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

    set_cached(cache_key, response)
    return jsonify(response)

def generate_solar_wind_chart(times, values, label, color, current_value=None, unit='', title=''):
    """
    Generate a small matplotlib chart for solar wind parameters matching frontend theme
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from io import BytesIO
    
    # Larger figure for better quality
    fig, ax = plt.subplots(figsize=(6, 2.2), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Plot data with better styling
    if len(times) > 0 and len(values) > 0:
        # Filter out None values
        valid_data = [(t, v) for t, v in zip(times, values) if v is not None]
        if valid_data:
            valid_times, valid_values = zip(*valid_data)
            ax.plot(valid_times, valid_values, color=color, linewidth=2.5, solid_capstyle='round')
            ax.fill_between(valid_times, valid_values, alpha=0.25, color=color)
    
    # Title with matching style
    if title:
        ax.set_title(title, color='#e2e8f0', fontsize=13, fontweight='bold', 
                    loc='left', pad=10, family='sans-serif')
    
    # Styling to match frontend
    ax.set_ylabel(label, color='#94a3b8', fontsize=10, fontweight='600')
    ax.tick_params(colors='#64748b', labelsize=9, length=4, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#475569')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.grid(True, alpha=0.15, color='#475569', linestyle='-', linewidth=0.8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    
    # Add current value badge
    if current_value is not None and not np.isnan(current_value):
        badge_text = f'{current_value:.1f}{unit}'
        ax.text(0.98, 0.92, badge_text, 
               transform=ax.transAxes, ha='right', va='top',
               color='white', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor=color, alpha=0.9, 
                        edgecolor='none'),
               family='sans-serif')
    
    plt.tight_layout()
    
    buf = _figure_to_png_buffer(
        fig,
        dpi=120,
        facecolor='#0f172a',
        edgecolor='none',
        bbox_inches='tight'
    )
    try:
        chart_img = Image.open(buf).convert('RGBA')
        chart_img.load()
        return chart_img
    finally:
        buf.close()


def generate_full_dashboard_image(
    frame_time=None,
    time_window_end=None,
    solar_wind_full=None,
    kp_data=None,
    scales=None,
    ovation_snapshot=None,
    map_image=None
):
    """
    Generate a complete dashboard image with map and all charts (server-side).
    This mimics what the frontend does with canvas composition.
    
    Args:
        frame_time: Timestamp for this frame (for display purposes)
        time_window_end: End time for the data window (creates animation effect)
    
    Returns:
        PIL Image object with the complete dashboard
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Fetch all current data
    if solar_wind_full is None:
        solar_wind_full = fetch_solar_wind_history()
    if kp_data is None:
        kp_data = fetch_kp_index()
    if scales is None:
        scales = fetch_noaa_scales()
    
    # Create sliding window effect for animation
    if time_window_end and solar_wind_full:
        # Filter data up to the time_window_end to create animation
        times = solar_wind_full['times']
        cutoff_idx = len(times)
        for i, t in enumerate(times):
            if t > time_window_end:
                cutoff_idx = i
                break
        
        # Create windowed data (show last 2 hours up to cutoff)
        solar_wind = {
            'times': times[:cutoff_idx],
            'speeds': solar_wind_full['speeds'][:cutoff_idx],
            'densities': solar_wind_full['densities'][:cutoff_idx],
            'bts': solar_wind_full['bts'][:cutoff_idx],
            'bzs': solar_wind_full['bzs'][:cutoff_idx]
        }
    else:
        solar_wind = solar_wind_full
    
    # Set up the canvas (matching frontend dimensions)
    canvas_width = 2400
    canvas_height = 1500
    bg_color = (15, 23, 42)  # #0f172a
    
    # Create the main canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Resize map to fit left half
    map_width = 1150
    map_height = canvas_height

    if map_image is None:
        map_buffer = generate_map_image(ovation_snapshot=ovation_snapshot)
        try:
            with Image.open(map_buffer) as map_src:
                source_map = map_src.convert('RGBA')
        finally:
            map_buffer.close()
        close_source_map = True
    else:
        source_map = map_image
        close_source_map = False

    try:
        resized_map = source_map.resize((map_width, map_height), Image.Resampling.LANCZOS)
        canvas.paste(resized_map, (0, 0), resized_map if resized_map.mode == 'RGBA' else None)
    finally:
        if 'resized_map' in locals():
            resized_map.close()
        if close_source_map:
            source_map.close()
    
    # Right side: Generate solar wind charts
    right_x = 1180
    chart_y_start = 180
    chart_spacing = 310
    
    if solar_wind and len(solar_wind.get('times', [])) > 0:
        times = solar_wind['times']
        
        # Generate each chart with better styling
        speed_chart = generate_solar_wind_chart(
            times, solar_wind['speeds'], 
            'km/s', '#3b82f6',
            solar_wind['speeds'][-1] if len(solar_wind['speeds']) > 0 and solar_wind['speeds'][-1] is not None else None,
            ' km/s',
            'SOLAR WIND SPEED'
        )
        try:
            canvas.paste(speed_chart, (right_x, chart_y_start), speed_chart if speed_chart.mode == 'RGBA' else None)
        finally:
            speed_chart.close()
        
        density_chart = generate_solar_wind_chart(
            times, solar_wind['densities'],
            'p/cm³', '#a855f7',
            solar_wind['densities'][-1] if len(solar_wind['densities']) > 0 and solar_wind['densities'][-1] is not None else None,
            ' p/cm³',
            'PROTON DENSITY'
        )
        try:
            canvas.paste(density_chart, (right_x, chart_y_start + chart_spacing), density_chart if density_chart.mode == 'RGBA' else None)
        finally:
            density_chart.close()
        
        bt_chart = generate_solar_wind_chart(
            times, solar_wind['bts'],
            'nT', '#f59e0b',
            solar_wind['bts'][-1] if len(solar_wind['bts']) > 0 and solar_wind['bts'][-1] is not None else None,
            ' nT',
            'IMF Bt'
        )
        try:
            canvas.paste(bt_chart, (right_x, chart_y_start + chart_spacing * 2), bt_chart if bt_chart.mode == 'RGBA' else None)
        finally:
            bt_chart.close()
        
        bz_chart = generate_solar_wind_chart(
            times, solar_wind['bzs'],
            'nT', '#ec4899',
            solar_wind['bzs'][-1] if len(solar_wind['bzs']) > 0 and solar_wind['bzs'][-1] is not None else None,
            ' nT',
            'IMF Bz'
        )
        try:
            canvas.paste(bz_chart, (right_x, chart_y_start + chart_spacing * 3), bz_chart if bz_chart.mode == 'RGBA' else None)
        finally:
            bz_chart.close()
    
    # Add Kp index panel (top right)
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        value_font = ImageFont.truetype("arialbd.ttf", 56)
        label_font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    if kp_data:
        kp_val = kp_data.get('kp', 0)
        g_scale = scales.get('g_scale', '0') if scales else '0'
        
        # Draw Kp panel with modern styling
        panel_x = right_x
        panel_y = 20
        panel_w = 700
        panel_h = 140
        draw.rectangle([panel_x, panel_y, panel_x + panel_w, panel_y + panel_h], 
                      fill=(30, 41, 59), outline=(71, 85, 105), width=2)
        
        draw.text((panel_x + 20, panel_y + 15), "Kp Index", fill=(148, 163, 184), font=title_font)
        draw.text((panel_x + 20, panel_y + 50), f"{kp_val:.1f}", fill=(255, 255, 255), font=value_font)
        
        # G-scale badge
        if g_scale and g_scale != '0':
            draw.text((panel_x + 180, panel_y + 75), f"G{g_scale}", fill=(251, 146, 60), font=label_font)
    
    # Add timestamp at bottom right
    if frame_time:
        timestamp_text = frame_time.strftime('%Y-%m-%d %H:%M UTC')
    else:
        timestamp_text = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    draw.text((right_x + 20, canvas_height - 50), timestamp_text, fill=(100, 116, 139), font=label_font)
    
    return canvas


def _prepare_gif_frame(image):
    """Convert a full RGB/RGBA frame to GIF palette mode before storing it."""
    palette_mode = getattr(getattr(Image, 'Palette', None), 'ADAPTIVE', None)
    if palette_mode is None:
        palette_mode = Image.ADAPTIVE
    frame = image.convert('P', palette=palette_mode, colors=256)
    frame.load()
    return frame


def generate_historical_gif(hours_back=2, frame_interval_minutes=5, output_filename=None, frame_duration=500):
    """
    Generate an animated GIF from HISTORICAL aurora data.
    This is MUCH faster than capturing live - generates all frames immediately.
    
    Args:
        hours_back: How many hours of historical data to use (default: 2)
        frame_interval_minutes: Minutes between each frame (default: 5)
        output_filename: Output filename (default: aurora_animation_YYYYMMDD_HHMM.gif)
        frame_duration: Duration of each frame in milliseconds (default: 500ms)
    
    Returns:
        Path to the generated GIF file
    """
    from datetime import timedelta
    
    # Calculate frame timestamps
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)
    total_minutes = int(hours_back * 60)
    num_frames = int(total_minutes / frame_interval_minutes) + 1
    
    print(f"🎬 Generating animated GIF from HISTORICAL data...")
    print(f"   Time range: {start_time.strftime('%Y-%m-%d %H:%M UTC')} to {end_time.strftime('%H:%M UTC')}")
    print(f"   Duration: {hours_back} hours")
    print(f"   Frame interval: {frame_interval_minutes} minute(s)")
    print(f"   Total frames: {num_frames}")
    print(f"   Frame display duration: {frame_duration}ms")
    print(f"")
    
    # Create output directory if it doesn't exist
    output_dir = "aurora_animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        output_filename = f"aurora_historical_{timestamp}.gif"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Generate frames from historical data. Source data and the map are reused
    # across frames so only the time-windowed charts need to change.
    frames = []
    solar_wind_full = fetch_solar_wind_history() or {}
    kp_data = fetch_kp_index() or {}
    scales = fetch_noaa_scales() or {}
    ovation_snapshot = None
    shared_map_image = None

    try:
        ovation_snapshot = fetch_ovation_latest_snapshot()
        map_buffer = generate_map_image(ovation_snapshot=ovation_snapshot)
        try:
            with Image.open(map_buffer) as map_src:
                shared_map_image = map_src.convert('RGBA')
                shared_map_image.load()
        finally:
            map_buffer.close()
    except Exception as e:
        print(f"   Shared map pre-render failed, falling back per frame: {e}")
    
    for i in range(num_frames):
        frame_time = start_time + timedelta(minutes=i * frame_interval_minutes)
        
        print(f"   [{i+1}/{num_frames}] Generating frame for {frame_time.strftime('%H:%M UTC')}...", end='')
        
        try:
            # Generate the FULL dashboard image with sliding time window for animation
            img = generate_full_dashboard_image(
                frame_time,
                time_window_end=frame_time,
                solar_wind_full=solar_wind_full,
                kp_data=kp_data,
                scales=scales,
                ovation_snapshot=ovation_snapshot,
                map_image=shared_map_image
            )
            try:
                frames.append(_prepare_gif_frame(img))
            finally:
                img.close()
            
            print(f" ✓")
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(frames) == 0:
        if shared_map_image is not None:
            shared_map_image.close()
        raise Exception("Failed to generate any frames")
    
    # Save as animated GIF
    print(f"\n💾 Saving GIF to {output_path}...")
    frame_count = len(frames)
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=False
        )
    finally:
        for frame in frames:
            frame.close()
        frames.clear()
        if shared_map_image is not None:
            shared_map_image.close()
    
    # Calculate stats
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    total_animation_duration = (frame_count * frame_duration) / 1000
    
    print(f"\n✅ Animation complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Frames: {frame_count}")
    print(f"   Playback duration: {total_animation_duration:.1f} seconds")
    print(f"   Time period: {hours_back} hours")
    
    return output_path


def generate_animated_gif(hours_duration=2, frame_interval_minutes=1, output_filename=None, frame_duration=500):
    """
    Generate an animated GIF by capturing LIVE aurora dashboard data over time.
    NOTE: This function waits in real-time - use generate_historical_gif() for instant results!
    
    Args:
        hours_duration: How many hours to capture (default: 2)
        frame_interval_minutes: Minutes between each frame capture (default: 1)
        output_filename: Output filename (default: aurora_animation_YYYYMMDD_HHMM.gif)
        frame_duration: Duration of each frame in milliseconds (default: 500ms)
    
    Returns:
        Path to the generated GIF file
    """
    import time
    
    # Calculate total frames and time required
    total_minutes = int(hours_duration * 60)
    num_frames = int(total_minutes / frame_interval_minutes) + 1
    
    print(f"🎬 Generating animated GIF by capturing LIVE data...")
    print(f"   Duration: {hours_duration} hours ({total_minutes} minutes)")
    print(f"   Frame interval: {frame_interval_minutes} minute(s)")
    print(f"   Total frames: {num_frames}")
    print(f"   Estimated capture time: {total_minutes} minutes")
    print(f"   Frame display duration: {frame_duration}ms")
    print(f"")
    print(f"⏰ Starting capture... This will take approximately {total_minutes} minutes.")
    print(f"   You can close the browser - the GIF will be saved when complete.")
    print(f"")
    
    # Create output directory if it doesn't exist
    output_dir = "aurora_animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        output_filename = f"aurora_animation_{timestamp}.gif"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Capture frames in real-time
    frames = []
    start_capture_time = datetime.now(timezone.utc)
    
    for i in range(num_frames):
        capture_time = datetime.now(timezone.utc)
        elapsed_minutes = (capture_time - start_capture_time).total_seconds() / 60
        
        print(f"   [{i+1}/{num_frames}] Capturing frame at {capture_time.strftime('%H:%M:%S UTC')} (elapsed: {elapsed_minutes:.1f}min)...", end='')
        
        try:
            # Capture current dashboard state
            img_buffer = generate_aurora_image()
            
            # Convert to a palette frame before storing to keep long captures lighter.
            img_buffer.seek(0)
            try:
                with Image.open(img_buffer) as frame_src:
                    frame = _prepare_gif_frame(frame_src)
            finally:
                img_buffer.close()
            
            frames.append(frame)
            print(" ✓")
            
            # Wait for next frame (unless this is the last frame)
            if i < num_frames - 1:
                wait_seconds = frame_interval_minutes * 60
                print(f"      ⏳ Waiting {frame_interval_minutes} minute(s) until next capture...")
                time.sleep(wait_seconds)
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            # Continue capturing even if one frame fails
            if i < num_frames - 1:
                time.sleep(frame_interval_minutes * 60)
            continue
    
    if not frames:
        print("❌ No frames captured!")
        return None
    
    # Save as animated GIF
    print(f"")
    print(f"💾 Saving animated GIF to: {output_path}")
    frame_count = len(frames)
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )
    finally:
        for frame in frames:
            frame.close()
        frames.clear()
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    total_animation_duration = frame_count * frame_duration / 1000
    
    print(f"✅ Animation complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Frames: {frame_count}")
    print(f"   Playback duration: {total_animation_duration:.1f} seconds")
    print(f"   Time period captured: {hours_duration} hours")
    
    return output_path



@app.route('/generate-gif')
def generate_gif_endpoint():
    """Web endpoint to trigger animated GIF generation from historical data"""
    try:
        # Get parameters from query string
        from flask import request
        hours = float(request.args.get('hours', 2))
        interval = int(request.args.get('interval', 5))  # Default to 5 min for faster generation
        duration = int(request.args.get('duration', 500))
        mode = request.args.get('mode', 'historical')  # 'historical' or 'live'
        
        # Validate parameters
        if hours < 0.5 or hours > 24:
            return jsonify({'error': 'hours must be between 0.5 and 24'}), 400
        if interval < 1 or interval > 120:
            return jsonify({'error': 'interval must be between 1 and 120 minutes'}), 400
        if duration < 100 or duration > 5000:
            return jsonify({'error': 'duration must be between 100 and 5000 ms'}), 400
        
        # Generate GIF based on mode
        if mode == 'live':
            # Legacy real-time capture (slow)
            gif_path = generate_animated_gif(
                hours_duration=hours,
                frame_interval_minutes=interval,
                frame_duration=duration
            )
        else:
            # Historical data (fast, instant generation)
            gif_path = generate_historical_gif(
                hours_back=hours,
                frame_interval_minutes=interval,
                frame_duration=duration
            )
        
        if gif_path:
            return send_file(gif_path, mimetype='image/gif', as_attachment=True,
                           download_name=os.path.basename(gif_path))
        else:
            return jsonify({'error': 'Failed to generate animation'}), 500
            
    except Exception as e:
        print(f"Error generating GIF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate-solar-overview')
def generate_solar_overview():
    """Generate a PNG image of the solar activity overview"""
    try:
        import matplotlib.dates as mdates
        
        print("[SOLAR_OVERVIEW] Generating solar overview PNG...")
        
        # Fetch current data (2-hour for X-ray, 6-hour for protons)
        xray_response = requests.get('https://services.swpc.noaa.gov/json/goes/primary/xrays-2-hour.json', timeout=10)
        proton_response = requests.get('https://services.swpc.noaa.gov/json/goes/primary/integral-protons-plot-6-hour.json', timeout=10)
        
        xray_data = xray_response.json() if xray_response.status_code == 200 else []
        proton_data = proton_response.json() if proton_response.status_code == 200 else []
        
        print(f"[SOLAR_OVERVIEW] X-ray data points: {len(xray_data)}, Proton data points: {len(proton_data)}")
        
        # URLs for imagery
        goes_131_url = 'https://services.swpc.noaa.gov/images/animations/suvi/primary/131/latest.png'
        goes_ccor_url = 'https://services.swpc.noaa.gov/images/animations/ccor1/latest.jpg'
        
        # Create figure with improved 2x2 grid layout
        fig = plt.figure(figsize=(18, 10), facecolor='#0f172a')
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1],
                     hspace=0.35, wspace=0.35, left=0.06, right=0.96, top=0.90, bottom=0.08)
        
        # Title and timestamp
        fig.text(0.5, 0.96, 'Solar Activity Overview', ha='center', va='top',
                fontsize=26, fontweight='bold', color='#e2e8f0', family='sans-serif')
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        fig.text(0.5, 0.92, timestamp, ha='center', va='top',
                fontsize=13, color='#94a3b8')
        
        # 1. X-Ray Flux Chart (top-left) - FIXED DATA HANDLING
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#1e293b')
        
        xray_plotted = False
        if xray_data and len(xray_data) > 0:
            try:
                # Parse times and flux values
                times = []
                flux_values = []
                for d in xray_data:
                    try:
                        t = datetime.fromisoformat(d['time_tag'].replace('Z', '+00:00'))
                        f = float(d['flux']) if d.get('flux') else None
                        if f and f > 0:
                            times.append(t)
                            flux_values.append(f)
                    except:
                        continue
                
                if len(times) > 0 and len(flux_values) > 0:
                    # Plot the line
                    ax1.plot(times, flux_values, color='#f59e0b', linewidth=3, solid_capstyle='round', zorder=3)
                    ax1.fill_between(times, flux_values, alpha=0.25, color='#f59e0b', zorder=2)
                    xray_plotted = True
                    
                    # Set log scale and limits
                    ax1.set_yscale('log')
                    ax1.set_ylim(1e-8, 1e-2)
                    
                    # Add colored zone backgrounds
                    ax1.axhspan(1e-9, 1e-6, facecolor='#4ade80', alpha=0.08, zorder=0)  # A/B
                    ax1.axhspan(1e-6, 1e-5, facecolor='#fbbf24', alpha=0.08, zorder=0)  # C
                    ax1.axhspan(1e-5, 1e-4, facecolor='#fb923c', alpha=0.08, zorder=0)  # M
                    ax1.axhspan(1e-4, 1e-3, facecolor='#f87171', alpha=0.08, zorder=0)  # X
                    ax1.axhspan(1e-3, 1, facecolor='#e879f9', alpha=0.08, zorder=0)    # X10
                    
                    # Add threshold lines
                    ax1.axhline(y=1e-6, color='#fbbf24', linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)
                    ax1.axhline(y=1e-5, color='#fb923c', linestyle='--', alpha=0.45, linewidth=1.5, zorder=1)
                    ax1.axhline(y=1e-4, color='#f87171', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
                    ax1.axhline(y=1e-3, color='#e879f9', linestyle='--', alpha=0.55, linewidth=1.5, zorder=1)
                    
                    # Add class labels
                    ax1.text(0.99, 0.10, 'C', transform=ax1.transAxes, color='#fbbf24', 
                            va='center', ha='right', fontsize=13, fontweight='bold')
                    ax1.text(0.99, 0.30, 'M', transform=ax1.transAxes, color='#fb923c',
                            va='center', ha='right', fontsize=13, fontweight='bold')
                    ax1.text(0.99, 0.52, 'X', transform=ax1.transAxes, color='#f87171',
                            va='center', ha='right', fontsize=13, fontweight='bold')
                    ax1.text(0.99, 0.73, 'X10', transform=ax1.transAxes, color='#e879f9',
                            va='center', ha='right', fontsize=13, fontweight='bold')
                    
                    # Format x-axis
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
                    
            except Exception as e:
                print(f"[SOLAR_OVERVIEW] Error plotting X-ray: {e}")
        
        if not xray_plotted:
            ax1.text(0.5, 0.5, 'No X-Ray Data Available', ha='center', va='center',
                    transform=ax1.transAxes, color='#64748b', fontsize=14)
        
        ax1.set_title('GOES X-Ray Flux (2-Hour)', color='#e2e8f0', fontsize=16, 
                     fontweight='bold', pad=12, loc='left')
        ax1.set_ylabel('Watts/m²', color='#94a3b8', fontsize=12, fontweight='600')
        ax1.tick_params(axis='both', colors='#64748b', labelsize=10, length=5, width=1.2)
        ax1.grid(True, alpha=0.12, color='#475569', linewidth=0.8, linestyle='-')
        for spine in ax1.spines.values():
            spine.set_color('#475569')
            spine.set_linewidth(1.2)
        
        # 2. Proton Flux Chart (bottom-left) - FIXED DATA HANDLING
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor('#1e293b')
        
        proton_plotted = False
        if proton_data and len(proton_data) > 0:
            try:
                times = []
                p10_vals = []
                p100_vals = []
                
                for d in proton_data:
                    try:
                        t = datetime.fromisoformat(d['time_tag'].replace('Z', '+00:00'))
                        p10 = float(d.get('>=10 MeV', 0.01))
                        p100 = float(d.get('>=100 MeV', 0.01))
                        times.append(t)
                        p10_vals.append(max(p10, 0.01))  # Ensure positive for log scale
                        p100_vals.append(max(p100, 0.01))
                    except:
                        continue
                
                if len(times) > 0:
                    ax2.plot(times, p10_vals, color='#86efac', linewidth=3, label='≥10 MeV', 
                            solid_capstyle='round', zorder=3)
                    ax2.fill_between(times, p10_vals, alpha=0.15, color='#86efac', zorder=1)
                    
                    ax2.plot(times, p100_vals, color='#a78bfa', linewidth=3, label='≥100 MeV',
                            solid_capstyle='round', zorder=3)
                    ax2.fill_between(times, p100_vals, alpha=0.15, color='#a78bfa', zorder=1)
                    
                    proton_plotted = True
                    ax2.set_yscale('log')
                    ax2.set_ylim(0.01, 1e5)
                    
                    # Add S-scale threshold lines
                    ax2.axhline(y=10, color='#fbbf24', linestyle='--', alpha=0.35, linewidth=1.5, zorder=2)
                    ax2.axhline(y=100, color='#fb923c', linestyle='--', alpha=0.4, linewidth=1.5, zorder=2)
                    ax2.axhline(y=1000, color='#ef4444', linestyle='--', alpha=0.45, linewidth=1.5, zorder=2)
                    
                    # Legend
                    ax2.legend(loc='upper left', fontsize=11, facecolor='#1e293b', 
                              edgecolor='#475569', framealpha=0.95, labelcolor='#e2e8f0')
                    
                    # Format x-axis
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')
                    
            except Exception as e:
                print(f"[SOLAR_OVERVIEW] Error plotting protons: {e}")
        
        if not proton_plotted:
            ax2.text(0.5, 0.5, 'No Proton Data Available', ha='center', va='center',
                    transform=ax2.transAxes, color='#64748b', fontsize=14)
        
        ax2.set_title('GOES Proton Flux (6-Hour)', color='#e2e8f0', fontsize=16,
                     fontweight='bold', pad=12, loc='left')
        ax2.set_ylabel('Particles/(cm²·sr·s)', color='#94a3b8', fontsize=12, fontweight='600')
        ax2.set_xlabel('Time (UTC)', color='#94a3b8', fontsize=11, fontweight='600')
        ax2.tick_params(axis='both', colors='#64748b', labelsize=10, length=5, width=1.2)
        ax2.grid(True, alpha=0.12, color='#475569', linewidth=0.8, linestyle='-')
        for spine in ax2.spines.values():
            spine.set_color('#475569')
            spine.set_linewidth(1.2)
        
        # 3. GOES SUVI 131Å (top-right)
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.set_facecolor('#0a0a0a')
        ax3.axis('off')
        try:
            img_response = requests.get(goes_131_url, timeout=10)
            if img_response.status_code == 200:
                img = Image.open(BytesIO(img_response.content))
                ax3.imshow(img, aspect='equal')
                ax3.set_title('GOES SUVI 131Å', color='#e2e8f0', 
                            fontsize=16, fontweight='bold', pad=12, loc='center')
        except Exception as e:
            ax3.text(0.5, 0.5, 'Image Unavailable', ha='center', va='center',
                    transform=ax3.transAxes, color='#64748b', fontsize=14)
            ax3.set_title('GOES SUVI 131Å', color='#e2e8f0', 
                         fontsize=16, fontweight='bold', pad=12, loc='center')
        
        # 4. GOES CCOR-1 Coronagraph (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#0a0a0a')
        ax4.axis('off')
        try:
            img_response = requests.get(goes_ccor_url, timeout=10)
            if img_response.status_code == 200:
                img = Image.open(BytesIO(img_response.content))
                ax4.imshow(img, aspect='equal')
                ax4.set_title('GOES-19 CCOR-1 Coronagraph', color='#e2e8f0', 
                            fontsize=16, fontweight='bold', pad=12, loc='center')
        except Exception as e:
            ax4.text(0.5, 0.5, 'Image Unavailable', ha='center', va='center',
                    transform=ax4.transAxes, color='#64748b', fontsize=14)
            ax4.set_title('GOES-19 CCOR-1 Coronagraph', color='#e2e8f0', 
                         fontsize=16, fontweight='bold', pad=12, loc='center')
        
        buf = _figure_to_png_buffer(
            fig,
            dpi=150,
            facecolor='#0f172a',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.3
        )
        
        print("[SOLAR_OVERVIEW] Solar overview PNG generated successfully")
        return send_file(buf, mimetype='image/png', as_attachment=True,
                        download_name=f'solar-overview-{datetime.now(timezone.utc).strftime("%Y%m%d")}.png')
        
    except Exception as e:
        print(f"[SOLAR_OVERVIEW] Error generating solar overview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_map_image(ovation_snapshot=None):
    """Generate the auroral oval map image using OVATION raw data."""
    import time
    start_time = time.time()
    print("[MAP] Starting generation...")

    if ovation_snapshot is None:
        ovation_snapshot = fetch_ovation_latest_snapshot()

    ovation_lons, ovation_lats, ovation_aurora, ovation_time, _ = extract_ovation_north_frame(ovation_snapshot)
    
    # Create figure with transparent background and reserve a slim top band for header text.
    fig = plt.figure(figsize=(10, 10), facecolor='none')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.94)
    # Adjusted projection to focus more on North America
    ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-95, 55))
    ax_map.set_extent([-135, -55, 25, 85], crs=ccrs.PlateCarree())
    
    # ── Map features (50m) ────────────────────────────────────────────
    # Use 50m Natural Earth data for balance of detail and performance (10m is too slow)
    scale = '50m'
    
    # Use 50m for polygons and primary boundaries
    land_feature = cfeature.NaturalEarthFeature('physical', 'land', scale,
                                        facecolor='#1e293b', edgecolor='none')
    ocean_feature = cfeature.NaturalEarthFeature('physical', 'ocean', scale,
                                         facecolor='#020617', edgecolor='none')
    lakes_feature = cfeature.NaturalEarthFeature('physical', 'lakes', scale,
                                         facecolor='#020617', edgecolor='none')
    coast_feature = cfeature.NaturalEarthFeature('physical', 'coastline', scale,
                                         facecolor='none', edgecolor='#38bdf8')
    borders_feature = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', scale,
                                           facecolor='none', edgecolor='#475569')
    states_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', scale,
                                          facecolor='none', edgecolor="#BBBFC5")
    
    # Use 50m for rivers
    rivers_feature = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale,
                                          facecolor='none', edgecolor='#0e4577')
    
    ax_map.add_feature(ocean_feature, zorder=0)
    ax_map.add_feature(land_feature, zorder=0)
    ax_map.add_feature(lakes_feature, zorder=0)
    
    # Lines
    ax_map.add_feature(rivers_feature, linewidth=0.2, alpha=0.3, zorder=1)
    ax_map.add_feature(coast_feature, linewidth=0.6, alpha=0.8, zorder=1)
    ax_map.add_feature(borders_feature, linewidth=0.5, alpha=0.5, zorder=1)
    ax_map.add_feature(states_feature, linewidth=0.6, linestyle=':', alpha=0.4, zorder=1)
    
    # Add lat/lon grid
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.3, color='#475569', 
                          alpha=0.3, linestyle='--', zorder=2)
    
    # Plot OVATION auroral probability data if available
    if ovation_lons is not None and len(ovation_lons) > 0:
        from scipy.interpolate import griddata
        
        # OVATION data: aurora values range 0-100 (percentage), but typically 0-40
        # Filter to show only significant aurora activity
        threshold = 5
        mask = ovation_aurora >= threshold
        filtered_lons = ovation_lons[mask]
        filtered_lats = ovation_lats[mask]
        filtered_aurora = ovation_aurora[mask]
        
        if len(filtered_lons) > 0:
            # Convert longitude from 0-360 to -180 to 180 BEFORE interpolation
            adjusted_lons = np.where(filtered_lons > 180, filtered_lons - 360, filtered_lons)
            
            # Fix 180° gap by wrapping boundary data
            # Duplicate points near -180 to +180 and vice versa
            wrap_mask_left = adjusted_lons < -170
            wrap_mask_right = adjusted_lons > 170
            
            wrapped_lons = np.concatenate([
                adjusted_lons,
                adjusted_lons[wrap_mask_left] + 360,  # Duplicate left edge to right
                adjusted_lons[wrap_mask_right] - 360  # Duplicate right edge to left
            ])
            wrapped_lats = np.concatenate([
                filtered_lats,
                filtered_lats[wrap_mask_left],
                filtered_lats[wrap_mask_right]
            ])
            wrapped_aurora = np.concatenate([
                filtered_aurora,
                filtered_aurora[wrap_mask_left],
                filtered_aurora[wrap_mask_right]
            ])
            
            # Create a smooth-enough grid without keeping several huge arrays resident.
            lon_grid = np.linspace(-180, 180, 720)
            lat_grid = np.linspace(35, 85, 240)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Interpolate using cubic method for smoother results
            aurora_grid = griddata(
                (wrapped_lons, wrapped_lats), 
                wrapped_aurora,
                (lon_mesh, lat_mesh),
                method='cubic',
                fill_value=np.nan
            )
            
            # Clamp values to valid range (cubic can overshoot)
            aurora_grid = np.clip(aurora_grid, 0, 100)
            
            # Apply multi-pass smoothing for natural-looking contours
            from scipy.ndimage import gaussian_filter
            valid = ~np.isnan(aurora_grid)
            if np.any(valid):
                # Stronger smoothing with multiple passes
                smoothed = np.nan_to_num(aurora_grid, 0)
                smoothed = gaussian_filter(smoothed, sigma=3.5)
                smoothed = gaussian_filter(smoothed, sigma=2.5)
                aurora_grid = np.where(valid, smoothed, np.nan)
            
            # Strict masking: only show aurora above threshold and with valid data
            aurora_grid = np.ma.masked_where((aurora_grid < threshold + 1) | np.isnan(aurora_grid), aurora_grid)
            
            lon_mesh_adjusted = lon_mesh
            
            # Create custom colormap for aurora matching SWPC reference
            # Green (low) -> Yellow (50%) -> Orange (75%) -> Red (90%+)
            colors = ['#10b981', '#84cc16', '#facc15', '#fb923c', '#ef4444']
            positions = [0.0, 0.25, 0.5, 0.75, 1.0]
            cmap = LinearSegmentedColormap.from_list('aurora', list(zip(positions, colors)), N=256)
            
            # Plot using pcolormesh for smooth blending
            # vmax=100 to show full probability range, alpha=0.8 for map visibility
            mesh = ax_map.pcolormesh(lon_mesh_adjusted, lat_mesh, aurora_grid,
                                    cmap=cmap, alpha=0.6, shading='gouraud',
                                    transform=ccrs.PlateCarree(), zorder=3,
                                    vmin=threshold, vmax=100)
            
            # Add colorbar legend with specific ticks matching reference
            cbar = plt.colorbar(mesh, ax=ax_map, orientation='horizontal', 
                               pad=0.02, shrink=0.7, aspect=20, 
                               ticks=[10, 50, 90])
            cbar.set_label('Aurora Probability (%)', fontsize=10, color='#10b981', fontweight='bold')
            cbar.ax.set_xticklabels(['10%', '50%', '90%'])
            cbar.ax.tick_params(labelsize=8, colors='#94a3b8')
            cbar.outline.set_edgecolor('#475569')
    
    # Add city markers
    # Expanded list of North American cities with more US locations
    cities = {
        'Seattle': (-122.3, 47.6),
        'Vancouver': (-123.1, 49.3),
        'Calgary': (-114.1, 51.0),
        'Yellowknife': (-114.4, 62.5),
        'Fairbanks': (-147.7, 64.8),
        'Anchorage': (-149.9, 61.2),
        'Winnipeg': (-97.1, 49.9),
        'Chicago': (-87.6, 41.9),
        'Toronto': (-79.4, 43.7),
        'New York': (-74.0, 40.7),
        'Minneapolis': (-93.3, 45.0),
        'Denver': (-105.0, 39.7),
        'Boston': (-71.1, 42.4),
        'Bismarck': (-100.8, 46.8),
        'Detroit': (-83.0, 42.3),
        'Salt Lake City': (-111.9, 40.8),
        'Quebec City': (-71.2, 46.8),
        'Ottawa': (-75.7, 45.4),
        'Portland': (-122.7, 45.5),
        'Los Angeles': (-118.2, 34.0),
        'Dallas': (-96.8, 32.8),
        'Atlanta': (-84.4, 33.7),
        'Orlando': (-81.4, 28.5),
        'Norfolk': (-76.3, 36.8),
        'Nashville': (-86.8, 36.16),
        'St. Louis': (-90.2, 38.63)
    }
    
    for city, (lon, lat) in cities.items():
        # Plot city marker
        ax_map.plot(lon, lat, 'o', color='#fbbf24', markersize=5, 
                   markeredgecolor='white', markeredgewidth=1.2, 
                   transform=ccrs.PlateCarree(), zorder=5)
        
        # Plot city label with outline for better readability
        # Place label slightly to the right of marker to avoid overlap
        text = ax_map.text(lon + 1.0, lat, city, fontsize=8, ha='left', va='center', 
                  color='#f8fafc', fontweight='bold',
                  transform=ccrs.PlateCarree(), zorder=5)
                          
        # Add black outline to text instead of box
        text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2.5, foreground='#020617')])
    
    # In-image header text, matching the requested top-left/top-right treatment.
    header_title = 'LIVE AURORAL OVAL'
    header_timestamp = ovation_time or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    header_left = fig.text(
        0.015,
        0.985,
        header_title,
        ha='left',
        va='top',
        fontsize=17,
        color='#f8fafc',
        fontweight='bold'
    )
    header_left.set_path_effects([
        matplotlib.patheffects.withStroke(linewidth=3, foreground='#0f172a', alpha=0.9)
    ])

    header_right = fig.text(
        0.985,
        0.985,
        header_timestamp,
        ha='right',
        va='top',
        fontsize=15,
        color='#f8fafc',
        fontweight='bold'
    )
    header_right.set_path_effects([
        matplotlib.patheffects.withStroke(linewidth=3, foreground='#0f172a', alpha=0.9)
    ])

    buf = _figure_to_png_buffer(
        fig,
        dpi=150,
        facecolor='none',
        edgecolor='none',
        bbox_inches='tight'
    )

    # Composite the WTUS logo after rendering so it always sits above the map layers.
    base_img = None
    logo_img = None
    logo_resized = None
    composited = None
    try:
        with Image.open(buf) as base_src:
            base_img = base_src.convert('RGBA')
        logo_path = os.path.join(os.path.dirname(__file__), 'wtusredlogotransparentx.png')
        with Image.open(logo_path) as logo_src:
            logo_img = logo_src.convert('RGBA')

        base_width, base_height = base_img.size
        target_logo_width = max(82, int(base_width * 0.115))
        aspect_ratio = logo_img.height / logo_img.width if logo_img.width else 1
        target_logo_height = max(24, int(target_logo_width * aspect_ratio))
        logo_resized = logo_img.resize((target_logo_width, target_logo_height), Image.Resampling.LANCZOS)

        margin_x = max(42, int(base_width * 0.07))
        margin_y = max(62, int(base_height * 0.115))
        logo_pos = (margin_x, base_height - target_logo_height - margin_y)

        composited = base_img.copy()
        composited.alpha_composite(logo_resized, logo_pos)

        output_buf = BytesIO()
        composited.save(output_buf, format='PNG')
        output_buf.seek(0)
        buf.close()
        buf = output_buf
    except Exception as e:
        print(f"Could not composite WTUS logo on aurora map: {e}")
    finally:
        if logo_img is not None:
            logo_img.close()
        if logo_resized is not None:
            logo_resized.close()
        if composited is not None:
            composited.close()
        if base_img is not None:
            base_img.close()
    
    elapsed = time.time() - start_time
    print(f"[MAP] Generation completed in {elapsed:.2f}s")
    
    return buf


def build_auroral_oval_payload(threshold=1):
    """Return current OVATION auroral oval cells for the interactive map."""
    cache_key = f'auroral_oval_payload_{threshold}'
    cached = get_cached(cache_key, max_age_seconds=30)
    if cached is not None:
        return cached

    ovation_snapshot = fetch_ovation_latest_snapshot()
    lons, lats, values, frame_label, _ = extract_ovation_north_frame(ovation_snapshot)
    if lons is None or lats is None or values is None:
        payload = {
            'points': [],
            'time_tag': frame_label,
            'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'source': 'NOAA SWPC OVATION Aurora 30-minute forecast'
        }
        set_cached(cache_key, payload, timeout=30)
        return payload

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    values = np.asarray(values, dtype=float)
    adjusted_lons = np.where(lons > 180, lons - 360, lons)
    mask = (
        np.isfinite(adjusted_lons) &
        np.isfinite(lats) &
        np.isfinite(values) &
        (values >= threshold) &
        (lats >= 20) &
        (lats <= 85)
    )

    points = [
        {
            'lat': round(float(lat), 3),
            'lon': round(float(lon), 3),
            'value': round(float(value), 1)
        }
        for lat, lon, value in zip(lats[mask], adjusted_lons[mask], values[mask])
    ]

    payload = {
        'points': points,
        'time_tag': frame_label,
        'max_probability': round(float(values[mask].max()), 1) if np.any(mask) else None,
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source': 'NOAA SWPC OVATION Aurora 30-minute forecast'
    }
    set_cached(cache_key, payload, timeout=30)
    return payload


def _request_float(name, default):
    try:
        return float(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def _request_int(name, default, minimum, maximum):
    try:
        value = int(float(request.args.get(name, default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _normalized_map_bounds():
    west = _request_float('west', -135)
    south = _request_float('south', 25)
    east = _request_float('east', -55)
    north = _request_float('north', 85)

    south = max(-80, min(84.5, south))
    north = max(-79.5, min(85, north))
    if north - south < 4:
        mid = (north + south) / 2
        south = max(-80, mid - 2)
        north = min(85, mid + 2)

    span = abs(east - west)
    if span >= 350:
        west, east = -180, 180
    else:
        west = max(-180, min(180, west))
        east = max(-180, min(180, east))
        if east <= west:
            west, east = -180, 180

    if east - west < 6:
        mid = (east + west) / 2
        west = max(-180, mid - 3)
        east = min(180, mid + 3)

    return west, south, east, north


_AURORA_EXPORT_STOP_VALUES = np.array([5, 15, 40, 65, 90, 100], dtype=float)
_AURORA_EXPORT_STOP_COLORS = np.array([
    [16, 185, 129],
    [132, 204, 22],
    [250, 204, 21],
    [251, 146, 60],
    [239, 68, 68],
    [239, 68, 68],
], dtype=float)


def _aurora_export_colormap():
    return LinearSegmentedColormap.from_list(
        'aurora_export_overlay',
        [
            (0.0, '#10b981'),
            (0.25, '#84cc16'),
            (0.5, '#facc15'),
            (0.75, '#fb923c'),
            (1.0, '#ef4444'),
        ],
        N=256,
    )


def _mercator_y_for_latitudes(latitudes):
    limited = np.clip(np.asarray(latitudes, dtype=float), -85.05112878, 85.05112878)
    radians = np.deg2rad(limited)
    sin_values = np.sin(radians)
    return 0.5 - np.log((1 + sin_values) / (1 - sin_values)) / (4 * np.pi)


def _latitudes_for_mercator_y(y_values):
    radians = 2 * np.arctan(np.exp((0.5 - np.asarray(y_values, dtype=float)) * 2 * np.pi)) - (np.pi / 2)
    return np.rad2deg(radians)


def _build_aurora_source_grid(ovation_lons, ovation_lats, ovation_aurora):
    if ovation_lons is None or ovation_lats is None or ovation_aurora is None:
        return None

    lons = np.asarray(ovation_lons, dtype=float)
    lats = np.asarray(ovation_lats, dtype=float)
    values = np.asarray(ovation_aurora, dtype=float)
    valid = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return None

    rounded_lats = np.rint(lats[valid]).astype(int)
    unique_lats = np.unique(rounded_lats)
    if unique_lats.size == 0:
        return None

    south = int(unique_lats.min())
    north = int(unique_lats.max())
    lat_count = (north - south) + 1
    source_grid = np.zeros((lat_count, 360), dtype=np.float32)

    normalized_lons = ((lons[valid] + 180) % 360)
    lon_indices = np.rint(normalized_lons).astype(int) % 360
    lat_indices = rounded_lats - south
    np.maximum.at(source_grid, (lat_indices, lon_indices), values[valid].astype(np.float32))

    return source_grid, south, north


def _render_aurora_export_overlay(ovation_lons, ovation_lats, ovation_aurora, west, south, east, north, width, height, projection):
    source = _build_aurora_source_grid(ovation_lons, ovation_lats, ovation_aurora)
    if source is None:
        return None, None, None

    source_grid, source_south, source_north = source
    source_mask = source_grid > 0
    softened_values = gaussian_filter(source_grid, sigma=(0.8, 1.0))
    softened_weights = gaussian_filter(source_mask.astype(np.float32), sigma=(0.8, 1.0))
    source_grid = np.where(
        softened_weights > 0.02,
        np.maximum(source_grid * 0.8, softened_values / np.maximum(softened_weights, 1e-6)),
        0.0,
    ).astype(np.float32)

    raster_width = max(720, min(2400, int(width)))
    raster_height = max(420, min(1800, int(height)))
    sampled_values = np.full((raster_height, raster_width), np.nan, dtype=np.float32)

    x_samples = np.linspace(west, east, raster_width)
    lon_positions = np.mod(x_samples + 180.0, 360.0)
    lon_floor = np.floor(lon_positions).astype(int) % 360
    lon_ceil = (lon_floor + 1) % 360
    lon_mix = lon_positions - np.floor(lon_positions)

    north_y = _mercator_y_for_latitudes([north])[0]
    south_y = _mercator_y_for_latitudes([south])[0]
    y_samples = np.linspace(north_y, south_y, raster_height)
    sample_lats = _latitudes_for_mercator_y(y_samples)

    red_stops = _AURORA_EXPORT_STOP_COLORS[:, 0]
    green_stops = _AURORA_EXPORT_STOP_COLORS[:, 1]
    blue_stops = _AURORA_EXPORT_STOP_COLORS[:, 2]

    for row_index, sample_lat in enumerate(sample_lats):
        if sample_lat < source_south or sample_lat > source_north:
            continue

        lat_position = min(source_grid.shape[0] - 1.0001, max(0.0, sample_lat - source_south))
        lat_floor = int(np.floor(lat_position))
        lat_ceil = min(source_grid.shape[0] - 1, lat_floor + 1)
        lat_mix = lat_position - lat_floor

        top_left = source_grid[lat_floor, lon_floor]
        top_right = source_grid[lat_floor, lon_ceil]
        bottom_left = source_grid[lat_ceil, lon_floor]
        bottom_right = source_grid[lat_ceil, lon_ceil]
        top_blend = top_left + ((top_right - top_left) * lon_mix)
        bottom_blend = bottom_left + ((bottom_right - bottom_left) * lon_mix)
        sampled_values[row_index, :] = top_blend + ((bottom_blend - top_blend) * lat_mix)

    valid_mask = np.isfinite(sampled_values) & (sampled_values > 0)
    if not np.any(valid_mask):
        return None, None, None

    normalized = np.clip((sampled_values - 5.0) / 95.0, 0.0, 1.0)
    alpha = np.where(np.isfinite(normalized), np.minimum(1.0, 0.15 + np.power(normalized, 1.01) * 0.98), 0.0)
    alpha = gaussian_filter(alpha, sigma=(0.55, 0.6))
    alpha *= np.where(valid_mask, 1.0, 0.0)
    visible = alpha > 0.025

    rgba = np.zeros((raster_height, raster_width, 4), dtype=np.uint8)
    if np.any(visible):
        clipped_values = np.clip(np.where(np.isfinite(sampled_values), sampled_values, 0.0), 5.0, 100.0)
        rgba[:, :, 0] = np.rint(np.interp(clipped_values, _AURORA_EXPORT_STOP_VALUES, red_stops)).astype(np.uint8)
        rgba[:, :, 1] = np.rint(np.interp(clipped_values, _AURORA_EXPORT_STOP_VALUES, green_stops)).astype(np.uint8)
        rgba[:, :, 2] = np.rint(np.interp(clipped_values, _AURORA_EXPORT_STOP_VALUES, blue_stops)).astype(np.uint8)
        rgba[:, :, 3] = np.where(visible, np.rint(np.clip(alpha, 0.0, 1.0) * 255), 0).astype(np.uint8)

    if not np.any(rgba[:, :, 3]):
        return None, None, None

    overlay_image = Image.fromarray(rgba, mode='RGBA')
    glow_radius = max(0.9, min(width, height) / 1500)
    glow_array = np.asarray(overlay_image.filter(ImageFilter.GaussianBlur(radius=glow_radius)))
    overlay_array = np.asarray(overlay_image)

    x_extent = projection.transform_points(ccrs.PlateCarree(), np.array([west, east]), np.array([0.0, 0.0]))[:, 0]
    y_extent = projection.transform_points(ccrs.PlateCarree(), np.array([0.0, 0.0]), np.array([south, north]))[:, 1]
    extent = (float(x_extent[0]), float(x_extent[1]), float(y_extent[0]), float(y_extent[1]))

    return glow_array, overlay_array, extent


def _add_aurora_export_legend(fig):
    legend_box = mpatches.FancyBboxPatch(
        (0.028, 0.03),
        0.26,
        0.11,
        boxstyle='round,pad=0.012,rounding_size=0.016',
        transform=fig.transFigure,
        facecolor=(7 / 255, 16 / 255, 20 / 255, 0.9),
        edgecolor=(71 / 255, 85 / 255, 105 / 255, 0.8),
        linewidth=1.0,
        zorder=10,
    )
    fig.add_artist(legend_box)

    fig.text(0.045, 0.117, 'Aurora probability', ha='left', va='center', fontsize=8.5, color='#edf3f7', fontweight='bold', zorder=11)

    legend_ax = fig.add_axes([0.045, 0.082, 0.225, 0.016], zorder=11)
    legend_ax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=_aurora_export_colormap(), origin='lower')
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_edgecolor('#475569')
        spine.set_linewidth(0.8)

    fig.text(0.045, 0.055, '5%', ha='left', va='center', fontsize=8, color='#cbd5e1', fontweight='bold', zorder=11)
    fig.text(0.1575, 0.055, '50%', ha='center', va='center', fontsize=8, color='#cbd5e1', fontweight='bold', zorder=11)
    fig.text(0.27, 0.055, '90%+', ha='right', va='center', fontsize=8, color='#cbd5e1', fontweight='bold', zorder=11)


def generate_map_view_image(west, south, east, north, width=1400, height=1000, ovation_snapshot=None):
    """Generate an export PNG for the current interactive map bounds."""
    if ovation_snapshot is None:
        ovation_snapshot = fetch_ovation_latest_snapshot()

    ovation_lons, ovation_lats, ovation_aurora, ovation_time, _ = extract_ovation_north_frame(ovation_snapshot)

    dpi = 150
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='#071014')
    projection = ccrs.Mercator()
    ax_map = fig.add_axes([0, 0, 1, 1], projection=projection)
    ax_map.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
    ax_map.set_aspect('auto')
    ax_map.set_facecolor('#071014')

    scale = '50m'
    ax_map.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', scale, facecolor='#071014', edgecolor='none'), zorder=0)
    ax_map.add_feature(cfeature.NaturalEarthFeature('physical', 'land', scale, facecolor='#15212a', edgecolor='none'), zorder=0)
    ax_map.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', scale, facecolor='#071014', edgecolor='none'), zorder=0)
    ax_map.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', scale, facecolor='none', edgecolor='#5fa6bb'), linewidth=0.55, alpha=0.75, zorder=1)
    ax_map.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', scale, facecolor='none', edgecolor='#667887'), linewidth=0.45, alpha=0.6, zorder=1)
    ax_map.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', scale, facecolor='none', edgecolor='#8fa2b1'), linewidth=0.38, linestyle=':', alpha=0.45, zorder=1)
    ax_map.gridlines(draw_labels=False, linewidth=0.25, color='#475569', alpha=0.35, linestyle='--', zorder=2)

    if ovation_lons is not None and len(ovation_lons) > 0:
        glow_overlay, overlay_image, overlay_extent = _render_aurora_export_overlay(
            ovation_lons,
            ovation_lats,
            ovation_aurora,
            west,
            south,
            east,
            north,
            width,
            height,
            projection,
        )

        if overlay_image is not None:
            ax_map.imshow(
                glow_overlay,
                origin='upper',
                extent=overlay_extent,
                transform=projection,
                interpolation='bilinear',
                alpha=0.82,
                zorder=3,
            )
            ax_map.imshow(
                overlay_image,
                origin='upper',
                extent=overlay_extent,
                transform=projection,
                interpolation='bilinear',
                zorder=4,
            )

    header_time = ovation_time or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    title_text = fig.text(0.015, 0.975, 'LIVE AURORAL OVAL', ha='left', va='top', fontsize=13, color='#edf3f7', fontweight='bold')
    title_text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2.5, foreground='#071014', alpha=0.92)])
    timestamp_text = fig.text(0.985, 0.975, header_time, ha='right', va='top', fontsize=10, color='#cbd5e1', fontweight='bold')
    timestamp_text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2.2, foreground='#071014', alpha=0.92)])
    fig.text(0.985, 0.015, 'NOAA SWPC OVATION Aurora 30-minute forecast', ha='right', va='bottom', fontsize=8, color='#94a3b8')

    _add_aurora_export_legend(fig)

    return _figure_to_png_buffer(fig, dpi=dpi, facecolor='#071014', edgecolor='none', pad_inches=0)


@app.route('/api/auroral-oval-data')
def get_auroral_oval_data():
    """Return SWPC OVATION cells for the interactive auroral oval map."""
    try:
        threshold = max(0, min(100, _request_int('threshold', 1, 0, 100)))
        return jsonify(build_auroral_oval_payload(threshold=threshold))
    except Exception as e:
        print(f"Error fetching auroral oval data: {e}")
        return jsonify({'points': [], 'error': str(e)}), 500


@app.route('/aurora-map-view.png')
def get_aurora_map_view_image():
    """Export the auroral oval for the current interactive map viewport."""
    try:
        west, south, east, north = _normalized_map_bounds()
        width = _request_int('width', 1400, 600, 2600)
        height = _request_int('height', 1000, 500, 2200)
        img_buffer = generate_map_view_image(west, south, east, north, width=width, height=height)
        force_download = request.args.get('download', '').lower() in {'1', 'true', 'yes'}
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=force_download,
            download_name=f'auroral_oval_view_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")}.png'
        )
    except Exception as e:
        print(f"Error generating aurora map view: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/aurora-map.png')
def get_aurora_map_image():
    """Generate and serve just the auroral oval map image with caching and concurrency control"""
    global _map_generating
    
    try:
        # Check cache first (30 second cache)
        cached = get_cached('aurora_map', max_age_seconds=30)
        if cached:
            # Return cached image
            buf = BytesIO(cached)
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        
        # Try to acquire lock for generation
        # If another request is already generating, wait for it with timeout
        # Increased timeout to 120s to handle high-res map generation/download
        lock_acquired = _map_generation_lock.acquire(timeout=120)
        
        if not lock_acquired:
            # Timeout waiting for lock - return stale cache if available or error
            stale_cached = get_stale_cached('aurora_map')
            if stale_cached:
                print("Map generation busy, returning stale cache")
                buf = BytesIO(stale_cached)
                buf.seek(0)
                return send_file(buf, mimetype='image/png')
            return "Map generation busy, please retry", 503
        
        try:
            # Double-check cache after acquiring lock (another thread may have populated it)
            cached = get_cached('aurora_map', max_age_seconds=30)
            if cached:
                buf = BytesIO(cached)
                buf.seek(0)
                return send_file(buf, mimetype='image/png')
            
            _map_generating = True
            print("Generating new aurora map...")
            
            # Generate new image
            img_buffer = generate_map_image()
            
            # Cache the image bytes
            img_bytes = img_buffer.getvalue()
            set_cached('aurora_map', img_bytes)
            
            print("Aurora map generated and cached successfully")
            
            img_buffer.seek(0)
            return send_file(img_buffer, mimetype='image/png')
        finally:
            _map_generating = False
            _map_generation_lock.release()
            
    except Exception as e:
        print(f"Error generating map image: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to return stale cache on error
        stale_cached = get_stale_cached('aurora_map')
        if stale_cached:
            print("Returning stale cache due to generation error")
            buf = BytesIO(stale_cached)
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        
        return f"Error generating map image: {e}", 500


@app.route('/api/geomagnetic-alerts')
def get_geomagnetic_alerts():
    """Get SWPC 3-Day Geomagnetic Forecast"""
    try:
        from datetime import datetime, timedelta, timezone
        from collections import defaultdict
        
        # Use the 3-day Kp forecast which is more reliable
        KP_FORECAST_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        
        response = requests.get(KP_FORECAST_URL, timeout=10)
        
        if response.status_code != 200:
            print(f"Error: Got status {response.status_code} from SWPC")
            return jsonify({'forecast': []}), 500
        
        forecast_data = response.json()
        
        # Parse the 3-day forecast
        # Format: [timestamp, kp_value, observed/predicted]
        # Group by date and find max Kp per day
        daily_kp = defaultdict(list)
        current_time = datetime.now(timezone.utc)
        
        print(f"DEBUG: Current UTC time: {current_time}")
        print(f"DEBUG: Processing {len(forecast_data)-1} forecast rows")
        
        for row in forecast_data[1:]:  # Skip header
            try:
                timestamp_str = row[0]  # Format: "YYYY-MM-DD HH:MM:SS"
                kp_str = row[1]
                
                if not timestamp_str or not kp_str:
                    continue
                
                # Parse timestamp
                try:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    dt = dt.replace(tzinfo=timezone.utc)
                except:
                    continue
                
                # Only include current and future predictions
                if dt < current_time - timedelta(hours=3):  # Allow 3-hour lookback
                    continue
                
                # Extract date
                date = dt.strftime('%Y-%m-%d')
                
                # Parse Kp value
                try:
                    kp_value = float(kp_str)
                except (ValueError, TypeError):
                    continue
                
                daily_kp[date].append(kp_value)
                
            except Exception as e:
                continue
        
        print(f"DEBUG: Found {len(daily_kp)} unique dates with forecast data")
        
        # Build forecast for next 3 days starting from today
        daily_forecasts = []
        
        # Get dates for the next 3 days
        forecast_dates = []
        for i in range(3):
            future_date = current_time + timedelta(days=i)
            forecast_dates.append(future_date.strftime('%Y-%m-%d'))
        
        print(f"DEBUG: Forecast dates: {forecast_dates}")
        
        # Create forecast for each day
        for date in forecast_dates:
            if date in daily_kp and daily_kp[date]:
                kp_values = daily_kp[date]
                max_kp = max(kp_values)
                print(f"DEBUG: {date} - Kp values: {kp_values}, Max: {max_kp}")
            else:
                # No data for this day, use a default quiet value
                max_kp = 2.0
                print(f"DEBUG: {date} - No data, using default Kp 2.0")
            
            # Convert Kp to G-scale
            # G-scale mapping (adjusted thresholds)
            # G1 ≈ Kp 5
            # G2 ≈ Kp 6
            # G3 ≈ Kp 7
            # G4 ≈ Kp 8
            # G5 ≈ Kp 9
            # Use conservative thresholds so values like Kp=7.7 map to G3 (not G4)
            if max_kp > 8.67:
                g_scale = 5
            elif max_kp >= 8.0:
                g_scale = 4
            elif max_kp >= 6.67:
                g_scale = 3
            elif max_kp >= 5.67:
                g_scale = 2
            elif max_kp >= 5.00:
                g_scale = 1
            else:
                g_scale = 0
            
            print(f"[GEOMAG_FORECAST] {date} - G-scale: G{g_scale}, Kp: {max_kp:.1f}")
            
            daily_forecasts.append({
                'date': date,
                'max_kp': round(max_kp, 1),
                'g_scale': g_scale
            })
        
        print(f"[GEOMAG_FORECAST] Final forecast: {daily_forecasts}")
        
        return jsonify({'forecast': daily_forecasts})
        
    except Exception as e:
        print(f"[GEOMAG_FORECAST] ERROR fetching geomagnetic forecast: {e}")
        import traceback
        traceback.print_exc()
        
        # Return placeholder data on error
        from datetime import datetime, timedelta, timezone
        current_date = datetime.now(timezone.utc)
        placeholder = [
            {
                'date': (current_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'max_kp': 2.0,
                'g_scale': 0
            }
            for i in range(3)
        ]
        return jsonify({'forecast': placeholder})

@app.route('/api/region-flares/<region_number>')
def get_region_flares(region_number):
    """Get comprehensive flare history for a specific active region from LMSAL archives"""
    try:
        import re
        from bs4 import BeautifulSoup
        from datetime import datetime, timezone, timedelta
        
        region_flares = []
        
        # Fetch multiple days of LMSAL archives to get comprehensive history
        # We'll fetch the last 14 days of archives
        urls_to_fetch = [
            "https://www.lmsal.com/solarsoft/latest_events/",  # Today's latest
        ]
        
        # Add archived snapshots from recent days
        # Archives are at: /solarsoft/ssw/last_events-YYYY/last_events_YYYYMMDD_HHMM/index.html
        # We'll fetch key snapshots going back ~2 weeks
        now = datetime.now(timezone.utc)
        for days_ago in range(1, 15):  # Last 14 days
            date = now - timedelta(days=days_ago)
            # Try the typical 12:01 snapshot time
            archive_url = f"https://www.lmsal.com/solarsoft/ssw/last_events-{date.year}/last_events_{date.strftime('%Y%m%d')}_1201/index.html"
            urls_to_fetch.append(archive_url)
        
        print(f"DEBUG: Fetching flare data from {len(urls_to_fetch)} LMSAL sources...")
        
        # Track unique flares by event name to avoid duplicates
        seen_events = set()
        
        for url in urls_to_fetch:
            try:
                response = requests.get(url, timeout=15)
                if response.status_code != 200:
                    continue  # Skip failed requests
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                tables = soup.find_all('table')
                
                for table in tables:
                    headers = [th.get_text(strip=True) for th in table.find_all('th')]
                    if 'Event#' in headers and 'GOES Class' in headers and 'Derived Position' in headers:
                        rows = table.find_all('tr')
                        
                        for row in rows[1:]:  # Skip header
                            cells = [td.get_text(strip=True) for td in row.find_all('td')]
                            
                            if len(cells) >= 7:
                                event_name = cells[1]
                                
                                # Skip if we've already seen this event
                                if event_name in seen_events:
                                    continue
                                
                                start_time = cells[2]
                                stop_time = cells[3]
                                peak_time = cells[4]
                                goes_class = cells[5]
                                position = cells[6]
                                
                                # Extract region number from position
                                region_match = re.search(r'\(\s*(\d+)\s*\)', position)
                                if region_match:
                                    flare_region = int(region_match.group(1))
                                    
                                    # Only include flares from the requested region
                                    if flare_region == int(region_number):
                                        try:
                                            # Parse times
                                            start_dt = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                            
                                            if '/' in peak_time:
                                                peak_dt = datetime.strptime(peak_time, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                            else:
                                                peak_dt = datetime.strptime(f"{start_dt.strftime('%Y/%m/%d')} {peak_time}", '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                            
                                            if '/' in stop_time:
                                                end_dt = datetime.strptime(stop_time, '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                            else:
                                                end_dt = datetime.strptime(f"{start_dt.strftime('%Y/%m/%d')} {stop_time}", '%Y/%m/%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                            
                                            # Only include C-class and above
                                            if goes_class and (goes_class.startswith('C') or goes_class.startswith('M') or goes_class.startswith('X')):
                                                region_flares.append({
                                                    'begin_time': start_dt.isoformat(),
                                                    'max_time': peak_dt.isoformat(),
                                                    'end_time': end_dt.isoformat(),
                                                    'max_class': goes_class,
                                                    'class': goes_class,
                                                    'region': flare_region,
                                                    'event_name': event_name,
                                                    'position': position
                                                })
                                                seen_events.add(event_name)
                                        except (ValueError, IndexError) as e:
                                            continue
            except Exception as e:
                # Skip individual URL failures
                continue
        
        # Sort flares by time (most recent first)
        region_flares.sort(key=lambda x: x['max_time'], reverse=True)
        
        print(f"DEBUG: Found {len(region_flares)} flares for region {region_number} from LMSAL archives")
        
        return jsonify({
            'region': int(region_number),
            'flares': region_flares,
            'count': len(region_flares),
            'source': 'LMSAL SolarSoft Latest Events + Archives',
            'note': f'Flare history from this active region (past ~14 days).'
        })
        
    except Exception as e:
        print(f"Error fetching flares for region {region_number}:")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'flares': [], 'count': 0}), 500

# ===== NEW API ENDPOINTS FOR ENHANCED FEATURES =====

@app.route('/api/kp-history')
def get_kp_history():
    """Get 3-day Kp index history with observed and forecast data"""
    try:
        KP_FORECAST_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
        print(f"[KP_HISTORY] Fetching Kp history from {KP_FORECAST_URL}")
        response = requests.get(KP_FORECAST_URL, timeout=10)
        data = response.json()
        
        kp_values = []
        current_time = datetime.now(timezone.utc)
        obs_count = 0
        pred_count = 0
        
        for row in data[1:]:  # Skip header
            try:
                time_str = row[0]
                kp_str = row[1]
                obs_pred = row[2] if len(row) > 2 else 'predicted'  # 'observed' or 'predicted'
                
                if not kp_str:
                    continue
                    
                kp = float(kp_str)
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                
                # Include last 48 hours of observed + 72 hours forecast
                if dt >= current_time - timedelta(hours=48):
                    kp_values.append({
                        'time': dt.isoformat(),
                        'kp': kp,
                        'observed': obs_pred == 'observed'
                    })
                    if obs_pred == 'observed':
                        obs_count += 1
                    else:
                        pred_count += 1
            except Exception as e:
                continue
        
        print(f"[KP_HISTORY] Retrieved {obs_count} observed points and {pred_count} forecast points")
        print(f"[KP_HISTORY] Total Kp history points: {len(kp_values)}")
        return jsonify({'kp_values': kp_values})
    except Exception as e:
        print(f"[KP_HISTORY] Error fetching Kp history: {e}")
        return jsonify({'kp_values': []}), 500

@app.route('/api/f107')
def get_f107():
    """Get F10.7 solar radio flux (current value)"""
    try:
        F107_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
        response = requests.get(F107_URL, timeout=10)
        data = response.json()
        
        # Get most recent value
        if data:
            latest = data[-1]
            return jsonify({'flux': latest.get('f10.7', 0), 'time': latest.get('time-tag', '')})
        
        return jsonify({'flux': 0})
    except Exception as e:
        print(f"Error fetching F10.7: {e}")
        return jsonify({'flux': 0}), 500


def fetch_daily_sunspot_summary():
    """Aggregate the latest SWPC sunspot report into a daily Wolf-style number."""
    response = requests.get("https://services.swpc.noaa.gov/json/sunspot_report.json", timeout=10)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        return None

    valid_rows = [
        row for row in rows
        if row.get('Region') and row.get('Type') == 'spt'
    ]
    if not valid_rows:
        return None

    def row_time(row):
        return parse_swpc_datetime(row.get('time_tag')) or parse_swpc_datetime(row.get('Obsdate')) or datetime.min.replace(tzinfo=timezone.utc)

    latest_obsdate = max((row.get('Obsdate') for row in valid_rows if row.get('Obsdate')), default=None)
    latest_date_rows = [
        row for row in valid_rows
        if row.get('Obsdate') == latest_obsdate
    ] if latest_obsdate else valid_rows

    latest_time = max((row_time(row) for row in latest_date_rows), default=None)
    latest_rows = [
        row for row in latest_date_rows
        if row_time(row) == latest_time
    ] if latest_time else latest_date_rows

    region_rows = {}
    for row in latest_rows:
        region = row.get('Region')
        if region is not None:
            region_rows[str(region)] = row

    groups = len(region_rows)
    spots = sum(_safe_int(row.get('Numspot'), 0) for row in region_rows.values())
    area = sum(_safe_int(row.get('Area'), 0) for row in region_rows.values())
    sunspot_number = groups * 10 + spots
    observatories = sorted({
        str(row.get('Observatory'))
        for row in region_rows.values()
        if row.get('Observatory')
    })

    obs_dt = parse_swpc_datetime(latest_obsdate)
    return {
        'daily_sunspot_number': sunspot_number,
        'daily_spot_count': spots,
        'daily_group_count': groups,
        'daily_spot_area': area,
        'daily_sunspot_date': obs_dt.strftime('%Y-%m-%d') if obs_dt else latest_obsdate,
        'daily_sunspot_time': latest_time.strftime('%Y-%m-%dT%H:%M:%SZ') if latest_time else '',
        'daily_observatories': observatories,
        'daily_sunspot_source': 'NOAA SWPC sunspot_report.json'
    }


def fetch_smoothed_sunspot_summary():
    """Fetch the latest official centered 13-month smoothed SSN from SWPC."""
    response = requests.get(SUNSPOTS_SMOOTHED_URL, timeout=10)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        return None

    entries = _coerce_cycle_entries(rows, predicted=False)
    latest = _latest_entry_with(entries, 'smoothed_ssn')
    if not latest:
        return None

    return {
        'smoothed_sunspot_number': _format_number(latest.get('smoothed_ssn'), 1),
        'smoothed_time': latest.get('time') or '',
        'smoothed_source': 'NOAA SWPC sunspots-smoothed.json',
        'smoothed_note': 'Latest centered 13-month smoothed value; the final six months are unavailable by definition.'
    }


@app.route('/api/solar-summary')
def get_solar_summary():
    """Get compact solar-cycle and ephemeris values for the overview page."""
    try:
        cache_key = 'solar_summary'
        cached = get_cached(cache_key, max_age_seconds=1800)
        if cached is not None:
            return jsonify(cached)

        response = requests.get(SOLAR_CYCLE_INDICES_URL, timeout=10)
        response.raise_for_status()
        raw_entries = response.json()
        entries = _coerce_cycle_entries(raw_entries, predicted=False)
        latest = _latest_entry_with(entries, 'ssn', 'f107') or _latest_entry_with(entries, 'ssn') or {}
        latest_smoothed = _latest_entry_with(entries, 'smoothed_ssn') or {}
        daily = fetch_daily_sunspot_summary() or {}
        smoothed = fetch_smoothed_sunspot_summary() or {}

        now = datetime.now(timezone.utc)
        payload = {
            'sunspot_number': daily.get('daily_sunspot_number'),
            'sunspot_time': daily.get('daily_sunspot_date') or '',
            'daily_sunspot_number': daily.get('daily_sunspot_number'),
            'daily_spot_count': daily.get('daily_spot_count'),
            'daily_group_count': daily.get('daily_group_count'),
            'daily_spot_area': daily.get('daily_spot_area'),
            'daily_sunspot_date': daily.get('daily_sunspot_date') or '',
            'daily_sunspot_time': daily.get('daily_sunspot_time') or '',
            'daily_observatories': daily.get('daily_observatories') or [],
            'monthly_sunspot_number': _format_number(latest.get('ssn'), 1),
            'monthly_sunspot_time': latest.get('time') or '',
            'smoothed_sunspot_number': smoothed.get('smoothed_sunspot_number') or _format_number(latest_smoothed.get('smoothed_ssn'), 1),
            'smoothed_time': smoothed.get('smoothed_time') or latest_smoothed.get('time') or '',
            'smoothed_source': smoothed.get('smoothed_source') or 'NOAA SWPC observed solar-cycle indices',
            'smoothed_note': smoothed.get('smoothed_note') or 'Latest centered 13-month smoothed value.',
            'f107': _format_number(latest.get('f107'), 1),
            'f107_time': latest.get('time') or '',
            'carrington_rotation': current_carrington_rotation(now),
            'generated_at': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'source': 'NOAA SWPC sunspot report, smoothed sunspots, and observed solar-cycle indices'
        }
        set_cached(cache_key, payload, timeout=1800)
        return jsonify(payload)
    except Exception as e:
        print(f"Error fetching solar summary: {e}")
        return jsonify({
            'sunspot_number': None,
            'smoothed_sunspot_number': None,
            'f107': None,
            'carrington_rotation': current_carrington_rotation(),
            'error': str(e)
        }), 500


@app.route('/api/swpc-report-texts')
def get_swpc_report_texts():
    """List available SWPC text report products for the report modal."""
    return jsonify({
        'reports': [
            {'id': key, 'title': meta['title']}
            for key, meta in SWPC_REPORT_TEXTS.items()
        ]
    })


@app.route('/api/swpc-report-text/<report_id>')
def get_swpc_report_text(report_id):
    """Fetch one latest SWPC text report through the backend."""
    meta = SWPC_REPORT_TEXTS.get(report_id)
    if not meta:
        return jsonify({'error': 'Unknown report id'}), 404

    try:
        cache_key = f"swpc_report_{report_id}"
        cached = get_cached(cache_key, max_age_seconds=300)
        if cached is not None:
            return jsonify(cached)

        response = requests.get(meta['url'], timeout=10)
        response.raise_for_status()
        text = response.text.strip()
        payload = {
            'id': report_id,
            'title': meta['title'],
            'source_url': meta['url'],
            'text': text,
            'fetched_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        set_cached(cache_key, payload, timeout=300)
        return jsonify(payload)
    except Exception as e:
        print(f"Error fetching SWPC report {report_id}: {e}")
        return jsonify({'error': str(e), 'id': report_id, 'title': meta['title'], 'text': ''}), 500


SOLAR_CYCLE_MINIMA = {
    '24': datetime(2008, 12, 1),
    '25': datetime(2019, 12, 1)
}


def _parse_solar_cycle_date(value):
    if not value:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m', '%Y'):
        try:
            return datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
    return None


def _to_float(value):
    try:
        if value in (None, ''):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_nonnegative_float(value):
    parsed = _to_float(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def _month_diff(start_dt, end_dt):
    return (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)


def _format_month(value):
    if not value:
        return ''
    return value.strftime('%Y-%m')


def _coerce_cycle_entries(raw_entries, predicted=False):
    entries = []
    for entry in raw_entries or []:
        parsed_date = _parse_solar_cycle_date(entry.get('time-tag'))
        if not parsed_date:
            continue
        if predicted:
            entries.append({
                'date': parsed_date,
                'time': _format_month(parsed_date),
                'smoothed_ssn': _to_nonnegative_float(entry.get('predicted_ssn')),
                'f107': _to_nonnegative_float(entry.get('predicted_f10.7'))
            })
        else:
            observed_ssn = _to_nonnegative_float(entry.get('observed_swpc_ssn'))
            if observed_ssn is None:
                observed_ssn = _to_nonnegative_float(entry.get('ssn'))

            smoothed_ssn = _to_nonnegative_float(entry.get('smoothed_swpc_ssn'))
            if smoothed_ssn is None:
                smoothed_ssn = _to_nonnegative_float(entry.get('smoothed_ssn'))

            entries.append({
                'date': parsed_date,
                'time': _format_month(parsed_date),
                'ssn': observed_ssn,
                'smoothed_ssn': smoothed_ssn,
                'f107': _to_nonnegative_float(entry.get('f10.7'))
            })
    entries.sort(key=lambda item: item['date'])
    return entries


def _latest_entry_with(entries, *keys):
    for entry in reversed(entries):
        if all(entry.get(key) is not None for key in keys):
            return entry
    return None


def _entry_months_from_start(entries, cycle_start):
    series = []
    for entry in entries:
        month = _month_diff(cycle_start, entry['date'])
        if month >= 0:
            series.append({
                'month': month,
                'time': entry['time'],
                'ssn': entry.get('ssn'),
                'smoothed_ssn': entry.get('smoothed_ssn'),
                'f107': entry.get('f107')
            })
    return series


def _lookup_cycle_point(series, month):
    for point in series:
        if point['month'] == month:
            return point
    return None


def _format_number(value, digits=1):
    if value is None:
        return None
    return round(value, digits)


def _julian_date(dt):
    """Convert a UTC datetime to Julian Date."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    year = dt.year
    month = dt.month
    day = dt.day + (
        dt.hour + (dt.minute + (dt.second + dt.microsecond / 1_000_000) / 60) / 60
    ) / 24

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + (a // 4)
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5


def current_carrington_rotation(now=None):
    """Return the current Carrington rotation number using the standard synodic period."""
    now = now or datetime.now(timezone.utc)
    jd = _julian_date(now)
    return int((jd - 2398167.329) / 27.2753) + 1


def _describe_cycle_phase(current_month, observed_peak_month, six_month_change, twelve_month_change):
    if observed_peak_month is None:
        if six_month_change is not None and six_month_change > 0:
            return 'Rising'
        if six_month_change is not None and six_month_change < 0:
            return 'Cooling'
        return 'Active Cycle'

    months_since_peak = current_month - observed_peak_month
    if months_since_peak >= 6 and ((six_month_change is not None and six_month_change < 0) or (twelve_month_change is not None and twelve_month_change < 0)):
        return 'Declining'
    if months_since_peak >= 0:
        return 'Near Peak / Easing'
    if six_month_change is not None and six_month_change > 0:
        return 'Rising'
    return 'Active Cycle'


def _describe_cycle_phase_detail(phase, latest, observed_peak, six_month_change):
    latest_month = latest['time'] if latest else 'the latest month'
    if phase == 'Declining' and observed_peak:
        change_text = ''
        if six_month_change is not None:
            change_text = f" Smoothed activity is {abs(_format_number(six_month_change)):.1f} lower than six months earlier."
        return f"Cycle 25 is easing from the observed smoothed high reached in {observed_peak['time']}.{change_text}"
    if phase == 'Near Peak / Easing' and observed_peak:
        return f"Cycle 25 remains elevated after the observed smoothed high in {observed_peak['time']}."
    if phase == 'Rising':
        return f"Cycle 25 is still climbing in the latest observed data through {latest_month}."
    if phase == 'Cooling':
        return f"Cycle 25 has softened in the latest observed data through {latest_month}."
    return f"Cycle 25 remains active in the latest observed data through {latest_month}."


def _build_solar_cycle_summary(observed_entries, predicted_entries):
    latest = _latest_entry_with(observed_entries, 'ssn', 'smoothed_ssn', 'f107')
    if not latest:
        return {
            'current': {},
            'cycle25': {},
            'comparison': {},
            'milestones': []
        }

    current_cycle_start = SOLAR_CYCLE_MINIMA['25']
    cycle25_observed = _entry_months_from_start(observed_entries, current_cycle_start)
    cycle25_predicted = _entry_months_from_start(predicted_entries, current_cycle_start)

    current_month = _month_diff(current_cycle_start, latest['date'])
    prediction_now = _lookup_cycle_point(cycle25_predicted, current_month)

    six_month_reference = next((entry for entry in reversed(observed_entries) if _month_diff(entry['date'], latest['date']) >= 6 and entry.get('smoothed_ssn') is not None), None)
    twelve_month_reference = next((entry for entry in reversed(observed_entries) if _month_diff(entry['date'], latest['date']) >= 12 and entry.get('smoothed_ssn') is not None), None)

    predicted_peak = max(
        (entry for entry in cycle25_predicted if entry.get('smoothed_ssn') is not None),
        key=lambda item: item['smoothed_ssn'],
        default=None
    )
    observed_peak = max(
        (entry for entry in cycle25_observed if entry.get('smoothed_ssn') is not None),
        key=lambda item: item['smoothed_ssn'],
        default=None
    )

    six_month_change = None
    if six_month_reference:
        six_month_change = latest['smoothed_ssn'] - six_month_reference['smoothed_ssn']

    twelve_month_change = None
    if twelve_month_reference:
        twelve_month_change = latest['smoothed_ssn'] - twelve_month_reference['smoothed_ssn']

    months_to_peak = predicted_peak['month'] - current_month if predicted_peak else None
    observed_peak_month = observed_peak['month'] if observed_peak else None
    phase = _describe_cycle_phase(current_month, observed_peak_month, six_month_change, twelve_month_change)
    phase_detail = _describe_cycle_phase_detail(phase, latest, observed_peak, six_month_change)

    return {
        'current': {
            'time': latest['time'],
            'ssn': _format_number(latest['ssn']),
            'smoothed_ssn': _format_number(latest['smoothed_ssn']),
            'f107': _format_number(latest['f107']),
            'six_month_change': _format_number(six_month_change),
            'twelve_month_change': _format_number(twelve_month_change),
            'predicted_ssn_now': _format_number(prediction_now.get('smoothed_ssn') if prediction_now else None)
        },
        'cycle25': {
            'start': _format_month(current_cycle_start),
            'months_elapsed': current_month,
            'phase': phase,
            'phase_detail': phase_detail,
            'predicted_peak_time': predicted_peak['time'] if predicted_peak else None,
            'predicted_peak_ssn': _format_number(predicted_peak.get('smoothed_ssn') if predicted_peak else None),
            'months_to_predicted_peak': months_to_peak,
            'observed_peak_time': observed_peak['time'] if observed_peak else None,
            'observed_peak_ssn': _format_number(observed_peak.get('smoothed_ssn') if observed_peak else None)
        },
        'comparison': {},
        'milestones': []
    }

@app.route('/api/solar-cycle-data')
def get_solar_cycle_data():
    """Get solar cycle progression data (sunspot numbers over time)"""
    try:
        SOLAR_CYCLE_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
        PREDICTED_URL = "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json"
        
        # Fetch observed data
        obs_response = requests.get(SOLAR_CYCLE_URL, timeout=10)
        obs_data = obs_response.json()
        
        # Fetch predicted data
        pred_response = requests.get(PREDICTED_URL, timeout=10)
        pred_data = pred_response.json()

        observed_entries = _coerce_cycle_entries(obs_data, predicted=False)
        predicted_entries = _coerce_cycle_entries(pred_data, predicted=True)
        
        # Process observed data (all available data for full cycle view)
        observed = {
            'times': [],
            'sunspot_numbers': [],
            'smoothed_ssn': [],
            'f107': []
        }
        
        if observed_entries:
            for entry in observed_entries:
                observed['times'].append(entry['time'])
                observed['sunspot_numbers'].append(entry.get('ssn'))
                observed['smoothed_ssn'].append(entry.get('smoothed_ssn'))
                observed['f107'].append(entry.get('f107'))
        
        # Process predicted data
        predicted = {
            'times': [],
            'smoothed_ssn': [],
            'f107': []
        }
        
        if predicted_entries:
            for entry in predicted_entries:
                predicted['times'].append(entry['time'])
                predicted['smoothed_ssn'].append(entry.get('smoothed_ssn'))
                predicted['f107'].append(entry.get('f107'))

        summary = _build_solar_cycle_summary(observed_entries, predicted_entries)
        normalized_comparison = {
            'cycle24': _entry_months_from_start(observed_entries, SOLAR_CYCLE_MINIMA['24']),
            'cycle25': _entry_months_from_start(observed_entries, SOLAR_CYCLE_MINIMA['25'])
        }
        
        return jsonify({
            'observed': observed,
            'predicted': predicted,
            'summary': summary,
            'normalized_comparison': normalized_comparison
        })
    except Exception as e:
        print(f"Error fetching solar cycle data: {e}")
        return jsonify({
            'observed': {'times': [], 'sunspot_numbers': [], 'smoothed_ssn': [], 'f107': []},
            'predicted': {'times': [], 'smoothed_ssn': [], 'f107': []},
            'summary': {'current': {}, 'cycle25': {}, 'comparison': {}, 'milestones': []},
            'normalized_comparison': {'cycle24': [], 'cycle25': []}
        }), 500

@app.route('/api/cme-events')
def get_cme_events():
    """Get CME events from NASA DONKI with improved Earth-direction detection"""
    try:
        # NASA DONKI CME endpoint (last 7 days)
        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        CME_URL = f"https://api.nasa.gov/DONKI/CME?startDate={start_date}&endDate={end_date}&api_key=bgduJ4idKoFqHnlU7nUkToH4QJtrg7F44xhiuAwm"
        
        response = requests.get(CME_URL, timeout=15)
        if response.status_code != 200:
            print(f"Error fetching CME events: NASA DONKI returned {response.status_code}")
            return jsonify({
                'events': [],
                'error': 'NASA DONKI is unavailable or the API quota has been exceeded.'
            }), 502

        data = response.json()
        if not isinstance(data, list):
            print(f"Error fetching CME events: unexpected payload {type(data).__name__}")
            return jsonify({
                'events': [],
                'error': 'NASA DONKI returned an unexpected response.'
            }), 502
        
        events = []
        for cme in data[:15]:  # Check more CMEs for better filtering
            # Get the most accurate analysis
            analyses = cme.get('cmeAnalyses', [])
            if not analyses:
                continue
                
            # Find the most accurate analysis
            best_analysis = None
            for analysis in analyses:
                if analysis.get('isMostAccurate', False):
                    best_analysis = analysis
                    break
            
            # Fallback to latest analysis if no "most accurate" flag
            if not best_analysis and analyses:
                best_analysis = analyses[-1]
            
            if not best_analysis:
                continue
            
            # Improved Earth-direction detection
            # 1. Check halfAngle (width) - wider CMEs more likely to impact
            # 2. Check latitude/longitude relative to Earth position
            # 3. Prioritize CMEs explicitly marked with Earth in enlilList
            
            half_angle = best_analysis.get('halfAngle', 0) or 0
            latitude = best_analysis.get('latitude', 0) or 0
            longitude = best_analysis.get('longitude', 0) or 0
            speed = best_analysis.get('speed', 0) or 0
            
            # Earth is at latitude ~7.25° (solar equator tilt), longitude varies with time of year
            # For simplicity, CMEs within ±45° longitude and ±30° latitude with wide half-angles
            is_earth_directed = False
            impact_probability = 'Low'
            
            # Check ENLIL simulation list for explicit Earth mention
            enlil_list = best_analysis.get('enlilList', [])
            has_earth_impact_sim = any('EARTH' in str(sim).upper() for sim in enlil_list) if enlil_list else False
            
            # Scoring system for Earth impact probability
            score = 0
            
            # Half-angle scoring (wider = more likely to hit Earth)
            if half_angle > 60:
                score += 3
            elif half_angle > 40:
                score += 2
            elif half_angle > 20:
                score += 1
            
            # Latitude scoring (closer to ecliptic = higher chance)
            if abs(latitude) < 15:
                score += 2
            elif abs(latitude) < 30:
                score += 1
            
            # Longitude scoring (front-side CMEs)
            # Longitude ~0° is center, ±90° is limb
            if abs(longitude) < 30:
                score += 3
            elif abs(longitude) < 60:
                score += 2
            elif abs(longitude) < 90:
                score += 1
            
            # Speed scoring (faster = more dangerous)
            if speed > 1000:
                score += 2
            elif speed > 700:
                score += 1
            
            # ENLIL simulation boost
            if has_earth_impact_sim:
                score += 4
            
            # Determine if Earth-directed and impact probability
            if score >= 6 or has_earth_impact_sim:
                is_earth_directed = True
                if score >= 9:
                    impact_probability = 'High'
                elif score >= 7:
                    impact_probability = 'Moderate'
                else:
                    impact_probability = 'Low'
            
            # Only include Earth-directed CMEs
            if is_earth_directed:
                # Calculate approximate arrival time (simple model: distance / speed)
                # Sun-Earth distance: ~150 million km
                if speed > 100:  # Avoid division by very small numbers
                    travel_time_hours = 150000000 / (speed * 3600)  # Convert km/s to hours
                    launch_time = cme.get('startTime', '')
                    if launch_time:
                        try:
                            launch_dt = datetime.strptime(launch_time, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                            arrival_dt = launch_dt + timedelta(hours=travel_time_hours)
                            arrival_str = arrival_dt.strftime("%Y-%m-%d %H:%M UTC")
                        except:
                            arrival_str = 'TBD'
                    else:
                        arrival_str = 'TBD'
                else:
                    arrival_str = 'TBD'
                
                events.append({
                    'id': cme.get('activityID', 'Unknown')[:20],
                    'launch_time': cme.get('startTime', '').replace('T', ' ').replace('Z', ' UTC'),
                    'speed': int(speed),
                    'probability': impact_probability,
                    'arrival': arrival_str,
                    'source': cme.get('sourceLocation', 'Unknown'),
                    'half_angle': int(half_angle)
                })
        
        # Sort by launch time (most recent first)
        events.sort(key=lambda x: x['launch_time'], reverse=True)
        
        return jsonify({'events': events[:10]})  # Return top 10
    except Exception as e:
        print(f"Error fetching CME events: {e}")
        return jsonify({'events': []}), 500

@app.route('/api/electron-flux')
def get_electron_flux():
    """Get GOES electron flux data"""
    try:
        ELECTRON_URL = "https://services.swpc.noaa.gov/json/goes/primary/electrons-6-hour.json"
        response = requests.get(ELECTRON_URL, timeout=10)
        data = response.json()
        
        times = []
        flux = []
        
        for entry in data:
            try:
                time_str = entry.get('time_tag')
                e2 = entry.get('electron_flux_2')
                
                if time_str and e2 is not None:
                    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    times.append(dt.isoformat())
                    flux.append(float(e2))
            except:
                continue
        
        return jsonify({'times': times, 'flux': flux})
    except Exception as e:
        print(f"Error fetching electron flux: {e}")
        return jsonify({'times': [], 'flux': []}), 500

@app.route('/api/region-heatmap')
def get_region_heatmap():
    """Deprecated: avoid mock active-region heatmaps without historical source data."""
    try:
        return jsonify({
            'regions': [],
            'days': [],
            'activity': [],
            'note': 'Region heatmap disabled because historical per-region activity is not available from the current SWPC feed.'
        })
    except Exception as e:
        print(f"Error fetching region heatmap: {e}")
        return jsonify({'regions': [], 'days': [], 'activity': []}), 500

@app.route('/api/noaa-scales')
def get_noaa_scales_current():
    """Get current NOAA space weather scale values"""
    try:
        scales = fetch_noaa_scales()
        
        if scales:
            return jsonify({
                'r_current': int(scales.get('r_scale', '0')),
                's_current': int(scales.get('s_scale', '0')),
                'g_current': int(scales.get('g_scale', '0'))
            })
        
        return jsonify({'r_current': 0, 's_current': 0, 'g_current': 0})
    except Exception as e:
        print(f"Error fetching NOAA scales: {e}")
        return jsonify({'r_current': 0, 's_current': 0, 'g_current': 0}), 500

@app.route('/api/historical-events')
def get_historical_events():
    """Get notable historical solar events for today's date"""
    try:
        # Hardcoded notable events database (simplified)
        today = datetime.now(timezone.utc)
        month_day = today.strftime('%m-%d')
        
        events_db = {
            '09-01': [{'date': '1859-09-01', 'title': 'Carrington Event', 'description': 'The most powerful geomagnetic storm in recorded history.'}],
            '03-13': [{'date': '1989-03-13', 'title': 'Quebec Blackout', 'description': 'Geomagnetic storm caused 9-hour power outage in Quebec.'}],
            '10-28': [{'date': '2003-10-28', 'title': 'Halloween Storms', 'description': 'Series of X-class flares and CMEs.'}],
        }
        
        events = events_db.get(month_day, [])
        return jsonify({'events': events})
    except Exception as e:
        print(f"Error fetching historical events: {e}")
        return jsonify({'events': []}), 500

@app.route('/api/solar-flares')
def get_solar_flares():
    """Get recent solar flares from NOAA DONKI API"""
    try:
        # DONKI API endpoint for flares (last 30 days)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        url = f"https://api.nasa.gov/DONKI/FLR"
        params = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'api_key': 'bgduJ4idKoFqHnlU7nUkToH4QJtrg7F44xhiuAwm'
        }
        
        print(f"[SOLAR_FLARES] Fetching flares from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        response = requests.get(url, params=params, timeout=10)
        
        # Check if response is successful
        if response.status_code != 200:
            print(f"DONKI API error: {response.status_code}")
            # Return empty array on API error
            resp = jsonify({'flares': [], 'error': 'API quota exceeded or unavailable. Please add your own NASA API key.'})
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        
        flares_data = response.json()
        print(f"[SOLAR_FLARES] Retrieved {len(flares_data)} total flares from DONKI")
        
        # Include C, M, and X class flares (filtering done on frontend)
        significant_flares = []
        for flare in flares_data:
            class_type = flare.get('classType', '')
            if class_type and (class_type.startswith('C') or class_type.startswith('M') or class_type.startswith('X')):
                significant_flares.append({
                    'time': flare.get('beginTime', 'N/A'),
                    'class': class_type,
                    'source': flare.get('sourceLocation', 'N/A'),
                    'region': flare.get('activeRegionNum', 'N/A'),
                    'peak_time': flare.get('peakTime', 'N/A'),
                    'linked_events': flare.get('linkedEvents', [])
                })
        
        # Sort by time (most recent first)
        significant_flares.sort(key=lambda x: x['time'], reverse=True)
        
        print(f"[SOLAR_FLARES] Filtered to {len(significant_flares)} C/M/X class flares")
        if significant_flares:
            print(f"[SOLAR_FLARES] Most recent: {significant_flares[0]['class']} at {significant_flares[0]['time']}")
        
        resp = jsonify({'flares': significant_flares[:20]})  # Limit to 20 most recent
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    except requests.exceptions.RequestException as e:
        print(f"Error fetching solar flares (network): {e}")
        return jsonify({'flares': [], 'error': 'Network error. Check your connection or API key.'})
    except Exception as e:
        print(f"Error fetching solar flares: {e}")
        return jsonify({'flares': [], 'error': str(e)})

@app.route('/api/swpc-events')
def get_swpc_events():
    """Get real-time solar events from NOAA SWPC (supplements DONKI data)"""
    try:
        # NOAA SWPC provides a real-time solar event feed
        url = "https://services.swpc.noaa.gov/json/solar_events_last_30_days.json"
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"[SWPC_EVENTS] Error: {response.status_code}")
            return jsonify({'events': [], 'error': 'SWPC API unavailable'})
        
        events_data = response.json()
        print(f"[SWPC_EVENTS] Retrieved {len(events_data)} events from SWPC")
        
        # Filter for X-ray events (Type 1)
        flare_events = []
        for event in events_data:
            event_type = event.get('event_type', '')
            if event_type == '1':  # Type 1 = X-ray event (solar flare)
                particulars = event.get('particulars', '')
                # Parse the flare class from particulars (e.g., "M1.2" or "X2.5")
                flare_class = particulars.split()[0] if particulars else None
                
                if flare_class and (flare_class[0] in ['C', 'M', 'X']):
                    flare_events.append({
                        'time': event.get('event_starttime', 'N/A'),
                        'class': flare_class,
                        'source': event.get('active_region', 'N/A'),
                        'region': event.get('active_region_num', 'N/A'),
                        'peak_time': event.get('event_peaktime', 'N/A'),
                        'end_time': event.get('event_endtime', 'N/A'),
                        'linked_events': None,
                        'swpc_realtime': True
                    })
        
        print(f"[SWPC_EVENTS] Filtered to {len(flare_events)} flare events (C/M/X)")
        if flare_events:
            print(f"[SWPC_EVENTS] Most recent: {flare_events[0]['class']} at {flare_events[0]['time']}")
        
        resp = jsonify({'events': flare_events})
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    except Exception as e:
        print(f"[SWPC_EVENTS] Error: {e}")
        return jsonify({'events': [], 'error': str(e)})

@app.route('/api/aurora-probability')
def calculate_aurora_probability():
    """Calculate aurora visibility probability based on latitude and current conditions"""
    try:
        latitude = float(request.args.get('lat', 0))
        
        # Fetch current Kp index
        kp_data = fetch_kp_index()
        current_kp = kp_data.get('kp_index', 0) if kp_data else 0
        
        # Aurora probability calculation based on Kp and latitude
        # Reference: https://www.swpc.noaa.gov/products/aurora-30-minute-forecast
        
        if latitude < 0:
            latitude = abs(latitude)  # Southern hemisphere
        
        # Probability matrix [Kp][latitude_band]
        # Latitude bands: <50, 50-55, 55-60, 60-65, 65-70, >70
        prob_matrix = {
            0: [0, 0, 0, 5, 15, 40],
            1: [0, 0, 0, 10, 25, 50],
            2: [0, 0, 5, 20, 40, 65],
            3: [0, 0, 15, 35, 55, 75],
            4: [0, 5, 25, 50, 70, 85],
            5: [5, 15, 40, 65, 80, 95],
            6: [15, 30, 55, 75, 90, 95],
            7: [25, 45, 70, 85, 95, 95],
            8: [40, 60, 80, 90, 95, 95],
            9: [55, 75, 90, 95, 95, 95]
        }
        
        # Determine latitude band
        if latitude < 50:
            band = 0
        elif latitude < 55:
            band = 1
        elif latitude < 60:
            band = 2
        elif latitude < 65:
            band = 3
        elif latitude < 70:
            band = 4
        else:
            band = 5
        
        kp_rounded = min(9, max(0, round(current_kp)))
        probability = prob_matrix.get(kp_rounded, [0]*6)[band]
        
        # Visibility message
        if probability >= 75:
            visibility = "Excellent"
            color = "#22c55e"
        elif probability >= 50:
            visibility = "Good"
            color = "#86efac"
        elif probability >= 25:
            visibility = "Moderate"
            color = "#f59e0b"
        elif probability >= 10:
            visibility = "Low"
            color = "#fb923c"
        else:
            visibility = "Very Low"
            color = "#8b8b8b"
        
        return jsonify({
            'probability': probability,
            'visibility': visibility,
            'color': color,
            'kp': current_kp,
            'latitude': latitude
        })
    except Exception as e:
        print(f"Error calculating aurora probability: {e}")
        return jsonify({'probability': 0, 'visibility': 'Unknown', 'color': '#8b8b8b', 'kp': 0}), 500

@app.route('/api/magnetometer-stations')
def get_magnetometer_data():
    """Get magnetometer data from multiple ground stations"""
    try:
        # Fetch data from INTERMAGNET or USGS stations
        # For now, using GOES magnetometer as primary source
        goes_data = fetch_goes_magnetometer()
        
        if goes_data:
            return jsonify({
                'stations': [{
                    'name': 'GOES Satellite',
                    'data': goes_data,
                    'location': 'Geostationary Orbit'
                }]
            })
        
        return jsonify({'stations': []}), 200
    except Exception as e:
        print(f"Error fetching magnetometer data: {e}")
        return jsonify({'stations': []}), 500

@app.route('/api/coronal-holes')
def get_coronal_holes():
    """Get coronal hole information and high-speed stream predictions"""
    try:
        # NASA DONKI Coronal Hole (CH) Analysis
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        url = "https://api.nasa.gov/DONKI/CH"
        params = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'api_key': 'bgduJ4idKoFqHnlU7nUkToH4QJtrg7F44xhiuAwm'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        # Check if response is successful
        if response.status_code != 200:
            print(f"DONKI CH API error: {response.status_code}")
            return jsonify({'coronal_holes': [], 'error': 'API quota exceeded or unavailable.'})
        
        ch_data = response.json()
        
        # Process coronal holes and HSS predictions
        coronal_holes = []
        for ch in ch_data:
            # Calculate estimated arrival time of high-speed stream
            # Typical solar wind transit time: 2-4 days
            observed_date_str = ch.get('observedDate', '')
            if not observed_date_str:
                continue
                
            try:
                observed_date = datetime.fromisoformat(observed_date_str.replace('Z', '+00:00'))
            except:
                continue
            
            # Estimate HSS arrival (simplified - real calculation would use solar rotation and CH location)
            est_arrival = observed_date + timedelta(days=3)
            
            coronal_holes.append({
                'id': ch.get('chID', 'Unknown'),
                'observed_date': ch.get('observedDate', 'N/A'),
                'latitude': ch.get('latitude', 0),
                'longitude': ch.get('longitude', 0),
                'area': ch.get('area', 0),
                'hss_arrival_estimate': est_arrival.isoformat() if est_arrival > datetime.now(timezone.utc) else None,
                'speed_estimate': '500-700 km/s',  # Typical HSS speed
                'source': ch.get('observatory', 'SDO')
            })
        
        return jsonify({'coronal_holes': coronal_holes})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching coronal holes (network): {e}")
        return jsonify({'coronal_holes': [], 'error': 'Network error'})
    except Exception as e:
        print(f"Error fetching coronal holes: {e}")
        return jsonify({'coronal_holes': [], 'error': str(e)})

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical solar data for a specific date"""
    try:
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({'error': 'Date parameter required'}), 400
        
        # Parse the date
        target_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # NOAA archives X-ray and proton data for past dates
        # Format: https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json returns last 24h
        # For historical data, we need to construct URLs for that specific date
        
        # For simplicity, we'll fetch the most recent archived data
        # Note: NOAA's JSON endpoints typically only have recent data (last few days)
        # For true historical data, you'd need to access their archive FTP servers
        
        xray_data = []
        proton_data = []
        kp_data = []
        
        # Try to fetch X-ray data (NOAA keeps ~7 days of JSON data)
        try:
            xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
            response = requests.get(xray_url, timeout=10)
            if response.status_code == 200:
                all_xray = response.json()
                # Filter for the target date
                for entry in all_xray:
                    entry_time = datetime.strptime(entry['time_tag'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    if entry_time.date() == target_date.date():
                        xray_data.append(entry)
        except Exception as e:
            print(f"Error fetching historical X-ray data: {e}")
        
        # Try to fetch proton data
        try:
            proton_url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"
            response = requests.get(proton_url, timeout=10)
            if response.status_code == 200:
                all_proton = response.json()
                # Filter for the target date
                for entry in all_proton:
                    entry_time = datetime.strptime(entry['time_tag'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    if entry_time.date() == target_date.date():
                        proton_data.append(entry)
        except Exception as e:
            print(f"Error fetching historical proton data: {e}")
        
        # For Kp data, construct historical values from forecast endpoint
        try:
            kp_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
            response = requests.get(kp_url, timeout=10)
            if response.status_code == 200:
                all_kp = response.json()[1:]  # Skip header
                for row in all_kp:
                    try:
                        kp_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        if kp_time.date() == target_date.date() and row[2] == 'observed':
                            kp_data.append({
                                'time': kp_time.isoformat(),
                                'kp': float(row[1]),
                                'observed': True
                            })
                    except:
                        continue
        except Exception as e:
            print(f"Error fetching historical Kp data: {e}")
        
        # Check if we have any data
        if not xray_data and not proton_data and not kp_data:
            return jsonify({
                'error': f'No archived data available for {date_str}. NOAA JSON archives typically contain only the last 7 days.'
            }), 404
        
        return jsonify({
            'date': date_str,
            'xray': xray_data,
            'proton': proton_data,
            'kp': kp_data,
            'note': 'Data retrieved from NOAA archives (last 7 days available)'
        })
        
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dst-index')
def get_dst_index():
    """Get Dst (Disturbance Storm Time) index from Kyoto WDC with time range filtering"""
    try:
        # Get time range parameter (default to 24h)
        time_range = request.args.get('range', '24h')

        cache_key = f'dst_index_{time_range}'
        cached = get_cached(cache_key, max_age_seconds=300)
        if cached is not None:
            return jsonify(cached)
        
        # NOAA provides estimated Dst values
        DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
        response = requests.get(DST_URL, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        dst_values = []

        if isinstance(data, list):
            # SWPC currently returns objects:
            # [{'time_tag': '2026-05-14T19:00:00', 'dst': 13}, ...]
            # Older products may use a header row followed by arrays, so keep both.
            header = None
            rows = data
            if rows and isinstance(rows[0], list):
                header = [str(value).strip().lower() for value in rows[0]]
                rows = rows[1:]

            for row in rows:
                try:
                    if isinstance(row, dict):
                        raw_time = next(
                            (row.get(key) for key in ('time_tag', 'time', 'timestamp') if row.get(key) is not None),
                            None
                        )
                        raw_dst = next(
                            (row.get(key) for key in ('dst', 'DST', 'Dst') if row.get(key) is not None),
                            None
                        )
                    elif isinstance(row, list):
                        if header:
                            row_map = {
                                header[index]: row[index] if index < len(row) else None
                                for index in range(len(header))
                            }
                            raw_time = row_map.get('time_tag') or row_map.get('time') or row_map.get('timestamp')
                            raw_dst = row_map.get('dst')
                        else:
                            raw_time = row[0] if len(row) > 0 else None
                            raw_dst = row[1] if len(row) > 1 else None
                    else:
                        continue

                    point_time = parse_swpc_datetime(raw_time)
                    dst = float(raw_dst)
                    if point_time is None or abs(dst) >= 9999:
                        continue

                    dst_values.append({
                        'time': point_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'dst': dst
                    })
                except (TypeError, ValueError, IndexError):
                    continue

        range_hours = {
            '6h': 6,
            '12h': 12,
            '24h': 24,
            '3d': 72,
            '7d': 168
        }.get(time_range, 24)

        parsed_points = [
            (parse_swpc_datetime(point['time']), point)
            for point in dst_values
        ]
        parsed_points = [
            (point_time, point)
            for point_time, point in parsed_points
            if point_time is not None
        ]
        reference_time = max(
            (point_time for point_time, _ in parsed_points),
            default=datetime.now(timezone.utc)
        )
        cutoff_time = reference_time - timedelta(hours=range_hours)
        filtered_values = []
        for point_time, point in parsed_points:
            if point_time >= cutoff_time:
                filtered_values.append(point)

        payload = {
            'dst_values': filtered_values,
            'count': len(filtered_values),
            'source': 'NOAA SWPC / Kyoto WDC',
            'unit': 'nT',
            'range': time_range
        }
        set_cached(cache_key, payload, timeout=300)
        return jsonify(payload)
            
    except Exception as e:
        print(f"Error fetching Dst index: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'dst_values': []}), 500

@app.route('/api/system-alert')
def get_system_alert():
    """Get system-wide alert configuration for displaying known issues"""
    try:
        # Construct path securely - file must be in app directory
        app_dir = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.abspath(os.path.join(app_dir, 'alert_config.json'))
        
        # Verify the config file is in the expected directory (prevent traversal)
        if not config_path.startswith(app_dir):
            print(f"Security warning: Invalid config path attempted")
            return jsonify({
                'enabled': False,
                'message': '',
                'type': 'info',
                'dismissible': True
            })
        
        # Check if config file exists
        if not os.path.exists(config_path):
            # Return default disabled state if config doesn't exist
            return jsonify({
                'enabled': False,
                'message': '',
                'type': 'info',
                'dismissible': True
            })
        
        # Read config (support single object or an array of alerts)
        with open(config_path, 'r') as f:
            raw = json.load(f)

        # Normalize to a list of alerts for the frontend to consume
        if isinstance(raw, list):
            alerts = raw
        else:
            alerts = [raw]

        # Validate config structure and set defaults per-alert
        defaults = {
            'enabled': False,
            'message': '',
            'type': 'info',
            'dismissible': True
        }

        for alert in alerts:
            for key, default_value in defaults.items():
                if key not in alert:
                    alert[key] = default_value

        # Return a predictable object with an `alerts` array
        return jsonify({ 'alerts': alerts })
        
    except Exception as e:
        print(f"Error reading system alert config: {e}")
        import traceback
        traceback.print_exc()
        # Return disabled state on error
        return jsonify({
            'enabled': False,
            'message': '',
            'type': 'info',
            'dismissible': True
        })

@app.route('/api/proxy-magnetogram')
def proxy_magnetogram():
    """Proxy the JSOC magnetogram image to avoid CORS issues"""
    try:
        mag_url = 'https://jsoc1.stanford.edu/data/hmi/images/latest/HMI_latest_color_Mag_4096x4096.jpg'
        
        # Get query parameters for cropping
        x = request.args.get('x', type=float)
        y = request.args.get('y', type=float)
        size = request.args.get('size', type=float)
        output_size = request.args.get('output_size', type=int, default=512)
        
        response = requests.get(mag_url, timeout=30)
        
        if response.status_code == 200:
            # If crop parameters provided, crop the image
            if x is not None and y is not None and size is not None:
                img = Image.open(BytesIO(response.content))
                
                # Crop the image
                left = int(x - size / 2)
                top = int(y - size / 2)
                right = int(x + size / 2)
                bottom = int(y + size / 2)
                
                cropped = img.crop((left, top, right, bottom))
                cropped = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                output = BytesIO()
                cropped.save(output, format='JPEG', quality=95)
                output.seek(0)
                
                return send_file(output, mimetype='image/jpeg')
            else:
                # Return full image
                return send_file(BytesIO(response.content), mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Failed to fetch magnetogram'}), 500
            
    except Exception as e:
        print(f"Error proxying magnetogram: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/proxy-intensity')
def proxy_intensity():
    """Proxy the HMI intensitygram image to avoid CORS issues, with optional crop"""
    try:
        # Full-disk HMI intensitygram candidates (prefer 4096 for quality)
        intensity_urls = [
            'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_4096_HMII.jpg',
            'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_HMII.jpg',
            'https://soho.nascom.nasa.gov/data/realtime/hmi_igr/1024/latest.jpg',
        ]
        
        # Get query parameters for cropping
        x = request.args.get('x', type=float)
        y = request.args.get('y', type=float)
        size = request.args.get('size', type=float)
        output_size = request.args.get('output_size', type=int, default=512)
        
        img_data = None
        for url in intensity_urls:
            try:
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    img_data = response.content
                    break
            except Exception:
                continue
        
        if img_data is None:
            return jsonify({'error': 'Failed to fetch intensitygram from all sources'}), 500
        
        # If crop parameters provided, crop the image
        if x is not None and y is not None and size is not None:
            img = Image.open(BytesIO(img_data))
            
            # Crop the image
            left = int(x - size / 2)
            top = int(y - size / 2)
            right = int(x + size / 2)
            bottom = int(y + size / 2)
            
            # Clamp to image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(img.width, right)
            bottom = min(img.height, bottom)
            
            cropped = img.crop((left, top, right, bottom))
            cropped = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output = BytesIO()
            cropped.save(output, format='JPEG', quality=95)
            output.seek(0)
            
            return send_file(output, mimetype='image/jpeg')
        else:
            # Return full image
            return send_file(BytesIO(img_data), mimetype='image/jpeg')
            
    except Exception as e:
        print(f"Error proxying intensitygram: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys
    
    # Check if running in CLI mode for GIF generation
    if len(sys.argv) > 1 and sys.argv[1] == 'generate-gif':
        # CLI mode: generate animated GIF
        hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2
        interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        duration = int(sys.argv[4]) if len(sys.argv) > 4 else 500
        
        print("CLI Mode: Generating Animated GIF")
        gif_path = generate_animated_gif(
            hours_duration=hours,
            frame_interval_minutes=interval,
            frame_duration=duration
        )
        
        if gif_path:
            print(f"\nSuccess! Animation saved to: {gif_path}")
        else:
            print("\nFailed to generate animation")
        
    else:
        # Web server mode
        import os
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 3005))
        
        # Start background map refresh thread
        def background_map_refresh():
            """Periodically refresh the aurora map cache in the background"""
            import time
            # Wait a bit for server to start
            time.sleep(5)
            
            while True:
                try:
                    # Check if cache is stale or missing
                    cached = get_cached('aurora_map', max_age_seconds=25)
                    if not cached:
                        with _map_generation_lock:
                            # Double check after acquiring lock
                            cached = get_cached('aurora_map', max_age_seconds=25)
                            if not cached:
                                print("[BG] Refreshing aurora map cache...")
                                img_buffer = generate_map_image()
                                img_bytes = img_buffer.getvalue()
                                set_cached('aurora_map', img_bytes)
                                print("[BG] Aurora map cache refreshed")
                except Exception as e:
                    print(f"[BG] Error refreshing map cache: {e}")
                
                # Check every 20 seconds
                time.sleep(20)
        
        # Start background thread unless disabled for local smoke tests.
        if os.environ.get('AURORA_DISABLE_BG_REFRESH') == '1':
            print("Background map refresh thread disabled")
        else:
            bg_thread = threading.Thread(target=background_map_refresh, daemon=True)
            bg_thread.start()
            print("Background map refresh thread started")
        
        print("Aurora Dashboard Starting...")
        print(f"Open your browser to: http://localhost:{port}")
        print(f"Generate GIF: http://localhost:{port}/generate-gif?hours=2&interval=1&duration=500")
        print("\nCLI Mode: python aurora.py generate-gif [hours] [interval_minutes] [frame_duration_ms]")
        print("   Example: python aurora.py generate-gif 2 1 500  (captures 2 hours at 1-min intervals)")
        app.run(debug=debug_mode, host=host, port=port, threaded=True)
