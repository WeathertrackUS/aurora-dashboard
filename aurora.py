from flask import Flask, render_template, jsonify, send_file, request
import requests
from datetime import datetime, timezone, timedelta
import json
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon
import matplotlib.patheffects
import numpy as np
from io import BytesIO
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Set font to Metropolis Bold
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Metropolis', 'Inter', 'Arial', 'sans-serif']
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.facecolor'] = '#1e293b'
matplotlib.rcParams['figure.facecolor'] = '#0f172a'
matplotlib.rcParams['text.color'] = '#f8fafc'
matplotlib.rcParams['axes.labelcolor'] = '#94a3b8'
matplotlib.rcParams['xtick.color'] = '#64748b'
matplotlib.rcParams['ytick.color'] = '#64748b'
matplotlib.rcParams['grid.color'] = '#334155'
matplotlib.rcParams['axes.edgecolor'] = '#334155'

app = Flask(__name__)

# SWPC API endpoints
PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"
PLASMA_2HR_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
MAG_2HR_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SCALES_URL = "https://services.swpc.noaa.gov/products/noaa-scales.json"
OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
HEMI_POWER_URL = "https://services.swpc.noaa.gov/text/aurora-nowcast-hemi-power.txt"
GOES_MAG_PRIMARY_URL = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-6-hour.json"
GOES_MAG_SECONDARY_URL = "https://services.swpc.noaa.gov/json/goes/secondary/magnetometers-6-hour.json"
GOES_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
GOES_XRAY_1DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
GOES_PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
SOLAR_REGIONS_URL = "https://services.swpc.noaa.gov/json/solar_regions.json"

def fetch_solar_wind_data():
    """Fetch current solar wind plasma and magnetic field data"""
    try:
        # Fetch plasma data (speed, density)
        plasma_response = requests.get(PLASMA_URL, timeout=10)
        plasma_data = plasma_response.json()
        
        # Fetch magnetic field data (Bt, Bz)
        mag_response = requests.get(MAG_URL, timeout=10)
        mag_data = mag_response.json()
        
        # Get latest values (last entry in the array)
        if len(plasma_data) > 1 and len(mag_data) > 1:
            latest_plasma = plasma_data[-1]
            latest_mag = mag_data[-1]
            
            # Handle both array formats and potential null values
            speed = latest_plasma[2] if len(latest_plasma) > 2 and latest_plasma[2] not in [None, '', -999, -9999] else None
            density = latest_plasma[1] if len(latest_plasma) > 1 and latest_plasma[1] not in [None, '', -999, -9999] else None
            temperature = latest_plasma[3] if len(latest_plasma) > 3 and latest_plasma[3] not in [None, '', -999, -9999] else None
            bt = latest_mag[6] if len(latest_mag) > 6 and latest_mag[6] not in [None, '', -999, -9999] else None
            bz = latest_mag[3] if len(latest_mag) > 3 and latest_mag[3] not in [None, '', -999, -9999] else None
            bx = latest_mag[1] if len(latest_mag) > 1 and latest_mag[1] not in [None, '', -999, -9999] else None
            by = latest_mag[2] if len(latest_mag) > 2 and latest_mag[2] not in [None, '', -999, -9999] else None
            
            return {
                'time': latest_plasma[0],
                'speed': float(speed) if speed is not None else None,
                'density': float(density) if density is not None else None,
                'temperature': float(temperature) if temperature is not None else None,
                'bt': float(bt) if bt is not None else None,
                'bz': float(bz) if bz is not None else None,
                'bx': float(bx) if bx is not None else None,
                'by': float(by) if by is not None else None
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
            plasma_data = plasma_response.json()
            mag_data = mag_response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error from SWPC (data provider issue): {e}")
            # Try to use current data to create minimal history
            sw_current = fetch_solar_wind_data()
            if sw_current:
                current_time = datetime.now(timezone.utc)
                return {
                    'times': [current_time],
                    'speeds': [sw_current.get('speed')],
                    'densities': [sw_current.get('density')],
                    'bzs': [sw_current.get('bz')],
                    'bts': [sw_current.get('bt')]
                }
            return None
        
        # Parse data (skip header row)
        times = []
        speeds = []
        densities = []
        bzs = []
        bts = []
        
        for i in range(1, min(len(plasma_data), len(mag_data))):
            try:
                p_row = plasma_data[i]
                m_row = mag_data[i]
                
                time_str = p_row[0]
                # Parse time and make it timezone-aware (UTC)
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                times.append(dt.replace(tzinfo=timezone.utc))
                
                # Filter out invalid values (-999, -9999, None, empty strings)
                speed = p_row[2] if len(p_row) > 2 and p_row[2] not in [None, '', -999, -9999] else None
                density = p_row[1] if len(p_row) > 1 and p_row[1] not in [None, '', -999, -9999] else None
                bz = m_row[3] if len(m_row) > 3 and m_row[3] not in [None, '', -999, -9999] else None
                bt = m_row[6] if len(m_row) > 6 and m_row[6] not in [None, '', -999, -9999] else None
                
                speeds.append(float(speed) if speed is not None else None)
                densities.append(float(density) if density is not None else None)
                bzs.append(float(bz) if bz is not None else None)
                bts.append(float(bt) if bt is not None else None)
            except Exception as row_error:
                print(f"Error parsing row {i}: {row_error}")
                continue
        
        if len(times) == 0:
            print("Warning: No valid historical data parsed")
            return None
        
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
                'location': entry.get('location') or '',
                'lo': str(entry.get('longitude') or ''),
                'area': str(entry.get('area') or ''),
                'z': entry.get('spot_class') or '',
                'll': str(entry.get('latitude') or ''),
                'nn': '', 
                'mag_type': mag_type,
                'c_flares': entry.get('c_xray_events', 0),
                'm_flares': entry.get('m_xray_events', 0),
                'x_flares': entry.get('x_xray_events', 0)
            })
            
    # Sort by region number
    regions.sort(key=lambda x: int(x['number']) if x['number'].isdigit() else 0, reverse=True)
    
    return regions

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
    try:
        # Fetch X-ray flux (6-hour)
        xray_response = requests.get(GOES_XRAY_URL, timeout=10)
        xray_data = xray_response.json()

        # Fetch X-ray flux (1-day)
        xray_1day_response = requests.get(GOES_XRAY_1DAY_URL, timeout=10)
        xray_1day_data = xray_1day_response.json()
        
        # Fetch Proton flux
        proton_response = requests.get(GOES_PROTON_URL, timeout=10)
        proton_data = proton_response.json()
        
        # Fetch Sunspot Regions with retry and longer timeout
        regions_data = fetch_json_with_retry(SOLAR_REGIONS_URL, retries=3, timeout=30)
        sunspots = parse_solar_regions(regions_data)
        
        return {
            'xray': xray_data,
            'xray_1day': xray_1day_data,
            'proton': proton_data,
            'sunspots': sunspots
        }
    except Exception as e:
        print(f"Error fetching solar data: {e}")
        return None

def fetch_kp_index():
    """Fetch current Kp index"""
    try:
        response = requests.get(KP_URL, timeout=10)
        kp_data = response.json()
        
        # Get the most recent Kp value
        if len(kp_data) > 1:
            latest = kp_data[-1]
            return {
                'time': latest[0],
                'kp': float(latest[1])
            }
    except Exception as e:
        print(f"Error fetching Kp index: {e}")
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

def fetch_ovation_data():
    """Fetch the latest OVATION auroral probability data from SWPC"""
    try:
        response = requests.get(OVATION_URL, timeout=10)
        data = response.json()
        
        if data and 'coordinates' in data:
            # Extract metadata
            obs_time = data.get('Observation Time', '')
            forecast_time = data.get('Forecast Time', '')
            
            # Parse coordinates: [Longitude, Latitude, Aurora]
            coords = np.array(data['coordinates'])
            
            # Separate into lon, lat, aurora values
            lons = coords[:, 0]
            lats = coords[:, 1]
            aurora_vals = coords[:, 2]
            
            # Filter for Northern Hemisphere only (positive latitudes)
            north_mask = lats > 0
            lons_north = lons[north_mask]
            lats_north = lats[north_mask]
            aurora_north = aurora_vals[north_mask]
            
            return lons_north, lats_north, aurora_north, forecast_time
    except Exception as e:
        print(f"Error fetching OVATION data: {e}")
        return None, None, None, None

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
        
        # Fetch primary satellite (GOES-19)
        try:
            response_primary = requests.get(GOES_MAG_PRIMARY_URL, timeout=10)
            if response_primary.status_code == 200:
                try:
                    data_primary = response_primary.json()
                except json.JSONDecodeError:
                    print("Warning: GOES primary magnetometer JSON decode error, skipping")
                    data_primary = []
                
                for entry in data_primary:
                    try:
                        time_str = entry.get('time_tag')
                        satellite = entry.get('satellite')
                        hp = entry.get('Hp')
                        
                        if time_str and hp is not None:
                            time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                            if satellite == 19:
                                goes19_times.append(time_obj)
                                goes19_hp.append(float(hp))
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"Warning: Could not fetch GOES primary magnetometer: {e}")
        
        # Fetch secondary satellite (GOES-18)
        try:
            response_secondary = requests.get(GOES_MAG_SECONDARY_URL, timeout=10)
            if response_secondary.status_code == 200:
                try:
                    data_secondary = response_secondary.json()
                except json.JSONDecodeError:
                    print("Warning: GOES secondary magnetometer JSON decode error, skipping")
                    data_secondary = []
                
                for entry in data_secondary:
                    try:
                        time_str = entry.get('time_tag')
                        satellite = entry.get('satellite')
                        hp = entry.get('Hp')
                        
                        if time_str and hp is not None:
                            time_obj = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                            if satellite == 18:
                                goes18_times.append(time_obj)
                                goes18_hp.append(float(hp))
                    except Exception as e:
                        continue
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
    # 5 rows: header, auroral oval (larger), speed/density/bt on right, bz/magnetometer on bottom
    gs = GridSpec(5, 3, figure=fig, width_ratios=[1.5, 1, 1], height_ratios=[0.25, 2.8, 1, 1, 1],
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
    ax_map = fig.add_subplot(gs[1:4, 0], projection=ccrs.Orthographic(-100, 60))
    ax_map.set_facecolor('#1e293b')
    # Add subtle border to map panel
    for spine in ax_map.spines.values():
        spine.set_edgecolor('#334155')
        spine.set_linewidth(1)
    
    # Add map features with updated colors
    ax_map.add_feature(cfeature.LAND, facecolor='#1e293b', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='#0f172a', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE, edgecolor='#38bdf8', linewidth=1, alpha=0.4)
    ax_map.add_feature(cfeature.BORDERS, edgecolor='#475569', linewidth=0.5, linestyle=':')
    ax_map.add_feature(cfeature.STATES, edgecolor='#475569', linewidth=0.3, linestyle=':')
    
    # Add lat/lon grid
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, color='#475569', 
                          alpha=0.3, linestyle='--')
    
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
            lon_grid = np.linspace(-180, 180, 720)  # Doubled resolution
            lat_grid = np.linspace(35, 85, 200)  # Doubled resolution
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
                smoothed = gaussian_filter(smoothed, sigma=2.0)
                smoothed = gaussian_filter(smoothed, sigma=1.5)
                aurora_grid = np.where(valid, smoothed, np.nan)
            
            # Strict masking: only show aurora above threshold and with valid data
            aurora_grid = np.ma.masked_where((aurora_grid < threshold + 1) | np.isnan(aurora_grid), aurora_grid)
            
            lon_mesh_adjusted = lon_mesh
            
            # Create custom colormap for aurora matching SWPC reference
            # Green (low) -> Yellow (50%) -> Orange (75%) -> Red (90%+)
            colors = ['#00ff00', '#88ff00', '#ffff00', '#ffaa00', '#ff0000']
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
                               pad=0.05, shrink=0.8, aspect=15, 
                               ticks=[10, 50, 90])
            cbar.set_label('Aurora Probability (%)', fontsize=12, color='#00ff88', fontweight='bold')
            cbar.ax.set_xticklabels(['10%', '50%', '90%'])
            cbar.ax.tick_params(labelsize=10, colors='#8899aa')
            cbar.outline.set_edgecolor('#4a5a6a')
    
    # Add city markers
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
    }
    
    for city, (lon, lat) in cities.items():
        ax_map.plot(lon, lat, 'o', color='#ffff00', markersize=6, 
                   markeredgecolor='white', markeredgewidth=1.5, 
                   transform=ccrs.PlateCarree(), zorder=5)
        ax_map.text(lon, lat-2, city, fontsize=9, ha='center', color='white',
                   transform=ccrs.PlateCarree(), zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000', 
                            alpha=0.7, edgecolor='none'))
    
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
        
        # Substorm detection logic
        # Active: Rapid increase (40+ nT spike) within last 15-20 minutes AND still elevated
        # Inactive: No recent spike OR spike occurred but has been declining for 15+ minutes
        substorm_active = False
        detection_window = 20  # minutes to look back
        threshold_change = 40  # nT spike threshold
        recent_window = 15  # minutes - must have spike within this window to be "active"
        
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
        
        # Modern boxy badge with shadow
        substorm_badge_style = dict(boxstyle='round,pad=0.6', facecolor='#1e293b', 
                                   edgecolor=substorm_color, linewidth=2,
                                   path_effects=[matplotlib.patheffects.withSimplePatchShadow(offset=(2, -2), 
                                                                                              shadow_rgbFace='#000000', 
                                                                                              alpha=0.3)])
        
        fig.text(0.21, 0.205, substorm_text, 
                 fontsize=12, color=substorm_color, fontweight='bold',
                 bbox=substorm_badge_style, ha='center')
    
    ax_goes.set_ylabel('Hp (nT)', fontsize=11, color='#94a3b8', fontweight='bold')
    title_goes = ax_goes.set_title('GOES MAGNETOMETER (Hp)', fontsize=12, color='#38bdf8',
                     fontweight='bold', pad=8, loc='left')
    title_goes.set_path_effects([matplotlib.patheffects.withStroke(linewidth=2, foreground='#0f172a', alpha=0.7)])
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
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

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

@app.route('/api/solar-data')
def get_solar_data():
    """Get solar activity data"""
    data = fetch_solar_data()
    if data:
        return jsonify(data)
    return jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/xray-data')
def get_xray_data():
    """Get X-ray flux data with specified time range"""
    time_range = request.args.get('range', '6h')
    
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
        data = response.json()
        
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
        img_buffer = generate_aurora_image()
        return send_file(img_buffer, mimetype='image/png', as_attachment=False, 
                        download_name=f'aurora_dashboard_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")}.png')
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating image: {e}", 500

@app.route('/api/aurora-data')
def get_aurora_data():
    """API endpoint to get all aurora-related data"""
    solar_wind = fetch_solar_wind_data()
    kp_data = fetch_kp_index()
    noaa_scales = fetch_noaa_scales()
    
    # Fetch historical data for charts
    sw_history = fetch_solar_wind_history()
    hemi_power = fetch_hemispheric_power()
    goes_mag = fetch_goes_magnetometer()
    
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
    
    # Save to buffer with higher DPI
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#0f172a', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)


def generate_full_dashboard_image(frame_time=None, time_window_end=None):
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
    import io
    
    # Fetch all current data
    solar_wind_full = fetch_solar_wind_history()
    kp_data = fetch_kp_index()
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
    
    # Generate aurora map (left side)
    map_buffer = generate_map_image()
    map_img = Image.open(map_buffer)
    
    # Resize map to fit left half
    map_width = 1150
    map_height = canvas_height
    map_img = map_img.resize((map_width, map_height), Image.Resampling.LANCZOS)
    canvas.paste(map_img, (0, 0))
    
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
        canvas.paste(speed_chart, (right_x, chart_y_start), speed_chart if speed_chart.mode == 'RGBA' else None)
        
        density_chart = generate_solar_wind_chart(
            times, solar_wind['densities'],
            'p/cm³', '#a855f7',
            solar_wind['densities'][-1] if len(solar_wind['densities']) > 0 and solar_wind['densities'][-1] is not None else None,
            ' p/cm³',
            'PROTON DENSITY'
        )
        canvas.paste(density_chart, (right_x, chart_y_start + chart_spacing), density_chart if density_chart.mode == 'RGBA' else None)
        
        bt_chart = generate_solar_wind_chart(
            times, solar_wind['bts'],
            'nT', '#f59e0b',
            solar_wind['bts'][-1] if len(solar_wind['bts']) > 0 and solar_wind['bts'][-1] is not None else None,
            ' nT',
            'IMF Bt'
        )
        canvas.paste(bt_chart, (right_x, chart_y_start + chart_spacing * 2), bt_chart if bt_chart.mode == 'RGBA' else None)
        
        bz_chart = generate_solar_wind_chart(
            times, solar_wind['bzs'],
            'nT', '#ec4899',
            solar_wind['bzs'][-1] if len(solar_wind['bzs']) > 0 and solar_wind['bzs'][-1] is not None else None,
            ' nT',
            'IMF Bz'
        )
        canvas.paste(bz_chart, (right_x, chart_y_start + chart_spacing * 3), bz_chart if bz_chart.mode == 'RGBA' else None)
    
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
    
    # Generate frames from historical data
    frames = []
    
    for i in range(num_frames):
        frame_time = start_time + timedelta(minutes=i * frame_interval_minutes)
        
        print(f"   [{i+1}/{num_frames}] Generating frame for {frame_time.strftime('%H:%M UTC')}...", end='')
        
        try:
            # Generate the FULL dashboard image with sliding time window for animation
            img = generate_full_dashboard_image(frame_time, time_window_end=frame_time)
            frames.append(img)
            
            print(f" ✓")
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(frames) == 0:
        raise Exception("Failed to generate any frames")
    
    # Save as animated GIF
    print(f"\n💾 Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=False
    )
    
    # Calculate stats
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    total_animation_duration = (len(frames) * frame_duration) / 1000
    
    print(f"\n✅ Animation complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Frames: {len(frames)}")
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
            
            # Convert to PIL Image
            img_buffer.seek(0)
            frame = Image.open(img_buffer).copy()  # Copy to ensure it's in memory
            
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
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    total_animation_duration = len(frames) * frame_duration / 1000
    
    print(f"✅ Animation complete!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Frames: {len(frames)}")
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

def generate_map_image():
    """Generate just the auroral oval map image"""
    ovation_lons, ovation_lats, ovation_aurora, ovation_time = fetch_ovation_data()
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(10, 10), facecolor='none')
    ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-100, 60))
    
    # Add map features with updated colors
    ax_map.add_feature(cfeature.LAND, facecolor='#334155', edgecolor='none')
    ax_map.add_feature(cfeature.OCEAN, facecolor='#0f172a', edgecolor='none')
    ax_map.add_feature(cfeature.COASTLINE, edgecolor='#38bdf8', linewidth=1.5, alpha=0.6)
    ax_map.add_feature(cfeature.BORDERS, edgecolor='#475569', linewidth=1, linestyle=':')
    ax_map.add_feature(cfeature.STATES, edgecolor='#475569', linewidth=0.5, linestyle=':')
    
    # Add lat/lon grid
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, color='#475569', 
                          alpha=0.3, linestyle='--')
    
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
            lon_grid = np.linspace(-180, 180, 720)  # Doubled resolution
            lat_grid = np.linspace(35, 85, 200)  # Doubled resolution
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
                smoothed = gaussian_filter(smoothed, sigma=2.0)
                smoothed = gaussian_filter(smoothed, sigma=1.5)
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
    }
    
    for city, (lon, lat) in cities.items():
        ax_map.plot(lon, lat, 'o', color='#facc15', markersize=6, 
                   markeredgecolor='white', markeredgewidth=1.5, 
                   transform=ccrs.PlateCarree(), zorder=5)
        ax_map.text(lon, lat-2, city, fontsize=9, ha='center', color='white',
                   transform=ccrs.PlateCarree(), zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', 
                            alpha=0.7, edgecolor='none'))
    
    # Title
    title_map = ax_map.set_title(f'REAL-TIME AURORAL OVAL\n{ovation_time if ovation_time else ""}',
                    fontsize=14, color='#10b981', fontweight='bold', pad=10)
    title_map.set_path_effects([matplotlib.patheffects.withStroke(linewidth=3, foreground='#0f172a', alpha=0.8)])
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='none', edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

@app.route('/aurora-map.png')
def get_aurora_map_image():
    """Generate and serve just the auroral oval map image"""
    try:
        img_buffer = generate_map_image()
        return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        print(f"Error generating map image: {e}")
        import traceback
        traceback.print_exc()
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
            # G1 = Kp 5 (5.00-5.33)
            # G2 = Kp 6- (5.67-6.00)
            # G3 = Kp 7- (6.67-7.00)
            # G4 = Kp 8- (7.67-8.00)
            # G5 = Kp 9- (8.67+)
            if max_kp >= 8.67:
                g_scale = 5
            elif max_kp >= 7.67:
                g_scale = 4
            elif max_kp >= 6.67:
                g_scale = 3
            elif max_kp >= 5.67:
                g_scale = 2
            elif max_kp >= 5.00:
                g_scale = 1
            else:
                g_scale = 0
            
            daily_forecasts.append({
                'date': date,
                'max_kp': round(max_kp, 1),
                'g_scale': g_scale
            })
        
        print(f"DEBUG: Final forecast: {daily_forecasts}")
        
        return jsonify({'forecast': daily_forecasts})
        
    except Exception as e:
        print(f"Error fetching geomagnetic forecast: {e}")
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
        response = requests.get(KP_FORECAST_URL, timeout=10)
        data = response.json()
        
        kp_values = []
        current_time = datetime.now(timezone.utc)
        
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
            except Exception as e:
                continue
        
        return jsonify({'kp_values': kp_values})
    except Exception as e:
        print(f"Error fetching Kp history: {e}")
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
        
        # Process observed data (all available data for full cycle view)
        observed = {
            'times': [],
            'sunspot_numbers': [],
            'smoothed_ssn': [],
            'f107': []
        }
        
        if obs_data and len(obs_data) > 0:
            for entry in obs_data:
                observed['times'].append(entry.get('time-tag', ''))
                observed['sunspot_numbers'].append(entry.get('ssn', None))
                observed['smoothed_ssn'].append(entry.get('smoothed_ssn', None))
                observed['f107'].append(entry.get('f10.7', None))
        
        # Process predicted data
        predicted = {
            'times': [],
            'smoothed_ssn': [],
            'f107': []
        }
        
        if pred_data and len(pred_data) > 0:
            for entry in pred_data:
                predicted['times'].append(entry.get('time-tag', ''))
                predicted['smoothed_ssn'].append(entry.get('predicted_ssn', None))
                predicted['f107'].append(entry.get('predicted_f10.7', None))
        
        return jsonify({
            'observed': observed,
            'predicted': predicted
        })
    except Exception as e:
        print(f"Error fetching solar cycle data: {e}")
        return jsonify({
            'observed': {'times': [], 'sunspot_numbers': [], 'smoothed_ssn': [], 'f107': []},
            'predicted': {'times': [], 'smoothed_ssn': [], 'f107': []}
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
        data = response.json()
        
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
    """Get active region heatmap (flare activity over time)"""
    try:
        # Fetch sunspot regions
        regions_data = fetch_json_with_retry(SOLAR_REGIONS_URL, retries=3, timeout=30)
        sunspots = parse_solar_regions(regions_data)
        
        # Create mock heatmap data based on flare counts
        regions = [spot['number'] for spot in sunspots[:5]]  # Top 5 regions
        days = [(datetime.now(timezone.utc) - timedelta(days=i)).strftime('%m/%d') for i in range(7, -1, -1)]
        
        # Create activity matrix (regions x days)
        activity = []
        for spot in sunspots[:5]:
            # Calculate activity score based on flare counts
            c_flares = spot.get('c_flares', 0)
            m_flares = spot.get('m_flares', 0)
            x_flares = spot.get('x_flares', 0)
            
            # Create activity pattern (simplified - same for all days for now)
            region_activity = []
            base_activity = (c_flares * 1 + m_flares * 5 + x_flares * 10) / 8
            
            for _ in range(8):
                # Add some variation
                variation = np.random.uniform(0.7, 1.3)
                region_activity.append(max(0, base_activity * variation))
            
            activity.append(region_activity)
        
        return jsonify({'regions': regions, 'days': days, 'activity': activity})
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
        
        response = requests.get(url, params=params, timeout=10)
        
        # Check if response is successful
        if response.status_code != 200:
            print(f"DONKI API error: {response.status_code}")
            # Return empty array on API error
            return jsonify({'flares': [], 'error': 'API quota exceeded or unavailable. Please add your own NASA API key.'})
        
        flares_data = response.json()
        
        # Filter for M and X class flares only
        significant_flares = []
        for flare in flares_data:
            class_type = flare.get('classType', '')
            if class_type and (class_type.startswith('M') or class_type.startswith('X')):
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
        
        return jsonify({'flares': significant_flares[:20]})  # Limit to 20 most recent
    except requests.exceptions.RequestException as e:
        print(f"Error fetching solar flares (network): {e}")
        return jsonify({'flares': [], 'error': 'Network error. Check your connection or API key.'})
    except Exception as e:
        print(f"Error fetching solar flares: {e}")
        return jsonify({'flares': [], 'error': str(e)})

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
    """Get Dst (Disturbance Storm Time) index from Kyoto WDC"""
    try:
        # NOAA provides estimated Dst values
        DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
        response = requests.get(DST_URL, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Skip header row and convert to structured format
            dst_values = []
            for row in data[1:]:  # Skip header
                if len(row) >= 2:
                    try:
                        dst_values.append({
                            'time': row[0],  # ISO timestamp
                            'dst': float(row[1])  # Dst value in nT
                        })
                    except (ValueError, IndexError):
                        continue
            
            return jsonify({
                'dst_values': dst_values,
                'count': len(dst_values),
                'source': 'NOAA SWPC / Kyoto WDC',
                'unit': 'nT'
            })
        else:
            return jsonify({'error': 'Failed to fetch Dst data', 'dst_values': []}), 500
            
    except Exception as e:
        print(f"Error fetching Dst index: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'dst_values': []}), 500

if __name__ == '__main__':
    import sys
    
    # Check if running in CLI mode for GIF generation
    if len(sys.argv) > 1 and sys.argv[1] == 'generate-gif':
        # CLI mode: generate animated GIF
        hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2
        interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        duration = int(sys.argv[4]) if len(sys.argv) > 4 else 500
        
        print(f"🎬 CLI Mode: Generating Animated GIF")
        gif_path = generate_animated_gif(
            hours_duration=hours,
            frame_interval_minutes=interval,
            frame_duration=duration
        )
        
        if gif_path:
            print(f"\n✅ Success! Animation saved to: {gif_path}")
        else:
            print(f"\n❌ Failed to generate animation")
        
    else:
        # Web server mode
        import os
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 3005))
        
        print("🌌 Aurora Dashboard Starting...")
        print(f"🌐 Open your browser to: http://localhost:{port}")
        print(f"🎬 Generate GIF: http://localhost:{port}/generate-gif?hours=2&interval=1&duration=500")
        print("\n💡 CLI Mode: python aurora.py generate-gif [hours] [interval_minutes] [frame_duration_ms]")
        print("   Example: python aurora.py generate-gif 2 1 500  (captures 2 hours at 1-min intervals)")
        app.run(debug=debug_mode, host=host, port=port)
