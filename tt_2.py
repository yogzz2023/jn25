import numpy as np
from scipy.stats import chi2

def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def mahalanobis_distance(x, y, cov_inv):
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def chi_squared_test(measurement, track, cov_inv):
    distances = [mahalanobis_distance(np.array(track_measurement), np.array(measurement), cov_inv) for track_measurement in track]
    min_distance = min(distances)
    return min_distance < chi2_threshold

def initialize_tracks(measurements, cov_inv):
    tracks = []
    track_ids = []

    for measurement in measurements:
        assigned = False
        for track_id, track in enumerate(tracks):
            if chi_squared_test(measurement, track, cov_inv):
                tracks[track_id].append(measurement)
                assigned = True
                break
        if not assigned:
            track_id = len(tracks)
            tracks.append([measurement])
            track_ids.append(track_id)
    
    return tracks, track_ids

# Sample measurements (azimuth, elevation, range, time)
sample_measurements = [
    (10, 5, 100, 0.1), (12, 6, 105, 0.2), (9, 4, 98, 0.3),
    (50, 20, 500, 0.4), (52, 22, 505, 0.5), (11, 5, 102, 0.6),
    (53, 23, 510, 0.7), (55, 25, 515, 0.8), (13, 7, 107, 0.9),
    (60, 30, 600, 1.0), (62, 32, 605, 1.1), (15, 8, 110, 1.2),
    (65, 33, 610, 1.3), (67, 35, 615, 1.4), (17, 9, 112, 1.5),
    (70, 40, 700, 1.6), (72, 42, 705, 1.7), (19, 10, 115, 1.8),
    (75, 43, 710, 1.9), (77, 45, 715, 2.0), (21, 11, 120, 2.1)
]

# Convert sample measurements to Cartesian coordinates
converted_measurements = [sph2cart(az, el, r) for az, el, r, _ in sample_measurements]

# Covariance matrix (assuming identity for simplicity)
cov_matrix = np.eye(3)
cov_inv = np.linalg.inv(cov_matrix)

# Chi-squared threshold
state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

# Initialize tracks
tracks, track_ids = initialize_tracks(converted_measurements, cov_inv)

# Output the tracks and their associated measurements
for track_id, track in enumerate(tracks):
    print(f"Track ID {track_id}:")
    for measurement in track:
        print(f"  Measurement: {measurement}")
