import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import mplcursors

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self, report):
        Inn = report - np.dot(self.H, self.Sf)  # Calculate innovation using associated report
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    az = np.degrees(np.arctan2(y, x))
    if az < 0:
        az += 360
    return r, az, el

# Function to convert Cartesian coordinates to spherical coordinates for arrays
def cart2sph2(x, y, z, filtered_values_csv):
    r = []
    az = []
    el = []
    for i in range(len(filtered_values_csv)):
        r_val = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        el_val = np.degrees(np.arctan2(z[i], np.sqrt(x[i]**2 + y[i]**2)))
        az_val = np.degrees(np.arctan2(y[i], x[i]))
        if az_val < 0:
            az_val += 360
        r.append(r_val)
        az.append(az_val)
        el.append(el_val)
    return r, az, el

def read_and_group_measurements(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    
    measurements.sort(key=lambda x: x[3])  # Sort by mt
    grouped_measurements = []
    current_group = []
    for measurement in measurements:
        if current_group and abs(measurement[3] - current_group[-1][3]) >= 0.050:
            grouped_measurements.append(current_group)
            current_group = []
        current_group.append(measurement)
    if current_group:
        grouped_measurements.append(current_group)
    
    return grouped_measurements

def initialize_tracks(grouped_measurements, cov_inv):
    tracks = []
    track_ids = []

    for i, group in enumerate(grouped_measurements):
        if i == 0:
            for measurement in group:
                track_id = len(tracks)
                tracks.append([measurement])
                track_ids.append(track_id)
        else:
            for measurement in group:
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

def chi_squared_test(measurement, track, cov_inv):
    distances = []
    for track_measurement in track:
        distance = mahalanobis_distance(np.array(track_measurement[:3]), np.array(measurement[:3]), cov_inv)
        distances.append(distance)
    min_distance = min(distances)
    return min_distance < chi2_threshold

def generate_clusters(tracks, measurements, cov_inv):
    clusters = {}
    for track_id, track in enumerate(tracks):
        clusters[track_id] = []
        for measurement in measurements:
            if chi_squared_test(measurement, track, cov_inv):
                clusters[track_id].append(measurement)
    return clusters

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    x = x.reshape(-1)[:3]
    y = y.reshape(-1)[:3]
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def generate_hypotheses(clusters):
    hypotheses = []
    for track_id, cluster in clusters.items():
        num_tracks = len(cluster)
        base = len(cluster) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in range(num_tracks):
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_id, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)
    return hypotheses

def calculate_joint_probabilities(hypotheses, tracks, clusters, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_id, report_idx in hypothesis:
            if report_idx == -1:
                continue
            track = tracks[track_id]
            report = clusters[track_id][report_idx]
            distances = [mahalanobis_distance(np.array(track_measurement[:3]), np.array(report[:3]), cov_inv)
                         for track_measurement in track]
            min_distance = min(distances)
            prob *= np.exp(-0.5 * min_distance)
        probabilities.append(prob)
    return probabilities

def update_filter_with_max_probability(hypotheses, probabilities, clusters, kalman_filter):
    max_prob_idx = np.argmax(probabilities)
    best_hypothesis = hypotheses[max_prob_idx]
    updated_states = []
    for track_id, report_idx in best_hypothesis:
        if report_idx != -1:
            kalman_filter.update_step(np.array(clusters[track_id][report_idx][:3]).reshape(3, 1))
            updated_states.append(kalman_filter.Sf.flatten())
    return updated_states

file_path = 'ttk_84_2.csv'
grouped_measurements = read_and_group_measurements(file_path)
cov_inv = np.linalg.inv(np.diag([0.1, 0.1, 0.1]))
tracks, track_ids = initialize_tracks(grouped_measurements, cov_inv)
clusters = generate_clusters(tracks, grouped_measurements, cov_inv)
hypotheses = generate_hypotheses(clusters)
probabilities = calculate_joint_probabilities(hypotheses, tracks, clusters, cov_inv)
kalman_filter = CVFilter()

updated_states = []

for track_id in track_ids:
    track = tracks[track_id]
    if len(track) >= 3:
        x1, y1, z1, t1 = track[0]
        x2, y2, z2, t2 = track[1]
        vx = (x2 - x1) / (t2 - t1)
        vy = (y2 - y1) / (t2 - t1)
        vz = (z2 - z1) / (t2 - t1)
        kalman_filter.initialize_filter_state(x2, y2, z2, vx, vy, vz, t2)
        for measurement in track[2:]:
            x, y, z, t = measurement
            kalman_filter.predict_step(t)
            kalman_filter.initialize_measurement_for_filtering(x, y, z, t)
            updated_states = update_filter_with_max_probability(hypotheses, probabilities, clusters, kalman_filter)
            x_vals, y_vals, z_vals = zip(*[(state[0], state[1], state[2]) for state in updated_states])
            plt.plot(x_vals, y_vals, z_vals, label=f'Track {track_id}')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
