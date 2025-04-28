import numpy as np
from VIP.logData import load_data_csv


def calculate_noise_score(encoder_data, sampling_interval=0.001):
    if len(encoder_data) < 3:
        return 0.0

    data = np.array(encoder_data)
    velocity = np.diff(data) / sampling_interval
    acceleration = np.diff(velocity) / sampling_interval
    acc_std = np.std(acceleration)
    sign_changes = np.diff(np.sign(acceleration))
    direction_flips = np.sum(sign_changes != 0)
    noise_score = acc_std + direction_flips * 10

    return noise_score

movements = load_data_csv("encoder.csv")
for movement in movements:
  value = calculate_noise_score(movement)
  print(f"Noise: {value:.6f}")