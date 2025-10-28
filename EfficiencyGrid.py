import numpy as np

class EfficiencyGrid:
    def __init__(self, grid_path="epsilon_grid.npz"):
        grid = np.load(grid_path)
        self.energies = grid["energies"]
        self.radii = grid["radii"]
        self.thresholds = grid["thresholds"]
        self.pitches = grid["pitches"]
        self.epsilon = grid["epsilon"]

    def get_efficiency(self, true_post, meta, idx, thresh_idx):
        energy = true_post[idx, 0]
        pitch_angle = true_post[idx, 1]
        radius = meta[idx, 2]
        energy_idx = np.digitize(energy, self.energies, right=True)
        pitch_angle_idx = np.digitize(pitch_angle, self.pitches, right=True)
        radius_idx = np.digitize(radius, self.radii, right=True)
        energy_idx = np.clip(energy_idx, 0, len(self.energies) - 1)
        pitch_angle_idx = np.clip(pitch_angle_idx, 0, len(self.pitches) - 1)
        radius_idx = np.clip(radius_idx, 0, len(self.radii) - 1)
        return self.epsilon[thresh_idx, energy_idx, pitch_angle_idx, radius_idx]
