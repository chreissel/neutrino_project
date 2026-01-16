import numpy as np

class NoiseScheduler:
    def __init__(self, schedule_type='linear', max_noise=1.0, total_epochs=100):
        self.schedule_type = schedule_type.lower()
        self.max_noise = max_noise
        self.total_epochs = total_epochs
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer.")

    def get_noise_const(self, current_epoch):
        if current_epoch >= self.total_epochs:
            return self.max_noise

        progress = current_epoch / self.total_epochs

        if self.schedule_type == 'linear':
            # Linear: C_t = max_noise * progress
            schedule_factor = progress
        elif self.schedule_type == 'root':
            # Root: C_t = max_noise * sqrt(progress)
            schedule_factor = np.sqrt(progress)
        elif self.schedule_type == 'quadratic':
            # Quadratic: C_t = max * (progress)^2
            schedule_factor = progress ** 2 
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return float(self.max_noise * schedule_factor)
