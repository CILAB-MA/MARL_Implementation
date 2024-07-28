
class EpsilonScheduler:
    def __init__(self, start=0.4, end=0.1, decay_steps=10000, interval=100):
        self.epsilon = start
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.interval = interval
        self.step_count = 0

    def step(self):
        if self.step_count % self.interval == 0 and self.epsilon > self.end:
            reduction = (self.start - self.end) / self.decay_steps
            self.epsilon = max(self.epsilon - reduction * self.interval, self.end)
        self.step_count += 1

    def get_epsilon(self):
        return self.epsilon
