import math

'''
9/16 anisotropy: x*9/16 = x*0.5625, xy*16/9 = 1.77777777777778
(3/4)(9/16) = (27/64) = (3/4)^2 anisotropy: x*27/64 = x*0.421875, xy*16/9 = 2.3703703703703703703703703703704
(4/3)(9/16) = (28/48) = 7/12 anisotropy: x*7/12 = x*0.58333333333333333333333333333333, xy*12/7 = 1.7142857142857142857142857142857

The same curve produced by changing sensitivity can be reproduced with sensitivity=1 and a clever choice of crossover.
crossover = 38.54813549926318, sensitivity = 16, misses limiter entirely
crossover = 12.75, sensitivity = 4, hits limiter of 5@70 before 45
crossover = 3.504998498211076, sensitivity = 1, hits 5@70 before 15

Using sensitivity = 4 keeps a hint of the limiter. Is it too high to be useful?
Could we just keep sensitivity = 1 and raise the limiter?
'''

t_max = 50
crossover = 12.75
sensitivity = 4.0
nonlinearity = 1.0
smooth = 0.5
saturation = 5.0
saturation_rate = 1

# soft limits using tanh
class limiter_t:
    def apply(self, t):
        normalized = t/self.limit
        compressed = math.pow(normalized, self.rate)
        limited = math.tanh(compressed)
        expanded = math.pow(limited, 1.0/self.rate)
        rescaled = expanded*self.limit
        return rescaled

    def __init__(self, limit, rate):
        self.limit = limit
        self.rate = rate

# chooses sample locations based on curvature
class sampler_curvature_t:
    sample_density = 1.5

    def sample_location(self, t):
        # The curve should have more samples where the sensitivity changes most. For now, just oversample small t.
        return (math.exp(math.pow(t, sampler_curvature_t.sample_density)) - 1)/(math.exp(1) - 1)

    def __init__(self, num_samples):
        self.num_samples = num_samples

# chooses sample locations uniformly
class sampler_uniform_t:
    def sample_location(self, t):
        return t*self.dt

    def __init__(self, dt):
        self.dt = dt

# raw accel supports up to 256 samples with arbitrary locations
class output_raw_accel_t:
    num_samples = 256

    def __call__(self, t):
        x = self.sampler.sample_location(t)
        y = self.generator.generate(x)

        x *= t_max
        y *= t_max
        print(f"{x:.20f},{y:.20f};")

    def __init__(self, generator):
        self.generator = generator
        self.sampler = sampler_curvature_t(t_max/output_raw_accel_t.num_samples)

# libinput supports up to 64 uniformly-spaced samples
class output_libinput_t:
    num_samples = 64

    def __call__(self, t):
        x = self.sampler.sample_location(t)
        y = self.generator.generate(x)

        y *= t_max
        print(f"{y:.20f};")

    def __init__(self, generator):
        self.generator = generator
        self.sampler = sampler_uniform_t(t_max/output_libinput_t.num_samples)

class generator_t:
    def generate(self, x):
        unfiltered = (math.exp(self.nonlinearity*x) - 1)/(math.exp(self.nonlinearity*self.crossover) - 1)
        filtered = self.sensitivity*self.saturation_limiter.apply(unfiltered)
        return x*filtered

    def __init__(self, sensitivity, crossover, nonlinearity, tangent_limiter, saturation_limiter):
        self.sensitivity = sensitivity
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.tangent_limiter = tangent_limiter
        self.saturation_limiter = saturation_limiter

class app_t:
    generator = generator_t(sensitivity, crossover/t_max, nonlinearity, limiter_t(1.0, 0.5/smooth), limiter_t(saturation, saturation_rate))
    output = output_raw_accel_t(generator)

    def run(self):
        num_samples = self.output.num_samples
        dt = 1.0/num_samples
        t = 0.0
        for sample in range(num_samples):
            t += dt
            self.output(t)

app_t().run()
