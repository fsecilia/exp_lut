import math

t_max = 50
crossover = 10
sensitivity = 1.0
nonlinearity = 1.0
saturation = 5.0
saturation_rate = 1.0

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

    def on_begin(self):
        pass

    def on_end(self):
        pass

    def __call__(self, t):
        x = self.sampler.sample_location(t)
        y = self.generator.generate(x)

        x *= t_max
        y *= x
        print(f"{x:.24f},{y:.24f};")

    def __init__(self, generator):
        self.generator = generator
        self.sampler = sampler_curvature_t(t_max/output_raw_accel_t.num_samples)

# libinput supports up to 64 uniformly-spaced samples
class output_libinput_t:
    num_samples = 63
    motion_step = t_max/(num_samples + 1)

    def on_begin(self):
        print("0 ", end="")

    def on_end(self):
        print("")

    def __call__(self, t):
        x = self.sampler.sample_location(t)
        y = self.generator.generate(x)

        y *= x

        # there is some scalar conversion I'm missing because it is too fast. probably dpi
        y /= 4

        print(f"{y:.24f} ", end="")

    def __init__(self, generator):
        self.generator = generator
        self.sampler = sampler_uniform_t(output_libinput_t.motion_step)

class generator_t:
    def generate(self, x):
        unfiltered = (math.exp(self.nonlinearity*x) - 1)/(math.exp(self.nonlinearity*self.crossover) - 1)
        filtered = self.sensitivity*self.saturation_limiter.apply(unfiltered)
        return filtered

    def __init__(self, sensitivity, crossover, nonlinearity, saturation_limiter):
        self.sensitivity = sensitivity
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.saturation_limiter = saturation_limiter

class app_t:
    generator = generator_t(sensitivity, crossover/t_max, nonlinearity, limiter_t(saturation, saturation_rate))
    #output = output_raw_accel_t(generator)
    output = output_libinput_t(generator)

    def run(self):
        num_samples = self.output.num_samples
        dt = 1.0/num_samples
        t = 0.0
        self.output.on_begin()
        for sample in range(num_samples):
            t += dt
            self.output(t)
        self.output.on_end()

app_t().run()

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

With no limiter, you can produce the same curve by varying sensitivity or crossover, but as knobs they do different
things. Sensitivity is a final scalar applied after limiting to the numerator. Crossover is applied before limiting,
and it is summed in the denominator, so it isn't as direct of a control over the final output scale as sensitivity is.
To scale the final output by 4, you just increase the sensitivity by 4. To increase the final output scale using
crossover requires solving the denominator. They affect the output in related ways, but they control different
intuitive things.

new_crossover = ln((e^old_crossover - 1)/relative_scale + 1)
= ln(e^old_crossover + relative_scale - 1) - ln(relative_scale)


in (e^tx - 1), t is literally the slope at x=0
in (e^nx - 1)/(e^nc - 1), the slope is more complicated, but related

c(e^nx - 1)/(e^nc - 1)
= ce^nx/(e^nc - 1) - c/(e^nc - 1)

So ths slope should be something like cn/(e^nc - 1)
'''
