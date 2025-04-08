#! /usr/bin/env python3

import math
import argparse

default_crossover = 20
default_nonlinearity = 2
default_magnitude = 1
default_sensitivity = 1
default_limit = 4.0
default_limit_rate = 4.0

table_size = 50

# exponential curve:
# - shifted to go through (0, 0)
# - scaled to go through (c, c)
# - exponentiated by n to flatten entry and sharpen transition
# - scaled exponentially by m relative to (c, c) to apply exponentiation gradually relative to linear
# ce^(m(x - c))(e^nx - 1)/(e^nc - 1)
class curve_exponential_t:
    def __call__(self, x):
        return self.crossover*math.exp(self.magnitude*(x - self.crossover))*(math.exp(self.nonlinearity*x) - 1)/(math.exp(self.nonlinearity*self.crossover) - 1)

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# Similar to curve_exponential_t, but scaled by softplus instead of the exponential term, log(1 + e^x).
class curve_exponential_by_softplus_t:
    def __call__(self, x):
        return math.log(1 + math.exp(x))*(math.exp(self.nonlinearity*x) - 1)/(math.exp(self.nonlinearity*self.crossover) - 1)

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# The logistic function has slope 1 at the crossover and is symmetric: 1/(1 + e^-((n/c)(x - c)))
class curve_logistic_t:
    def __call__(self, x):
        return 2*self.crossover/(1 + math.exp(-(self.nonlinearity/self.crossover)*(x - self.crossover)))

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# non-analytic smooth transition function
class curve_smooth_t:
    def __call__(self, x):
        offset = x - self.crossover
        return self.crossover*(math.tanh(self.nonlinearity*offset/(1 + offset*offset)) + 1)

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# symmetric cubic hermite: 3x^2 - 2x^3
class curve_smoothstep_t:
    def __call__(self, x):
        if x >= 2*self.crossover: return 2*self.crossover

        d = 1.0/(2*self.crossover)
        u = d*x
        v = d*self.crossover
        uu = u*u
        vv = v*v
        return self.crossover*(3*uu - 2*uu*u)/(3*vv - 2*vv*v)

    def __init__(self, crossover, _nonlinearity, _magnitude):
        self.crossover = crossover

# smooth ramp: log(1 + e^x)
class curve_softplus_t:
    def __call__(self, x):
        return self.crossover*math.log(1 + math.exp(self.nonlinearity*(x - self.crossover)))/math.log(2)

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# log diff, limited by tanh: tanh(n*ln(x)) + 1
class curve_log_diff_t:
    def __call__(self, x):
        return self.crossover*(math.tanh(self.nonlinearity*math.log(x/self.crossover)) + 1)

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# log diff, exponentiated: e^tanh(n*ln(x))
class curve_log_diff_exponentiated_t:
    def __call__(self, x):
        return self.crossover*(math.exp(math.tanh(self.nonlinearity*math.log(x/self.crossover)) + 1) - 1)/(math.exp(1) - 1)

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# product of exponential_t and log_diff_t
# this needs steepness and individual contributions
class curve_product_exponential_log_diff_t:
    def __call__(self, x):
        exp = (math.exp(self.nonlinearity*x) - 1)/(math.exp(self.nonlinearity*self.crossover) - 1)
        log_diff = math.tanh(self.nonlinearity*math.log(x/self.crossover)) + 1
        return exp*log_diff

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

class curve_linear_t:
    def __call__(self, _):
        return 1

    def __init__(self, crossover, nonlinearity, _):
        pass

# combines a curve with a limiter and sensitivity
class generator_t:
    def __call__(self, x):
        unlimited = self.curve(x)
        limited = self.sensitivity*self.limiter(unlimited)
        return limited

    def __init__(self, curve, limiter, sensitivity):
        self.curve = curve
        self.limiter = limiter
        self.sensitivity = sensitivity

# soft limits using tanh
class limiter_t:
    def __call__(self, t):
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

    def __call__(self, t):
        # The curve should have more samples where the sensitivity changes most. For now, just oversample small t.
        # The change in sensitivity is the derivative of the whole limited function, which is difficult.
        # Alternatively, we can find where the limiter comes on and sample 2 points there, and calc the rest using
        # just the derivative of the curve, which is much simpler.
        return (math.exp(math.pow(t, sampler_curvature_t.sample_density)) - 1)/(math.exp(1) - 1)

    def __init__(self, num_samples):
        self.num_samples = num_samples

# chooses sample locations uniformly
class sampler_uniform_t:
    def __call__(self, t):
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
        x = self.sampler(t)
        y = self.generator(x)

        y *= x

        y *= table_size
        x *= table_size
        print(f"{x:.24f},{y:.24f};")

    def __init__(self, generator):
        self.generator = generator
        self.sampler = sampler_curvature_t(table_size/output_raw_accel_t.num_samples)

# libinput supports up to 64 uniformly-spaced samples, but this includes 0.
class output_libinput_t:
    num_samples = 63
    motion_step = (table_size/(num_samples + 1))/2

    def on_begin(self):
        print("0 ", end="")

    def on_end(self):
        print("")

    def __call__(self, t):
        x = self.sampler(t)
        y = self.generator(x)

        y *= output_libinput_t.motion_step
        y *= x

        print(f"{y:.24f} ", end="")

    def __init__(self, generator):
        self.generator = generator
        self.sampler = sampler_uniform_t(output_libinput_t.motion_step)

class app_t:
    def run(self, output):
        num_samples = output.num_samples
        dt = 1.0/num_samples
        t = 0.0
        output.on_begin()
        for sample in range(num_samples):
            t += dt
            output(t)
        output.on_end()

def create_arg_parser():
    impl = argparse.ArgumentParser(prog="exp_lut.py",
        description="Generates a lookup table for exponential input curves.")
    impl.add_argument('-c', '--crossover', type=float, default=default_crossover)
    impl.add_argument('-n', '--nonlinearity', type=float, default=default_nonlinearity)
    impl.add_argument('-m', '--magnitude', type=float, default=default_magnitude)
    impl.add_argument('-s', '--sensitivity', type=float, default=default_sensitivity)
    impl.add_argument('-l', '--limit', type=float, default=default_limit)
    impl.add_argument('-r', '--limit_rate', type=float, default=default_limit_rate)

    curve_choices={
        "exponential": curve_exponential_t,
        "exponential_by_softplus": curve_exponential_by_softplus_t,
        "logistic": curve_logistic_t,
        "smooth": curve_smooth_t,
        "smoothstep": curve_smoothstep_t,
        "softplus": curve_softplus_t,
        "log_diff": curve_log_diff_t,
        "log_diff_exponentiated": curve_log_diff_exponentiated_t,
        "product_exponential_log_diff": curve_product_exponential_log_diff_t,
        "linear": curve_linear_t,
    }
    impl.add_argument('-x', '--curve', choices=curve_choices.keys(), default="exponential")

    format_choices={
       "raw_accel": output_raw_accel_t,
       "libinput": output_libinput_t,
    }
    impl.add_argument('-f', '--format', choices=format_choices.keys(), default="raw_accel")

    result = impl.parse_args()

    result.curve_t = curve_choices[result.curve]
    result.output_t = format_choices[result.format]

    return result

args = create_arg_parser()
app_t().run(args.output_t(generator_t(args.curve_t(args.crossover/table_size, args.nonlinearity, args.magnitude),
    limiter_t(args.limit/args.sensitivity, args.limit_rate), args.sensitivity)))

'''
9/16 anisotropy: x*9/16 = x*0.5625, xy*16/9 = 1.77777777777778
(3/4)(9/16) = (27/64) = (3/4)^2 anisotropy: x*27/64 = x*0.421875, xy*16/9 = 2.3703703703703703703703703703704
(4/3)(9/16) = (28/48) = 7/12 anisotropy: x*7/12 = x*0.58333333333333333333333333333333, xy*12/7 = 1.7142857142857142857142857142857

(ln((e^old_crossover - 1)/relative_scale + 1))
= ln(e^old_crossover + relative_scale - 1) - ln(relative_scale)
'''
