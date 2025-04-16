#! /usr/bin/env python3

import math
import argparse

table_size = 50

default_in_game_sensitivity = 1/5
default_crossover = 4
default_nonlinearity = 5.2
default_magnitude = 0.25
default_sensitivity = 0.25
default_limit = 16*default_in_game_sensitivity
default_limit_rate = 25
default_curve = "exponential_by_logistic"

# exponential scaled by right half of logistic. similar to by power with m=1, but the linear term tapers
# so the range above the crossover should deviate little from pure exp
class curve_exponential_by_logistic_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        unity_scale = 6
        logistic = 2/(1 + math.exp(-x*unity_scale*self.magnitude/self.crossover)) - 1

        return exponential*logistic

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

class curve_exponential_by_unit_logistic_log_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        c = self.crossover
        m = self.magnitude

        # this can maybe be simplified if we break open tanh. e^log(m(x/c)) is m(x/c)
        u = math.log(m*(x/c))
        t = (math.tanh(u) + 1)/2
        f = c*t

        return exponential*f

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# Similar to curve_exponential_t, but scaled by x^m: x^m(e^(x - c))^n = e^(n(x - c) + m*ln(x))
# d/dx se^(n(x - c))x^m = sx^(m - 1)e^(n(x - c))(nx + m)
# d/dx sce^(n(x - c))(x/c)^m = se^(n(x - c))(x/c)^(m - 1)(nx + m)
# d/dx sce^(n(x/c - 1)/c)(x/c)^m = se^(n(x/c - 1))(x/c)^m(cm + nx)/x
class curve_exponential_by_power_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))
        power = math.pow(x, self.magnitude)/math.pow(self.crossover, self.magnitude)
        return exponential*power

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# exponential curve:
# d/dx e^(n(x - c))) = ne^(n(x - c))
class curve_exponential_t:
    def __call__(self, x):
        return math.exp(self.nonlinearity*(x - self.crossover))

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# doesn't scale by c arbitrarily
class curve_normalized_logistic_log_t:
    def __call__(self, x):
        c = self.crossover
        n = self.nonlinearity
        m = self.magnitude

        p = m
        q = .5/p

        t = math.log(x/c)
        k = -1 if t < 0 else 1
        f = (k*math.pow(math.tanh(math.pow(k*n*t, q)), 1/q) + 1)/2

        return f

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# crossfades a horizontal line into the exponential
class curve_horizontal_into_exponential_t:
    def __call__(self, x):
        # this needs to either be calculated or configured
        crossfade_offset = -0.25*self.crossover

        # using the y intercept gives a weird ratio between the original curve that doesn't happen with a constant
        # line_height = self.crossover*math.exp(-self.nonlinearity*self.crossover)
        line_height = 1e-2

        exp = math.exp(self.nonlinearity*(x - self.crossover))
        crossfade = (math.tanh((self.magnitude/2)*(x + crossfade_offset)) + 1)/2
        return self.crossover*math.pow(line_height, 1 - crossfade)*math.pow(exp, crossfade)

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# This is the weighted product of two functions, the exponential and the logistic, both centered on the crossover.
# The weights themselves are from another instance of the logistic, and they smoothly transition from the logistic
# to the exponential, with equal weights at the crossover.
class curve_exponential_by_logistic_log_t:
    def __call__(self, x):
        exp = math.exp(self.nonlinearity*(x - self.crossover))
        logistic = math.tanh(self.nonlinearity*math.log(x/(2*self.crossover))/2) + 1
        crossfade = (math.tanh(self.magnitude*x) + 1)/2
        return self.crossover*math.pow(logistic, 1 - crossfade)*math.pow(exp, crossfade)

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

class curve_constant_t:
    def __call__(self, _):
        return 1

    def __init__(self, crossover, nonlinearity, _):
        pass

class curve_linear_t:
    def __call__(self, x):
        return x

    def __init__(self, crossover, nonlinearity, _):
        pass

class curve_power_t:
    def __call__(self, x):
        return math.pow(x, self.magnitude)/math.pow(self.crossover, self.magnitude - 1)

    def __init__(self, crossover, _, magnitude):
        self.crossover = crossover
        self.magnitude = magnitude

# Similar to curve_exponential_t, but scaled by softplus, log(1 + e^mx)/m.
class curve_exponential_by_softplus_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        offset = 1.6
        softplus = (math.log(1 + math.exp(self.magnitude*offset*(x - 1))) - math.log(1 + math.exp(-self.magnitude*offset)))/self.magnitude
        return exponential*softplus

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# unidentified negative exponential curve: c(x/c)^m*(e^(-nx) - 1)/(e^(-nc) - 1)
# I found this from an older version of the exponential when I accidentaly put in a negative value.
# The curve is naturally s-shaped, with no initial tangent and it approaches linear(!), but it may be the limiter.
class curve_negative_exponential_t:
    def __call__(self, x):
        polynomial = math.pow(x/self.crossover, self.magnitude)
        exponential = (math.exp(-self.nonlinearity*x) - 1)/(math.exp(-self.nonlinearity*self.crossover) - 1)
        return self.crossover*polynomial*exponential

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# smooth ramp: log(1 + e^x)
class curve_softplus_t:
    def __call__(self, x):
        return math.log(1 + math.exp(self.magnitude*(x - self.crossover)))/self.magnitude

    def __init__(self, crossover, _nonlinearity, magnitude):
        self.crossover = crossover
        self.magnitude = magnitude

class curve_synchronous_t:
    def __call__(self, x):
        sign = -1 if x < self.crossover else 1
        l = sign*(self.gamma/math.log(self.motivity))*(math.log(x) - math.log(self.crossover))
        k = 0.5/self.smooth
        p = math.pow(l, k)
        h = sign*math.pow((math.tanh(p) + 1)/2, 1/k)
        return self.crossover*(math.pow(self.motivity, h) - 1)/(math.pow(self.motivity, 1/2) - 1)

    def __init__(self, crossover, gamma, _):
        self.crossover = crossover
        self.gamma = gamma
        self.motivity = 5
        self.smooth = .27

# The logistic function has slope 1 at the crossover and is symmetric: 1/(1 + e^-((n/c)(x - c)))
class curve_logistic_t:
    def __call__(self, x):
        return 2*self.crossover/(1 + math.exp(-(self.nonlinearity/self.crossover)*(x - self.crossover)))

    def __init__(self, crossover, nonlinearity, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity

# logistic of the log of x/c
# d/dx c^(1 - m)((x/c)^(-n/c) + 1)^-m = (mn(c^-m)((x/c)^(-n/c) + 1)^-m)/(x((x/c)^(n/c) + 1))
class curve_logistic_log_t:
    def __call__(self, x):
        c = self.crossover
        n = self.nonlinearity
        m = self.magnitude
        return math.pow(c, 1 - m)*math.pow(math.pow(x/c, -n/c) + 1, -m)

    def __init__(self, crossover, nonlinearity, magnitude):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

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

# combines a curve with a limiter and sensitivity
class generator_t:
    def __call__(self, x):
        unlimited = self.curve(x)
        limited = self.limiter(unlimited)
        return self.sensitivity*limited

    def __init__(self, curve, limiter, sensitivity):
        self.curve = curve
        self.limiter = limiter
        self.sensitivity = sensitivity

# soft limits using tanh
class limiter_tanh_t:
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

class limiter_null_t:
    def __call__(self, t):
        return t

    def __init__(self, _limit, _rate):
        pass

# chooses sample locations based on curvature
class sampler_curvature_t:
    sample_density = 1.5

    def __call__(self, t):
        # The curve should have more samples where the sensitivity changes most. For now, just oversample small t,
        # since the information there is more important. The change in sensitivity is the derivative of the whole
        # function, including limiting, which isn't trivial enough to bother with yet.
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

    impl.add_argument('-i', '--in-game-sensitivity', type=float, default=default_in_game_sensitivity,
        help="Multiply in-game sensitivity by this value. Scales final output by the inverse.")

    curve_choices={
        "constant": curve_constant_t,
        "linear": curve_linear_t,
        "power": curve_power_t,
        "exponential": curve_exponential_t,
        "exponential_by_power": curve_exponential_by_power_t,
        "exponential_by_softplus": curve_exponential_by_softplus_t,
        "exponential_by_logistic": curve_exponential_by_logistic_t,
        "negative_exponential": curve_negative_exponential_t,
        "softplus": curve_softplus_t,
        "synchronous": curve_synchronous_t,
        "logistic": curve_logistic_t,
        "logistic_log": curve_logistic_log_t,
        "normalized_logistic_log": curve_normalized_logistic_log_t,
        "smooth": curve_smooth_t,
        "smoothstep": curve_smoothstep_t,
        "exponential_by_logistic_log": curve_exponential_by_logistic_log_t,
        "horizontal_into_exponential": curve_horizontal_into_exponential_t,
        "exponential_by_unit_logistic_log": curve_exponential_by_unit_logistic_log_t,
    }
    impl.add_argument('-x', '--curve', choices=curve_choices.keys(), default=default_curve)

    format_choices={
       "raw_accel": output_raw_accel_t,
       "libinput": output_libinput_t,
    }
    impl.add_argument('-f', '--format', choices=format_choices.keys(), default="raw_accel")

    result = impl.parse_args()

    result.curve_t = curve_choices[result.curve]
    result.output_t = format_choices[result.format]

    result.limiter_t = limiter_null_t if result.curve_t == curve_normalized_logistic_log_t else limiter_tanh_t
    # result.limiter_t = limiter_null_t

    return result

args = create_arg_parser()
app_t().run(args.output_t(generator_t(args.curve_t(args.crossover/table_size, args.nonlinearity, args.magnitude),
    args.limiter_t(args.limit/args.sensitivity, args.limit_rate), args.sensitivity/args.in_game_sensitivity)))
