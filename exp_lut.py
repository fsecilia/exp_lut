#! /usr/bin/env python3

import math
import argparse

table_size = 50

default_in_game_sensitivity = 1
default_crossover = 10
default_nonlinearity = 2
default_sensitivity = 5
default_magnitude = 0.01
default_limit = 4
default_limit_rate = 8
default_curve = "limited_floored_power_law_log"

def logistic(t, r):
    return (math.pow(math.tanh(math.pow(t/2, r)), 1/r) + 1)/2

def unit_logistic(t, r):
    return 2*logistic(2*t, r) - 1

def taper(t, r):
    return t if t < 1 else unit_logistic(t - 1, r) + 1

'''
It's not clear in the literature whether human perception is logrithmic or follows a power law, so I've added both
a power law curve and a power law curve of the log. Setting n=1 is linear for the power law, and just the log for the
power law log.

These curves are pure power laws, ax^n. The floored versions are pure when no floor is set. Setting the floor sets the
initial value, changing the result to m + ax^n. The same result is possible by offsetting x instead, but the terms are
more complex. Because adding m is equivalent to shifting x, this still feels like a power law.

They feel smooth because the initial tangent is always 0. The limited versions follow the original curve exactly until
after the crossover, where they start to roll off smoothly. The limit is exactly sensitivity, -s. The rate is still
controlled by limit rate, -r.

The log version is a*log(x + 1)^n. It feels slower and smoother, but the params are the same.

The purpose of floored versions is to tune how much the mouse sits down when stopping and starting. Lower floors are
more accurate, but stop dead and feel sluggish to start again. It's most noticable when changing directions. Setting a
very low floor, on the order of .001 to .01, will pick up the minimum sensitivity. Too little and it will still feel
sticky and sluggish. Too much and it will feel floaty and inaccurate. The floor is controlled by -m, magnitude. It is
affected by sensitivity, but maybe shouldn't be.
'''

# same as power law, but magnitude specifies a min other than 0
class curve_floored_power_law_t:
    def __call__(self, x):
        return self.magnitude + (1 - self.magnitude)*math.pow(2, -self.nonlinearity)*math.pow(x/self.crossover, self.nonlinearity)

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# same as floored power law, but with a natural limiter
class curve_limited_floored_power_law_t:
    limited = True

    def __call__(self, x):
        tapered = taper(x/self.crossover, self.limit_rate)
        return self.magnitude + (1 - self.magnitude)*math.pow(2, -self.nonlinearity)*math.pow(tapered, self.nonlinearity)

    def __init__(self, crossover, nonlinearity, magnitude, limit_rate):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.limit_rate = limit_rate

# same as floored power law, but of the log
class curve_floored_power_law_log_t:
    def __call__(self, x):
        # when taking the log, the crossover moves by this much, so scale it back
        crossover_scale = math.exp(1) - 1
        return self.magnitude + (1 - self.magnitude)*math.pow(2, -self.nonlinearity)*math.pow(math.log(crossover_scale*x/self.crossover + 1), self.nonlinearity)

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# same as floored power law log, but with a natural limiter
class curve_limited_floored_power_law_log_t:
    limited = True

    def __call__(self, x):
        crossover_scale = math.exp(1) - 1
        tapered = taper(math.log(crossover_scale*x/self.crossover + 1), self.limit_rate)
        return self.magnitude + (1 - self.magnitude)*math.pow(2, - self.nonlinearity)*math.pow(tapered, self.nonlinearity)

    def __init__(self, crossover, nonlinearity, magnitude, limit_rate):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.limit_rate = limit_rate

# (m(x/c))^n = ax^n, a = mc^-n
class curve_power_law_t:
    def __call__(self, x):
        return math.pow(x/self.crossover, self.nonlinearity)

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# same as power law, but naturally limited in a way that very closely approximates the original power law up until c.
class curve_limited_power_law_t:
    limited = True

    def __call__(self, x):
        return unit_logistic(self.nonlinearity*math.pow(x/self.crossover, self.nonlinearity), self.limit_rate)

    def __init__(self, crossover, nonlinearity, magnitude, limit_rate):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.limit_rate = limit_rate

# same as power law, but of the log
class curve_power_law_log_t:
    def __call__(self, x):
        return math.pow(math.log(x/self.crossover + 1), self.nonlinearity)

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# same as power law log, but naturally limited
class curve_limited_power_law_log_t:
    limited = True

    def __call__(self, x):
        return unit_logistic(self.nonlinearity*math.pow(math.log(x/self.crossover + 1), self.nonlinearity), self.limit_rate)

    def __init__(self, crossover, nonlinearity, magnitude, limit_rate):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.limit_rate = limit_rate

# exponential scaled by right half of logistic. similar to by power with m=1, but the linear term tapers
# so the range above the crossover should deviate little from pure exp
class curve_exponential_by_logistic_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        unity_scale = 6
        logistic = unit_logistic(unity_scale*self.magnitude*(x/self.crossover), self.limit_rate)

        return exponential*logistic

    def __init__(self, crossover, nonlinearity, magnitude, limit_rate):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.limit_rate = limit_rate

class curve_exponential_by_unit_logistic_log_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        unity_scale = 6
        logistic = unit_logistic(-unity_scale*self.magnitude*(math.log(x/self.crossover)), self.limit_rate)

        return exponential*logistic

    def __init__(self, crossover, nonlinearity, magnitude, limit_rate):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.limit_rate = limit_rate

# Similar to curve_exponential_t, but scaled by x^m: x^m(e^(x - c))^n = e^(n(x - c) + m*ln(x))
# d/dx se^(n(x - c))x^m = sx^(m - 1)e^(n(x - c))(nx + m)
# d/dx sce^(n(x - c))(x/c)^m = se^(n(x - c))(x/c)^(m - 1)(nx + m)
# d/dx sce^(n(x/c - 1)/c)(x/c)^m = se^(n(x/c - 1))(x/c)^m(cm + nx)/x
class curve_exponential_by_power_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))
        power = math.pow(x, self.magnitude)/math.pow(self.crossover, self.magnitude)
        return exponential*power

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# exponential curve:
# d/dx e^(n(x - c))) = ne^(n(x - c))
class curve_exponential_t:
    def __call__(self, x):
        return math.exp(self.nonlinearity*(x - self.crossover))

    def __init__(self, crossover, nonlinearity, magnitude, _):
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

    def __init__(self, crossover, nonlinearity, magnitude, _):
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

    def __init__(self, crossover, nonlinearity, magnitude, _):
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

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

class curve_constant_t:
    def __call__(self, _):
        return 1

    def __init__(self, crossover, nonlinearity, _magnitude, _):
        pass

class curve_linear_t:
    def __call__(self, x):
        return x

    def __init__(self, crossover, nonlinearity, _magnitude, _):
        pass

class curve_power_t:
    def __call__(self, x):
        return math.pow(x, self.magnitude)/math.pow(self.crossover, self.magnitude - 1)

    def __init__(self, crossover, _nonlinearity, magnitude, _):
        self.crossover = crossover
        self.magnitude = magnitude

# Similar to curve_exponential_t, but scaled by softplus, log(1 + e^mx)/m.
class curve_exponential_by_softplus_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        offset = 1.6
        softplus = (math.log(1 + math.exp(self.magnitude*offset*(x - 1))) - math.log(1 + math.exp(-self.magnitude*offset)))/self.magnitude
        return exponential*softplus

    def __init__(self, crossover, nonlinearity, magnitude, _):
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

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# smooth ramp: log(1 + e^x)
class curve_softplus_t:
    def __call__(self, x):
        return math.log(1 + math.exp(self.magnitude*(x - self.crossover)))/self.magnitude

    def __init__(self, crossover, _nonlinearity, magnitude, _):
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

    def __init__(self, crossover, gamma, motivity, smooth):
        self.crossover = crossover
        self.gamma = gamma
        self.motivity = motivity
        self.smooth = smooth

# The logistic function has slope 1 at the crossover and is symmetric: 1/(1 + e^-((n/c)(x - c)))
class curve_logistic_t:
    def __call__(self, x):
        return 2*self.crossover/(1 + math.exp(-(self.nonlinearity/self.crossover)*(x - self.crossover)))

    def __init__(self, crossover, nonlinearity, magnitude, _):
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

    def __init__(self, crossover, nonlinearity, magnitude, _):
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude

# non-analytic smooth transition function
class curve_smooth_t:
    def __call__(self, x):
        offset = x - self.crossover
        return self.crossover*(math.tanh(self.nonlinearity*offset/(1 + offset*offset)) + 1)

    def __init__(self, crossover, nonlinearity, _magnitude, _):
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

    def __init__(self, crossover, _nonlinearity, _magnitude, _):
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
        "power_law": curve_power_law_t,
        "limited_power_law": curve_limited_power_law_t,
        "power_law_log": curve_power_law_log_t,
        "limited_power_law_log": curve_limited_power_law_log_t,
        "floored_power_law": curve_floored_power_law_t,
        "limited_floored_power_law": curve_limited_floored_power_law_t,
        "floored_power_law_log": curve_floored_power_law_log_t,
        "limited_floored_power_law_log": curve_limited_floored_power_law_log_t,
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

    if hasattr(result.curve_t, "limited") and result.curve_t.limited:
        result.limiter_t = limiter_null_t
    else:
        result.limiter_t = limiter_tanh_t

    return result

args = create_arg_parser()
app_t().run(args.output_t(
    generator_t(args.curve_t(args.crossover/table_size, args.nonlinearity, args.magnitude, args.limit_rate),
    args.limiter_t(args.limit/args.sensitivity, args.limit_rate), args.sensitivity/args.in_game_sensitivity)))
