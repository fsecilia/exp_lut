#! /usr/bin/env python3
'''
In Advanced->Device Menu settings:
- check Override defaults
- uncheck Disable
- set DPI to 16000
- set polling rate to 0

Start with the current settings: python exp_lut.py -s 50 -c 25 -m 0.5 -f 0.009

This should be close. You'll likely have to adjust sensitivity to start, -s. Once that feels usable, keep using it until
you notice it getting fast too soon or too late, then adjust crossover, -c. The rest are pretty subtle. It'll likely
take a bit until you notice what they affect enough to adjust them, but when you do, here's how to tune this curve.

tl;dr: -s is how much, -c is how soon, -m is how smooth, -f is how sticky

The graph of the curve is here: https://www.desmos.com/calculator/iwmeivjih6
You can use it to see your current settings and compare what happens when you change a parameter. The first set of
variables there match parameters you pass to the script on the command line, or edit in default_params in the script
itself. These control the purple curve. There is also another set of variables with mostly random names, but the same
order, that control the orange curve. You can use them to show your current settings in orange as a reference while you
change purple, then input the purple's settings to the script.

Proceed in this order. They are ordered broadly by increasing subtlety, but adjusting floor requires correct magnitude.

1) Sensitivity, -s, controls how fast fast is. This is a scalar applied to the final result, stretching or squishing
the whole output. It has nothing to do with curve shape or onset time, just the absolute magnitude. If it feels too
fast at all speeds, lower -s. If it feels too slow, raise -s.

2) Crossover, -c, controls when fast is fast. This is the speed where the function switches from minimizing to
maximizing, from scaling down to scaling up. All speeds to the left of this value are slower than 1.0, all to the
right, faster. It feels like it controls how soon the nonlinearity kicks in. If it feels like it gets too fast too
quickly, increase -c. If it feels like it takes too long to get going, decrease -c. This moves inversely because it is
in the same units as input velocity, which puts it in the denominator.

On the graph, crossover ranges from 0 to 1 because all the ways to make it range from 0 to 50 make using the graph more
difficult. When you go to put the crossover from the graph in the script, multiply it by 50. The graph even has a
50c entry under c, so you can see the literal value. This is the only value that has this weirdness; the rest are all
input literally.

3) -m controls the starting tangent. You must use the graph to inspect this visually. Zoom way in. You want it to just
kiss horizontally off the start of the graph, like a parabola. If it is laying down too much, with an obvious flat
spot, increase -m. If it is starting at too steep of an angle, decrease -m. This one should not need much, and if it
does, much above .6 or below .4 is probably too much. .5 is neutral.

4) -f controls the floor, shifting the whole graph up or down. Only adjust this after the tangent looks correct because
they affect similar things, but the tangent can be inspected visually to be sure it is correct. If it feels like it
sits down too hard when stopping and is difficult to get it moving at all, increase -f. If it feels like it is skating
away and never stops enough, decrease -f. The floor should be very small, on the order of .01 for glass, maybe .1 for
cloth.
'''

import math
import argparse

table_size = 50

# (x + 1)^(n*ln(x + 1)) - 1
# x*ln(x + 1)
# sxe^(sign(ln(x/c))abs(ln(x/c))^n) starts of flat, grows quickly, flattens out near crossover to s, then grows quickly again.

class params_t:
    def __init__(self, curve, sensitivity, crossover, nonlinearity, magnitude, floor,
        limit, limit_rate):
        self.curve = curve
        self.sensitivity = sensitivity
        self.crossover = crossover
        self.nonlinearity = nonlinearity
        self.magnitude = magnitude
        self.floor = floor
        self.limit = limit
        self.limit_rate = limit_rate

default_params = params_t(
    curve = "floored_log",
    floor = 0.011,
    limit = 0.0,
    limit_rate = 0.0,
    sensitivity = 50,
    crossover = 25,
    nonlinearity = 1.0,
    magnitude = 0.5,
)

def logistic(t, r):
    sign = -1 if t < 0 else 1
    return (sign*math.pow(math.tanh(math.pow(sign*(t/2), r)), 1/r) + 1)/2

def unit_logistic(t, r):
    return 2*logistic(2*t, r) - 1

def taper(t, r):
    return t if t < 1 else unit_logistic(t - 1, r) + 1

def taper_output(t, c, l, r):
    return t if t < c else (l - c)*unit_logistic((t - c)/(l - c), r) + c

def taper_input(t, l, r):
    return unit_logistic((t/l)*(1 + 1/r), r)

def floor(t, s, c, f):
    return t*(s*c - f) + f

class curve_floored_log_t:
    limited = True

    def __call__(self, x):
        # aliases to match graph
        f = self.floor
        s = self.sensitivity
        c = self.crossover
        n = self.nonlinearity
        m = self.magnitude

        r = 0.5/m

        t = n*math.log(x/c)
        k = -1 if t < 0 else 1
        g = (k*math.pow(math.tanh(math.pow(k*t, r)), 1/r) + 1)/2
        y = g*s + f

        return y/s

    def __init__(self, params):
        self.floor = params.floor
        self.sensitivity = params.sensitivity
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

class curve_input_limited_exponential_t:
    limited = True

    def __call__(self, x):
        # aliases to match graph
        c = self.crossover
        n = self.nonlinearity
        l = self.limit
        r = self.limit_rate

        # limit input
        u = 2*logistic(2*(x + 1 - l), r) - 2*logistic(2*(1 - l), r)

        # calc normally with limited input
        y = math.exp(n*(u - c))

        return y

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.limit = params.limit
        self.limit_rate = params.limit_rate

# same as input_limited_tapered_tangent_exponential, but of the log
class curve_input_limited_log_t:
    limited = True

    def __call__(self, x):
        # aliases to match graph
        s = self.sensitivity
        c = self.crossover
        n = self.nonlinearity
        m = self.magnitude
        f = self.floor
        l = self.limit
        r = self.limit_rate

        # limit input
        u = 2*logistic(2*(x + 1 - l), r) - 2*logistic(2*(1 - l), r)
        #u = x

        # calc normally with limited input
        y = math.pow(math.log(m*u/c + math.exp(1)/2), n)
        y0 = math.pow(math.log(math.exp(1)/2), n)

        return y - y0 + f/s

    def __init__(self, params):
        self.sensitivity = params.sensitivity
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude
        self.floor = params.floor
        self.limit = params.limit
        self.limit_rate = params.limit_rate

# floored exponential, but the input is limited and the tangent is tapered
# this is very similar to curve_input_limited_floored_exponential_t, but with a tangent suitable for glass.
# https://www.desmos.com/calculator/nwefns0msj
class curve_input_limited_tapered_tangent_exponential_t:
    limited = True

    def __call__(self, x):
        # aliases to match graph
        s = self.sensitivity
        c = self.crossover
        n = self.nonlinearity
        m = self.magnitude
        f = self.floor
        l = self.limit
        r = self.limit_rate

        # taper input tangent
        t = 2*logistic(2*(x - (m + 1)/m), m)

        # limit input
        u = 2*logistic(2*(t + 1 - l), r) - 2*logistic(2*(1 - l), r)

        # calc normally with limited input
        y = math.exp(n*(u - c))

        # floor output
        y0 = math.exp(-n*c) - f/s

        return y - y0

    def __init__(self, params):
        self.sensitivity = params.sensitivity
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude
        self.floor = params.floor
        self.limit = params.limit
        self.limit_rate = params.limit_rate

# exponential, but limited on input side, with simple additive (nonscaling) floor
# the tangent is unmodified and you can feel it on glass.
class curve_input_limited_floored_exponential_t:
    limited = True

    def __call__(self, x):
        # aliases to match graph
        s = self.sensitivity
        c = self.crossover
        n = self.nonlinearity
        f = self.floor
        l = self.limit
        r = self.limit_rate
        k = -1 if x < 0 else 1

        # limit input
        t = k*l*math.pow(math.tanh(k*math.pow(x/l, r)), 1/r)

        # calc normally with limited input
        y = math.exp((n/c)*(t - c))

        # floor output
        y0 = math.exp(-n) - f/s

        return y - y0

    def __init__(self, params):
        self.sensitivity = params.sensitivity
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.floor = params.floor
        self.limit = params.limit
        self.limit_rate = params.limit_rate

# same as exponential_t, but naturally tapers output and applies a floor
class curve_limited_floored_exponential_t:
    limited = True

    def __call__(self, x):
        y0 = math.exp(-self.nonlinearity*self.crossover)
        y = math.exp(self.nonlinearity*(x - self.crossover)) - y0
        f = y + self.floor
        return taper_output(f, self.crossover, self.limit, self.limit_rate)

    def __init__(self, params):
        self.sensitivity = params.sensitivity
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.floor = params.floor
        self.limit = params.limit
        self.limit_rate = params.limit_rate

# same as power law, but with a simple floor
class curve_floored_power_law_t:
    def __call__(self, x):
        return self.floor + math.pow(2, -self.nonlinearity)*math.pow(x/self.crossover, self.nonlinearity)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.floor = params.floor

# same as floored power law, but with a natural output limiter
class curve_limited_floored_power_law_t:
    limited = True

    def __call__(self, x):
        k = math.pow(self.limit, 1/self.nonlinearity)
        tapered = k*taper(x/(k*self.crossover), self.limit_rate)
        return self.floor + math.pow(2, -self.nonlinearity)*math.pow(tapered, self.nonlinearity)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.floor = params.floor
        self.limit = params.limit
        self.limit_rate = params.limit_rate

# same as floored power law, but of the log
class curve_floored_power_law_log_t:
    def __call__(self, x):
        # when taking the log, the crossover moves by this much, so scale it back
        crossover_scale = math.exp(1) - 1
        return self.floor + math.pow(2, -self.nonlinearity)*math.pow(math.log(crossover_scale*x/self.crossover + 1), self.nonlinearity)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.floor = params.floor

# same as floored power law log, but with a natural output limit
class curve_limited_floored_power_law_log_t:
    limited = True

    def __call__(self, x):
        crossover_scale = math.exp(1) - 1
        tapered = taper(math.log(crossover_scale*x/self.crossover + 1), self.limit_rate)
        return self.floor + math.pow(2, - self.nonlinearity)*math.pow(tapered, self.nonlinearity)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.floor = params.floor
        self.limit_rate = params.limit_rate

# (m(x/c))^n = ax^n, a = mc^-n
class curve_power_law_t:
    def __call__(self, x):
        return math.pow(x/self.crossover, self.nonlinearity)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity

# same as power law, but naturally limited in a way that very closely approximates the original power law up until c.
class curve_limited_power_law_t:
    limited = True

    def __call__(self, x):
        return unit_logistic(self.nonlinearity*math.pow(x/self.crossover, self.nonlinearity), self.limit_rate)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude
        self.limit_rate = params.limit_rate

# same as power law, but of the log
class curve_power_law_log_t:
    def __call__(self, x):
        return math.pow(math.log(x/self.crossover + 1), self.nonlinearity)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

# same as power law log, but naturally limited
class curve_limited_power_law_log_t:
    limited = True

    def __call__(self, x):
        return unit_logistic(self.nonlinearity*math.pow(math.log(x/self.crossover + 1), self.nonlinearity), self.limit_rate)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude
        self.limit_rate = params.limit_rate

# exponential scaled by right half of logistic. similar to by power with m=1, but the linear term tapers
# so the range above the crossover should deviate little from pure exp
class curve_exponential_by_logistic_t:
    def __call__(self, x):
        return math.exp(self.nonlinearity*(x - self.crossover))*unit_logistic(self.magnitude*(x/self.crossover), self.limit_rate)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude
        self.limit_rate = params.limit_rate

class curve_exponential_by_unit_logistic_log_t:
    def __call__(self, x):
        return math.exp(self.nonlinearity*(x - self.crossover))*unit_logistic(-self.magnitude*(math.log(x/self.crossover)), self.limit_rate)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude
        self.limit_rate = params.limit_rate

# Similar to curve_exponential_t, but scaled by x^m: x^m(e^(x - c))^n = e^(n(x - c) + m*ln(x))
# d/dx se^(n(x - c))x^m = sx^(m - 1)e^(n(x - c))(nx + m)
# d/dx sce^(n(x - c))(x/c)^m = se^(n(x - c))(x/c)^(m - 1)(nx + m)
# d/dx sce^(n(x/c - 1)/c)(x/c)^m = se^(n(x/c - 1))(x/c)^m(cm + nx)/x
class curve_exponential_by_power_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))
        power = math.pow(x, self.magnitude)/math.pow(self.crossover, self.magnitude)
        return exponential*power

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

# exponential curve:
# d/dx e^(n(x - c))) = ne^(n(x - c))
class curve_exponential_t:
    def __call__(self, x):
        return math.exp(self.nonlinearity*(x - self.crossover))

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

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

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

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

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

# This is the weighted product of two functions, the exponential and the logistic, both centered on the crossover.
# The weights themselves are from another instance of the logistic, and they smoothly transition from the logistic
# to the exponential, with equal weights at the crossover.
class curve_exponential_by_logistic_log_t:
    def __call__(self, x):
        exp = math.exp(self.nonlinearity*(x - self.crossover))
        logistic = math.tanh(self.nonlinearity*math.log(x/(2*self.crossover))/2) + 1
        crossfade = (math.tanh(self.magnitude*x) + 1)/2
        return self.crossover*math.pow(logistic, 1 - crossfade)*math.pow(exp, crossfade)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

class curve_constant_t:
    def __call__(self, _):
        return 1

    def __init__(self, _):
        pass

class curve_linear_t:
    def __call__(self, x):
        return x

    def __init__(self, _):
        pass

class curve_power_t:
    def __call__(self, x):
        return math.pow(x, self.magnitude)/math.pow(self.crossover, self.magnitude - 1)

    def __init__(self, params):
        self.crossover = params.crossover
        self.magnitude = params.magnitude

# Similar to curve_exponential_t, but scaled by softplus, log(1 + e^mx)/m.
class curve_exponential_by_softplus_t:
    def __call__(self, x):
        exponential = math.exp(self.nonlinearity*(x - self.crossover))

        offset = 1.6
        softplus = (math.log(1 + math.exp(self.magnitude*offset*(x - 1))) - math.log(1 + math.exp(-self.magnitude*offset)))/self.magnitude
        return exponential*softplus

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

# unidentified negative exponential curve: c(x/c)^m*(e^(-nx) - 1)/(e^(-nc) - 1)
# I found this from an older version of the exponential when I accidentaly put in a negative value.
# The curve is naturally s-shaped, with no initial tangent and it approaches linear(!), but it may be the limiter.
class curve_negative_exponential_t:
    def __call__(self, x):
        polynomial = math.pow(x/self.crossover, self.magnitude)
        exponential = (math.exp(-self.nonlinearity*x) - 1)/(math.exp(-self.nonlinearity*self.crossover) - 1)
        return self.crossover*polynomial*exponential

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

# smooth ramp: log(1 + e^x)
class curve_softplus_t:
    def __call__(self, x):
        return math.log(1 + math.exp(self.magnitude*(x - self.crossover)))/self.magnitude

    def __init__(self, params):
        self.crossover = params.crossover
        self.magnitude = params.magnitude

class curve_synchronous_t:
    def __call__(self, x):
        sign = -1 if x < self.crossover else 1
        l = sign*(self.gamma/math.log(self.motivity))*(math.log(x) - math.log(self.crossover))
        k = 0.5/self.smooth
        p = math.pow(l, k)
        h = sign*math.pow((math.tanh(p) + 1)/2, 1/k)
        return self.crossover*(math.pow(self.motivity, h) - 1)/(math.pow(self.motivity, 1/2) - 1)

    def __init__(self, params):
        self.crossover = params.crossover
        self.gamma = params.gamma
        self.motivity = params.motivity
        self.smooth = params.smooth

# The logistic function has slope 1 at the crossover and is symmetric: 1/(1 + e^-((n/c)(x - c)))
class curve_logistic_t:
    def __call__(self, x):
        return 2*self.crossover/(1 + math.exp(-(self.nonlinearity/self.crossover)*(x - self.crossover)))

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity

# logistic of the log of x/c
# d/dx c^(1 - m)((x/c)^(-n/c) + 1)^-m = (mn(c^-m)((x/c)^(-n/c) + 1)^-m)/(x((x/c)^(n/c) + 1))
class curve_logistic_log_t:
    def __call__(self, x):
        c = self.crossover
        n = self.nonlinearity
        m = self.magnitude
        return math.pow(c, 1 - m)*math.pow(math.pow(x/c, -n/c) + 1, -m)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity
        self.magnitude = params.magnitude

# non-analytic smooth transition function
class curve_smooth_t:
    def __call__(self, x):
        offset = x - self.crossover
        return self.crossover*(math.tanh(self.nonlinearity*offset/(1 + offset*offset)) + 1)

    def __init__(self, params):
        self.crossover = params.crossover
        self.nonlinearity = params.nonlinearity

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

    def __init__(self, params):
        self.crossover = params.crossover

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

    def __init__(self, params):
        self.limit = params.limit/params.sensitivity
        self.rate = params.limit_rate

class limiter_null_t:
    def __call__(self, t):
        return t

    def __init__(self, _):
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
    impl.add_argument('-s', '--sensitivity', type=float, default=default_params.sensitivity)
    impl.add_argument('-c', '--crossover', type=float, default=default_params.crossover)
    impl.add_argument('-n', '--nonlinearity', type=float, default=default_params.nonlinearity)
    impl.add_argument('-m', '--magnitude', type=float, default=default_params.magnitude)
    impl.add_argument('-f', '--floor', type=float, default=default_params.floor)
    impl.add_argument('-l', '--limit', type=float, default=default_params.limit)
    impl.add_argument('-r', '--limit-rate', type=float, default=default_params.limit_rate)

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
        "limited_floored_exponential": curve_limited_floored_exponential_t,
        "input_limited_floored_exponential": curve_input_limited_floored_exponential_t,
        "input_limited_tapered_tangent_exponential": curve_input_limited_tapered_tangent_exponential_t,
        "input_limited_log": curve_input_limited_log_t,
        "input_limited_exponential": curve_input_limited_exponential_t,
        "floored_log": curve_floored_log_t,
    }
    impl.add_argument('-x', '--curve', choices=curve_choices.keys(), default=default_params.curve)

    output_format_choices={
       "raw_accel": output_raw_accel_t,
       "libinput": output_libinput_t,
    }
    impl.add_argument('-o', '--output-format', choices=output_format_choices.keys(), default="raw_accel")

    result = impl.parse_args()

    result.curve_t = curve_choices[result.curve]
    result.output_t = output_format_choices[result.output_format]

    if not result.limit_rate or (hasattr(result.curve_t, "limited") and result.curve_t.limited):
        result.limiter_t = limiter_null_t
    else:
        result.limiter_t = limiter_tanh_t

    result.params = params_t(
        curve = result.curve,
        sensitivity = result.sensitivity,
        crossover = result.crossover/table_size,
        nonlinearity = result.nonlinearity,
        magnitude = result.magnitude,
        floor = result.floor,
        limit_rate = result.limit_rate,
        limit = result.limit
    )

    return result

args = create_arg_parser()
app_t().run(args.output_t(generator_t(args.curve_t(args.params), args.limiter_t(args.params), args.params.sensitivity)))
