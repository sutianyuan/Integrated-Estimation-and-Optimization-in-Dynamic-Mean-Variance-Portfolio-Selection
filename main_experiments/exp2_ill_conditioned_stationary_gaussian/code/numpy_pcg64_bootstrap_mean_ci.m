function [ci_low, ci_high, sample_mean, audit] = ...
    numpy_pcg64_bootstrap_mean_ci(values, resamples, seed)
%NUMPY_PCG64_BOOTSTRAP_MEAN_CI Reproduce the approved NumPy bootstrap band.
%   Rows of VALUES are Monte Carlo replications and columns are stages.
%   Each bootstrap draw resamples complete replication rows. CI_LOW and
%   CI_HIGH are the pointwise 2.5% and 97.5% percentile limits for the
%   Monte Carlo mean. This is not mean +/- 1.96 SE and is not a
%   simultaneous confidence band.
%
%   The approved paper figures were produced with NumPy 2.0.2
%   default_rng(seed).integers(..., dtype=int32). This implementation uses
%   the identical PCG64 state transition and Lemire uint32 bounded-integer
%   mapping, so MATLAB recreates the accepted bootstrap indices without a
%   Python runtime. The two approved reporting seeds have fixed initial
%   PCG64 states below. The simulation RNG is never read or changed.

validateattributes(values, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'});
validateattributes(resamples, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'});
validateattributes(seed, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'});

[replications, stages] = size(values);
assert(replications <= double(intmax('uint32')), ...
    'Bootstrap replication count exceeds the uint32 generator range.');

generator = approved_generator_state(seed);
boot_means = zeros(resamples, stages);
first_indices = zeros(1, min(20, replications), 'uint32');
for draw = 1:resamples
    indices = zeros(replications, 1, 'uint32');
    for j = 1:replications
        [zero_based, generator] = bounded_uint32(generator, replications);
        indices(j) = zero_based + 1;
    end
    if draw == 1
        first_indices = indices(1:numel(first_indices))';
    end
    counts = accumarray(double(indices), 1, [replications, 1]);
    boot_means(draw, :) = (counts' * double(values)) / replications;
end

sample_mean = mean(values, 1);
ci_low = numpy_linear_quantile(boot_means, 0.025);
ci_high = numpy_linear_quantile(boot_means, 0.975);
audit = struct('method', 'pointwise_percentile_bootstrap_mean', ...
    'bootstrap_unit', 'Monte Carlo replication', ...
    'quantiles', [0.025, 0.975], 'resamples', resamples, ...
    'seed', seed, 'bit_generator', 'NumPy PCG64', ...
    'first_indices_one_based', first_indices);

if replications == 2000 && resamples > 0
    expected = approved_first_indices(seed);
    assert(isequal(first_indices, expected), ...
        'PCG64 compatibility self-check failed for approved seed %d.', seed);
end
end

function q = numpy_linear_quantile(values, probability)
% NumPy default quantile method='linear': h=(n-1)*p.
sorted_values = sort(values, 1);
h = (size(sorted_values, 1) - 1) * probability;
lower_row = floor(h) + 1;
upper_row = ceil(h) + 1;
fraction = h - floor(h);
q = (1 - fraction) * sorted_values(lower_row, :) + ...
    fraction * sorted_values(upper_row, :);
end

function generator = approved_generator_state(seed)
% Little-endian base-2^16 limbs of the NumPy PCG64 state and increment.
switch double(seed)
    case 20260914
        generator.state = hex_limbs({'d171','ac27','f6d5','8144', ...
            '205b','179c','d2f8','ba30'});
        generator.increment = hex_limbs({'7d9f','aa7a','d1c5','ea53', ...
            '4bfe','cb28','5d4a','8ba9'});
    case 20260915
        generator.state = hex_limbs({'ed58','6337','3dd9','9a65', ...
            'b964','be02','e5a7','7170'});
        generator.increment = hex_limbs({'f855','9c73','8b2e','671e', ...
            'b533','40c4','2f5f','2dd3'});
    otherwise
        error('Bootstrap:UnsupportedApprovedSeed', ...
            ['Seed %d is not an approved paper-figure reporting seed. ' ...
             'Use the reporting seed documented by the calling figure.'], seed);
end
generator.multiplier = hex_limbs({'f645','9fcc','df64','4385', ...
    '5da4','1fc6','ed05','2360'});
generator.has_uint32 = false;
generator.cached_uint32 = uint32(0);
end

function limbs = hex_limbs(parts)
limbs = zeros(1, numel(parts));
for k = 1:numel(parts)
    limbs(k) = hex2dec(parts{k});
end
end

function expected = approved_first_indices(seed)
switch double(seed)
    case 20260914
        zero_based = [1298,415,1808,918,0,1525,1157,1841,615,1268, ...
            1633,1193,1645,1334,1414,666,1312,472,639,585];
    case 20260915
        zero_based = [1111,1486,1833,1180,371,383,1039,1294,1844,492, ...
            112,1209,650,818,1861,1697,1238,1361,1593,538];
    otherwise
        error('Bootstrap:UnsupportedApprovedSeed', ...
            'Unsupported approved reporting seed %d.', seed);
end
expected = uint32(zero_based + 1);
end

function [value, generator] = bounded_uint32(generator, exclusive_high)
% NumPy's Lemire uint32 mapping for integers in [0, exclusive_high).
range = uint64(exclusive_high);
threshold = uint32(mod(2^32, double(range)));
while true
    [random32, generator] = next_uint32(generator);
    product = uint64(random32) * range;
    leftover = uint32(bitand(product, uint64(4294967295)));
    if leftover >= threshold
        value = uint32(floor(double(product) / 2^32));
        return
    end
end
end

function [value, generator] = next_uint32(generator)
if generator.has_uint32
    value = generator.cached_uint32;
    generator.has_uint32 = false;
    return
end
[random64, generator] = next_uint64(generator);
value = uint32(bitand(random64, uint64(4294967295)));
generator.cached_uint32 = uint32(bitshift(random64, -32));
generator.has_uint32 = true;
end

function [value, generator] = next_uint64(generator)
% PCG XSL-RR 128/64: advance first, then rotate high XOR low.
base = 65536;
next_state = zeros(1, 8);
carry = 0;
for k = 1:8
    total = carry + generator.increment(k);
    for i = 1:k
        total = total + generator.state(i) * generator.multiplier(k-i+1);
    end
    next_state(k) = mod(total, base);
    carry = floor(total / base);
end
generator.state = next_state;

low64 = limbs_to_uint64(next_state(1:4));
high64 = limbs_to_uint64(next_state(5:8));
xor_value = bitxor(high64, low64);
rotation = double(bitshift(high64, -58));
if rotation == 0
    value = xor_value;
else
    value = bitor(bitshift(xor_value, -rotation), ...
        bitshift(xor_value, 64-rotation));
end
end

function value = limbs_to_uint64(limbs)
value = uint64(limbs(1));
value = bitor(value, bitshift(uint64(limbs(2)), 16));
value = bitor(value, bitshift(uint64(limbs(3)), 32));
value = bitor(value, bitshift(uint64(limbs(4)), 48));
end
