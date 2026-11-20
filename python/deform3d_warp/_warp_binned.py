"""Reuse Warp 1.19's exact RUN_TO_RUN float accumulator without record sorting.

The algorithm is Warp's implementation of reproducible binned summation, not an
original deform3d algorithm. We load its reviewed definition from the installed
Warp runtime, retain its arithmetic, and adapt only the local C++ declaration
syntax. No third-party implementation is vendored here. See the source's
attribution to Ahrens, Demmel, Nguyen and reproducible_floating_sums.
"""
import hashlib
from pathlib import Path
import re
import warp as wp

SOURCE_SHA256 = 'e092c78963d32491dbb11c54e2730867c6e7bc33223a24076822960174719aa4'


def _definition():
    path = Path(wp.__file__).resolve().parent/'native/deterministic.cu'
    try:
        source = path.read_bytes()
    except OSError as error:
        raise RuntimeError('owned material requires the reviewed Warp native/deterministic.cu source') from error
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('unreviewed Warp binned accumulator source; requires reviewed Warp 1.19.0')
    source = source.decode()
    start = source.index('struct BinnedFloatAccumulator {')
    end = source.index('\nstruct FloatToBinnedAccumulator {',start)
    definition = source[start:end]
    # Local C++ classes cannot have static data members. Integral constants
    # become enum constants; the two exact power-of-two float constants inline.
    definition = re.sub(r'static constexpr int (\w+) = (.*?);',r'enum { \1 = \2 };',definition)
    definition = definition.replace('    static constexpr float COMPRESSION = 1.0f / 4096.0f;\n','')
    definition = definition.replace('    static constexpr float EXPANSION = 4096.0f;\n','')
    definition = re.sub(r'\bCOMPRESSION\b','(1.0f / 4096.0f)',definition)
    definition = re.sub(r'\bEXPANSION\b','4096.0f',definition)
    return definition


_DEFINITION = _definition()
Bins = wp.types.vector(length=6,dtype=wp.float32)


@wp.func_native(_DEFINITION + '''
    BinnedFloatAccumulator total, record;
    for (int i=0; i<3; ++i) { total.primary[i]=state[i]; total.carry[i]=state[3+i]; }
    record.add(value);
    total.add(record);
    wp::vec_t<6,float> result;
    for (int i=0; i<3; ++i) { result[i]=total.primary[i]; result[3+i]=total.carry[i]; }
    return result;
''')
def add_record(state: Bins, value: float) -> Bins: ...


@wp.func_native(_DEFINITION + '''
    BinnedFloatAccumulator total;
    for (int i=0; i<3; ++i) { total.primary[i]=state[i]; total.carry[i]=state[3+i]; }
    return total.to_float();
''')
def rounded_sum(state: Bins) -> float: ...
