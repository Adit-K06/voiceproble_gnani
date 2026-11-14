import glob

import numpy as np
from scipy.io import wavfile

for p in sorted(glob.glob("output/audio/*.wav")):
    sr, d = wavfile.read(p)
    print(p, sr, d.dtype, d.shape, round(len(d) / sr, 2), "s", "peak", int(np.abs(d).max()))