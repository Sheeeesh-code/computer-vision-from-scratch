Image 120x90, template 15x15 (grayscale, best of 5 runs; naive run once)

| implementation | time (s) | speed-up vs naive |
|---|---|---|
| naive (4 loops) | 0.4704 | 1x |
| fast (2 loops) | 0.0274 | 17x |
| matrix (im2col) | 0.0021 | 225x |
| OpenCV matchTemplate (C++, FFT) | 0.0002 | 2,872x |
