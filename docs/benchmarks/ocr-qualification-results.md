# OCR qualification

## ja (PepperCarrotJA, 1 chapter(s))

| candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors |
|---|---:|---:|---:|---:|---:|---:|
| ppocr-v6-medium | 0.832 | 0.899 | 0.095 | 2.527 | 2.668 | 0 |
| ppocr-v6-small | 0.778 | 0.898 | 0.095 | 1.362 | 2.024 | 0 |
| paddleocr-vl-1.6 | 0.757 | 0.901 | 0.870 | 81.968 | 3.989 | 0 |
| ppocr-v5-server-multi ★ | 0.746 | 0.897 | 0.092 | 3.382 | 4.964 | 0 |
| manga-ocr-base | 0.689 | 0.889 | 0.783 | 3.998 | 1.294 | 0 |
| manga-ocr-2025 | 0.686 | 0.902 | 0.836 | 4.639 | 1.174 | 0 |
| ppocr-v6-tiny | 0.152 | 0.846 | 0.266 | 0.991 | 0.922 | 0 |

Recommendation: PROMOTE ppocr-v6-medium (gains 0.086 chrF with -0.002 char recall loss)

## ko (PepperCarrotKR, 1 chapter(s))

| candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors |
|---|---:|---:|---:|---:|---:|---:|
| paddleocr-vl-1.6 | 0.567 | 0.908 | 0.965 | 100.010 | 3.988 | 0 |
| ppocr-v5-ko ★ | 0.475 | 0.852 | 0.113 | 16.690 | 3.671 | 0 |
| ppocr-v6-medium | 0.105 | 0.743 | 0.113 | 3.846 | 3.805 | 0 |
| ppocr-v6-small | 0.095 | 0.737 | 0.113 | 2.162 | 2.963 | 0 |
| ppocr-v6-tiny | 0.089 | 0.759 | 0.161 | 1.658 | 1.223 | 0 |

Recommendation: PROMOTE paddleocr-vl-1.6 (gains 0.091 chrF with -0.056 char recall loss)

## zh (PepperCarrotCN, 1 chapter(s))

| candidate | chrF | char recall | CER (box, not comparable) | s/chapter | VRAM GiB | errors |
|---|---:|---:|---:|---:|---:|---:|
| paddleocr-vl-1.6 | 0.741 | 0.949 | 0.719 | 56.013 | 3.979 | 0 |
| ppocr-v6-small | 0.642 | 0.945 | 0.084 | 1.353 | 2.707 | 0 |
| ppocr-v5-server-multi ★ | 0.582 | 0.938 | 0.098 | 3.473 | 7.024 | 0 |
| ppocr-v6-medium | 0.557 | 0.937 | 0.092 | 2.812 | 3.946 | 0 |
| ppocr-v6-tiny | 0.474 | 0.235 | 0.435 | 0.911 | 1.175 | 0 |

Recommendation: PROMOTE paddleocr-vl-1.6 (gains 0.159 chrF with -0.011 char recall loss)
