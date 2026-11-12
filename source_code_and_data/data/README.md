# Ambient vibration records — pile-group foundation of offshore wind turbine T18

Dong Hai 1 wind farm, Tra Vinh province, Viet Nam. Two ambient vibration
campaigns recorded on the top surface of the pile cap, **campaign 1 before the
tower was erected and campaign 2 after**.

These are the records behind the paper *A geometric criterion for sway and
rocking in inclined pile groups*. The analysis code that reads them directly is
at <https://github.com/chungntu/t18-foundation-oma>.

## The structure

A circular reinforced concrete cap, 20.5 m in diameter and 4 m thick, on 34
prestressed high-strength concrete piles of 1.0 m outside diameter and 46 m
length, in three concentric rings. The inner ring of 4 piles is vertical, the
middle ring of 12 is raked 7:1, the outer ring of 18 is raked 5:1. Each pile
stands free in water over 12.34 m between the cap soffit at +5.435 and the
seabed at −6.900.

## Acquisition

Accelerometers on the cap, sampled at **1651 Hz**. The instrumented area is a
5 × 5 m square frame with sensors at the four corners and at the centre; in
campaign 2 a sixth channel sits on the tower two metres above the centre. The
position of the frame on the cap is not documented.

| Directory | File | Variable | Setup | Ch. | Samples | Minutes |
|---|---|---|---|---|---|---|
| `dot1` | `xl_thangdung.mat` | `xl_thangdung` | Vertical, campaign 1 | 5 | 2 535 936 | 25.6 |
| `dot1` | `xl_ngang1.mat` | `xl_ngang1` | Lateral 1, campaign 1 | 5 | 1 523 873 | 15.4 |
| `dot1` | `xl_ngang2.mat` | `xl_ngang2` | Lateral 2, campaign 1 | 5 | 1 385 189 | 14.0 |
| `dot1` | `xl_ga.mat` | `xl_ga` | Three-channel, campaign 1 | 3 | 1 545 336 | 15.6 |
| `dot2` | `xl_ngang.mat` | `xl_ngang` | Lateral, campaign 2 | 6 | 906 399 | 9.2 |
| `dot2` | `xl_td1.mat` | `xl_td1` | Vertical d1, campaign 2 | 6 | 1 875 536 | 18.9 |
| `dot2` | `xl_td2.mat` | `xl_td2` | Vertical d2ke, campaign 2 | 6 | 2 260 219 | 22.8 |

Each `.mat` array is samples × channels.

The directory and file names are the original Vietnamese ones and are kept
unchanged so that the published code runs without modification. `dot1` and
`dot2` are campaign 1 and campaign 2; `thangdung` and `td` mean vertical,
`ngang` means lateral, `xl` marks the extracted array.

## What is here

Raw LabVIEW `.tdms` records, the MATLAB arrays derived from them, the
sensitivity files, and the conversion scripts used to go from one to the other
(`convertTDMS.m`, `simpleConvertTDMS.m`, `code.m`, `logfile.m`). `MANIFEST.txt`
lists every file with its size and SHA-256 checksum.

Three kinds of file were left out: the `.tdms_index` files, which LabVIEW
regenerates on opening a record; the `*_ssi_cov*.mat` and `*_proc.mat` files,
which are intermediate outputs of an earlier analysis rather than measurements;
and an internal report document.

The foundation design drawings are **not** included. They belong to the project
owner and are not ours to redistribute.

## A warning about the records

The excitation is not the stationary broadband process that output-only
identification assumes, and the departure is large. In the lateral setup of
campaign 2 a single event reaches 139 standard deviations of its channel and
carries the kurtosis of a nine-minute record to 3883; that event appears on one
channel only. In the lateral 2 setup of campaign 1 the segment RMS varies by a
factor of 112 within one record, and in lateral 1 one channel is 47 times
weaker than another in the same setup. Between 50 and 200 Hz the records carry
78.6 % of their energy, against 0.7 % in the 0.2–5 Hz structural band, and they
contain mains harmonics at high level.

Anyone reanalysing these records should expect the processing configuration to
decide the outcome. Retaining the acquisition rate with a short covariance
window places a 1 Hz mode outside what the estimator can resolve, and a 5 Hz
high-pass removes it outright; either alone is enough to suppress the
structural band entirely.

## Citation

Please cite both this dataset and the paper it supports.

## Licence

CC BY 4.0.
