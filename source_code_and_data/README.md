# Code and data of WT1

Everything needed to reproduce the numbers, tables and figures of the article. Every
number quoted in the article is produced by one of the scripts in `scripts/`.

## Layout

```text
requirements.txt  Python packages
scripts/
  paths.py        single definition of all project paths (T18_DATA overrides the data path)
  run_all.py      reruns every step below, in order
  oma/            identification: library, campaign runners, diagnostic analyses,
                  table generation
  model/          six-degree-of-freedom rigid-cap model, axial pile stiffness,
                  pile-group interaction, sensitivity to the tower
  figures/        figure generation; writes into ../manuscript/figures/
results/          identified modes (CSV), MAC matrices, stabilisation diagrams,
                  FDD plots, and the data behind the tables of the article
data/
  README.md       description of the measurement records
  MANIFEST.txt    the record files with SHA-256 checksums
  raw/            the records, dot1/ and dot2/ (not in Git)
```

## Setup

Python 3 with the packages in `requirements.txt`:

```bash
pip install -r source_code_and_data/requirements.txt
```

The measurement records (2.4 GB; seven `.mat` files of about 470 MB are all the analysis
reads) are openly available at https://doi.org/10.5281/zenodo.23177722. Place the folders
`dot1/` and `dot2/` under `source_code_and_data/data/raw/`, or set the environment
variable `T18_DATA` to the folder that holds them. The record folders and files keep the
names of the published dataset.

## Reproducing the results

```bash
python source_code_and_data/scripts/run_all.py   # everything, about 10 minutes
```

`run_all.py` runs, in order (paths relative to `scripts/`):

```bash
python oma/run_campaign_1.py         # campaign 1: identification, 4 setups
python oma/run_campaign_2.py         # campaign 2: identification, 3 setups
python oma/participation.py          # effective channel count, both frequency bands
python oma/configuration_experiment.py   # covariance window against high-pass cut-off
python oma/low_band_search.py        # search below 0.5 Hz for a tower mode
python oma/criterion_calibration.py  # N_eff threshold and sensitivity, by simulation
python oma/make_tables.py            # data behind the tables of the article
python oma/rotation_centre_test.py   # vertical/horizontal ratio, shape of the 1.052 Hz pole
python oma/segment_uncertainty.py    # scatter of the low-band poles between record thirds
python model/foundation_stiffness.py # geometry, focal heights, forward axial stiffness
python model/rigid_cap_6dof.py       # 6-DOF rigid cap: modes, centres, L_eff inversion
python model/group_interaction.py    # group efficiency, forward axial pile stiffness
python model/tower_sensitivity.py    # centre of rotation with a tower mode as an oscillator
python figures/make_figures.py       # pile_layout, foundation_section, mode_shape
python figures/make_figures_extra.py # the remaining figures except processing_configurations
python figures/fig_harmonics.py      # processing_configurations
```

The scripts in `model/` need no records and run on their own.
`oma/criterion_calibration.py` is a simulation and takes several minutes.

`oma/oma_lib.py` holds the identification routines (pre-processing, SSI-COV, FDD,
MAC/MPC/MPD, harmonic screening); `oma/pipeline.py` is the processing chain shared by both
campaigns, so that the two are treated identically, a precondition for comparing
frequencies between them.

## Which script produces what

| Item of the article | Script | Result file |
|---|---|---|
| Table 1, focal heights | `model/foundation_stiffness.py` | printed |
| Table 3, Table 6 (Appendix A), Appendix B inversion | `model/rigid_cap_6dof.py` | `results/tables/table3_coupled_model.csv`, `table8_L_eff_inversion.csv` |
| Table 7 (Appendix B), axial stiffness | `model/group_interaction.py`, `model/foundation_stiffness.py` | printed |
| Table 5, vertical/horizontal ratio | `oma/rotation_centre_test.py` | `results/tables/table_rotation_centre.csv` |
| Effective channel count (Figure 6, Table 4) | `oma/participation.py` | printed |
| Online Resource 1, Tables S1, S2, S3, S5 | `oma/make_tables.py`, campaign runners | `results/tables/table2_records.csv`, `table4_poles.csv`, `table5_phase.csv`, `table9_shift.csv`, `results/campaign_1/`, `results/campaign_2/` |
| Online Resource 1, S2, threshold simulation | `oma/criterion_calibration.py` | printed |
| Online Resource 1, S4, processing settings | `oma/configuration_experiment.py` | printed |
| Online Resource 1, S5, low-band search | `oma/low_band_search.py` | printed |
| Online Resource 1, S7, tower sensitivity | `model/tower_sensitivity.py` | `results/tables/table_tower_sensitivity.csv` |
| Segment scatter | `oma/segment_uncertainty.py` | `results/tables/table_segment_uncertainty.csv` |

The model inputs were corrected from the design report on 2026-11-12 (cap volume 830 m3,
precast bottom slab 93 m3, pile modulus 42.9 GPa). The result files are CSV and PNG; they were not converted to JSON.
