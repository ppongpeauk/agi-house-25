# agi-house-25

## Diseases
| Disease Index | Disease Name       |
|---------------|--------------------|
| 0             | Cholera          |
| 1             | Malaria          |
| 2             | Unused          |
| 3             | Unused          |
| 4             | Unused          |
| 5             | Unused          |
| 6             | Unused          |
| 7             | Unused          |
| 8             | Unused          |
| 9             | Unused          |

## Getting Started
Create a new conda environment using `environment.yml`.

## How to Train
1. Place disease outbreak data in `./data/outbreaks`. Ensure it follows the format of `./data/outbreaks/example.csv`.
2. Run `aggregate.py`
3. Run `ptw.py`
4. Move the newly generated `merged_data.csv` file to the `./backend/model` directory.
5. Run `train.py`.
