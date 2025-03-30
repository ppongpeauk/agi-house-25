# agi-house-25

## Getting Started
Create a new conda environment using `environment.yml`.

## How to Train
1. Place disease outbreak data in `./data/outbreaks`. Ensure it follows the format of `./data/outbreaks/example.csv`.
2. Run `aggregate.py`
3. Run `ptw.py`
4. Move the newly generated `merged_data.csv` file to the `./backend/model` directory.
5. Run `train.py`.

## Diseases
| Disease Index | Disease Name               |
|---------------|----------------------------|
| 0             | Cholera                    |
| 1             | Malaria                    |
| 2             | Dengue                     |
| 3             | Chikungunya                |
| 4             | COVID-19                   |
| 5             | Cutaneous leishmaniasis    |
| 6             | Dracunculiasis             |
| 7             | Visceral leishmaniasis     |
| 8             | Measles                    |
| 9             | Meningitis                 |

<img src="./assets/Capture-2025-03-29-222524.png">
