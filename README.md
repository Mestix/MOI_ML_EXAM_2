# ECG-heartslagclassificatie met convolutionele neurale netwerken

**Naam:** Melissa Landwerd  
**Studentnummer:** 1878919  
**Cursus:** Machine Learning 2025

Dit project onderzoekt hoe verschillende CNN-keuzes de classificatie van ECG-hartslagen uit de MIT-BIH Arrhythmia Dataset beïnvloeden. De nadruk ligt op een iteratief onderzoeksproces en op **macro-recall en recall per klasse**, omdat de dataset sterk ongebalanceerd is.

## Onderzochte experimenten

1. **1D-CNN versus 2D-CNN** – vergelijking van de oorspronkelijke tijdreeksrepresentatie met een 16×12-representatie.
2. **Class weights** – inverse-frequency gewichten om zeldzame klassen zwaarder mee te laten tellen in de loss.
3. **Parallelle 2D-CNN** – twee parallelle routes: een vaste 3×3-kernel en een tweede route met een grotere kernel.
4. **Ray Tune** – hyperparameteroptimalisatie van de parallelle CNN op validation macro-recall.

De modellen staan in `src/networks.py`. Datasetlogica staat in `src/datasets.py`, de train/validation-split en class weights in `src/data_utils.py`, de experimenteerlogica in `src/helpers.py` en de Ray Tune-logica in `src/tuning.py`.

Gedeelde instellingen zoals datasetpaden, klasselabels, het aantal ECG-features
en de 2D-matrixvorm staan in `src/config.py`. De `FullDataStreamer` zorgt ervoor
dat ook de laatste onvolledige batch wordt verwerkt.

## Methodologische splitsing

De meegeleverde `heart_big_train.parq` wordt met een vaste seed **stratified** opgesplitst in train (80%) en validation (20%). De officiële `heart_big_test.parq` blijft buiten modelselectie en Ray Tune en wordt alleen gebruikt voor de uiteindelijke evaluatie van een getraind/geselecteerd model.

Dit voorkomt dat de testset invloed heeft op hyperparameterselectie.

> Let op: uit alleen deze lokale parquetbestanden/code kan niet worden vastgesteld of de oorspronkelijke train/test-splitsing patiëntonafhankelijk is. Dat moet uit de datasetdocumentatie worden geverifieerd en als beperking in het verslag worden beschreven als dit niet aantoonbaar is.

## Installatie

Dit project gebruikt Python 3.12 en `uv`.

```bash
uv sync
```

De ECG-data staat in:

```text
data/heart_big_train.parq
data/heart_big_test.parq
```

## Experimenten starten vanaf de commandline

Gebruik `main.py` als centrale ingang.

```bash
# Baselines
uv run python main.py --experiment baseline_1d
uv run python main.py --experiment baseline_2d

# Gewogen modellen
uv run python main.py --experiment weighted_1d
uv run python main.py --experiment weighted_2d

# Parallel model
uv run python main.py --experiment parallel

# Ray Tune (standaard 20 configuraties, 10 epochs per trial aanbevolen)
uv run python main.py --experiment tune --epochs 10 --num-samples 20

# Train/evalueer beste tuningconfiguratie op de officiële testset
uv run python main.py --experiment final --epochs 15 --runs 3
```

Instellingen kunnen via flags worden aangepast:

```bash
uv run python main.py --experiment parallel \
  --epochs 5 \
  --runs 3 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --filters 16 \
  --large-kernel 5 \
  --dropout 0.0 \
  --seed 42
```

## Ray Tune zoekruimte

De tuning gebruikt:

| Hyperparameter | Zoekruimte |
|---|---|
| Learning rate | `1e-4` – `1e-2`, loguniform |
| Filters | `8, 16, 32, 64` |
| Batch size | `16, 32, 64` |
| Weight decay | `1e-6` – `1e-3`, loguniform |
| Dropout | `0.0` – `0.4`, uniform |
| Tweede kernelroute | `5×5` of `7×7` |

De eerste parallelle route blijft vast op **3×3**. Ray Tune selecteert op macro-recall van de validation set. ASHA wordt gebruikt om zwakke configuraties vroegtijdig te stoppen.

## Resultaten

Experimenten schrijven resultaten weg naar:

```text
results/results.csv
results/<experiment>_summary.csv
results/ray_tuning_results.csv
figures/<experiment>_mean_confusion_matrix.png
logs/<experiment>/run_<n>/
```

Per run worden onder andere opgeslagen:

- seed;
- accuracy;
- macro-recall;
- recall per klasse N/S/V/F/Q;
- trainingstijd;
- relevante traininginstellingen.

Bij meerdere runs wordt ook een CSV met gemiddelde en standaarddeviatie gemaakt.

## Projectstructuur

```text
.
├── data/                  # ECG parquetbestanden
├── notebooks/             # eigen eindopdracht-notebooks en resultaten
├── src/
│   ├── config.py          # gedeelde paden en modelinstellingen
│   ├── datasets.py        # 1D/2D datasetrepresentatie
│   ├── data_utils.py      # split, seed en class weights
│   ├── helpers.py         # trainen, testen, resultaten opslaan
│   ├── networks.py        # CNN1D, CNN2D en ParallelCNN2D
│   └── tuning.py          # Ray Tune + ASHA
├── main.py                # commandline-aansturing
├── pyproject.toml
└── README.md
```

## Codekwaliteit

Ruff, mypy en pyright staan als ontwikkeltools in het project. Voer de
kwaliteitscontrole uit met:

```bash
uv run ruff check src main.py
uv run mypy src main.py
```

