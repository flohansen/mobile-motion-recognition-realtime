# Master-Thesis

## Tools
`tools/convert_ucf101.py`: Konvertiert bestimmte Klassen des UFC-101-Datensatzes zu Schlüsselpunktanimationen.

## Datensätze
`datasets/motions2021_10`: Schlüsselpunktanimationen mit 10 Frames als Bilder kodiert.
`datasets/motions2021_20`: Schlüsselpunktanimationen mit 20 Frames als Bilder kodiert.
`datasets/motions2021_60`: Schlüsselpunktanimationen mit 60 Frames als Bilder kodiert.

## Trainieren von KpGAN
`python train_kpgan.py --help`

Beispiel für 3000 Epochen und einer Batch-Size von 128:
`python train_kpgan.py --epochs 3000 --save-interval 100 --batch-size 128`

## Trainieren von ViGAN
`python train_vigan.py --help`

Beispiel für 3000 Epochen und einer Batch-Size von 4:
`python train_vigan.py --epochs 3000 --save-interval 100 --batch-size 4`