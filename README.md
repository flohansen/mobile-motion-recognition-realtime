# Master-Thesis: Bewegungserkennung auf mobilen Geräten mit Verwendung von GANs für eine automatische Datensatzgenerierung

## Inhaltsverzeichnis
- [Master-Thesis](#master-thesis)
  * [Quickstart](#quickstart)
  * [Datensätze](#datens-tze)
  * [Abhängigkeiten installieren](#abh-ngigkeiten-installieren)
  * [Trainieren von KpGAN](#trainieren-von-kpgan)
    + [Schritt 1: Herunterladen der Datensätze](#schritt-1--herunterladen-der-datens-tze)
    + [Schritt 2: Trainingsschleife starten](#schritt-2--trainingsschleife-starten)
  * [Trainieren von ViGAN](#trainieren-von-vigan)
    + [Schritt 1: Herunterladen der Datensätze](#schritt-1--herunterladen-der-datens-tze-1)
    + [Schritt 2: Trainingsschleife starten](#schritt-2--trainingsschleife-starten-1)
  * [Trainieren von MotionNet](#trainieren-von-motionnet)
    + [Schritt 1: Datensatz festlegen](#schritt-1--datensatz-festlegen)
    + [Schritt 2: Trainingsschleife starten](#schritt-2--trainingsschleife-starten-2)
  * [Tools](#tools)
    + [Erstellen eines Datensatzes für KpGAN](#erstellen-eines-datensatzes-f-r-kpgan)

## Quickstart

**Umwandeln eines trainierten MotionNet-Modells in das TFLite-Format:**

    tflite_convert --saved_model_dir=checkpoints/motionnet --output_file=checkpoints/motionnet_20.tflite

---

**Trainieren von MotionNet, welches 20 Frames einer Bewegung klassifiziert:**

    python .\train_motionnet.py 20 .\datasets\motions2021_20 --epochs 300 --save-interval 50 --batch-size 128 --use-dataset-generator

---

**Trainieren von KpGAN, welches Bewegungen bestehend aus 20 Schlüsselpunktframes erzeugt:**

    python train_kpgan.py 20 --epochs 500 --save-interval 100 --batch-size 128

## Datensätze
Es wurden verschiedene Datensätze zum Trainieren der verschiedenen GANs
erstellt. Diese sind in folgender Tabelle dokumentiert. Der Pfad zu den
Datensatzordnern kann beliebig gewählt werden. Dieser kann in den Python-Skripts
übergeben werden. Die Datensätze selbst müssen aber **zwingend** wie folgt
benannt werden. 

| Pfad | Beschreibung | Modell |
| ---- | ------------ | ------ |
| `motions2021_10` | Schlüsselpunktanimationen mit 10 Frames als Bilder kodiert. | KpGAN |
| `motions2021_20` | Schlüsselpunktanimationen mit 20 Frames als Bilder kodiert. | KpGAN |
| `motions2021_60` | Schlüsselpunktanimationen mit 60 Frames als Bilder kodiert. | KpGAN |
| `videos`         | Videos von Hantelübungen mit unterschiedlichen Personen. | ViGAN |

## Abhängigkeiten installieren
Damit die Python-Programme ausgeführt werden können, müssen die entsprechenden Abhängigkeiten installiert werden. Dies kann mithilfe der `requirements.txt` im Hauptverzeichnis des Projektes getan werden.

    pip install -r requirements.txt

## Trainieren von KpGAN
### Schritt 1: Herunterladen der Datensätze
In der Arbeit wurden unterschiedliche Datensätze verwendet, um KpGAN Bewegungssequenzen generieren zu lassen, die jeweils aus 10, 20 oder 60 Frames bestehen. Falls die Datensätze nicht bereits vorhanden sind, können diese aus dem [Google Drive](https://drive.google.com/drive/folders/1eP3lIetg4uYdd5ICSJtjb5AHvnZgCGpg?usp=sharing) heruntergeladen werden. Die Datensätze werden am besten unter `/path/to/project/datasets` entpackt.

```
/path/to/project
├───datasets
|   ├───motions2021_10
|   ├───motions2021_20
|   └───motions2021_60
├───modules
├───tools
...
```

### Schritt 2: Trainingsschleife starten
Nachdem die Datensätze lokal zur Verfügung stehen, kann das KpGAN trainiert werden. Hierfür wird das Skript `train_kpgan.py` verwendet. Einzelheiten über die Parameter können mithilfe des Kommandos `python train_kpgan.py --help` nachgelesen werden. Diese sind im Folgenden trotzdem nochmal kurz beschrieben.

| Parameter | Default | Beschreibung |
| --------- | ------- | ------------ |
| `frames` | - | Kann `10`, `20` oder `60` sein. Gibt die Anzahl der Frames an, die eine Bewegung darstellen. |
| `--batch-size` | 32 | Größe der zum Training verwendeten Batches |
| `--n-critic` | 5 | Anzahl der Iterationen zum Trainieren des Kritisierers |
| `--epochs` | 100 | Anzahl der zum Training verwendeten Epochen |
| `--save-interval` | 10 | Gibt an, nach wie vielen Epochen das Modell gespeichert werden soll |
| `--checkpoint-dir` | `checkpoints` | Der Ordner, in welchem das trainierte Modell gespeichert werden soll |
| `--log-dir` | `logs` | Der Ordner, in welchem der Trainingsprozess dokumentiert abgelegt werden soll (kann über [Tensorboard](https://www.tensorflow.org/tensorboard) ausgelesen und dargestellt werden) |

Beispiel für ein Training des KpGAN, welches 20 Frames einer Bewegung erzeugt, mit 3000 Epochen und einer Batch-Größe von 128:

    python train_kpgan.py 20 --epochs 500 --save-interval 100 --batch-size 128

## Trainieren von ViGAN
### Schritt 1: Herunterladen der Datensätze
ViGAN wurde mithilfe eines Videodatensatzes trainiert. Falls die Datensätze nicht bereits vorhanden sind, können diese aus dem [Google Drive](https://drive.google.com/drive/folders/1eP3lIetg4uYdd5ICSJtjb5AHvnZgCGpg?usp=sharing) heruntergeladen werden. Die Datensätze werden am besten unter `/path/to/project/datasets` entpackt.

```
/path/to/project
├───datasets
|   └───videos
├───modules
├───tools
...
```

### Schritt 2: Trainingsschleife starten
Nachdem die Datensätze lokal zur Verfügung stehen, kann das KpGAN trainiert werden. Hierfür wird das Skript `train_vigan.py` verwendet. Einzelheiten über die Parameter können mithilfe des Kommandos `python train_vigan.py --help` nachgelesen werden. Diese sind im Folgenden trotzdem nochmal kurz beschrieben.


| Parameter | Default | Beschreibung |
| --------- | ------- | ------------ |
| `--batch-size` | 4 | Größe der zum Training verwendeten Batches |
| `--n-critic` | 5 | Anzahl der Iterationen zum Trainieren des Kritisierers |
| `--epochs` | 100 | Anzahl der zum Training verwendeten Epochen |
| `--save-interval` | 100 | Gibt an, nach wie vielen Epochen das Modell gespeichert werden soll |
| `--checkpoint-dir` | `checkpoints` | Der Ordner, in welchem das trainierte Modell gespeichert werden soll |
| `--dataset-dir` | `datasets/videos` | Der Ordner des Datensatzes für das Training |
| `--log-dir` | `logs` | Der Ordner, in welchem der Trainingsprozess dokumentiert abgelegt werden soll (kann über [Tensorboard](https://www.tensorflow.org/tensorboard) ausgelesen und dargestellt werden) |
| `--checkpoint` | None | Gibt den Speicherpunkt einer vergangenen Trainingssession an, die weiter trainiert werden soll |

Beispiel für 3000 Epochen und einer Batch-Größe von 4:

    python train_vigan.py --epochs 500 --save-interval 100 --batch-size 4

## Trainieren von MotionNet
### Schritt 1: Datensatz festlegen
Beim Training des MotionNets kann zwischen zwei Arten von Datensätzen gewählt werden. Zum einen kann ein realer Datensatz, wie beim Training von KpGAN verwendet werden. Zum anderen kann ein KpGAN selbst als Datensatzgenerator verwendet werden, um dynamisch auf Nachfrage neue Trainingsdaten zu erzeugen.

### Schritt 2: Trainingsschleife starten
Das Training mit realen Daten verhält sich ähnlich wie beim Trainieren eines KpGANs (siehe oben). Ein Beispiel für das Trainieren des MotionNets mit 20 Frames ist folgendes Kommando.

    python train_motionnet.py 20 --epochs 300 --save-interval 50 --batch-size 128

Um MotionNet mithilfe eines KpGANs zu trainieren, kann folgendes Kommando angepasst werden.

    python .\train_motionnet.py 20 --epochs 300 --save-interval 50 --batch-size 128 --use-dataset-generator

Der reale Datensatz wird in diesem Fall als Testdatensatz verwendet.

## Tools

### Erstellen eines Datensatzes für KpGAN
Um die Datensätze für KpGAN zu erzeugen, wurde ein Werkzeug entwickelt, welches Videos des [UCF-101-Datensatzes](https://www.crcv.ucf.edu/data/UCF101.php) in Schlüsselpunktanimationen umwandelt. Dieses Werkzeug befindet sich unter `tools/convert_ucf101.py`. Um die Datensätze aus der Thesis zu erzeugen, wurden folgende Kommandos verwendet.

    python .\tools\convert_ucf101.py --number-output-frames 60 /path/to/UCF-101 datasets/motions2021_60 PushUps PullUps BenchPress

    python .\tools\convert_ucf101.py --number-output-frames 20 /path/to/UCF-101 datasets/motions2021_20 PushUps PullUps BenchPress

    python .\tools\convert_ucf101.py --number-output-frames 10 /path/to/UCF-101 datasets/motions2021_10 PushUps PullUps BenchPress

Diese Kommandos erzeugen Schlüsselpunktanimationsdatensätze bestehend aus den Klassen `PushUps`, `PullUps` und `BenchPress` mit 60, 20 und 10 Frames pro Bewegung. Natürlich kann anstatt dieser Klassen alle anderen des UCF-101-Datensatzes verwendet werden.