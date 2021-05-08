# Master-Thesis: Bewegungserkennung auf mobilen Geräten mit Verwendung von GANs für eine automatische Datensatzgenerierung

## Literatur
* K. Liu, Y. Ye, X. Li and Y. Li: **A Real-Time Method to Estimate Speed of Object Based on Object Detection and Optical Flow Calculation** ([PDF](https://iopscience.iop.org/article/10.1088/1742-6596/1004/1/012003/pdf), [BibTeX](https://iopscience.iop.org/export?articleId=1742-6596/1004/1/012003&doi=10.1088/1742-6596/1004/1/012003&exportFormat=iopexport_bib&exportType=abs&navsubmit=Export+abstract))
* M. Mandal, L. K. Kumar, M. S. Saran and S. K. Vipparthi: **MotionRec: A Unified Deep Framework for Moving Object Recognition** ([PDF](https://openaccess.thecvf.com/content_WACV_2020/papers/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.pdf), [BibTeX](https://openaccess.thecvf.com/content_WACV_2020/html/Mandal_MotionRec_A_Unified_Deep_Framework_for_Moving_Object_Recognition_WACV_2020_paper.html))
* Y. Yang, A. Loquercio, D. Scaramuzza, S. Soatto: **Unsupervised Moving Object Detection via Contextual Information Separation** ([PDF](https://arxiv.org/pdf/1901.03360), [BibTeX](https://arxiv.org/abs/1901.03360))
* X. Mao, Q. Li, H. Xie, R. Y.K. Lau, Z. Wang, S. P. Smolley: **Least Squares Generative Adversarial Networks** ([PDF](https://arxiv.org/pdf/1611.04076.pdf), [BibTeX](https://arxiv.org/abs/1611.04076))
* M. Arjovsky, S. Chintala, L. Bottou: **Wasserstein GAN** ([PDF](https://arxiv.org/pdf/1701.07875) [BibTeX](https://arxiv.org/abs/1701.07875))
* I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, A. Courville: **Improved Training of Wasserstein GANs**, ([PDF](https://arxiv.org/pdf/1704.00028), [BibTeX](https://arxiv.org/abs/1704.00028))

## Videos
* [Creating Videos with Neural Networks using GAN](https://www.youtube.com/watch?v=CIua95jUD_I)

## Blogs
* [Meow Generator](https://ajolicoeur.wordpress.com/cats/): Vergleich zwischen GAN-Modellen zum Generieren von Katzenbildern

## Experimente und Messungen

| #   | Datensatz | Epochen | Trainingszeit | Ergebnisse |
| --- | --------- | ------- | ------------- | ---------- |
| 1 | 2021-05-01-164113 | 9870 | 1619896607.382972 | ![](./evaluation/2021-05-01-164113/results.gif) |
| 2 | 2021-05-02-135730 | 9870 | 1619972648.515409 | ![](./evaluation/2021-05-02-135730/results.gif) |
| 3 | 2021-05-03-172829 | 2000 | 1620059589.828395 | ![](./evaluation/2021-05-03-172829/results.gif) |
| 4 | 2021-05-04-023451 | 4950 | 1620098297.972187 | ![](./evaluation/2021-05-04-023451/results.gif) |
| 5 | 2021-05-04-064157 | 5000 | 1620113920.317127 | ![](./evaluation/2021-05-04-064157/results.gif) |
| 6 | 2021-05-06-024533 | 3000 | 1620274451.146458 | ![](./evaluation/2021-05-06-024533/results.gif) |
| 7 | 2021-05-07-124008 | 1449 | 1620389590.842311 | ![](./evaluation/2021-05-07-124008/results.gif) |
| 8 | 2021-05-07-135545 | 3000 | 1620418631.520207 | ![](./evaluation/2021-05-07-135545/results.gif) |


## Inhalt der Thesis

* Einführung in GANs
    - Theorie
    - Mode-Collapse
    - Deep Convolution GAN
    - Wasserstein GAN
    - Wasserstein GAN-GP
    - Unrolled GAN
    - Least Squares GAN
* Erstellen eines Datensatzes
    - Rahmenbedingungen
    - Verwendung von GANs
    - Messung unterschiedlicher GANs
    - Analyse der Ergebnisse
