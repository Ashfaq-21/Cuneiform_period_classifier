# Overview: what the system is for

## Purpose

Dating a cuneiform tablet usually takes a specialist reading its script, content and
physical form. This prototype tests one narrow question:

> Can a standard image classifier tell **Ur III** tablets (ca. 2100-2000 BC) from
> **Old Babylonian** tablets (ca. 1900-1600 BC) using only the **3D surface geometry** of one face?

It is a research prototype and demo, not a dating tool. It shows a reproducible pipeline from public 3D scans to a
measured result and an inspectable prediction.

## What it does

1. **Gets real data.** It reads the public [HeiCuBeDa Hilprecht](https://doi.org/10.11588/data/IE8CCN)
   dataset (1,977 tablet scans from the Hilprecht Collection, Jena) and its metadata, where the period label
   comes from the CDLI catalogue. It downloads only the scans it needs, one tablet at a time. Your own labelled `.ply` scans also work.
2. **Turns each 3D scan into a depth map.** It orients the tablet automatically and renders one face as a
   256×256 height image in millimetres. The cuneiform wedges show up as small dents.
3. **Trains a classifier.** It fine-tunes an ImageNet-pretrained ResNet18 on those depth maps. Train, validation
   and test splits are made **per tablet**, so the model is always tested on tablets it has never seen.
4. **Measures honestly.** It reports accuracy, macro F1 and a confusion matrix on the held-out test split
   against a majority-class baseline, the score you would get by always guessing the larger class.
5. **Explains individual predictions.** A local web app lets you pick a held-out tablet (or upload a `.ply`), click
   **Analyze**, and see:
   * the depth map,
   * the predicted period,
   * the model's confidence,
   * a **Grad-CAM** overlay marking the regions that influenced the prediction.

## Who it is for

* **Developers** extending it to more periods, other surface representations (normals, curvature) or other models.
  See [ARCHITECTURE.md](ARCHITECTURE.md#extension-points).
* **Researchers** who want a reproducible baseline on HeiCuBeDa for period classification.

## What it does not claim

* **Not a general dating tool.** It covers two periods and one museum collection, mostly from Nippur.
  It may pick up collection- or site-specific cues.
* **Labels are catalogue metadata** (CDLI `period`), not re-verified by the authors.
* **Confidence is not calibrated.** It is the network's softmax probability.
* **Grad-CAM is not a scholarly explanation.** It shows *where* the network's evidence came from, not *why*
  the tablet belongs to a period.
* **Face choice is heuristic.** Geometry alone can't tell the obverse from the reverse. The system uses the face with more fine relief.

## Data licence and citation

HeiCuBeDa is CC BY-SA 4.0. Cite: Mara, H. & Bogacz, B. (2019), *HeiCuBeDa Hilprecht – Heidelberg Cuneiform
Benchmark Dataset for the Hilprecht Collection*, heiDATA, doi:10.11588/data/IE8CCN.
