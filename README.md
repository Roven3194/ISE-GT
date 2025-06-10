# ISE-GT: Interaction strength-enhanced graph Transformer for explainable multi-agent trajectory prediction

![](./fig/ISEGT.png)

## Abstract ##

Accurate trajectory prediction is crucial for autonomous vehicles to understand surrounding vehicles and make appropriate planning decisions. However, capturing interactions in complex scenarios remains a key challenge, limiting the performance of trajectory prediction algorithms. To address this, we divides the driver trajectory prediction process into two stages: identifying interaction tendencies and generating predicted trajectories. Subsequently, we integrate interaction tendencies and other physical information into an encoding framework and propose an interaction-strength encoding Graph Transformer trajectory prediction network (ISE-GT). To evaluate whether the interaction tendencies understood by ISE-GT align with those recognized by human drivers, we develop an interaction tendency recognition method (ITRM). ITRM quantifies interaction tendencies by calculating the interaction strength between the trajectories of surrounding vehicles and the ego vehicle. Experimental results demonstrate that ISE-GT outperforms state-of-the-art baselines in trajectory prediction performance and aligns with human drivers in its understanding of interaction tendencies. Additionally, ITRM provides explanations for the predictions of ISE-GT that are consistent with human intuition, endowing the proposed method with passive interpretability.

## Highlights

* **Propose interaction strength encoding for social information in interactions.**
* **Develop a graph Transformer for social, physical, and rule information integration.**
* **Construct a mechanism model to identify interaction tendencies from predictions.**
* **Align algorithm-identified interaction tendencies with human expectations.**
* **Explain trajectory predictions in a human-intuitive way by interaction tendencies.**

## Getting Started
**Step 1**: clone this repository:
```
git clone https://github.com/Roven3194/ISE-GT.git && cd ISE-GT
```

**Step 2**: create a conda environment and install the dependencies:
Requires:
* Python ≥ 3.6
* PyTorch ≥ 1.6
* pytorch_lightning
* pytorch-geometric

**Step 3**: install the **Argoverse API** and download the  **Argoverse Motion Forecasting Dataset v1.1.**
## Training & Validation

### Training
To train ISE-GT:

```
python train.py --root /path/to/dataset_root/
```
**Note:** When running the training script for the first time, it will take several hours to preprocess the data (depend on your platform). Training on an RTX 4090D GPU takes 20~25 minutes per epoch.

During training, the checkpoints will be saved in `lightning_logs/` automatically. To monitor the training process:

### Validation

To evaluate on the validation set:
```
python eval.py --root /path/to/dataset_root/
```