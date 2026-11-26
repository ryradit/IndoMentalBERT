# Model Outputs

Trained model weights (`indomentalbert_best.pt`, ~502 MB) are saved in this directory.

To train the checkpoint:
```bash
python3 src/train.py --data_path data.csv --epochs 3 --batch_size 16
```
This generates `outputs/indomentalbert/indomentalbert_best.pt` with ~79.84% validation accuracy.
