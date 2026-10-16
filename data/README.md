# Data

This project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle:
https://www.kaggle.com/c/ieee-fraud-detection

The data files are **not committed** to this repository (they are large, and the competition rules restrict redistribution).

## On Kaggle (recommended)
1. Open the competition page and click **"Join Competition"** to accept the rules (needed once).
2. Create a new notebook and choose **Add Input → Competitions → IEEE-CIS Fraud Detection**.
3. The files appear under `/kaggle/input/ieee-fraud-detection/`.

## Locally
Download the files and place them here:
```
data/
├── train_transaction.csv   # main file used in this project (has the isFraud label)
└── train_identity.csv      # optional, device/browser info
```

Only the **train** files have fraud labels, so they are the only ones used. The test set is built by splitting the train data by time.
