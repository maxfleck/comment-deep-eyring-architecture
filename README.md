# comment_deep_eyring
Code for Comment on Paper: Advancing material property prediction: using physics-informed machine learning models for viscosity

## Technical NOTES:

- SAGE Pooling leads to best model performance on most train/test/val splits
- GraphConv Pooling leads to lower variance between train/test/val splits i.e. might be easier to train
- when trained on external features, eyring parameters may vary for different temperature
