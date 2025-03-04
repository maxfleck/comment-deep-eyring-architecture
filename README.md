# comment-deep-eyring-architecture

### Code for Comment on Paper: 
## Advancing material property prediction: using physics-informed machine learning models for viscosity

The Code enables reproducing the best performing model from the comment.
It also allows you to build and train your own models. A more detailed documentation and support is available on request. 

## Technical NOTES:

- SAGE Pooling leads to best model performance on most train/test/val splits
- GraphConv Pooling leads to lower variance between train/test/val splits i.e. might be easier to train
- when trained on external features, eyring parameters may vary for different temperature
