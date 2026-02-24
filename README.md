# Binary classification using an XGBoost model on Sagemaker AI

This is a proyect developed on AWS Sagemaker AI in which a dataset is pre-proceesed, aplitted and uploaded to an S3 bucket. 

The data is fed to an XGBoost model extracted from the xgboost container available on AWS. 

Then, an endpoint is deployed for the model to be able to make inferences asynchornously.

Finally, the endpont is invoked to evaluate the test set and it returns an accuracy over 70%.
