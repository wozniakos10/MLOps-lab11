# MLOPS-LAB11


Screenshots taken during exercise can be found in [assets](assets/)

## Notes

### Laboratory

I was following lab instructions and initially for integration tests we've been using the previous app configuration with torch/transformers-based inference approach. I added credential configuration to that step and the download_artifacts script to test the application properly. However, later in the deployment step we are exporting model artifacts to ONNX format, building a docker image, pushing to ECR, and deploying a CloudFormation stack, so to test the app properly we would need to actually add deployment dependencies to the integration part, also exporting the model to ONNX and using it for running pytest.

I've made some workarounds and I created:
- [deprecated_app.py](src/lab11_lib/deprecated_app.py): previous app configuration for integration testing with pytest
- [app.py](src/lab11_lib/app.py): updated app configuration with ONNX and Mangum for Lambda inference

It's not a robust solution as I'm performing integration tests on a different configuration than I have actually deployed, but I am aware of that and I believe for learning purposes it's okay.

Possible approaches were:
- My approach
- Add deployment dependencies to integration tests, export the model to ONNX, and run integration tests with [app.py](src/lab11_lib/app.py)
- Run integration tests with [app.py](src/lab11_lib/app.py) but exclude pytest, or make sure the pytest part doesn't require loading models

### Homework

I didn't understand Exercise 2 from the homework, as the steps described there were also done for the laboratory, so I didn't see any reason to update the workflow as those steps were already covered.