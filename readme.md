# MLOps with GitHub Actions

This repository demonstrates how to implement workflows for continuous integration and continuous deployment of a simple machine learning models using GitHub Actions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
MLOps (Machine Learning Operations) is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. This project showcases how to leverage GitHub Actions to automate the MLOps pipeline.

## Features
- Automated training and validation of ML models
- Continuous integration of the serving layer and deployment workflows
- Model versioning and tracking
- Integration with cloud services for model deployment

## Getting Started
To get started with this project, clone the repository and set up the necessary dependencies.

```bash
git clone https://github.com/yourusername/MLOps-githubactions.git
cd MLOps-githubactions
```

### Prerequisites
- git
- a text editor

### Installation
No installation is required, everything is done on github actions


## Usage
1. Push your changes to the repository.
2. Trigger the training workflows tagging with `v*`
3. Trigger the serving layer workflow tagging with `s*`
4. Update the [values.yaml](./deploy/values.yaml) with the versions to deploy (model and serving layer)
5. Lets ArgoCD sync the changes
3. Monitor the deployment.

## Testing
To test the model you can use a simple `curl`:

```bash
curl -X POST "http://localhost:8080/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Volare.com is amazing!"}'
```

## Test environment

To deploy the model on a local Kubernetes cluster you can use [kind](https://kind.sigs.k8s.io/).

Then you can deploy [ArgoCD](https://argo-cd.readthedocs.io/en/stable/) following [this doc](https://argo-cd.readthedocs.io/en/latest/try_argo_cd_locally/)
