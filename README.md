# BuildFL

BuildFL (Building Federated) is a HVAC Analytic Platform through Federated Learning. With BuildFL, we can easily test a large number of different models on different dataset.

BuildFL is published as a note paper in The Eleventh ACM International Conference on Future Energy Systems (ACM e-Energy '20): Towards Federated Learning for HVAC Analytics: A Measurement Study.

BuildFL provide an abstraction API of model training, intermediate model update, global model distribute in federated learning and support machine learning models used in HVAC analytics.

## Document 

The document of BuildFL is placed in `document\`.


## Quick Start

BuildFL is written in Python,  for now you can use `python run.py` to stimulate a federated learning training process. 
Another version of BuildFL which can execute on actual parameter server and participants' device will be published as a python library.


## Cite Us

You are welcome to cite our research paper:

```
@inproceedings{10.1145/3396851.3397717,
author = {Yunzhe Guo, Dan Wang, Arun Vishwanath, Cheng Xu and Qi Li},
title = {Towards Federated Learning for HVAC Analytics: A Measurement Study},
year = {2020},
isbn = {9781450380096},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3396851.3397717},
doi = {10.1145/3396851.3397717},
booktitle = {Proceedings of the Eleventh ACM International Conference on Future Energy Systems},
pages = {68–73},
numpages = {6},
keywords = {Federated Learning, Applied Machine Learning, HVAC Analytics},
location = {Virtual Event, Australia},
series = {e-Energy ’20}
}
```
