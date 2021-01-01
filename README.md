# AU-Expression Knowledge Constrained Representation Learning for Facial Expression Recognition

Implementation of paper:   

- [AU-Expression  Knowledge  Constrained  Representation  Learning for  Facial  Expression  Recognition](https://arxiv.org/abs/2012.14587)   
  Technical Report.   
  Tao Pu, Tianshui Chen, Yuan Xie, Hefeng Wu, and Liang Lin.

![Pipeline](./Images/framework.pdf)

## Environment
Ubuntu 16.04 LTS, Python 3.5, PyTorch 1.3   

## Usage

```
# Step 1: Train the branch of facial expression recognition
python main.py --Model ResNet-101 --Experiment EM
# Step 2: Train the branch of facial AU recognition
python main.py --Model ResNet-101 --Experiment AU --Resume_Model <yourCheckpointPath>
# Step 3: Train whole model
python main.py --Model ResNet-101 --Experiment Fuse --Resume_Model <yourCheckpointPath>
```
**Note:** At step 2 and 3, you should load the checkpoint from the previous step. 

## Result

### Result on RAF-DB

| Methods | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprised | Ave. acc |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **DCNN-DA** | 78.4 | 64.4 | 62.2 | 91.1 | 80.6 | 81.2 | 84.5 | 77.5 |
| **WSLGRN** | 75.3 | 56.9 | 63.5 | 93.8 | 85.4 | 83.5 | 85.4 | 77.7 |
| **CP** | 80.0 | 61.0 | 61.0 | 93.0 | **89.0** | **86.0** | 86.0 | 79.4 |
| **CompactDLM** | 74.5 | 67.6 | 46.9 | 82.3 | 59.1 | 58.0 | 84.6 | 67.6 |
| **FSN** | 72.8 | 46.9 | 56.8 | 90.5 | 76.9 | 81.6 | 81.8 | 72.5 |
| **DLP-CNN** | 71.6 | 52.2 | 62.2 | 92.8 | 80.3 | 80.1 | 81.2 | 74.2 |
| **MRE-CNN** | **84.0** | 57.5 | 60.8 | 88.8 | 80.2 | 79.9 | 86.0 | 76.7 |
| **Ours** | 80.5 | **67.6** | **68.9** | **94.1** | 85.8 | 83.6 | **86.4** | **81.0** |


### Result on SFEW2.0

| Methods | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprised | Ave. acc |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **CP** | 66.0 | 0.0 | 14.0 | **90.0** | 86.0 | **66.0** | 29.0 | 50.1 |
| **DLP-CNN** | - | - | - | - | - | - | - | 51.1 |
| **IA-CNN** | 70.7 | 0.0 | 8.9 | 70.4 | 60.3 | 58.8 | 28.9 | 42.6 |
| **IL** | 61.0 | 0.0 | 6.4 | 89.0 | 66.2 | 48.0 | 33.3 | 43.4 |
| **Ours** | **75.3** | **17.4** | **25.5** | 86.3 | **72.1** | 50.7 | **42.1** | **52.8** |


## Citation

```
@misc{pu2020auexpression,
      title={AU-Expression Knowledge Constrained Representation Learning for Facial Expression Recognition}, 
      author={Tao Pu and Tianshui Chen and Yuan Xie and Hefeng Wu and Liang Lin},
      year={2020},
      eprint={2012.14587},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* putao537@gmail.com
* tianshuichen@gmail.com
* phoenixsysu@gmail.com