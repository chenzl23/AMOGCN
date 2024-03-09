# Codes for AMOGCN

## Introduction
- This is an implement of AMOGCN with PyTorch, which was run on a machine with AMD R9-5900HX CPU, RTX 3080 16G GPU and 32G RAM. The corresponding paper has been accepted by Neural Networks.
- *Chen, Zhaoliang and Wu, Zhihao and Zhong, Luying and Plant, Claudia and Wang, Shiping and Guo, Wenzhong. Attributed Multi-order Graph Convolutional Network for Heterogeneous Graphs. Neural Networks. in press, 2024*

## Requirements

- torch: 2.0.1+cu117
- torch-cluster: 1.6.1+pt20cu117
- torch-geometric: 2.3.1
- torch-scatter: 2.1.1+pt20cu117
- torch-sparse: 0.6.17+pt20cu117
- torch-spline-conv: 1.2.2+pt20cu117
- torch-geometric: 2.3.1
- numpy: 1.20.1
- texttable: 1.6.4
- scikit-learn: 1.1.2
- scipy: 1.6.2

## Demos
- Here are some commands for quick running of datasets used in this paper.

  - For AMOGCN on the DBLP dataset: 
  ```
  python ./main.py --dataset-name DBLP4057 --k 30 --gamma 0.05 --train-ratio 0.6
  ``` 

  - For AMOGCN on the YELP dataset: 
  ```
  python ./main.py --dataset-name yelp --gamma 0.02 --k 50 --train-ratio 0.6    
  ``` 

    - For AMOGCN on the IMDB dataset: 
  ```
  python ./main.py --dataset-name imdb5k --k 100 --gamma 0.1 --train-ratio 0.6
  ``` 

   - For AMOGCN on the ACM dataset: 
  ```
  python ./main.py --dataset-name ACM3025 --k 40 --gamma 0.8 --train-ratio 0.6
  ``` 
