Drug-drug Interactions Extraction
====
This is the code for paper "[Extracting Drug-drug Interactions with a Dependency-based Graph Convolution Neural Network](https://ieeexplore.ieee.org/document/8983150)".

The preprocessed dataset is  available at [here](https://drive.google.com/drive/folders/15px_dODJjww8l1OaIYkzbdOgbXR1lZdu?usp=sharing). In order to run the code, you need to put it in the dataset directory. The original dataset can be obtained from [here](https://www.cs.york.ac.uk/semeval-2013/task9/index.php%3Fid=data.html).


Requirements:
-------  

python 3.6

pytorch 1.1.0

sklearn

numpy

 Train:
-------  
 
```
 bash run.sh
```

Evaluation:
-------  
 
```
 python3 eval.py
```
