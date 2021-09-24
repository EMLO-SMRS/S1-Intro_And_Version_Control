# Session 1 MLOPs Introduction and Version Control

## DVC
Data Version Control, or DVC, is a data and ML experiment management tool that takes advantage of the existing toolset (Git, CI/CD).

1. [Intstallation Guide](https://dvc.org/doc/install)
2. [Documentation](https://dvc.org/doc)
3. [Tutorial](https://dvc.org/doc/use-cases/versioning-data-and-model-files/tutorial)
 

## Steps To Run

1. Clone the Repo
2. Install dependencies from requirements.txt
3. To Run train.py : 

```python
    python train.py
```

4. To run with DVC: 



```python
    dvc run  -f -n train -d train.py -d data \
    -o model.pt -O Summary.md \
    -O loss.jpg -O accuracy.jpg -O class_wise.jpg -M metrics.csv \
   python train.py
```
#### Abbreviations
1. -f : force to run the stage
2. -n : stage name
3. -d : dependency
4. -o : output 
5. -O : output with no cache
6. -M : metrics with no cache

### Implementation 
1. Basic pytorch train test code for binary Image classification.
2. This is to learn the basics of dvc.
3. Model is trained twice, first with 1000 train set and 800 test set, secondly 2000 train and 800 test data for 10 Epochs

### Add On Feature
1. It also geneates the **[`Summary.md`](Summary.md)** file at the end automatically, which gives you the brief idea how/what is used to train the model.
