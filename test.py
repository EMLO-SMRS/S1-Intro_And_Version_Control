import pytest
import yaml
import csv

from os import path

def test_data_exists():
    assert path.exists('data') is not True, "Data Folder should not be uploaded"

def test_data_exists():
    assert path.exists('data.zip') is not True, "Data Zip should not be uploaded"

def test_data_exists():
    assert path.exists('model.pt') is not True, "Model should not be uploaded"

def test_data_in_yaml_file():
    file = open('dvc.yaml')
    yml_op =  yaml.load(file, Loader=yaml.FullLoader)
    assert 'data' in yml_op['stages']['train']['deps'], "data should be present in dvc pipeline dependencies"

def test_model_in_yaml_file():
    file = open('dvc.yaml')
    yml_op =  yaml.load(file, Loader=yaml.FullLoader)
    assert 'model.pt' in yml_op['stages']['train']['outs'], "model should be present in dvc pipeline outputs"

def test_data_in_lock_file():
    file = open('dvc.lock')
    yml_op =  yaml.load(file, Loader=yaml.FullLoader)
    dep_paths = [ i['path']  for i in yml_op['stages']['train']['deps']]
    assert 'data' in dep_paths, "data should be present in dvc lock file dependencies"

def test_model_in_lock_file():
    file = open('dvc.lock')
    yml_op =  yaml.load(file, Loader=yaml.FullLoader)
    outs_paths = [ i['path']  for i in yml_op['stages']['train']['outs']] 
    assert 'model.pt' in outs_paths , "model should be present in dvc lock file output"

def test_overal_acc():
    file = open('metrics.csv')
    rows = list(csv.DictReader(file))
    for row in rows[5: 10]:
        assert float(row['test_accuracy']) > 70, "Test Accuracy of final 5 epochs should be greater than 70"

def test_class_wise_acc():
    file = open('metrics.csv')
    rows = list(csv.DictReader(file))
    for row in rows[5: 10]:
        assert float(row['cats']) > 70, "Test Accuracy(Cats) of final 5 epochs should be greater than 70"
        assert float(row['dogs']) > 70, "Test Accuracy(Dogs) of final 5 epochs should be greater than 70"


