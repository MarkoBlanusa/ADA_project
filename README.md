
For dataset and checkpoints
https://drive.google.com/drive/folders/17MIKdY7reX_TAYMpGl4GhkvzxCW3687o?usp=drive_link

For train and training with the states being in numpy, use "main.py" and "trade_env_numpy.py". For training  with ray, use "main_ray.py" and "trade_env_ray.py"

Then use "main_test_results.py" and "test_trade_env_numpy.py" for both training. The testing is always done in numpy. 
In the "main_test_results.py", adapt the directory of your folder for checkpoints. 

If you want to create or store new data in the dataset, use the "data_collector.py" module by instantiating
