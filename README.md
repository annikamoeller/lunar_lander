This program uses python version 3.8. 

There is a requirements.txt file which contains all the packages needed to run this program. Run "pip install -r requirements.txt" to install them.

The program can be run by running 'main.py --run_type train' for training the model, or 'main.py --run_type test' for testing the model. If you want to change the (hyper)parameters, you can edit them in the main function. 

All model checkpoints are saved in the folder 'checkpoints' - be careful before training because old checkpoints will be overwritten if they have the same name. We would recommend to copy them to a separate folder if you want to save them. 

The log file of each training session and the corresponding plot are saved in the 'metrics' folder. These will also be overwritten if they are given the same name, so be careful. 