External Libraries Used
numpy
scikit-learn
matplotlib

Usage instructions:
Running main.py will run a python script for each of the algorithms (decision tree, random forests, and adaboost)as well
as the k-fold cross-validation code. The scripts will produce graphs for each algorithm using the same specifications
noted in the report. The k-fold cross-validation algorithm implemented is the k_fold method inside of
k_fold_cross_validation/k_cross_fold.py. The graphs will be saved as .svg files inside of the directories associated
with each algorithm. Additionally, a .txt file will appear containing lowest error achieved by each algorithm for each
varied hyperparameter.

If you wish to run main.py multiple times you'll need to manually delete each generated file (the graphs and the .txt
stats files) beforehand or the subsequent runs will append data to those files instead of overwriting them. Sorry about
that but it ended up being a hassle and caused me some grief so I didn't automate it.

Happy marking!