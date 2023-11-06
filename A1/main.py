from A1.random_forest import random_forest
from A1.ada_boost import adaboost
from A1.decision_tree import tree
from A1.k_fold_cross_validation import k_cross_fold
import os


def main():
    working_dir = os.getcwd()
    os.chdir(working_dir+'/ada_boost')
    adaboost.main()
    os.chdir(working_dir+'/random_forest')
    random_forest.main()
    os.chdir(working_dir+'/decision_tree')
    tree.main()
    os.chdir(working_dir+'/k_fold_cross_validation')
    k_cross_fold.main()


if __name__ == "__main__":
    main()