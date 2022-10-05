SFDA Method

In SFDA_main.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.



Change the task in office-train-config.yaml file (n_total and n_share) depending of the task that you want to test, example: n_share = 3 and n_total = 3 in case of task with 3 labels.

TO TRAIN SFDA: !python SFDA_train.py --config office-train-config.yaml

TO TEST SFDA: !python SFDA_test.py --config office-train-config.yaml

