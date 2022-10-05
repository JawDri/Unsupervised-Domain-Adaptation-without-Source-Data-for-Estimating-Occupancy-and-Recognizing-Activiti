HoMM Method

In HoMM.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.



Change the task in TrainLenet.py (class_num in main function) depending of the task that you want to test, example: class_num=3 in case of task with 3 labels.

Change the task in Lenet.py (net = slim.fully_connected(net,##number of classes##, activation_fn=None, scope='fc5')) depending of the task that you want to test, example: ##number of classes##=3 in case of task with 3 labels.

TO TEST HoMM: !python TrainLenet.py

