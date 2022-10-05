SHOT Method
In SHOT_IM_V1.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced).
for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.

In case of AR, set the input features to 32 and  self.in_features = 768 in the network.py file DTNBase class.
In case of OE, set the input features to 9 and self.in_features = 384 in the network.py file DTNBase class.

Change the task in uda_digit.py (args.class_num) depending of the task that you want to test, example: args.class_num = 3 in case of task with 3 labels.

TO TEST SHOT-IM:
!python uda_digit.py --dset s2m --gpu_id 0 --cls_par 0.0 --output ckps_digits 

TO TEST FULL SHOT:
!python uda_digit.py --dset s2m --gpu_id 0 --cls_par 0.1 --output ckps_digits 
