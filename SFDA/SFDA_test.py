from data import *
from net import *
from lib import  *
from torch import optim
from APM_update import *
import torch.backends.cudnn as cudnn
import time
from sklearn.metrics import precision_recall_fscore_support as score

cudnn.benchmark = True
cudnn.deterministic = True

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

save_model_path = 'pretrained_weights/'+str(args.data.dataset.source)+str(args.data.dataset.target)+'/'+'domain'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'model1.pth.tar'
save_model_statedict = torch.load(save_model_path)['state_dict']

model_dict = {
    'resnet50': ResNet50Fc
}

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    confusion_vector = torch.from_numpy(confusion_vector)

    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


class F1ScoreCounter:
    """
    in supervised learning, we often want to count the test accuracy.
    but the dataset size maybe is not dividable by batch size, causing a remainder fraction which is annoying.
    also, sometimes we want to keep trace with accuracy in each mini-batch(like in train mode)
    this class is a simple class for counting accuracy.

    usage::

        counter = AccuracyCounter()
        iterate over test set:
            counter.addOneBatch(predict, label) -> return accuracy in this mini-batch
        counter.reportAccuracy() -> return accuracy over whole test set
    """
    def __init__(self):
        self.true_positives = 0.0
        self.false_positives = 0.0
        self.true_negatives = 0.0
        self.false_negatives = 0.0
        self.fscore = np.ndarray((len(source_classes),))


    def addOneBatch(self, predict, label):
        assert predict.shape == label.shape
        self.fscore+= score(np.argmax(predict, 1), np.argmax(label, 1))[2]
        true_positives, false_positives, true_negatives, false_negatives = confusion(np.argmax(predict, 1), np.argmax(label, 1))
        '''correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)'''
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.true_negatives += true_negatives
        self.false_negatives += false_negatives
        self.F1score = true_positives/(0.5*(false_positives+false_negatives)+ true_positives)

        return self.F1score, self.fscore

    
    def reportF1score(self):
        """
        :return: **return nan when 0 / 0**
        """
        return np.asarray(self.F1score, dtype=float), np.asarray(self.fscore, dtype=float) 

# ======= network architecture =======
class Target_TrainableNet(nn.Module):
    def __init__(self):
        super(Target_TrainableNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.cls_multibranch = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)


# ======= target network =======
trainable_tragetNet = Target_TrainableNet()
trainable_tragetNet.load_state_dict(save_model_statedict)

feature_extractor_t =(trainable_tragetNet.feature_extractor).cuda()
classifier_s2t = (trainable_tragetNet.classifier).cuda()
classifier_t = (trainable_tragetNet.cls_multibranch).cuda()
print ("Finish model loaded...")


domains=['Source', 'Target']
print ('domain....'+domains[args.data.dataset.source]+'>>>>>>'+domains[args.data.dataset.target])

counter = AccuracyCounter()
F1Counter = F1ScoreCounter()
with TrainingModeManager([feature_extractor_t, classifier_t], train=False) as mgr, torch.no_grad():
        i = 0
        for i, img_ in enumerate(target_test_dl):
            i+=1
            img = img_['X']
            label = img_['Y']
            img = img.cuda()
            label = label.cuda()

            feature = feature_extractor_t.forward(img)
            ___, __, before_softmax, predict_prob = classifier_t.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))
            F1Counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))
        acc_test = counter.reportAccuracy()
        F1_test, F1_labels = F1Counter.reportF1score()
        F1_labels = F1_labels/i

        print('>>>>>>>Test Accuracy>>>>>>>>>>.')
        print(acc_test)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
        print('>>>>>>>>>>>F1Score>>>>>>>>>>>>>>>>.')
        print(F1_test)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
        print('>>>>>>>>>>>F1labels>>>>>>>>>>>>>>>>.')
        print(F1_labels)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

exit()