from data import *
from net import *
from lib import  *
from torch import optim
import torch
from APM_update import *
import torch.backends.cudnn as cudnn
import time
import pandas as pd
import numpy as np
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

save_model_path = 'source_pretrained_weights/'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'/'+'model1.pth.tar'
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



    def addOneBatch(self, predict, label):
        assert predict.shape == label.shape
  
        true_positives, false_positives, true_negatives, false_negatives = confusion(np.argmax(predict, 1), np.argmax(label, 1))
        '''correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)'''
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.true_negatives += true_negatives
        self.false_negatives += false_negatives
        self.F1score = true_positives/(0.5*(false_positives+false_negatives)+ true_positives)

        return self.F1score

    
    def reportF1score(self):
        """
        :return: **return nan when 0 / 0**
        """
        return np.asarray(self.F1score, dtype=float)


# ======= network architecture =======
class Source_FixedNet(nn.Module):
    def __init__(self):
        super(Source_FixedNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)

class Target_TrainableNet(nn.Module):
    def __init__(self):
        super(Target_TrainableNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.cls_multibranch = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)


# ======= pre-trained source network =======
fixed_sourceNet = Source_FixedNet()
fixed_sourceNet.load_state_dict(save_model_statedict)
fixed_feature_extractor_s =(fixed_sourceNet.feature_extractor).cuda()
fixed_classifier_s = (fixed_sourceNet.classifier).cuda()
fixed_feature_extractor_s.eval()
fixed_classifier_s.eval()

# ======= trainable target network =======
trainable_tragetNet = Target_TrainableNet()
feature_extractor_t =(trainable_tragetNet.feature_extractor).cuda()
feature_extractor_t.load_state_dict(fixed_sourceNet.feature_extractor.state_dict())
classifier_s2t = (trainable_tragetNet.classifier).cuda()
classifier_s2t.load_state_dict(fixed_sourceNet.classifier.state_dict())
classifier_t = (trainable_tragetNet.cls_multibranch).cuda()
classifier_t.load_state_dict(fixed_sourceNet.classifier.state_dict())


model_dict = {
            'global_step':0,
            'state_dict': trainable_tragetNet.state_dict(),
            'accuracy': 0}


feature_extractor_t.train()
classifier_s2t.train()
classifier_t.train()
print ("Finish model loaded...")

domains=['Source','Target']
print ('domain....'+domains[args.data.dataset.source]+'>>>>>>'+domains[args.data.dataset.target])

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=(args.train.min_step))

optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor_t.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_s2t = OptimWithSheduler(
    optim.SGD(classifier_s2t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_t= OptimWithSheduler(
    optim.SGD(classifier_t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0
epoch_id = 0
class_num =  args.data.dataset.n_total
pt_memory_update_frequncy =  args.train.update_freq

while global_step < args.train.min_step:

    epoch_id += 1

    for i, img in enumerate(target_train_dl) :
        
        img_target = img['X']
        label_target = img['Y']
        # APM init/update
        if (global_step) % pt_memory_update_frequncy == 0:
            prototype_memory, num_prototype_,prototype_memory_dict = APM_init_update(feature_extractor_t, classifier_t)


        img_target = img_target.cuda()

        # forward pass:  source-pretrained network
        fixed_fc1_s = fixed_feature_extractor_s.forward(img_target)
        _, _, _, logit_s = fixed_classifier_s.forward(fixed_fc1_s)
        pseudo_label_s = torch.argmax(logit_s, dim=1)

        # forward pass:  target network
        fc1_t = feature_extractor_t.forward(img_target)
        _, _, logit_s2t, _ = classifier_s2t.forward(fc1_t)
        _, _, logit_t, _  = classifier_t(fc1_t)

        # compute pseudo labels
        proto_feat_tensor = torch.Tensor(prototype_memory) # (B * 2048)
        feature_embed_tensor = fc1_t.cpu()
        proto_feat_tensor = tensor_l2normalization(proto_feat_tensor)
        batch_feat_tensor = tensor_l2normalization(feature_embed_tensor)

        sim_mat = torch.mm(batch_feat_tensor, proto_feat_tensor.permute(1,0))
        sim_mat = F.avg_pool1d(sim_mat.unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0)# (B, #class)
        pseudo_label_t = torch.argmax(sim_mat, dim=1).cuda()

        # confidence-based filtering
        arg_idxs = torch.argsort(sim_mat, dim=1, descending=True) # (B, #class)

        first_group_idx = arg_idxs[:, 0]
        second_group_idx = arg_idxs[:, 1]

        first_group_feat = [prototype_memory_dict[int(x.data.numpy())] for x in first_group_idx]
        first_group_feat_tensor = torch.tensor(np.concatenate(first_group_feat, axis=0)) # (B*P, 2048)
        first_group_feat_tensor = tensor_l2normalization(first_group_feat_tensor)

        second_group_feat = [prototype_memory_dict[int(x.data.numpy())] for x in second_group_idx]
        second_group_feat_tensor = torch.tensor(np.concatenate(second_group_feat, axis=0)) # (B*P, 2048)
        second_group_feat_tensor = tensor_l2normalization(second_group_feat_tensor)

        feature_embed_tensor_repeat = torch.Tensor(np.repeat(feature_embed_tensor.cpu().data.numpy(), repeats=num_prototype_, axis=0))
        feature_embed_tensor_repeat = tensor_l2normalization(feature_embed_tensor_repeat)

        first_dist_mat = 1 - torch.mm(first_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0)) # distance = 1  - simialirty
        second_dist_mat = 1 - torch.mm(second_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0))

        first_dist_mat = F.max_pool2d(first_dist_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)
        second_dist_mat = -1*F.max_pool2d(-1* second_dist_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)

        first_dist_vec = torch.diag(first_dist_mat) #(B)
        second_dist_vec = torch.diag(second_dist_mat) # B

        confidence_mask = ((first_dist_vec- second_dist_vec) < 0).cuda()

        # optimize target network using two types of pseudo labels
        ce_from_s2t = nn.CrossEntropyLoss()(logit_s2t, pseudo_label_s)
        ce_from_t = nn.CrossEntropyLoss(reduction='none')(logit_t, pseudo_label_t).view(-1, 1).squeeze(1)
        ce_from_t = torch.mean(ce_from_t * confidence_mask, dim=0, keepdim=True)

        alpha = np.float(2.0 / (1.0 + np.exp(-10 * global_step / float(args.train.min_step//2))) - 1.0)
        ce_total = (1 - alpha) * ce_from_s2t + alpha * ce_from_t

        with OptimizerManager([optimizer_finetune, optimizer_classifier_s2t, optimizer_classifier_t]):
            loss = ce_total
            loss.backward()

        global_step += 1

        # evaluation during training
        if global_step % args.test.test_interval == 0:

            counter = AccuracyCounter()
            F1Counter = F1ScoreCounter()
            with TrainingModeManager([feature_extractor_t, classifier_t], train=False) as mgr, torch.no_grad():
              
                for i, img_ in enumerate(target_test_dl):
                  
                    img = img_['X']
                    label = img_['Y']
                    img = img.cuda()
                    label = label.cuda()

                    feature = feature_extractor_t.forward(img)
                    _, _, _, predict_prob_t = classifier_t.forward(feature)

                    counter.addOneBatch(variable_to_numpy(predict_prob_t), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))
                    F1Counter.addOneBatch(variable_to_numpy(predict_prob_t), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

            acc_test = counter.reportAccuracy()
            F1_test = F1Counter.reportF1score()
          
            print('>>>>>>>>>>>accuracy>>>>>>>>>>>>>>>>.')
            print(acc_test)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

            print('>>>>>>>>>>>F1Score>>>>>>>>>>>>>>>>.')
            print(F1_test)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
            
            if best_acc < acc_test:
                best_acc = acc_test
                model_dict = {
                        'global_step': global_step + 1,
                        'state_dict': trainable_tragetNet.state_dict(),
                        'accuracy': acc_test}

                torch.save(model_dict, join('pretrained_weights/'+str(args.data.dataset.source) + str(args.data.dataset.target) +'/' + 'domain'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'model1.pth.tar'))



counter = AccuracyCounter()
with TrainingModeManager([feature_extractor_t, classifier_t], train=False) as mgr, torch.no_grad():

        for i, img_ in enumerate(target_test_dl):
            img = img_['X']
            label = img_['Y']
            img = img.cuda()
            label = label.cuda()

            feature = feature_extractor_t.forward(img)
            _, _, _, predict_prob_t = classifier_t.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob_t), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

        acc_test = counter.reportAccuracy()
        print('>>>>>>>Final accuracy>>>>>>>>>>.')
        print(acc_test)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

        if best_acc < acc_test:
            best_acc = acc_test
            model_dict = {
                'global_step': global_step + 1,
                'state_dict': trainable_tragetNet.state_dict(),
                'accuracy': acc_test}

            torch.save(model_dict, join('pretrained_weights/'+str(args.data.dataset.source) + str(
                args.data.dataset.target) +'/' + 'domain' + str(args.data.dataset.source) + str(
                args.data.dataset.target) + 'model1.pth.tar'))


exit()