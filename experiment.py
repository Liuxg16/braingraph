import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
import torch


class Experiment():
    """
    This class handles all experiments related activties, 
    including training, testing, early stop, and visualize
    results, such as get attentions and get rules. 

    Args:
        sess: a TensorFlow session 
        saver: a TensorFlow saver
        option: an Option object that contains hyper parameters
        learner: an inductive learner that can  
                 update its parameters and perform inference.
        data: a Data object that can be used to obtain 
              num_batch_train/valid/test,
              next_train/valid/test,
              and a parser for get rules.
    """
    
    def __init__(self, option,device, learner=None, data=None):
        self.device = device
        self.option = option
        self.learner = learner
        self.data = data
        # helpers
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600., 
                        (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

        param_optimizer = list(self.learner.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        # t_total = -1 # self.data.num_batch_train
        # self.optimizer = BertAdam(optimizer_grouped_parameters,
        #                  lr= self.option.learning_rate,
        #                  warmup= 0.01,
        #                  t_total=t_total)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), self.option.learning_rate)

    def one_epoch(self, mode):
        epoch_loss = []
        epoch_in_top = []
		            

        if mode == "train":
            edge1 = self.data['train_edge1'] # query(relation), head(target), tails
            edge2 = self.data['train_edge2'] # query(relation), head(target), tails
            labels = self.data['train_labels'] # query(relation), head(target), tails
            
            edge1 = torch.tensor(edge1, dtype=torch.float).to(self.device)
            edge2 = torch.tensor(edge2, dtype=torch.float).to(self.device)
            labels = torch.tensor(labels, dtype=torch.float).to(self.device)
            loss, in_top = self.learner(edge1,edge2,labels)
            self.optimizer.zero_grad()
            loss.backward()
            # for name, p in self.learner.named_parameters():
            #     print name,p.data
            self.optimizer.step()
        else:
            edge1 = self.data['valid_edge1'] # query(relation), head(target), tails
            edge2 = self.data['valid_edge2'] # query(relation), head(target), tails
            labels = self.data['valid_labels'] # query(relation), head(target), tails
            
            edge1 = torch.tensor(edge1, dtype=torch.float).to(self.device)
            edge2 = torch.tensor(edge2, dtype=torch.float).to(self.device)
            labels = torch.tensor(labels, dtype=torch.float).to(self.device)

            loss, in_top = self.learner(edge1,edge2,labels)
        epoch_loss += [loss.item()]
        epoch_in_top += [in_top.item()]

        msg = self.msg_with_time(
                "Epoch %d mode %s Loss %0.4f In top %0.4f." 
                % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_in_top)))
        print(msg)
        self.log_file.write(msg + "\n")
        return epoch_loss, epoch_in_top

    def one_epoch_train(self):
        self.learner.train()
        if self.epoch > 0 and self.option.resplit:
            self.data.train_resplit(self.option.no_link_percent)
        loss, in_top = self.one_epoch("train")
        with open(self.learner.name+'2.pkl', 'wb') as f:
            torch.save(self.learner.state_dict(), f)

        self.train_stats.append([loss, in_top])
        
    def one_epoch_valid(self):
        self.learner.eval()
        loss, in_top = self.one_epoch("valid")
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = min(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        self.learner.eval()
        loss, in_top = self.one_epoch("test")
        self.test_stats.append([loss, in_top])
    
    def early_stop(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])
        if loss_improve or in_top_improve:
            return False
        else:
            if self.epoch < self.option.min_epoch:
                return False
            else:
                return True

    def train(self):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()
            self.one_epoch_valid()
            self.one_epoch_test()
            self.epoch += 1
            # model_path = self.saver.save(self.sess, 
            #                              self.option.model_path,
            #                              global_step=self.epoch)
            # print("Model saved at %s" % model_path)
            # 

            # if self.early_stop():
            #     self.early_stopped = True
            #     print("Early stopped at epoch %d" % (self.epoch))
        
        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmin(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]
        
        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)       
        print(msg)
        self.log_file.write(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "w"))

    def close_log_file(self):
        self.log_file.close()

