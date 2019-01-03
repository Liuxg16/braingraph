import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainGraph(nn.Module):
    """ 
    pytorch version
    """

    def __init__(self, option):
        super(BrainGraph, self).__init__()
        self.name = 'braingraph'
        self.seed = option.seed        
        self.num_step = option.num_step
        self.num_layer = option.num_layer
        self.hids = option.rnn_state_size
        
        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k
        self.regressor = nn.Linear(self.hids,1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
        self.to(self.device)
        # n_gpu = torch.cuda.device_count()

        print('-'*10+"Neual LP is built."+'-'*10)

    def forward(self, edge1, edge2,labels):
        '''
        300,246,246
        300,246,246
        300,1
        '''
        batch_size = labels.size(0)
        print '9999999'
        print edge1.size()

        tensort = torch.tensor(t, dtype=torch.long).to(self.device)
        tensorh = torch.tensor(h, dtype=torch.long).to(self.device)
    
        databases = []
        for r in range(len(mdb)):
            ids = torch.LongTensor(mdb[r][0])
            values = torch.FloatTensor(mdb[r][1])
            map1 = torch.sparse.FloatTensor(ids.t(), values, mdb[r][2]).to(self.device) # (3007,3007)
            databases.append(map1)
            # print ids.size()
            # print values.size()
            # print map1.size()

        rnn_inputs = self.qembedding(tensorq.t()) # (bs,3,emb)
        h0 = torch.zeros(1, batch_size, self.rnn_state_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.rnn_state_size).to(self.device)
        rnn_outputs,(ht,ct) = self.qrnn(rnn_inputs, (h0,c0)) # (bs,3,hids)

        # a_t, (bs, 3, 24)
        a_t = self.att_at(rnn_outputs.contiguous().view(-1, self.rnn_state_size)).view(batch_size, self.num_step,
                self.num_operator)
        a_t = nn.Softmax(2)(a_t)

        # u_t, actually the tails at first
        u_ts = onehot(tensort, self.num_entity).unsqueeze(1) # (bs,1,3007)

        for t in range(self.num_step):
            # b, (bs,t,1)
            b = torch.bmm(rnn_outputs[:,:t+1,:] ,rnn_outputs[:,t:t+1,:].permute(0,2,1))
            b = nn.Softmax(1)(b) #

            # u_{t+1} = \sum b_t u_t, (bs,3007)
            u_tplus1 = torch.sum(b * u_ts,1)
            
            # construct  memories to store u_t
            if t < self.num_step - 1:
                # database_results: (will be) a list of num_operator tensors,
                # each of size (batch_size, num_entity).
                database_results = []    
                for r in xrange(self.num_operator/2): # 12
                    for op_matrix, op_attn in zip(
                                    [databases[r], 
                                     databases[r].transpose(1,0)], # 
                                    [a_t[:,t,r],  a_t[:,t,r+self.num_operator/2]]): # (bs,)
                        # M_R_k sum b_t u_t
                        product = torch.mm(u_tplus1,op_matrix.to_dense())  # (bs,3007)
                        # a_k M_R_k sum b_t _u_t
                        sumat = product*op_attn.unsqueeze(1)
                        database_results.append(sumat.unsqueeze(1))
                # u_t 
                added_database_results = torch.sum(torch.cat(database_results,1),1, keepdim = True) # 128,1,3007

                if self.norm:
                    added_database_results /= torch.max(torch.sum(added_database_results,2,
                        keepdim=True), torch.cuda.FloatTensor([self.thr]))
                
                if self.dropout > 0.:
                    added_database_results = nn.Dropout(self.dropout)(added_database_results)

                # Populate a new cell in memory by concatenating.  
                u_ts = torch.cat( 
                    [u_ts, added_database_results],1) #(bs,t+1,3007)
            else:
                predictions = u_tplus1 #(bs,3007)
        loss = - torch.mean(onehot(tensorh, self.num_entity) * torch.log(torch.max(predictions, \
                torch.cuda.FloatTensor([self.thr]))))
        # loss = self.criterion(predictions, tensorh)

        values,index_topk = torch.topk(predictions,self.top_k, sorted=False)      
        # print values
        acc_res = torch.eq(index_topk, tensorh.unsqueeze(1))
        in_top = torch.sum(acc_res,1)
        return loss, in_top, predictions, index_topk
 
    def get_predictions_given_queries(self, qq, hh, tt, mdb):
        loss, in_top, predictions, top_k = self.forward(qq, hh, tt, mdb)
        return in_top.detach().cpu().numpy(), predictions.detach().cpu().numpy()

class CNN(nn.Module):
    """ 
    pytorch version
    """
    def __init__(self, option):
        super(CNN, self).__init__()

        self.hids = option.rnn_state_size
        self.name = 'cnn'
        self.conv1 = nn.Conv2d(2, 8, 9)
        self.conv2 = nn.Conv2d(8, 16, 9)
        self.conv3 = nn.Conv2d(16, 16, 9)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 10)
        self.fc3   = nn.Linear(10, 1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
        self.to(self.device)
        # n_gpu = torch.cuda.device_count()

        print('-'*10+"Neual LP is built."+'-'*10)

    def forward(self, edge1, edge2,labels):
        '''
        300,246,246
        300,246,246
        300,1
        '''
        batch_size = labels.size(0)
        x = torch.cat([edge1.unsqueeze(1), edge2.unsqueeze(1)],1) # bs,2,246,246

        out = F.relu(self.conv1(x)) # bs,6,238,238
        out = F.max_pool2d(out, 3) # bs,6,119,119
        out = F.relu(self.conv2(out)) # bs,16,111
        out = F.max_pool2d(out, 3) # bs,8, 55
        out = F.relu(self.conv3(out)) # bs,16,47,47
        out = F.max_pool2d(out, 3) # bs,8, 23,23


        out = out.view(out.size(0), -1) # bs,440
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out)).view(-1)

        loss = (out-labels)*(out-labels)
        loss = torch.mean(loss)

       
        return loss, loss
 
    def get_predictions_given_queries(self, qq, hh, tt, mdb):
        loss, in_top, predictions, top_k = self.forward(qq, hh, tt, mdb)
        return in_top.detach().cpu().numpy(), predictions.detach().cpu().numpy()


class StackedGAT(nn.Module): # inheriting from nn.Module!
    
    def __init__(self, option):
        super(StackedGAT, self).__init__()
        self.name = 'gat'
        self.num_layers = 1
        nhid = 24
        nfeat = 246*2
        dropout = 0.1
        alpha = 0.2
        nheads = 4
        self.dropout = dropout
        self.thr = 1e-20
        self.att_layers = []
        self.fc_layers = []
        for j in range(self.num_layers):
            self.attentions = [GraphAttentionLayer(nhid, nhid/nheads, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_layer_{}_{}'.format(j,i), attention) # important, add module to torch
            self.att_layers.append(self.attentions)

        self.layer_norm = nn.LayerNorm(nhid)
        self.feature2rep = nn.Linear(nfeat,nhid)
        self.clf = nn.Linear(nhid,1)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
        self.to(self.device)

        print('The basic gat is built!')
        
    def forward(self, edge1, edge2,labels):
        '''
        A, bs,nnode, nnode
        X, bs,1,nnode,fea
        P, bs,
        '''
        batch_size = labels.size(0)
        x = torch.cat([edge1, edge2],2).unsqueeze(1) # bs,1,246,246*2

        P = labels.view(-1,1)
        adj = torch.eye(246).to(self.device)
        N_node = x.size(2)
        # x: bs,1,node, fea; adj: bs,n,n
        x_input = F.elu(self.feature2rep(x))
        for attention_layer in self.att_layers:

            x = F.dropout(x_input, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in attention_layer], dim=3) # bs, 1,node,nhids*8
            x = F.dropout(x, self.dropout, training=self.training)
            x_input = self.layer_norm( x + x_input)
        #bs,1,node,nhids
        # poolx = nn.MaxPool2d((N_node,1))(x).squeeze()
        poolx = torch.mean(x_input,2).squeeze()
        Ppred = F.sigmoid(self.clf(poolx))

        loss = torch.mean((Ppred-P)*(Ppred-P))
        return  loss,loss




class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
		# input: (bs,1,nnode,fea)
		# adj: (bs,nnode, nnode)
        # out: bs,1,node,fea
        h = torch.matmul(input, self.W) # bs,1,nnode, outfea
        N = h.size()[2]
        batchsize = h.size()[0]
        # (N*N,fea),(N*N,fea)'-> (N,N,2*fea): (N,N,ifea+jfea)
        a_input = torch.cat([h.repeat(1,1, N,1).view(batchsize,N * N, -1), h.repeat(1,N,
            1,1).view(batchsize,N*N,-1)], dim=2)
        a_input = a_input.view(batchsize,N, N, 2 * self.out_features)
        attention_score = torch.matmul(a_input, self.a) #bs,N*N,1
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # bs,N,N
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention.unsqueeze(1), h) # bs,n,n  bs,1,n,fea
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'







