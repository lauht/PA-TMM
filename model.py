import math
import torch
import torch.nn as nn
import torch.nn.functional as F

inf = math.inf

class PriceEmbedding(nn.Module):
    def __init__(self, n_feat, out_dim, rnn_layers, dropout):
        super(PriceEmbedding, self).__init__()
        self.trans = nn.LSTM(
            input_size=n_feat,
            hidden_size=out_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, X):
        x_hidden, _ = self.trans(X)
        output = x_hidden[:, -1, :]
        return output
    
class HMON(nn.Module):
    def __init__(self, n_feat, pretrained_path=None, dropout=0.2, n_out=2):
        super(HMON, self).__init__()
        self.num_node = 118
        # pseudo_message
        self.pseudo_message = torch.load('./pseudo_message.pth')

        # dim setting
        dm = 768
        dp = 128
        dr = 128
        dh = 256
        self.dn = 256
        dphi = 192
        de = 256

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.rnn = PriceEmbedding(n_feat=n_feat, out_dim=dp, rnn_layers=2, dropout=dropout)
        self.Wum = nn.Parameter(torch.Tensor(dm+1, dr))
        self.Wvm = nn.Parameter(torch.Tensor(dm+1, dr))
        self.Wup = nn.Parameter(torch.Tensor(dp+1, dr))
        self.Wvp = nn.Parameter(torch.Tensor(dp+1, dr))
        self.prompt_encoding = nn.Linear(3*dr, 2)
        self.hybrid_embedding = nn.Linear(3*dr, dh)
        self.act = nn.Linear(dh+2, self.dn, bias=True)
        self.inact = nn.Linear(dh, self.dn, bias=True)
        self.Wphin = nn.Parameter(torch.Tensor(2*self.dn, dphi))
        self.aphi = nn.Parameter(torch.Tensor(dphi, 1))
        self.Won = nn.Parameter(torch.Tensor(2*self.dn, self.dn))
        self.Weo = nn.Parameter(torch.Tensor(3*self.dn, de))
        self.Wi = nn.Parameter(torch.Tensor(self.num_node, self.dn+2*de, n_out))
        self.bi = nn.Parameter(torch.Tensor(self.num_node, n_out))

        self._init_params_()
        if pretrained_path is not None:
            self._load_params_(pretrained_path)

    def _load_params_(self, pretrained_path):
        pretrained_state_dict = torch.load(pretrained_path)
        asl_name = [
            'Wum',
            'Wvm',
            'Wup',
            'Wvp',
            'Wphin',
            'aphi',
            'Won',
            'Weo',
            'Wi',
            'bi',
            'rnn.trans.weight_ih_l0',
            'rnn.trans.weight_hh_l0',
            'rnn.trans.bias_ih_l0',
            'rnn.trans.bias_hh_l0',
            'rnn.trans.weight_ih_l1',
            'rnn.trans.weight_hh_l1',
            'rnn.trans.bias_ih_l1',
            'rnn.trans.bias_hh_l1',
            'prompt_encoding.weight',
            'prompt_encoding.bias',
            'hybrid_embedding.weight',
            'hybrid_embedding.bias',
            'act.weight',
            'act.bias',
            'inact.weight',
            'inact.bias',
        ]
        for layer in asl_name:
            self.state_dict()[layer].copy_(pretrained_state_dict[layer])

    def _init_params_(self):
        nn.init.xavier_uniform_(self.Wum)
        nn.init.xavier_uniform_(self.Wvm)
        nn.init.xavier_uniform_(self.Wup)
        nn.init.xavier_uniform_(self.Wvp)

        nn.init.xavier_uniform_(self.Wphin)
        nn.init.xavier_uniform_(self.aphi)

        nn.init.xavier_uniform_(self.Won)
        nn.init.xavier_uniform_(self.Weo)

        nn.init.xavier_uniform_(self.Wi)
        nn.init.xavier_uniform_(self.bi)

    def forward(self, x, x_message): 
        N = x.size(0)

        # 1. Cross-modal Prompt and Fusion
        ## representation learning
        m = x_message 
        p = self.rnn(x)
        p = self.leaky_relu(p)

        ## pseudo message
        activation_status = (x_message.sum(dim=-1) == 0).float()
        m[activation_status==1] = self.pseudo_message.clone()
        V1 = (activation_status == 0)
        V0 = (activation_status == 1)
        
        activation_status = activation_status.unsqueeze(1) 
        
        
        ## multi-modal decomposition
        u_m = self.leaky_relu(torch.cat([m, activation_status], dim=1).matmul(self.Wum))
        v_m = self.leaky_relu(torch.cat([m, activation_status], dim=1).matmul(self.Wvm)) 
        u_p = self.leaky_relu(torch.cat([p, activation_status], dim=1).matmul(self.Wup)) 
        v_p = self.leaky_relu(torch.cat([p, activation_status], dim=1).matmul(self.Wvp)) 

        ## multi-modal integration
        h_pmt = self.prompt_encoding(torch.cat([u_m, u_m * v_p, v_p], dim=1)) 
        h_pmt = self.softmax(h_pmt) 
        
        h_hyb = self.hybrid_embedding(torch.cat([u_p, u_p + v_m, v_m], dim=1))
        h_hyb = self.leaky_relu(h_hyb)
        
        # 2. Hyper-Message Modeling
        ## stock polarized activation
        n_act = self.act(torch.cat([h_hyb, h_pmt], dim=1)).mul((1-activation_status).repeat(1, self.dn))
        n_inact = self.inact(h_hyb).mul((activation_status).repeat(1, self.dn))
        n = self.leaky_relu(n_act + n_inact) 
        
        ## interactions inference
        n_paircat = torch.cat([n.unsqueeze(0).repeat(N, 1, 1), n.unsqueeze(1).repeat(1, N, 1)], dim=-1)
        phi = self.leaky_relu(n_paircat.matmul(self.Wphin)).matmul(self.aphi).squeeze()
        if V1.sum().item() != 0:
            phi[V1] = nn.Softmax(dim=0)(phi[V1])
            phi[V0] = nn.Softmax(dim=0)(phi[V0])
        else:
            phi = nn.Softmax(dim=0)(phi)
        alpha = phi

        ## information overflow
        e = self.leaky_relu(n_paircat.matmul(self.Won))
        e = torch.cat([e, n_paircat], dim=-1) 
        e = e.matmul(self.Weo) 
        alpha1 = ((1-activation_status).repeat(1, N)*alpha).unsqueeze(1) 
        alpha0 = (activation_status.repeat(1, N)*alpha).unsqueeze(1)
        hyper_m1 = self.leaky_relu(alpha1.matmul(e)).squeeze() 
        hyper_m0 = self.leaky_relu(alpha0.matmul(e)).squeeze()
        hyper_m = torch.cat([hyper_m0, hyper_m1], dim=-1) 
        
        # 3. Prediction
        output = torch.cat([n, hyper_m], dim=-1)
        y_hat = output.unsqueeze(1).matmul(self.Wi).squeeze() + self.bi.squeeze()
        y_hat = self.softmax(y_hat)

        # 4. loss
        o1 = self.Wum.mm(self.Wvm.T)
        o2 = self.Wup.mm(self.Wvp.T)
        l_ort = torch.sqrt(torch.sum(o1*o1)) + torch.sqrt(torch.sum(o2*o2))
        # l_ort = 0
        
        l_pol = 0
        if V1.sum().item() != 0:
            n1 = n[V1]
            h_pmt1 = h_pmt[V1]
            for i in range(len(n1)):
                for j in range(len(n1)):
                    l_pol += F.cosine_similarity(n1[i], n1[j], dim=0) * torch.sign((h_pmt1[i,1]-h_pmt1[i,0])*(h_pmt1[j,1]-h_pmt1[j,0]))
        
        return y_hat, h_pmt, l_ort, l_pol, V1, V0
    
class MPA(nn.Module):
    def __init__(self, n_feat, pretrained_path=None, dropout=0.2, n_out=2):
        super(MPA, self).__init__()
        self.num_node = 118

        # pseudo_message
        self.pseudo_message = torch.load('./pseudo_message.pth')

        # dim setting
        dm = 768
        dp = 128
        dr = 128
        dh = 256
        self.dn = 256
        dphi = 192
        de = 256

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        # 1. Cross-modal Prompt and Fusion
        ## price embedding
        self.rnn = PriceEmbedding(n_feat=n_feat, out_dim=dp, rnn_layers=2, dropout=dropout)

        ## multi-modal decomposition
        self.Wum = nn.Parameter(torch.Tensor(dm+1, dr))
        self.Wvm = nn.Parameter(torch.Tensor(dm+1, dr))
        self.Wup = nn.Parameter(torch.Tensor(dp, dr))
        self.Wvp = nn.Parameter(torch.Tensor(dp, dr))

        nn.init.xavier_uniform_(self.Wum)
        nn.init.xavier_uniform_(self.Wvm)
        nn.init.xavier_uniform_(self.Wup)
        nn.init.xavier_uniform_(self.Wvp)

        ## multi-modal integration
        self.prompt_encoding = nn.Linear(3*dr, 2)
        self.hybrid_embedding = nn.Linear(3*dr, dh)

        # 2. Hyper-Message Modeling
        ## stock polarized activation
        self.act = nn.Linear(dh+2, self.dn, bias=True)
        self.inact = nn.Linear(dh, self.dn, bias=True)

        ## interactions inference
        self.Wphin = nn.Parameter(torch.Tensor(2*self.dn, dphi))
        self.aphi = nn.Parameter(torch.Tensor(dphi, 1))
        nn.init.xavier_uniform_(self.Wphin)
        nn.init.xavier_uniform_(self.aphi)

        ## information overflow
        self.Won = nn.Parameter(torch.Tensor(2*self.dn, self.dn))
        self.Weo = nn.Parameter(torch.Tensor(3*self.dn, de))
        nn.init.xavier_uniform_(self.Won)
        nn.init.xavier_uniform_(self.Weo)

        # 3. Prediction
        self.Wi = nn.Parameter(torch.Tensor(self.num_node, self.dn+2*de, n_out))
        self.bi = nn.Parameter(torch.Tensor(self.num_node, n_out))
        nn.init.xavier_uniform_(self.Wi)
        nn.init.xavier_uniform_(self.bi)


    def forward(self, x, x_tag): 
        N = x.size(0)
        # 1. Cross-modal Prompt and Fusion
        ## representation learning
        m = self.pseudo_message.repeat(N, 1) 
        p = self.rnn(x)
        p = self.leaky_relu(p)

        ## activation
        activation_status = x_tag[:,-1]
        V1 = (activation_status == 0)
        V0 = (activation_status == 1)
        
        activation_status = activation_status.unsqueeze(1)
        
        
        ## multi-modal decomposition
        # u_m = self.leaky_relu(torch.cat([m, activation_status], dim=1).matmul(self.Wum))
        v_m = self.leaky_relu(torch.cat([m, activation_status], dim=1).matmul(self.Wvm))
        u_p = self.leaky_relu(p.matmul(self.Wup)) 
        # v_p = self.leaky_relu(p.matmul(self.Wvp))

        ## multi-modal integration
        h_pmt = x_tag[:,0:2]
        h_hyb = self.hybrid_embedding(torch.cat([u_p, u_p + v_m, v_m], dim=1))
        h_hyb = self.leaky_relu(h_hyb)
        
        # 2. Hyper-Message Modeling
        ## stock polarized activation
        n_act = self.act(torch.cat([h_hyb, h_pmt], dim=1)).mul((1-activation_status).repeat(1, self.dn))
        n_inact = self.inact(h_hyb).mul((activation_status).repeat(1, self.dn))
        n = self.leaky_relu(n_act + n_inact)
        
        ## interactions inference
        n_paircat = torch.cat([n.unsqueeze(0).repeat(N, 1, 1), n.unsqueeze(1).repeat(1, N, 1)], dim=-1)
        phi = self.leaky_relu(n_paircat.matmul(self.Wphin)).matmul(self.aphi).squeeze()
        if V1.sum().item() != 0:
            phi[V1] = nn.Softmax(dim=0)(phi[V1])
            phi[V0] = nn.Softmax(dim=0)(phi[V0])
        else:
            phi = nn.Softmax(dim=0)(phi)
        alpha = phi

        ## information overflow
        e = self.leaky_relu(n_paircat.matmul(self.Won))
        e = torch.cat([e, n_paircat], dim=-1)
        e = e.matmul(self.Weo)
        alpha1 = ((1-activation_status).repeat(1, N)*alpha).unsqueeze(1)
        alpha0 = (activation_status.repeat(1, N)*alpha).unsqueeze(1)
        hyper_m1 = self.leaky_relu(alpha1.matmul(e)).squeeze()
        hyper_m0 = self.leaky_relu(alpha0.matmul(e)).squeeze() 
        hyper_m = torch.cat([hyper_m0, hyper_m1], dim=-1)
        
        # 3. Prediction
        output = torch.cat([n, hyper_m], dim=-1)
        y_hat = output.unsqueeze(1).matmul(self.Wi).squeeze() + self.bi.squeeze()
        y_hat = self.softmax(y_hat)

        # 4. loss
        o1 = self.Wum.mm(self.Wvm.T)
        o2 = self.Wup.mm(self.Wvp.T)
        l_ort = torch.sqrt(torch.sum(o1*o1)) + torch.sqrt(torch.sum(o2*o2))
        
        l_pol = 0
        if V1.sum().item() != 0:
            n1 = n[V1]
            h_pmt1 = h_pmt[V1]
            for i in range(len(n1)):
                for j in range(len(n1)):
                    l_pol += F.cosine_similarity(n1[i], n1[j], dim=0) * torch.sign((h_pmt1[i,1]-h_pmt1[i,0])*(h_pmt1[j,1]-h_pmt1[j,0]))
        
        return y_hat, h_pmt, l_ort, l_pol, V1, V0

    
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.beta = 0.15
        self.gamma = 0.20

    def forward(self, x, h_pmt, l_ort, l_pol, y, V1, V0):
        if V1.sum().item() != 0:
            l_mov = F.cross_entropy(input=x[V0], target=y[V0]) + F.cross_entropy(input=h_pmt[V1], target=y[V1])
        else:
            l_mov = F.cross_entropy(input=x, target=y)
        l_tot = l_mov + self.beta*l_ort + self.gamma*l_pol
        return l_tot