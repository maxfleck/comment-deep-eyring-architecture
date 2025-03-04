import numpy as np

import torch
import torch.nn.functional as F 
from torch.nn import Linear
import torch.nn as nn

from torch_geometric.nn import (
    EdgePooling,
    GraphConv,
    JumpingKnowledge,
    GCNConv,
    SAGEConv,
)
from torch_geometric.nn import global_mean_pool as gmep, global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gap

class EyringEdgePool_ini(torch.nn.Module):
    def __init__(self, num_features, graph_layers, graph_hidden,
                 net_layers, net_hidden,
                 aggr="mean", funnel_graph=False,
                 funnel_net=False, graph_conv="GCNConv",
                 pooling = "", readout_freq=2,
                ):
        super().__init__()

        self.readout_freq = readout_freq
        
        if graph_conv == "SAGEConv":
            GC = SAGEConv
        if graph_conv == "GCNConv":
            GC = GCNConv            
        else:
            GC = GraphConv      

        if pooling == "meanmax":
            self.poolings = [gmep,gmp]
        if pooling == "addmax":
            self.poolings = [gap,gmp]            
        elif pooling == "meanmaxadd":
            self.poolings = [gmep,gmp,gap]                
        elif pooling == "max":
            self.poolings = [gmp]  
        elif pooling == "add":
            self.poolings = [gap]              
        else:
            self.poolings = [gmep]
        p_len = len(self.poolings)
        
        self.conv1 = GC(num_features, graph_hidden, aggr=aggr)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        #self.convs.extend([
        #    GC(graph_hidden, graph_hidden, aggr=aggr)
        #    for i in range(graph_layers - 1)
        #])
        #self.pools.extend(
        #    [EdgePooling(graph_hidden) for i in range((graph_layers) // readout_freq)])
        #dummy = graph_layers//readout_freq
        #n_jump = p_len * dummy * graph_hidden
        
        ghs = [graph_hidden]*graph_layers
        ghs = np.array(ghs)
        n_jump = 0
        #print("ghs",ghs)
        if funnel_graph:
            a = np.arange(graph_layers)
            a[0] = 1
            ghs = ghs/a
        ghs = ghs.astype(int)
        #print("ghs",ghs)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(graph_layers-1):
            a = int(ghs[i])
            b = int(ghs[i+1])
            self.convs.extend( [GC( a, b, aggr=aggr)] )
            if i % 2 == 0 and i < graph_layers - 2:
                self.pools.extend([EdgePooling(b)])
                n_jump += b*p_len
        #print("n_jump",n_jump)
        self.jump = JumpingKnowledge(mode='cat')

        #print("n_jump",n_jump)
        #self.lin1 = Linear( n_jump, net_hidden)
        #self.lin2 = Linear(net_hidden, net_hidden)
        #self.lin3 = Linear(net_hidden, net_hidden)
        #self.lin4 = Linear(net_hidden, 2)

        nnhs = [n_jump]+[net_hidden]*(net_layers-1)
        nnhs = np.array(nnhs)        
        if funnel_net:
            a = np.arange(net_layers)
            a[0] = 1
            nnhs = nnhs/a          
        self.nnls = torch.nn.ModuleList()
        for i in range(net_layers-1):
            a = int(nnhs[i])
            b = int(nnhs[i+1])            
            self.nnls.extend([Linear(a,b)])
        b = int(nnhs[-1])
        #print("nnhs",nnhs)
        self.linX = Linear(b,2)
        
        return

    def pooling(self,xs,x,batch):
        for ping in self.poolings:
            xs += [ping(x, batch)]
        return xs

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for nnl in self.nnls:
            nnl.reset_parameters()   
        self.linX.reset_parameters()               
        #self.lin1.reset_parameters()
        #self.lin2.reset_parameters()
        #self.lin3.reset_parameters()
        #self.lin4.reset_parameters()
        return

    #def forward(self, data):
    def forward(self, x_in, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        #xs = [gmep(x, batch)]
        #xs = self.pooling([],x,batch)
        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            #xs += [gmep(x, batch)]
            #xs = self.pooling([],x,batch)
            if i % self.readout_freq == 0 and i < len(self.convs) - 1:
                xs = self.pooling(xs,x,batch)
                pool = self.pools[i // self.readout_freq]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        
        x = self.jump(xs)
        """
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin4(x)
        """
        for i, nnl in enumerate(self.nnls):
            x = F.relu(nnl(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.linX(x)
        
        a0,n = torch.split(x, [1, 1], 1)
        #print("a0",a0)
        #print("n",n)        
        dEv = x_in
        #_,dEv,_ = torch.split(x_in, [1, 1, 1], 1)
        #print("dEv",dEv)
        #print("log_a",log_a)
        n = 1+n
        #a0 = 1+a0
        log_eta = dEv*n - a0
        #print("log_eta",log_eta)
        return log_eta

    def __repr__(self):
        return self.__class__.__name__

class EyringEdgePool_graph_induce(torch.nn.Module):
    def __init__(self, num_features, graph_layers, graph_hidden,
                 net_layers, net_hidden,
                 aggr="mean", funnel_graph=False,
                 funnel_net=False, graph_conv="GCNConv",
                 pooling = "", readout_freq=2,
                ):
        super().__init__()

        num_features += 8
        self.readout_freq = readout_freq
        
        if graph_conv == "SAGEConv":
            GC = SAGEConv
        if graph_conv == "GCNConv":
            GC = GCNConv            
        else:
            GC = GraphConv      

        if pooling == "meanmax":
            self.poolings = [gmep,gmp]
        if pooling == "addmax":
            self.poolings = [gap,gmp]            
        elif pooling == "meanmaxadd":
            self.poolings = [gmep,gmp,gap]                
        elif pooling == "max":
            self.poolings = [gmp]  
        elif pooling == "add":
            self.poolings = [gap]              
        else:
            self.poolings = [gmep]
        p_len = len(self.poolings)
        
        self.conv1 = GC(num_features, graph_hidden, aggr=aggr)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        #self.convs.extend([
        #    GC(graph_hidden, graph_hidden, aggr=aggr)
        #    for i in range(graph_layers - 1)
        #])
        #self.pools.extend(
        #    [EdgePooling(graph_hidden) for i in range((graph_layers) // readout_freq)])
        #dummy = graph_layers//readout_freq
        #n_jump = p_len * dummy * graph_hidden
        
        ghs = [graph_hidden]*graph_layers
        ghs = np.array(ghs)
        n_jump = 0
        #print("ghs",ghs)
        if funnel_graph:
            a = np.arange(graph_layers)
            a[0] = 1
            ghs = ghs/a
        ghs = ghs.astype(int)
        #print("ghs",ghs)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(graph_layers-1):
            a = int(ghs[i])
            b = int(ghs[i+1])
            self.convs.extend( [GC( a, b, aggr=aggr)] )
            if i % 2 == 0 and i < graph_layers - 2:
                self.pools.extend([EdgePooling(b)])
                n_jump += b*p_len
        #print("n_jump",n_jump)
        self.jump = JumpingKnowledge(mode='cat')

        #print("n_jump",n_jump)
        #self.lin1 = Linear( n_jump, net_hidden)
        #self.lin2 = Linear(net_hidden, net_hidden)
        #self.lin3 = Linear(net_hidden, net_hidden)
        #self.lin4 = Linear(net_hidden, 2)

        nnhs = [n_jump]+[net_hidden]*(net_layers-1)
        nnhs = np.array(nnhs)        
        if funnel_net:
            a = np.arange(net_layers)
            a[0] = 1
            nnhs = nnhs/a          
        self.nnls = torch.nn.ModuleList()
        for i in range(net_layers-1):
            a = int(nnhs[i])
            b = int(nnhs[i+1])            
            self.nnls.extend([Linear(a,b)])
        b = int(nnhs[-1])
        #print("nnhs",nnhs)
        self.linX = Linear(b,2)
        
        return

    def pooling(self,xs,x,batch):
        for ping in self.poolings:
            xs += [ping(x, batch)]
        return xs

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for nnl in self.nnls:
            nnl.reset_parameters()   
        self.linX.reset_parameters()               
        #self.lin1.reset_parameters()
        #self.lin2.reset_parameters()
        #self.lin3.reset_parameters()
        #self.lin4.reset_parameters()
        return

    #def forward(self, data):
    def forward(self, x_in, x, edge_index, batch):
        #dEv = x_in
        #x_cat = dEv[batch]
        dEv,x_cat = torch.split(x_in, [1, 8], 1)
        #print(dEv.shape,x_in.shape)
        x_cat = x_cat[batch]
        #print(x.shape,x_cat.shape)
        x = torch.cat((x, x_cat), dim=1)
        #print(x.shape,x_cat.shape)
        
        x = F.relu(self.conv1(x, edge_index))
        #xs = [gmep(x, batch)]
        #xs = self.pooling([],x,batch)
        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            #xs += [gmep(x, batch)]
            #xs = self.pooling([],x,batch)
            if i % self.readout_freq == 0 and i < len(self.convs) - 1:
                xs = self.pooling(xs,x,batch)
                pool = self.pools[i // self.readout_freq]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        
        x = self.jump(xs)
        """
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin4(x)
        """
        for i, nnl in enumerate(self.nnls):
            x = F.relu(nnl(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.linX(x)
        
        a0,n = torch.split(x, [1, 1], 1)
        #print("a0",a0)
        #print("n",n)        
        #dEv = x_in
        #_,dEv,_ = torch.split(x_in, [1, 1, 1], 1)
        #print("dEv",dEv)
        #print("log_a",log_a)
        n = 1+n
        #a0 = 1+a0
        log_eta = dEv*n - a0
        #print("log_eta",log_eta)
        return log_eta

    def __repr__(self):
        return self.__class__.__name__


class EyringEdgePool_graph_vWind(torch.nn.Module):
    def __init__(self, num_features, graph_layers, graph_hidden,
                 net_layers, net_hidden,
                 aggr="mean", funnel_graph=False,
                 funnel_net=False, graph_conv="GCNConv",
                 pooling = "", readout_freq=2,
                ):
        super().__init__()

        num_features += 9
        self.readout_freq = readout_freq
        
        if graph_conv == "SAGEConv":
            GC = SAGEConv
        if graph_conv == "GCNConv":
            GC = GCNConv            
        else:
            GC = GraphConv      

        if pooling == "meanmax":
            self.poolings = [gmep,gmp]
        if pooling == "addmax":
            self.poolings = [gap,gmp]            
        elif pooling == "meanmaxadd":
            self.poolings = [gmep,gmp,gap]                
        elif pooling == "max":
            self.poolings = [gmp]  
        elif pooling == "add":
            self.poolings = [gap]              
        else:
            self.poolings = [gmep]
        p_len = len(self.poolings)
        
        self.conv1 = GC(num_features, graph_hidden, aggr=aggr)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        #self.convs.extend([
        #    GC(graph_hidden, graph_hidden, aggr=aggr)
        #    for i in range(graph_layers - 1)
        #])
        #self.pools.extend(
        #    [EdgePooling(graph_hidden) for i in range((graph_layers) // readout_freq)])
        #dummy = graph_layers//readout_freq
        #n_jump = p_len * dummy * graph_hidden
        
        ghs = [graph_hidden]*graph_layers
        ghs = np.array(ghs)
        n_jump = 0
        #print("ghs",ghs)
        if funnel_graph:
            a = np.arange(graph_layers)
            a[0] = 1
            ghs = ghs/a
        ghs = ghs.astype(int)
        #print("ghs",ghs)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(graph_layers-1):
            a = int(ghs[i])
            b = int(ghs[i+1])
            self.convs.extend( [GC( a, b, aggr=aggr)] )
            if i % 2 == 0 and i < graph_layers - 2:
                self.pools.extend([EdgePooling(b)])
                n_jump += b*p_len
        #print("n_jump",n_jump)
        self.jump = JumpingKnowledge(mode='cat')

        #print("n_jump",n_jump)
        #self.lin1 = Linear( n_jump, net_hidden)
        #self.lin2 = Linear(net_hidden, net_hidden)
        #self.lin3 = Linear(net_hidden, net_hidden)
        #self.lin4 = Linear(net_hidden, 2)

        nnhs = [n_jump]+[net_hidden]*(net_layers-1)
        nnhs = np.array(nnhs)        
        if funnel_net:
            a = np.arange(net_layers)
            a[0] = 1
            nnhs = nnhs/a          
        self.nnls = torch.nn.ModuleList()
        for i in range(net_layers-1):
            a = int(nnhs[i])
            b = int(nnhs[i+1])            
            self.nnls.extend([Linear(a,b)])
        b = int(nnhs[-1])
        #print("nnhs",nnhs)
        self.linX = Linear(b,2)
        
        return

    def pooling(self,xs,x,batch):
        for ping in self.poolings:
            xs += [ping(x, batch)]
        return xs

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for nnl in self.nnls:
            nnl.reset_parameters()   
        self.linX.reset_parameters()               
        #self.lin1.reset_parameters()
        #self.lin2.reset_parameters()
        #self.lin3.reset_parameters()
        #self.lin4.reset_parameters()
        return

    #def forward(self, data):
    def forward(self, x_in, x, edge_index, batch):
        #dEv = x_in
        #x_cat = dEv[batch]
        dEv,x_cat = torch.split(x_in, [1, 8], 1)
        #print(dEv.shape,x_in.shape)
        x_cat = x_cat[batch]
        #print(x.shape,x_cat.shape)
        x = torch.cat((x, x_cat), dim=1)
        #print(x.shape,x_cat.shape)
        
        x = F.relu(self.conv1(x, edge_index))
        #xs = [gmep(x, batch)]
        #xs = self.pooling([],x,batch)
        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            #xs += [gmep(x, batch)]
            #xs = self.pooling([],x,batch)
            if i % self.readout_freq == 0 and i < len(self.convs) - 1:
                xs = self.pooling(xs,x,batch)
                pool = self.pools[i // self.readout_freq]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        
        x = self.jump(xs)
        """
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin4(x)
        """
        for i, nnl in enumerate(self.nnls):
            x = F.relu(nnl(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.linX(x)
        
        a0,n = torch.split(x, [1, 1], 1)
        #print("a0",a0)
        #print("n",n)        
        #dEv = x_in
        #_,dEv,_ = torch.split(x_in, [1, 1, 1], 1)
        #print("dEv",dEv)
        #print("log_a",log_a)
        n = 1+n
        #a0 = 1+a0
        log_eta = dEv*n - a0
        #print("log_eta",log_eta)
        return log_eta

    def __repr__(self):
        return self.__class__.__name__

def get_batch_map(batch, batch_size=None):
    if not batch_size:
        batch_size = batch.shape[0]
    bb = batch.expand( batch_size, batch_size )#.clone()
    #b = bb - bb.T +1
    b = 1 + bb - bb.permute(*torch.arange(bb.ndim - 1, -1, -1))
    b[b!=1] = 0
    bm = b.type(torch.bool)
    return bb[bm], bm

def vanWesten_pool(m, sig, eps, q, batch, expanded_batch, batch_map):
    bm = batch_map
    bb = expanded_batch
    
    x_m = scatter(m,batch, reduce="sum")
    x_sig3 = scatter(m*sig**3,batch, reduce="sum")/x_m
    x_sig = x_sig3**(1/3)

    sig_i = sig.expand( -1, sig.shape[0] )    
    sig_ij = (sig_i[bm]+sig_i.T[bm])/2
    eps_i = eps.expand( -1, eps.shape[0] )
    eps_ij = eps_i[bm]*eps_i.T[bm]
    x_eps = scatter( eps_ij*sig_ij**3, bb, 0, reduce="sum")
    x_eps = torch.unsqueeze(x_eps,1)
    x_eps = x_eps/(x_m**2 *x_sig3)

    q_i = q.expand( -1, q.shape[0] )
    q_ij = q_i[bm]*q_i.T[bm]
    x_q = scatter( q_ij*sig_ij**3, bb, 0, reduce="sum")
    x_q = torch.unsqueeze(x_q,1)
    x_q = x_q/(x_m**2 *x_sig3)    

    x_out = torch.cat([x_m, x_sig, x_eps, x_q], 1)
    return x_out
