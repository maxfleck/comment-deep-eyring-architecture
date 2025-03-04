import numpy as np

import copy
import shutil
import os
import tempfile
import json
import matplotlib.pyplot as plt
import sklearn.metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ray.train import Checkpoint
from ray import train

from gnn_model import EyringEdgePool_graph_induce
from gnn_model import EyringEdgePool_ini
from gnn_dataset import dataset, minmax_from_json

def select_node_feats(batch_x, node_features):
    #print("0",batch_x.shape)
    if node_features == 1:
        batch_x = batch_x.float()[:,7]
        batch_x = torch.unsqueeze(batch_x, 1)
    else:
        batch_x = batch_x.float()  
    #print("1",batch_x.shape)
    return batch_x


def freeze_children(model, freeze):
    for name, child in model.named_children():
        if name in freeze:
            for param in child.parameters():
                param.requires_grad = False
    return


def unfreeze_children(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
    return


def check_freeze_children(model):
    for name, child in model.named_children():
        for param in child.parameters():
            if param.requires_grad:
                print(name, ": not freezed, grad = ", param.requires_grad)
            else:
                print(name, ": freezed, grad = ", param.requires_grad)
    return


def load_model0(config):
    # print(config)
    num_features = config["num_features"]
    graph_layers = config["graph_layers"]
    graph_hidden = config["graph_hidden"]
    
    net_hidden = config["net_hidden"]
    net_layers = config["net_layers"]

    graph_conv = config["graph_conv"]
    aggr = config["aggr"]
    funnel_net = config["funnel_net"]
    funnel_graph = config["funnel_graph"]
    pooling = config["pooling"]
    
    if  config["build"] == "EyringEdgePool_ini":
        model = EyringEdgePool_ini(num_features, graph_layers, graph_hidden,
                         net_layers, net_hidden, graph_conv=graph_conv,
                         aggr=aggr, funnel_net=funnel_net,funnel_graph=funnel_graph,
                         pooling=pooling)           
    #elif  config["build"] == "EyringEdgePool_induce":
    #    model = EyringEdgePool_ini(num_features, graph_layers, graph_hidden,
    #                     net_layers, net_hidden, graph_conv=graph_conv,
    #                     aggr=aggr, funnel_net=funnel_net,funnel_graph=funnel_graph,
    #                     pooling=pooling)           
    elif  config["build"] == "EyringEdgePool_graph_induce":
        model = EyringEdgePool_graph_induce(num_features, graph_layers, graph_hidden,
                         net_layers, net_hidden, graph_conv=graph_conv,
                         aggr=aggr, funnel_net=funnel_net,funnel_graph=funnel_graph,
                         pooling=pooling)           
  
    

    if "checkpoint_path" in config.keys() and config["checkpoint_path"]:
        try:
            model_state, _ = torch.load(config["checkpoint_path"],
                                        weights_only=True)
        except ValueError:
            model_state = torch.load(config["checkpoint_path"],
                                     weights_only=True)

        model.load_state_dict(model_state)

    return model


def train_loop(config_train, config, epochs=1000, freeze=[],
               n_restart=1, patience=40, report_result=True,
               ini_patience=200, node_features=9,
               ):   

    x_features = ["dE_nRT",'MD_density', 'MD_FV', 'MD_Rg', 'MD_SP_E', 'MD_SP_V', 'MD_SP', 'MD_HV', 'MD_RMSD']
    scale_x_features_with=["log_value",'MD_density', 'MD_FV', 'MD_Rg', 'MD_SP_E', 'MD_SP_V', 'MD_SP', 'MD_HV', 'MD_RMSD']
    
    data_train = dataset( config["data_path_train"], config["data_csv_train"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=False )
    data_val = dataset( config["data_path_val"], config["data_csv_val"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=False )

    dataloader = DataLoader(data_train, config_train["batch_size"],
                            shuffle=True, pin_memory=False)
    # Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_loss = 1e6
    best_l1 = 1e6
    best_val_loss = 1e6
    best_val_l1 = 1e6
    i_re_best = 0

    model = load_model0({**config_train, **config,})
    # ini weights
    best_model_weights = copy.deepcopy(model.state_dict())

    # re_losses = []
    # re_l1s = []
    # re_val_losses = []
    # re_val_l1s = []
    n_patience = patience

    for i_re in np.arange(n_restart):

        model.reset_parameters()

        # device = "cpu"
        # if torch.cuda.is_available():
        #     device = "cuda:0"
        #     if torch.cuda.device_count() > 1:
        #         model = nn.DataParallel(model)
        # model.to(device)

        loss_fn = nn.MSELoss()
        mead_fn = nn.L1Loss()
        # optimizer = optim.SGD(model.parameters(), lr=0.001,
        # momentum=0.9, weight_decay=1e-7)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config_train["lr"], weight_decay=1e-7)

        # losses = []
        # l1s = []
        # val_losses = []
        # val_l1s = []

        for epoch in range(epochs):  # loop over the dataset multiple times
            # print(epoch)
            model.train()
            for X_train_scaled, y_train_scaled, batch_index in dataloader:
    
                batch = data_train.build_batch(batch_index)
                batch_x = select_node_feats(batch.x, node_features)   
                y_pred = model(X_train_scaled, batch_x, batch.edge_index.to(torch.int64), batch.batch) 

                y_train_scaled = torch.unsqueeze(y_train_scaled[:,0],1)
                loss = loss_fn(y_pred, y_train_scaled)
    
                # Zero the gradients before running the backward pass.
                model.zero_grad()
    
                # Backward pass: compute gradient of the loss with respect
                # to all the learnable parameters of the model.
                # Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will
                # compute gradients for all learnable parameters in the model.
                loss.backward()
    
                # Update the weights using gradient descent.
                # Each parameter is a Tensor, so we can access
                # its gradients like we did before.
                # with torch.no_grad():
                #    for param in model.parameters():
                #        param -= learning_rate * param.grad
                optimizer.step()
                #break
                #print("y_pred.shape, y_train_scaled.shape")
                #print(y_pred.shape, y_train_scaled.shape)
    
            model.eval()
            batch = data_train.build_batch(data_train.graph_indexes)
            batch_x = select_node_feats(batch.x, node_features)      
            y_pred = model(data_train.X_scaled, batch_x, batch.edge_index.to(torch.int64), batch.batch) 
            y_train_scaled = torch.unsqueeze(data_train.Y_scaled[:,0],1)
            loss = loss_fn(y_pred, y_train_scaled)
            l1 = mead_fn(y_pred, y_train_scaled)
    
            batch = data_val.build_batch(data_val.graph_indexes)
            batch_x = select_node_feats(batch.x, node_features)            
            y_pred_val = model(data_val.X_scaled, batch_x, batch.edge_index.to(torch.int64), batch.batch) 
            y_val_scaled = torch.unsqueeze(data_val.Y_scaled[:,0],1)
            val_loss = loss_fn(y_pred_val, y_val_scaled)
            val_l1 = mead_fn(y_pred_val, y_val_scaled)

            # losses.append(loss)
            # l1s.append(l1)
            # val_losses.append(val_loss)
            # val_l1s.append(val_l1)

            # Early stopping
            if val_loss < best_val_loss:
                best_loss = loss
                best_l1 = l1
                best_val_loss = val_loss
                best_val_l1 = val_l1
                best_model_weights = copy.deepcopy(model.state_dict())
                n_patience = patience  # Reset patience counter
                i_re_best = i_re
                epoch_best = epoch

            elif epoch >= ini_patience:
                n_patience -= 1
                if n_patience == 0:
                    print("early termination")
                    break
            # print("")
            # gc.collect()

        # re_losses.append(losses)
        # re_l1s.append(l1s)
        # re_val_losses.append(val_losses)
        # re_val_l1s.append(val_l1s)
        print("fin RE")

    model.load_state_dict(best_model_weights)
    # if report_result:
    checkpoint_name = "checkpoint.pt"
    train_report = {"val_loss": float(best_val_loss),
                    "val_l1": float(best_val_l1),
                    "train_loss": float(best_loss), "train_l1": float(best_l1),
                    "epoch": epoch, "i_re_best": i_re_best, "epoch_best": epoch_best}
    if report_result:
        with tempfile.TemporaryDirectory() as tcd:
            path = os.path.join(tcd, checkpoint_name)
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(tcd)

            train.report(
                train_report,
                checkpoint=checkpoint,
            )
        print("Finished Training", epoch)
        return model
    else:
        print("Finished Training", epoch)
        return model, train_report


def train_looper(config_train, config):

    config["checkpoint_path"] = ""

    # train all
    freeze = []
    train_loop(config_train, config, epochs=60,
               patience=10, n_restart=1, freeze=freeze,
               report_result=True, ini_patience=10,
               node_features=config_train["num_features"],
              )
    return


def best_model(config, exp_error=True):

    config["checkpoint_path_native"] = config["checkpoint_path"]
    cpath = os.path.join(config["model_path"], "checkpoint.pt")
    config["checkpoint_path"] = cpath
    try:
        shutil.copyfile(config["checkpoint_path_native"], cpath)
    except shutil.SameFileError:
        print("checkpoint copy failed (probably already the same file.)")

    x_features = ["dE_nRT",'MD_density', 'MD_FV', 'MD_Rg', 'MD_SP_E', 'MD_SP_V', 'MD_SP', 'MD_HV', 'MD_RMSD']
    scale_x_features_with=["log_value",'MD_density', 'MD_FV', 'MD_Rg', 'MD_SP_E', 'MD_SP_V', 'MD_SP', 'MD_HV', 'MD_RMSD']    
    
    data_train = dataset( config["data_path_train"], config["data_csv_train"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=True )
    data_val = dataset( config["data_path_val"], config["data_csv_val"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=True )
    data_test = dataset( config["data_path_test"], config["data_csv_val"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=True )

    model = load_model0(config)
    model.eval()

    losses = {"training": 0, "validation": 0, "test": 0}
    l1s = {"training": 0, "validation": 0, "test": 0}
    loss_fn = nn.MSELoss()
    mead_fn = nn.L1Loss()

    a = ["training", "validation", "test"]
    b = [data_train, data_val, data_test]
    for dd, data in zip(a, b):
        for ii in range(data.n_species):
            X, Y, batch_index = data.get_species(ii)
            if len(Y.shape) > 1:
                # Y = scalerY.inverse_transform(Y)
                dummy = data.get_species_keep(ii)
                temperature = np.array(dummy["temperature"])
                log_a = np.array(dummy["log_a"])
                log_vis = np.array(dummy["log_viscosity"])
                a00 = np.array(dummy["a00"])
                Y = torch.unsqueeze(Y[:,0],1)

                # y_pred = scalerY.inverse_transform(model(X))
                batch = data.build_batch(batch_index)
                batch_x = select_node_feats(batch.x, config["num_features"])   
                y_pred = model(X, batch_x, batch.edge_index.to(torch.int64), batch.batch)
                losses[dd] += loss_fn(y_pred, Y).detach().numpy()
                l1s[dd] += mead_fn(y_pred, Y).detach().numpy()
                # resd_entropy *= m

                y_pred = data.scalerY.inverse_transform(y_pred.detach().numpy())
                y_pred = np.squeeze(y_pred)
                y_pred = y_pred + log_a - a00 

                yy = np.squeeze(log_vis)
                y_pred = np.squeeze(y_pred)
                # print( resd_entropy.shape, yy.shape, y_pred.shape )
                if ii == 0:
                    plt.plot(temperature, yy, "kx", label="experimental")
                else:
                    plt.plot(temperature, yy, "kx")
                if exp_error:
                    error = np.abs((np.exp(y_pred) - np.exp(yy)) / np.exp(yy))
                    error = np.mean(error)
                else:
                    error = np.mean(np.abs((y_pred - yy) / yy))
                # print(error)
                info = data.get_species_keep(ii)
                name = info["iupac_name"].iloc[0]

                #print( y_pred.shape, log_a.shape, a00.shape, temperature.shape  )
                #asda                 
                plt.plot(temperature, y_pred, ".",
                         label=str(name)+" "+str(round(error*100, 2))+"%"
                         )
        plt.title(dd)
        plt.legend(bbox_to_anchor=(1, 1.1))
        plt.savefig(config["model_path"]+dd+".png", bbox_inches="tight")
        plt.savefig(config["model_path"]+dd+".pdf", bbox_inches="tight")
        plt.show()
        plt.close()

    config = {**config, **{"loss": losses, "l1": l1s}}
    fname = os.path.join(config["model_path"], "best_result_config.json")
    with open(fname, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=False, default=str)

    return model, data_train

def plot_result(config):

    msize=12
    mwidth = 2
    fsize=16
    alpha=0.7
    lsize = 2    
    framewidth = 3
    
    #config["checkpoint_path_native"] = config["checkpoint_path"]
    #cpath = os.path.join(config["model_path"], "checkpoint.pt")
    #config["checkpoint_path"] = cpath

    x_features = ["dE_nRT",'MD_density', 'MD_FV', 'MD_Rg', 'MD_SP_E', 'MD_SP_V', 'MD_SP', 'MD_HV', 'MD_RMSD']
    scale_x_features_with=["log_value",'MD_density', 'MD_FV', 'MD_Rg', 'MD_SP_E', 'MD_SP_V', 'MD_SP', 'MD_HV', 'MD_RMSD']
    
    data_train = dataset( config["data_path_train"], config["data_csv_train"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=True )
    data_val = dataset( config["data_path_val"], config["data_csv_val"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=True )
    data_test = dataset( config["data_path_test"], config["data_csv_val"], log_transform=False,
         x_features=x_features,
         y_features=["log_value"],
         keep_features=["log_a","temperature","log_viscosity","a00","identifier","iupac_name","family","graph_id"],
         scale_x_features_with=scale_x_features_with,
         keepXY=True )

    model = load_model0(config)
    model.eval()

    losses = {"training": 0, "validation": 0, "test": 0}
    l1s = {"training": 0, "validation": 0, "test": 0}
    loss_fn = nn.MSELoss()
    mead_fn = nn.L1Loss()

    a = ["training", "validation", "test"]
    b = [data_train, data_val, data_test]
    
    fig = plt.figure(figsize=(6, 6))
    for dd, data in zip(a, b):
        #for ii in range(data.n_species):
        X = data.X_scaled
        Y = data.Y_scaled
        print(dd)
        print("Scaler Info")
        print(data.scalerX.data_min_, data.scalerX.data_max_)
        print(data.scalerY.data_min_, data.scalerY.data_max_)
        print()
        
        # Y = scalerY.inverse_transform(Y)
        temperature = np.array(data.keep["temperature"])
        log_a = np.array(data.keep["log_a"])
        log_vis = np.array(data.keep["log_viscosity"])
        a00 = np.array(data.keep["a00"])
        Y = torch.unsqueeze(Y[:,0],1)
        
        # y_pred = scalerY.inverse_transform(model(X))
        batch = data.build_batch(data.graph_indexes)
        batch_x = select_node_feats(batch.x, config["num_features"])   
        y_pred = model(X, batch_x, batch.edge_index.to(torch.int64), batch.batch)
        losses[dd] += loss_fn(y_pred, Y).detach().numpy()
        l1s[dd] += mead_fn(y_pred, Y).detach().numpy()
        # resd_entropy *= m

        y_pred = data.scalerY.inverse_transform(y_pred.detach().numpy())
        y_pred = np.squeeze(y_pred)
        y_pred = y_pred + log_a - a00         
        
        #Y = data.scalerY.inverse_transform(Y)
        #y_pred = data.scalerY.inverse_transform(y_pred)
        bbb = [-1.5,4]
        bbb = [-1.0, 2.416474079]
        bbb = [-9.210340371976182, -3.6298561136644754]
        bbb = [5.5804842583117065, -3.6298561136644754]
        bbb = [ 9.210340371976182 ,2.416474079/5.5804842583117065 ]
        log_vis = ( log_vis + bbb[0] )*bbb[1] -1
        y_pred = ( y_pred + bbb[0] )*bbb[1] -1
        #bbb += [ bbb[1]+bbb[0] ]
        
        yy = np.squeeze(log_vis)
        y_pred = np.squeeze(y_pred)
        # print( resd_entropy.shape, yy.shape, y_pred.shape )

        rmsd = np.sqrt( np.mean( ( (y_pred - yy)  )**2 ) )
        rmsd = str(round(rmsd,3))
        r2 = sklearn.metrics.r2_score(yy,y_pred)
        r2 = str(round(r2,3))
        label = dd+"\n RMSD:"+rmsd+" R2:"+r2
        
        plt.plot(yy, y_pred, ".", alpha=0.6, label=label, markersize=msize)


    #plt.title(dd)
    bbb = [-1.5,2.5]
    plt.plot( [bbb[0],bbb[1]],[bbb[0],bbb[1]],"k-",zorder=-100  )
    #plt.xlim(bbb[0],bbb[-1])
    #plt.ylim(bbb[0],bbb[-1])
    plt.xlabel(r"actual ln($\eta$)",fontsize=fsize)
    plt.ylabel(r"predicted ln($\eta$)",fontsize=fsize)
    plt.xticks(fontsize=fsize)  
    plt.yticks(fontsize=fsize)      
    plt.legend(fontsize=fsize-2,frameon=False, markerscale=3,) #loc="lower right")
    #plt.legend(frameon=False)

    
    plt.savefig(config["model_path"]+"_pub.png", bbox_inches="tight")
    plt.savefig(config["model_path"]+"_pub.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    return model, data_train