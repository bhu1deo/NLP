"""
We have our data and we have our models: 
Training is done for 50 epochs, every 10th epoch the plots are stored 
Alongwith the MSE/MAE values in a txt file 
Both Single TimeStep and MultiTimeStep Prediction is addressed here:: 

"""

import argparse
from torch.utils.data import TensorDataset, DataLoader   # A Pytorch dataloader from the numpy arrays here :: 
import pandas as pd
import numpy as np
from torch import Tensor
import torch
from data import train_test_data
from DARNN import darnn_model
from DARNN_1 import darnn_model_scaled_dot_product_attention
from DARNN_2 import darnn_model_all_prev_decoder_hidden_states
from DARNN_3 import darnn_model_just_input
from DARNN_multistep import darnn_model_multistep
from Transformer import transformer_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import torch.nn as nn
from matplotlib import pyplot as plt

def train(data_train_loader,data_val_loader,target_train_max,target_train_min,args):        # By default train for single time step prediction here 
    # train data and all the paths are available 

    if(args.prediction_horizon!=1):                            # For Multitimestep predcition horizon we should have darnn_model_multistep model 
        if(args.model!='DARNN_MULTISTEP'):
            print('\nError, model incompatible with prediction horizon')
            return
    else:
        if(args.model=='DARNN_MULTISTEP'):
            print('\nError, model incompatible with prediction horizon')
            return

    plot_path = args.plot_path+args.model+'/'
    result_path = args.result_path+args.model+'/'

    if(args.model=='TRANSFORMER'):                      # Change in instantiation here 
        model = transformer_model.Transformer(batch_first=True)
        model = model.to(args.device)
    elif(args.model=='DARNN'):                                               # The DARNN architectures here
        model = darnn_model.DARNN(81, 64, 64, 16)
        model = model.to(args.device)
    elif(args.model=='DARNN_SDPA'):
        model = darnn_model_scaled_dot_product_attention.DARNN_scaled_dot_product(81, 64, 64, 16)
        model = model.to(args.device)
    elif(args.model=='DARNN_PDHS'):
        model = darnn_model_all_prev_decoder_hidden_states.DARNN_prev_decoder_states(81, 64, 64, 16)
        model = model.to(args.device)
    elif(args.model=='DARNN_JI'):
        model = darnn_model_just_input.DARNN_just_input(81, 64, 64, 16)
        model = model.to(args.device)
    elif(args.model=='DARNN_MULTISTEP'):          # Prediction horizon is greater than 1 here 
        model = darnn_model_multistep.DARNN_three_timestep(81, 64, 64, 16)
        model = model.to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)         # learning rate scheduler here 

    if(args.prediction_horizon==1):
        # Better to run this on the server may take time here ::
        # Also for various proposed modifications do note the timing constriants for training and all 
        start = time.time()

        epochs = 50
        loss = nn.MSELoss()
        patience = 15
        min_val_loss = 9999
        counter = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        for i in range(epochs):
            mse_train = 0
            for batch_x, batch_y_h, batch_y in data_train_loader :
        #         batch_x = batch_x.cuda()
        #         batch_y = batch_y.cuda()
        #         batch_y_h = batch_y_h.cuda()
                opt.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_y_h = batch_y_h.to(device)
                
                model = model.to(device)
                
                y_pred = model(batch_x, batch_y_h)
                y_pred = y_pred.squeeze(1)             # Because we are computing only for the last sample here 
                batch_y = batch_y.squeeze(1)
                # print(y_pred.shape)
                # print(batch_y.shape)
                l = loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item()*batch_x.shape[0]
                opt.step()
            epoch_scheduler.step()
            with torch.no_grad():                       # Don't do any training in here :: after every epoch he is updating this here :: 
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y_h, batch_y in data_val_loader:      # On the validation data here :: 
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_y_h = batch_y_h.to(device)
                    model = model.to(device)
                    output = model(batch_x, batch_y_h)         # using history and driving compute the target series here 
                    output = output.squeeze(1)
                    batch_y = batch_y.squeeze(1)
                    preds.append(output.detach().cpu().numpy())
                    true.append(batch_y.detach().cpu().numpy())
                    mse_val += loss(output, batch_y).item()*batch_x.shape[0]
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            
            if min_val_loss > mse_val**0.5:
                min_val_loss = mse_val**0.5
                print("Saving...")
                torch.save(model.state_dict(), args.model_path+args.model+".pt")                 # Change according to the name here 
                counter = 0
            else: 
                counter += 1

            if counter == patience:        # Probably done to stop overfitting here 
                break
            # print("Iter: ", i, "train: ", (mse_train/len(X_train_t))**0.5, "val: ", (mse_val/len(X_val_t))**0.5)
            # Save the MSE/MAE values to a txt file here: 



            if(i % 10 == 0):         # Every 10 epochs here :: 
                preds = preds*(target_train_max - target_train_min) + target_train_min
                true = true*(target_train_max - target_train_min) + target_train_min
                plt.figure(figsize=(20, 10))
                plt.plot(preds)
                plt.plot(true)
                plt.savefig(plot_path+args.model+'training-'+str(i)+'.png')
                plt.show()

                mse = mean_squared_error(true, preds)
                mae = mean_absolute_error(true, preds)

                if(i==0):
                    with open(result_path+args.model+'.txt', 'w') as f:
                        f.write('MSE/MAE:{},{}'.format(mse,mae))
                        f.write('\n')
                else:
                    with open(result_path+args.model+'.txt', 'a') as f:
                        f.write('MSE/MAE:{},{}'.format(mse,mae))
                        f.write('\n')
                # print("mse: ", mse, "mae: ", mae)
        f.close()
        print("\nTime Required for Training\n")        
        print(time.time()-start)

    else:
        print("\nDoing Multistep prediction here using DARNN multistep model...\n")
        # The training process is just a bit different for multistep time series here: 
        # 3 timestep decoder : Training and predictions here : the 3 targets need to be passed and 3 predictions would be gotten at 
        # While Plotting do some kind of Intelligent Averaging here : Or maybe WE can just Plot at every 3 iterations here 

        start = time.time()

        epochs = 50
        loss = nn.MSELoss()
        patience = 15
        min_val_loss = 9999
        counter = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        for i in range(epochs):
            mse_train = 0
            for batch_x, batch_y_h, batch_y in data_train_loader :
                opt.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_y_h = batch_y_h.to(device)
                
                model = model.to(device)
                
                y_pred = model(batch_x, batch_y_h,batch_y,train=True)
                y_pred = torch.cat((y_pred[0],y_pred[1],y_pred[2]),-1)
                l = loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item()*batch_x.shape[0]
                opt.step()
            epoch_scheduler.step()
            with torch.no_grad():                       # Don't do any training in here :: after every epoch he is updating this here :: 
                mse_val = 0
                preds = []
                true = []
                # Plotting needs to be done with extreme care here : 
                for index,(batch_x, batch_y_h, batch_y) in enumerate(data_val_loader):      # On the validation data here :: 
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_y_h = batch_y_h.to(device)
                    model = model.to(device)
                    y_pred = model(batch_x, batch_y_h,batch_y,train=False)         # using history and driving compute the target series here 
        #             output = output.squeeze(1)
                    output = torch.cat((y_pred[0],y_pred[1],y_pred[2]),-1)
                    if(index%3==0):                                   # We are using every step predictions only to compute the Loss 
                        # we will only use every 3rd timestep prediction for plotting here :: 
                        # Another approach might be to average out the target and the prediction values :: that's also possible 
                        
                        out = output.detach().cpu().numpy()
                        bat = batch_y.detach().cpu().numpy()
                        
                        out = out*(target_train_max - target_train_min) + target_train_min
                        bat = bat*(target_train_max - target_train_min) + target_train_min
                        
                        preds.extend([out[:,0],out[:,1],out[:,2]])
                        true.extend([bat[:,0],bat[:,1],bat[:,2]])

                    mse_val += loss(output, batch_y).item()*batch_x.shape[0]
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            
            if min_val_loss > mse_val**0.5:
                min_val_loss = mse_val**0.5
                print("Saving...")
                torch.save(model.state_dict(), args.model_path+args.model+".pt")                  # Change according to the name here 
                counter = 0
            else: 
                counter += 1
            
            if counter == patience:        # Probably done to stop overfitting here 
                break
            # print("Iter: ", i, "train: ", (mse_train/len(X_train_t))**0.5, "val: ", (mse_val/len(X_val_t))**0.5)
            if(i % 10 == 0):         # Every 10 epochs here :: 
                plt.figure(figsize=(20, 10))
                plt.plot(preds)
                plt.plot(true)
                plt.savefig(plot_path+args.model+'DARNN-three-step-training-'+str(i)+'.png')
                plt.show()
                
                mse = mean_squared_error(true, preds)
                mae = mean_absolute_error(true, preds)
                if(i==0):
                    with open(result_path+args.model+'.txt', 'w') as f:
                        f.write('MSE/MAE:{},{}'.format(mse,mae))
                        f.write('\n')
                else:
                    with open(result_path+args.model+'.txt', 'a') as f:
                        f.write('MSE/MAE:{},{}'.format(mse,mae))
                        f.write('\n')
                # print("mse: ", mse, "mae: ", mae)
        f.close()
        print("\nTime Required for Training\n")        
        print(time.time()-start)
                







# run the training setup here: 


if __name__ == '__main__':

    # train_dataloader,model,device,plot_path,result_path,prediction_horizon=1


    parser = argparse.ArgumentParser(description='Training on the NASDAQ100 data')

    parser.add_argument('--input_file', type=str, default='/home/bhushan/Desktop/bhushan_env/bhushan/ADL_project/Codes/data/nasdaq.csv') 

    parser.add_argument('--device', default=torch.device("cuda:0"))          # Change to CPU if not available here 

    parser.add_argument('--model', type=str,default="DARNN")

    parser.add_argument('--plot_path', type=str, default='/home/bhushan/Desktop/bhushan_env/bhushan/ADL_project/training_plots/', help='Training PLOT Filepath')

    parser.add_argument('--result_path', type=str, default='/home/bhushan/Desktop/bhushan_env/bhushan/ADL_project/training_results/', help='Training result Filepath')

    parser.add_argument('--model_path', type=str, default='/home/bhushan/Desktop/bhushan_env/bhushan/ADL_project/training_models/', help='Training saved models Filepath')

    parser.add_argument('--prediction_horizon', type=int, default=1, help='prediction_horizon')




    # args, unknown = parser.parse_known_args()              # To Run this in Jupyter to avoid the errrors 

    args = parser.parse_args()                             # To Run it in Python 

    # print("\nDoing it for DARNN\n")

    # args.model = "DARNN"

    # Get the Training Data Here : 

    # target_train_max,target_train_min,train_dataloader,train_valloader= train_test_data.train_val_test(args.input_file)[0:4]

    # train(train_dataloader,train_valloader,target_train_max,target_train_min,args)


    # print("\nDoing it for DARNN_SDPA\n")

    # args.model = "DARNN_SDPA"

    # # Get the Training Data Here : 

    # target_train_max,target_train_min,train_dataloader,train_valloader= train_test_data.train_val_test(args.input_file)[0:4]

    # train(train_dataloader,train_valloader,target_train_max,target_train_min,args)


    # print("\nDoing it for DARNN_PDHS\n")

    # args.model = "DARNN_PDHS"

    # # Get the Training Data Here : 

    # target_train_max,target_train_min,train_dataloader,train_valloader= train_test_data.train_val_test(args.input_file)[0:4]

    # train(train_dataloader,train_valloader,target_train_max,target_train_min,args)


    # print("\nDoing it for TRANSFORMER\n")

    # args.model = "TRANSFORMER"

    # # Get the Training Data Here : 

    # target_train_max,target_train_min,train_dataloader,train_valloader= train_test_data.train_val_test(args.input_file)[0:4]

    # train(train_dataloader,train_valloader,target_train_max,target_train_min,args)


    print("\nDoing it for Multistep Time Series\n")

    args.model = "DARNN_MULTISTEP"

    args.prediction_horizon = 3

    # Get the Training Data Here : Doing it for 3 timestep prediction horizon here:: 

    target_train_max,target_train_min,train_dataloader,train_valloader= train_test_data.train_val_test(args.input_file,prediction_horizon=3)[0:4]

    train(train_dataloader,train_valloader,target_train_max,target_train_min,args)

