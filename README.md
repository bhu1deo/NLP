# NLP

This contains 2 sub-folders: 

1.) Time Series Prediction using DA-RNN and Transformers: 

pytorch 1.9.0 

Paper-1 : DA-RNN https://arxiv.org/abs/1704.02971
Paper-2 : Transformer https://arxiv.org/abs/1706.03762

Baseline Code : https://github.com/KurochkinAlexey/DA-RNN

My modifications : I did several modifications to the baseline architecture, which are:
a.) Scaled Dot Product Attention: Leads to good initial starting point, fast convergence. 
b.) Using All the Decoder states for prediction: To combat the Long Term dependency problem. 
c.) DARNN-Multistep: Modified the architecture to suit multi-step prediction. Useful in real world practice. 
d.) Transformer: Used modified Time2Vec Embeddings, different embedding model dimensions,heads for the Encoder and Decoder.

Results:

![Screenshot from 2021-09-21 21-40-58](https://user-images.githubusercontent.com/20145042/134206572-f543a5a3-ab5c-4a91-a83e-c24500791850.png)

Conclusions: 

1.) The DA-RNN is noticed to suffer from large initial errors on the training set. The Scaled Dot Product and Using All the Decoder states for prediction versions alleviate this issue by having less severe initializations. 
2.) The DA-RNN fits well for multi-step predictions, which might be useful in real scenarios. 
3.) The Transformer model has some way to go before applying it on the given exogenous time series data under consideration. 





2.) Fast Transformers for next level character prediction: All the following architectures were implemented from scratch in Tensorflow: 

tf 2.3.0 

a.) Linformer : https://arxiv.org/abs/2006.04768

b.) Fast Autoregressive Transformer : https://arxiv.org/abs/2006.16236

c.) Sparse Transformer : https://arxiv.org/abs/1904.10509

The models worked well on Next Level character prediction. The Kernel Transformer model has some problems as to it's loss diverges to nan. A deeper analysis is required. 

Results:
The time Required to compute Multi-Head Attention is as follows, for different query,key and value dimensions: 


![Screenshot (3)](https://user-images.githubusercontent.com/20145042/134819497-ba683f3b-b4b7-4dde-be57-019b62607474.png)

The Kernel Autoregressive Transformer takes nearly constant, as especially beneficial for very long sequences. Linformer on the other hand loses some amount of information as it compresses the embeddings into a lower dimensional space. It results in fast run times. Sparse Transformer implemented here doesn’t tackle the time issue. Rather it is implemented keeping in view the redundancy of the attention matrix.  Note that the methods are implemented in Python, hence have an overhead, the core CUDA implementations would be quite faster. 

Linformer : Heavy loss of information due to dimension reduction. 
Kernel Transformer : Operates in the original dimension, but is difficult to train. 

