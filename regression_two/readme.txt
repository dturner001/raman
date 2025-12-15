going to try develop a new model from scratch, not using ramanNet, as it was designed for classification.


https://www.mdpi.com/2673-4532/3/3/20

from this paper, it was found that a small CNN network was used as a way of cleaning the data before going into the main model. However performance might not be any better, as the model will do it implicitly


common types of DLN for this application:

CNN, ResNets, RNNs and GANs


classification and regression are mixed in practice -> regression with a threshold becomes classification

a GAN could be an interesting way of pre-training the model, or just using other similar raman data to innitialise the weights and biases as in ./extra data

truncating data - some frequencies are more related to the IR absorption - > wavelength of the lamps -> what range does PET absorb IR radation attempted

is the model considering the curve (i.e. relational) or individual data points - need to consider curve as it matters what 'trjectory' the point is on 
    -functional PCA

find papers on it
    -thermoscientific




from chatGPT:

a method to improve accuracy would be to package the feature vector with its
Corresponding rPET% to feed into the MLP. Adds a 'checmical background' to 
the prediction. Its like an internal classification that helps it differentiate. 
It can be seen from a plot that as rPET % increases, the temperature 
the the perform is heated to decreases, so this may help improve results.

In order to do so ill need to:

-pull rpet% out of each sample (done with classification)
-concat each feature vector with the rPET%
-feed into the regressor????
