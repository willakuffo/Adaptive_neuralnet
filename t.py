from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils


def adaptive_model(xshape,yshape,hidden_layers = 5 ): 
    model = Sequential()
    
    #output activation function
    output_activation = 'softmax' #multiclassification
    if yshape == 2:
        output_activation = 'sigmoid' #binary class 
    print(output_activation)

    #configure input and output layers
    il = Dense(units =xshape,activation = 'relu',input_dim = xshape)
    ol = Dense(units =yshape,activation = output_activation) #activation fnc dependent on num of classes
    

    if type(hidden_layers) == str:
        if hidden_layers == 'adaptive':
            #number of hidden layers are adaptive to features
            #in an attempt to go for depth
            hidden_layers = int(1.5*xshape)
            print('hidden layers set to',hidden_layers)
             
        else:
            raise ValueError('Value error for keyword argument hidden_layers when string.For hidden_layers\
            as type string,set hidden_layers = adaptive. This allows hidden layers to adapt to training features')
          
    #fully connecting layers
    # decoder to encoder type arch
    hidden_neuron = xshape
    hidden_layer_neurons = []
    model.add(il) #add input layer

    #add hidden layers
    for neuron in range(1,1+hidden_layers) : 
        if ((2*neuron//2) < (hidden_layers)//2): 
            hidden_neuron *= 2
            #print(hidden_neuron,'a')
            model.add(Dense(units = hidden_neuron,activation = 'relu' ))
        else:     
            hidden_neuron //=2 
            #print(hidden_neuron,'d')
            
            if hidden_neuron<= yshape:
                break
            model.add(Dense(units =hidden_neuron ,activation = 'relu' ))
        hidden_layer_neurons.append(hidden_neuron)
    print(hidden_layer_neurons)
    model.add(ol) #add output layer


            
    print(model.summary())

adaptive_model(4,2,hidden_layers = 'adaptive')