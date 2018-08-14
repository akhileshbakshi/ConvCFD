from keras.models import Model
from keras.layers import Conv2D, Activation, Input, Dropout
from keras.layers.normalization import BatchNormalization 
from keras import regularizers
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

def ConvCFD(nGridCellsX = 48, nGridCellsY = 48, nFeatures = 6, nFeaturesOut = 1, 
            kernelRegularizer = 0.01, biasRegularlizer = 0.01, 
            nKernels = [5, 3], nLayers = [1, 1], nFilters = [16, 16]):
  """Define convolution network for image processing 
  Args:
      nGridCellsX : int, # cells in x dimension 
      nGridCellsY : int, # cells in y dimension
      nFeatures : int, input depth of image 
      nFeaturesOut : int, output depth of image 
      kernelRegularizer : float, regularizer 
      biasRegularlizer : float, regularizer 
      nKernels : list, kernal sizes for convolution 
      nFilters : list, # filters for convolution 
      nLayers : list, # layers associated with each kernel and filter 
  Returns: 
      Model : keras model built using functional API 

  # ---------------------------------------------------------------------------

  # references: 
  # overall architecture: Tompson et al, https://arxiv.org/abs/1607.03597
  # bias and kernel initialization: He et al, https://arxiv.org/pdf/1502.01852.pdf
  # batch normalization: Ioffe and Szegedy, https://arxiv.org/pdf/1502.03167v2.pdf
    """
    
  paddingChoice = 'same'
  biasInitializer = 'zeros' 
  kernelInitializerRelu='he_uniform' 
  kernelInitializerOthers = 'glorot_uniform'

  inputData = Input(shape=(nGridCellsY, nGridCellsX, nFeatures))
  mainData = inputData

  for i in range(len(nKernels)): 
      kernel = nKernels[i]
      filters = nFilters[i]
      for n in range(nLayers[i]): # applying convolution nLayers[i] times 
          mainData = Conv2D(filters, (kernel, kernel), padding=paddingChoice, 
            kernel_initializer = kernelInitializerRelu, kernel_regularizer= regularizers.l2(kernelRegularizer),
            use_bias = True, bias_initializer = biasInitializer, bias_regularizer= regularizers.l2(biasRegularlizer))(mainData)
          mainData = BatchNormalization()(mainData)
          mainData = Activation('relu')(mainData)
          mainData = Dropout(0.2)(mainData)

  # last layer is 1x1 convolution with nFeaturesOut filters 
  mainData = Conv2D(nFeaturesOut, (1, 1), padding=paddingChoice, activation = 'linear', 
    kernel_initializer = kernelInitializerOthers,  kernel_regularizer= regularizers.l2(kernelRegularizer),
    use_bias = True, bias_initializer = biasInitializer, bias_regularizer= regularizers.l2(biasRegularlizer))(mainData)

  return Model(inputs = inputData, outputs= mainData) 




def trainNetwork(train_X, train_Y, model, lamda1 = 0, \
                 epochs = 2000, learnRate = 0.01, batchSize = 30, \
                 patience = 10, decayRate = 10**-7):
    """train convolution network using training data  
    Args:
      train_X : float, training X data set of size [nt x nx x ny x nfeaturesIn] 
      train_Y : float, training Y data set of size [nt x nx x ny x nfeaturesOut] 
      model : keras model 
      lamda1 : float, trainable parameter for mass balance 
      epochs : int, number of iteratios of training set 
      learnRate : float, initial learning rate of optimizer 
      batchSize : int, batch size for batch SGD 
      patience : float, # iterations for terminating training (see keras documentation)
      decayRate : float, decay in learnRate during training 
      
    Returns: 
      model : keras model trained using training data  
      Model : keras object for storing history of training, validation loss 
    """
    def customLoss(lamda1): 
        # regularizers/ arguments can be passed as inputs through customLoss
        def loss(y_true, y_pred):   
            weights = y_true[:,:,:,-1:]
            y_true = y_true[:,:,:,:-1]
            y_pred = y_pred[:,:,:,:-1]
            loss = K.mean(weights*K.square(y_pred[:,:,:,:]-y_true[:,:,:,:]) \
                          + lamda1*K.square(y_pred[:,:,:,0:1]-y_true[:,:,:,0:1]), axis=-1)
            return loss
        return loss 

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []
        def on_epoch_end(self, batch, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss')) 
    history = LossHistory()        

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, \
                                  patience=patience, verbose=0, mode='auto') 
    
    # saves model weights every 'period' iterations 
    modelCheckpoint = ModelCheckpoint(filepath = 'model_weights.{epoch:04d}-{val_loss:.5f}.hdf5', \
                                  monitor='val_loss', verbose=0, save_best_only=False, \
                                  save_weights_only=False, mode='auto', period=100)
    
    if decayRate == -1: decay = learnRate/epochs
    else: decay = decayRate
    
    sgd = SGD(lr=learnRate, momentum=0.8, decay=decay, nesterov=True)
    model.compile(loss=customLoss(lamda1 = lamda1), optimizer=sgd)
    Model = model.fit(train_X, train_Y,
              batch_size=batchSize,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[history, modelCheckpoint], 
              verbose = 1)
    
    return model, Model