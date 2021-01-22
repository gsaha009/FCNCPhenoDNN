import sys
import os
import uproot
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras import regularizers
from keras.layers import LeakyReLU
from sklearn.metrics import roc_curve, auc
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def loadConfig(configfile):
    with open(configfile, 'r') as inf:
        return yaml.safe_load(inf)

def unfoldEventInfoInNunpyArray(sampleDict, inputVariablesList, verbose):
    sampleArray = []
    xsecList    = []
    evWtSumList =[]
    for sample, info in sampleDict.items():
        #if verbose:
        logging.info('Sample :==>> {}'.format(sample))
        file_    = uproot.open(info.get('file'))
        tree_    = file_.get(b'RTree;2')
        xsec_    = info.get('xsec')
        evWtSum_ = info.get('nEvents')
        #if verbose:
        logging.info('File : {}'.format(info.get('file')))
        logging.info('Xsec : {}, nEventsProduced: {}'.format(xsec_, evWtSum_))
        evInfoArray = np.concatenate([tree_[var].array() for var in inputVariablesList])
        #if verbose:
        logging.info('concatenate all input vars along axis=0, shape : {}'.format(evInfoArray.shape))
        nev         = tree_[inputVariablesList[0]].array().shape[0]
        evInfoArray = np.transpose(evInfoArray.reshape(len(inputVariablesList), nev))
        #if verbose:
        logging.info('Reshaping to (nEvents, nInputVars), shape : {}'.format(evInfoArray.shape))
        logging.info('\n')
        sampleArray.append(evInfoArray)
        xsecList.append(xsec_)
        evWtSumList.append(evWtSum_)
    return sampleArray, xsecList, evWtSumList

def makeInputCorrelationPlot(all_in, inputVarList, handle):
    df = pd.DataFrame(all_in, columns=inputVarList)
    corrMatrix = df.corr()
    plt.figure(figsize=(11, 8.5))
    sns.heatmap(corrMatrix, cmap='YlGnBu', annot=True, annot_kws={'size':13}, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join('Outputs',handle+'_correation_plot.eps'), format='eps')


def prepareTrainAndTestDataSet(signalXZs, signalH2Zs, WZs, clsWtCoeff, inVarList, handle, trainTestSplit, stop, plot_correl=False):
    combineInfo = lambda evList : np.concatenate([item for item in evList])
    signalXZ_array  = combineInfo(signalXZs)
    signalH2Z_array = combineInfo(signalH2Zs)
    WZ_array        = combineInfo(WZs)
    #print(signalXZ_array.shape, signalH2Z_array.shape, WZ_array.shape)
    #print('asasasasasassa: ',signalXZ_array.shape)
    #print('asasasasasassa: ',signalH2Z_array.shape)
    
    zeros = np.zeros(signalXZ_array.shape[0]).reshape(signalXZ_array.shape[0],1)
    ones  = np.ones (signalH2Z_array.shape[0]).reshape(signalH2Z_array.shape[0],1)
    twos  = 2*np.ones(WZ_array.shape[0]).reshape(WZ_array.shape[0],1)

    SigXZ  = np.concatenate((signalXZ_array,zeros), 1)[:stop,:] if stop > 0 else np.concatenate((signalXZ_array,zeros), 1)
    SigH2Z = np.concatenate((signalH2Z_array,ones), 1)[:stop,:] if stop > 0 else np.concatenate((signalH2Z_array,ones), 1)
    WZ     = np.concatenate((WZ_array,twos), 1)
    
    all_in = np.concatenate((SigXZ,SigH2Z,WZ),0)
    #print(all_in[:100,:3])
    np.random.shuffle(all_in)

    if plot_correl:
        makeInputCorrelationPlot(all_in[:,:len(inVarList)], inVarList, handle)
    
    ntr = int(float(trainTestSplit)*all_in.shape[0])

    (data_train, data_test) = (all_in[0:ntr,:], all_in[ntr:,:])
    (x_train, _y_train) = (data_train[:,0:(data_train.shape[1]-1)], data_train[:,(data_train.shape[1]-1)])
    y_train = to_categorical(_y_train, 3)
    (x_test, _y_test) = (data_test[:,0:(data_test.shape[1]-1)], data_test[:,(data_test.shape[1]-1)])
    y_test = to_categorical(_y_test, 3)

    SigXZ_wt  = clsWtCoeff*((SigH2Z.shape[0]+WZ.shape[0])/all_in.shape[0])
    SigH2Z_wt = clsWtCoeff*((SigXZ.shape[0]+WZ.shape[0])/all_in.shape[0])
    WZ_wt     = clsWtCoeff*((SigH2Z.shape[0]+SigXZ.shape[0])/all_in.shape[0])
    
    return x_train, y_train, x_test, y_test, [_y_train, _y_test], {0 : SigXZ_wt, 1 : SigH2Z_wt, 2 : WZ_wt}


def scaleData(trainSet, testSet, scale=False):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    trainSet = scaler.fit_transform(trainSet) if scale else trainSet
    testSet  = scaler.transform(testSet) if scale else testSet
    return scaler, trainSet, testSet

def plotHist(nbins, xlow, xhigh, ylow, yhigh, inVar=[], classes=[], colors=[], islog=False):
    ymax = 0.0
    for i in range(0, len(classes)):
        y_,bins,patches=plt.hist(inVar[i],nbins,density=True, facecolor=colors[i],histtype='step', alpha=0.8, lw=2, log=islog)
        ymax_ = y_.max()
        if ymax_ > ymax:
            ymax = ymax_ 
    plt.xlim(xlow, xhigh)
    plt.ylim(ylow, ymax+0.02)
    plt.grid(True)
    plt.legend(classes, loc='upper right', fontsize=16)  
    return plt

# Feature Plots
def doFeaturePlots(allXZs, allHZs, WZs, RestBKGs, featureList, Dir, handle):
    sig1matrix = allXZs[0]
    sig2matrix = allHZs[0]
    Bkg1martix = WZs[0]
    Restmatrix = RestBKGs[0]
    features   = [item.split('/')[0] for item in featureList] 
    nbins      = [int(item.split('/')[1]) for item in featureList] 
    xlow       = [float(item.split('/')[2]) for item in featureList] 
    xhigh      = [float(item.split('/')[3]) for item in featureList]
    import matplotlib.pyplot as plt
    for i, ft in enumerate(features):
        plt.figure(figsize=(11, 8.5))
        plt = plotHist(nbins[i], xlow[i], xhigh[i], 0.001, 3.0, inVar=[sig1matrix[:,i:i+1], sig2matrix[:,i:i+1], Bkg1martix[:,i:i+1]], classes=['Sig_XZ', 'Sig_h2Z', 'WZTo3LNu_012Jets'], colors=['red','black','blue'])
        plt.savefig(os.path.join(Dir,handle+'_'+features[i]+'.pdf'), bbox_inches='tight')
        plt.close()

# Accuracy & Loss
def plotLossAndAccuracy(history, handle):
    #histKeys = [key for key in history.history.keys()]
    trainKeys = [key for key in history.history.keys() if not 'val' in key]
    valKeys   = [key for key in history.history.keys() if 'val' in key]
    plt.figure(figsize=(11, 8.5))
    plt.grid(True)
    plt.plot(history.history[trainKeys[1]])
    plt.plot(history.history[valKeys[1]])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join('Outputs',handle+'_accuracy.eps'), format='eps')
    plt.clf()
    # summarize history for loss
    plt.figure(figsize=(11, 8.5))
    plt.grid(True)
    plt.plot(history.history[trainKeys[0]])
    plt.plot(history.history[valKeys[0]])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join('Outputs',handle+'_loss.eps'), format='eps')
    plt.clf()

def buildModel(inputVarList, modelParams):
    LayerDims   = modelParams.get('Layers')
    IsBatchNorm = modelParams.get('BatchNorms')
    DropOuts    = modelParams.get('Dropout')
    L2regs      = modelParams.get('L2')
    Activations = modelParams.get('activ')
    
    model = Sequential()
    model.add(Dense(LayerDims[0], input_dim=len(inputVarList), activation=Activations[0], kernel_regularizer=regularizers.l2(L2regs[0])))
    if IsBatchNorm[0]:
        model.add(BatchNormalization())
    model.add(Dropout(DropOuts[0]))    
    for i in range(1, len(LayerDims)-1):
        model.add(Dense(LayerDims[i], activation=Activations[i], kernel_regularizer=regularizers.l2(L2regs[i])))
        if IsBatchNorm[i]:
            model.add(BatchNormalization())
        model.add(Dropout(DropOuts[i]))
    if IsBatchNorm[-1]:
        model.add(BatchNormalization())
    model.add(Dense(LayerDims[-1], activation=Activations[-1]))
    return model

def compilenfit(model, x_train, y_train, nEpoch, batchsize, classWeight, handle, trainingParams, stat=False, plotLossAcc=False):
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    #set early stopping monitor so the model stops training when it won't improve anymore
    #early_stopping_monitor = EarlyStopping(patience=3)
    # https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
    custom_early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=int(nEpoch/10), 
        min_delta=0.001,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )
    #https://keras.io/api/callbacks/reduce_lr_on_plateau/
    custom_ReduceLROnPlateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="min",
        cooldown=0,
        min_lr=0
    )
    
    LossFunc = trainingParams.get('Loss')
    LRate    = trainingParams.get('LR')
    Metrics  = trainingParams.get('Metrics')
    valSplit = trainingParams.get('valSplit')
    opt      = keras.optimizers.Adam(learning_rate=LRate)
    model.compile(loss=LossFunc, optimizer=opt, metrics=Metrics)
    if stat==True:
        model.summary()
    plot_model(model, to_file=os.path.join('Outputs',handle+'_modelDNN.png'),show_shapes=True,show_layer_names=True)
    history = model.fit(x_train, y_train, epochs=nEpoch, batch_size=batchsize, validation_split=valSplit, 
                        verbose=1, class_weight=classWeight, use_multiprocessing=True, callbacks=[custom_ReduceLROnPlateau, custom_early_stopping])

    if plotLossAcc:
        plotLossAndAccuracy(history, handle)
    return history


def plotROC(y_pred, y_true, n_classes, handle, outdir, classes=[], colours=[], style='solid', verbose=False):
    from itertools import cycle
    plt.grid(True)
    if (verbose):
        print('making prediction!')
        #print('true {} :: pred {}: '.format(y_true[:20,:], y_pred[:20,:]))
        prediction = plotHist(100,0.0,1.0,0.001,100.0, [y_pred[:,0],y_pred[:,1],y_pred[:,2]] ,classes, ['r','g','b'],False)
        prediction.xlabel('prediction probability')
        prediction.title('Class Prediction Response')
        prediction.savefig(os.path.join(outdir,handle+"_DNN_Response_onTrain.eps"), format='eps')
        prediction.clf()
    # ROC
    thr = dict() # thresholds
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i],thr[i] = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(tpr[i], 1-fpr[i])

    if (verbose):
        print ('# FPR')
        for key,val in fpr.items():
            print(key, val, val.shape)
        print ('# TPR')
        for key,val in tpr.items():
            print(key, val, val.shape)
        print ('# AUC')
        for key,val in roc_auc.items():
            print(key, val, val.shape)
        print ('# Thresholds')
        for key,val in thr.items():
            print(key, val, val.shape)
            

    colors = cycle(colours)
    #classes = ['XZ','H2Z',"WZ"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, ls=style, label='{0}\n(auc = {1:0.3f})'''.format(classes[i],roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, ls=style)
    plt.xlim([0.00001, 1.005])
    plt.ylim([0.00001, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: multi-class')
    plt.legend(loc="lower right", prop={'size': 10}, ncol=1)
    return plt


def saveModel(model, modelName):
    model_json = model.to_json()
    with open(modelName.split('.')[0]+".json", "w") as jfile:
        jfile.write(model_json)
    model.save_weights(modelName)
    print("Saved model to disk")

    
def loadModel(modelName):
    json_file  = open(modelName.split('.')[0]+'.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(modelName)
    print("Loaded model from disk")
    return model

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(11, 8.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def estimateSignificance(nSig, nBkg, isLog=False):
    # https://arxiv.org/pdf/1007.1727.pdf
    from math import sqrt, log
    signf = sqrt(2*(nSig+nBkg)*log(1+(nSig/nBkg)) - 2*nSig) if isLog else nSig/sqrt(nSig+nBkg)
    return signf
