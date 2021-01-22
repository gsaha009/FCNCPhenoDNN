import sys
import os
import copy
import uproot as up
import numpy as np
import pandas as pd
import keras
from keras.layers import LeakyReLU
import tensorflow as tf
import matplotlib.pyplot as plt
#import seaborn as sns
#np.set_printoptions(threshold=sys.maxsize)
import argparse
import logging
from UtilsML import *

def get_options():
    parser = argparse.ArgumentParser(description='ExoFCNC_MachineLearning')

    parser.add_argument('-v','--verbose', action='store_true', required=False, default=False,
                        help='Verbose')    
    parser.add_argument('--configName', action='store', required=True, type=str,
                        help='Name of the config')
    parser.add_argument('--train', action='store_true', required=False, default=False,
                        help='Do Training Only')
    parser.add_argument('--test', action='store_true', required=False, default=False,
                        help='Do Testing Only')
    parser.add_argument('--evaluate', action='store_true', required=False, default=False,
                        help='Do Testing Only')
    parser.add_argument('--minmaxscaling', action='store_true', required=False, default=False,
                        help='Use Min-Max Scaler or not')
    parser.add_argument('--plotInputs', action='store_true', required=False, default=False,
                        help='plot the fatures or not')
    
    
    opt = parser.parse_args()
    return opt

    
def main():
    # get parser
    opt = get_options()
    # Load Config-File
    config     = loadConfig(str(opt.configName))
    configKeys = [key for key in config.keys()]
    handle     = opt.configName.split('.')[0]
    OutDir     = config.get('outDir')
    print('Key for naming all outputs : {}, Output Directory : {}'.format(handle, OutDir))
    # logging
    logging.basicConfig(filename=os.path.join(OutDir,handle+'_INFO.log'), filemode='w',level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%M:%S')
    if not opt.verbose:
        logging.getLogger().setLevel(logging.INFO)

    logging.info('Config Keys: {}'.format(config.keys()))

    ModelDir    = config.get('modelDir')
    lumi        = config.get('Lumi')
    logging.info('Luminosity : {}'.format(lumi))
    AllSignals  = config.get('SignalSamples')
    featureInfoList = config.get('InputVariables')  
    inputVariablesList = [item.split('/')[0] for item in featureInfoList]
    logging.info('InputVariables : {}'.format(inputVariablesList))
    ModelParams = config.get('modelParams')
    TrainParams = config.get('trainingParams')
    train_test_split = TrainParams.get('trainTestSplit')
    batchSize   = TrainParams.get('BatchSize')
    epoch       = TrainParams.get('epoch')
    logging.info('Batch_Size: {}, nEpochs : {}'.format(batchSize, epoch))
    wtMultFac   = config.get('clsWeightFac')
    
    signalXZ_dict  = AllSignals.get('Signal_XZ')
    logging.info('signalXZ_dict : {}'.format(signalXZ_dict))
    signalH2Z_dict = AllSignals.get('Signal_H2Z')
    logging.info('signalH2Z_dict : {}'.format(signalH2Z_dict))
    TrainBkg_dict  = config.get('Train_Background') # bkgs used for training
    logging.info('TrainBkg_dict : {}'.format(TrainBkg_dict))
    OtherBkg_dict  = config.get('Other_Background') # used for test only 
    logging.info('OtherBkg_dict : {}'.format(OtherBkg_dict))

    couplings = [key for key in signalXZ_dict.keys()]
    
    allXZs, allXZxsecs, allXZnEvts       = unfoldEventInfoInNunpyArray(signalXZ_dict,  inputVariablesList, opt.verbose)
    allH2Zs, allH2Zxsecs, allH2ZnEvts    = unfoldEventInfoInNunpyArray(signalH2Z_dict, inputVariablesList, opt.verbose)
    WZs, WZxsecs, WZnEvts                = unfoldEventInfoInNunpyArray(TrainBkg_dict,  inputVariablesList, opt.verbose)
    RestBKGs, RestBKGxsecs, RestBKGnEvts = unfoldEventInfoInNunpyArray(OtherBkg_dict,  inputVariablesList, opt.verbose)

    # --------------- Feature Plots ---------------- #
    if opt.plotInputs and opt.train:
        doFeaturePlots(allXZs, allH2Zs, WZs, RestBKGs, featureInfoList, OutDir, handle)

    # ---------------------------------------------- #
    
    nEvPerSig = config.get('nEvPerSigSample') if opt.train or opt.test else -1
    # Preparing Train and Test dataset
    x_train_, y_train, x_test_, y_test, yRaws, clsWtDict = prepareTrainAndTestDataSet(allXZs,allH2Zs,WZs,wtMultFac,inputVariablesList,handle,trainTestSplit=train_test_split,stop=nEvPerSig,plot_correl=True)
    #logging.info('X_Train : {}'.format(x_train_[:10,:]))
    #logging.info('X_Test  : {}'.format(x_test_[:10, :]))

    logging.info('Min-Max Scaling : {}'.format(opt.minmaxscaling))
    
    # min-max scaling :: Keep the scaler==> will be used for evaluation
    scaler, x_train, x_test = scaleData(x_train_, x_test_, scale=opt.minmaxscaling)
    #logging.info('afterScaling_X_Train: {}'.format(x_train[:10,:]))
    #logging.info('afterScaling_X_Test : {}'.format(x_test[:10,:]))

    #logging.info('shapes: X-Train - {}, Y-Train - {}, X-Test - {}, Y-Test - {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    #logging.info('X-Train - {}, Y-Train - {}, X-Test - {}, Y-Test - {}'.format(x_train[:10],yRaws[0][:10],x_test[:10],yRaws[1][:10]))

    modelPath = os.path.join(ModelDir, handle+'_DNN_model.pb')
    if not opt.train and os.path.exists(modelPath) :
        model = loadModel(modelPath)
    # Train
    elif opt.train:
        print('Training ===>>>')
        model = buildModel(inputVariablesList, ModelParams)
        history = compilenfit(model,x_train,y_train,epoch,batchSize,clsWtDict,handle,TrainParams,stat=True,plotLossAcc=True)
        
        y_train_pred = model.predict(x_train)
        ROC_plot_onTrain = plotROC(y_train_pred, y_train, 3, handle, OutDir, ['XZ','H2Z',"WZ"], ['r','g','b'], 'solid', verbose=False)
        ROC_plot_onTrain.savefig(os.path.join(OutDir,handle+'_ROC_onTrain.eps'), format='eps')
        ROC_plot_onTrain.clf()
        
        saveModel(model, modelName=modelPath)
    else:
        print('no model found!')
        
    print('Testing ===>>>')    
    LossFunc = TrainParams.get('Loss')
    LRate    = TrainParams.get('LR')
    Metrics  = TrainParams.get('Metrics')
    valSplit = TrainParams.get('valSplit')
    optmz    = keras.optimizers.Adam(learning_rate=LRate)
    model.compile(loss=LossFunc, optimizer=optmz, metrics=Metrics)
    score    = model.evaluate(x_test, y_test, verbose=1)
    logging.info('Metrics_names : {}'.format(model.metrics_names))
    logging.info('Score : {}'.format(score))
    y_pred   = model.predict(x_test)
    #print(y_pred[:10,:])
    #y_predClass = model.predict_classes(x_test)

    #y_predClass = np.asarray([np.argmax(item) for item in y_pred])
    y_predClassNoCutOnProb = np.asarray([np.argmax(item) for item in y_pred])
    #print(y_predClassNoCutOnProb.shape)
    #print(type(y_predClass), y_predClass)
    #print(y_pred[:10], y_predClass[:10])

    # Confusion Matrix
    if opt.train or opt.test : 
        cnf_matrix = confusion_matrix(yRaws[1], y_predClassNoCutOnProb, labels=[0,1,2])
        cm = plot_confusion_matrix(cnf_matrix, classes=['SigXZ', 'SigH2Z', 'BkgWZ'],normalize=True, title='Confusion matrix')
        cm.savefig(os.path.join(OutDir,handle+'_confusion_matrix.eps'), format='eps')
        cm.clf()
    
        # test_ROC
        test_ROC = plotROC(y_pred, y_test, 3, handle, OutDir, ['SigXZ','SigH2Z',"BkgWZ"], ['r','g','b'],'solid',verbose=False)
        test_ROC.savefig(os.path.join(OutDir,handle+'_ROC_test.eps'), format='eps')
        test_ROC.clf()
    
    if opt.evaluate:
        logging.info('Evaluating ===>>>')
        logging.info('Maximum prediction probability > 0.7')
        y_predClass    = np.asarray([maxidx for i, maxidx in enumerate(list(y_predClassNoCutOnProb)) if y_pred[i][maxidx] > 0.7])
        print(y_predClass.shape)

        allXZsScaled      = [scaler.transform(item) for item in allXZs] if opt.minmaxscaling else [item for item in allXZs]
        allH2ZsScaled     = [scaler.transform(item) for item in allH2Zs] if opt.minmaxscaling else [item for item in allH2Zs]
        WZsScaled         = [scaler.transform(item) for item in WZs] if opt.minmaxscaling else [item for item in WZs]
        RestBKGsScaled    = [scaler.transform(item) for item in RestBKGs] if opt.minmaxscaling else [item for item in RestBKGs] 

        logging.info('XZ_Unweighted : Before Cut on PredProb ::')
        for i in allXZsScaled:
            logging.info(len(i))
        logging.info('HZ_Unweighted : Before Cut on PredProb ::')
        for i in allH2ZsScaled:
            logging.info(len(i))
        
        # If you don't want to put any cut on predictio probability and to use predict_classes instead of maxargs()
        #allXZsPredClass   = [model.predict_classes(item) for item in allXZsScaled]
        #allH2ZsPredClass  = [model.predict_classes(item) for item in allH2ZsScaled]
        #bkgWZsPredClass   = [model.predict_classes(item) for item in WZsScaled]
        #restBkgsPredClass = [model.predict_classes(item) for item in RestBKGsScaled]

        # Using argmax() and put a cut on prediction probability
        # [[ndarray(x,y,z), ndarray(a,b,c), ...], [], [], [], []]  
        allXZsScaledPredProb   = [model.predict(item) for item in allXZsScaled] 
        allH2ZsScaledPredProb  = [model.predict(item) for item in allH2ZsScaled] 
        WZsScaledPredProb      = [model.predict(item) for item in WZsScaled]
        RestBKGsScaledPredProb = [model.predict(item) for item in RestBKGsScaled]

        ##################### test #######################
        #print(allXZsScaledPredProb[0][0][0])
        plt.clf()
        sigPred = np.asarray([item[0] for item in allXZsScaledPredProb[0]])
        bkgPred = np.asarray([item[0] for item in WZsScaledPredProb[0]])
        x = np.arange(sigPred.shape[0], dtype=np.float)
        y = np.arange(bkgPred.shape[0], dtype=np.float)
        sigWt = np.full_like(x, (0.033*100000/500000))
        bkgWt = np.full_like(y, (3.517*100000/9910270))
        #print(sigPred.shape, bkgPred.shape, sigWt.shape, bkgWt.shape)
        #print(sigPred[:10], bkgPred[:10])
        plt.hist(sigPred, 100, density=False, facecolor='blue',alpha=0.6, lw=2, log=False, weights=sigWt)
        plt.hist(bkgPred, 100, density=False, facecolor='red',alpha=0.6, lw=2, log=False, weights=bkgWt)
        plt.xlim(0, 1)
        plt.ylim(0.0, 120.)
        plt.grid(True)
        plt.legend(['Sig', 'Bkg'], loc='upper right', fontsize=16)
        plt.savefig(os.path.join(OutDir,handle+'_PredProb.eps'), format='eps')
        plt.clf()
        #################################################
        
        # [ndarray(id1, id2, .....), ndarray(), nd(), nd(), nd()]
        allXZsPredClassNoCutOnProb   = [np.asarray([np.argmax(probs) for probs in sample])  for sample in allXZsScaledPredProb]
        allH2ZsPredClassNoCutOnProb  = [np.asarray([np.argmax(probs) for probs in sample])  for sample in allH2ZsScaledPredProb]
        WZsPredClassNoCutOnProb      = [np.asarray([np.argmax(probs) for probs in sample])  for sample in WZsScaledPredProb] 
        RestBKGsPredClassNoCutOnProb = [np.asarray([np.argmax(probs) for probs in sample])  for sample in RestBKGsScaledPredProb]

        allXZsPredClass   = [np.asarray([maxidx for j, maxidx in enumerate(list(sample)) if allXZsScaledPredProb[i][j][maxidx] > 0.7]) for i, sample in enumerate(allXZsPredClassNoCutOnProb)]
        allH2ZsPredClass  = [np.asarray([maxidx for j, maxidx in enumerate(list(sample)) if allH2ZsScaledPredProb[i][j][maxidx] > 0.7]) for i, sample in enumerate(allH2ZsPredClassNoCutOnProb)]
        bkgWZsPredClass   = [np.asarray([maxidx for j, maxidx in enumerate(list(sample)) if WZsScaledPredProb[i][j][maxidx] > 0.7]) for  i, sample in enumerate(WZsPredClassNoCutOnProb)]
        restBkgsPredClass = [np.asarray([maxidx for j, maxidx in enumerate(list(sample)) if RestBKGsScaledPredProb[i][j][maxidx] > 0.7]) for i, sample in enumerate(RestBKGsPredClassNoCutOnProb)]

        
        logging.info('XZ_Unweighted : After Cut on PredProb ::')
        for i in allXZsPredClass:
            logging.info(len(i))
        logging.info('HZ_Unweighted : After Cut on PredProb ::')
        for i in allH2ZsPredClass:
            logging.info(len(i))
            
        XZ_selectedDict = dict()
        HZ_selectedDict = dict()

        for i, coup in enumerate(couplings):
            predclassXZ = allXZsPredClass[i]
            XZ_selectedDict[coup] = [(predclassXZ == 0).sum(), (predclassXZ == 1).sum(), (predclassXZ == 2).sum()]
            predclassHZ = allH2ZsPredClass[i]
            HZ_selectedDict[coup] = [(predclassHZ == 0).sum(), (predclassHZ == 1).sum(), (predclassHZ == 2).sum()]
        
        '''
        for i, predclass in enumerate(list(allXZsPredClass)):
            print(couplings[i])
            XZ_selectedDict[couplings[i]] = [(predclass == 0).sum(), (predclass == 1).sum(), (predclass == 2).sum()]

        for i, predclass in enumerate(list(allH2ZsPredClass)):
            XZ_selectedDict[couplings[i]] = [(predclass == 0).sum(), (predclass == 1).sum(), (predclass == 2).sum()]

        
        XZ_Yl_Yq_01      = allXZsPredClass[0]
        XZ_Yl_Yq_03      = allXZsPredClass[1]
        XZ_Yl_Yq_0033    = allXZsPredClass[2]
        XZ_Yl_03_Yq_01   = allXZsPredClass[3]
        XZ_Yl_0033_Yq_01 = allXZsPredClass[4]
        #print(XZ_Yl_Yq_01, XZ_Yl_Yq_03, XZ_Yl_Yq_0033, XZ_Yl_03_Yq_01, XZ_Yl_0033_Yq_01)

        HZ_Yl_Yq_01      = allH2ZsPredClass[0]
        HZ_Yl_Yq_03      = allH2ZsPredClass[1]
        HZ_Yl_Yq_0033    = allH2ZsPredClass[2]
        HZ_Yl_03_Yq_01   = allH2ZsPredClass[3]
        HZ_Yl_0033_Yq_01 = allH2ZsPredClass[4]

        #print(bkgWZsPredClass)

        
        #couplings = ['Yl_Yq_01','Yl_Yq_03','Yl_Yq_0033','Yl_03_Yq_01','Yl_0033_Yq_01']
        #couplings = ['Yl_Yq_01']
        
        XZ_Yl_Yq_01_fracInAllNodes       = [(XZ_Yl_Yq_01 == 0).sum(), (XZ_Yl_Yq_01 == 1).sum(), (XZ_Yl_Yq_01 == 2).sum()]
        XZ_selectedDict['Yl_Yq_01']      = XZ_Yl_Yq_01_fracInAllNodes
        XZ_Yl_Yq_03_fracInAllNodes       = [(XZ_Yl_Yq_03 == 0).sum(), (XZ_Yl_Yq_03 == 1).sum(), (XZ_Yl_Yq_03 == 2).sum()]
        XZ_selectedDict['Yl_Yq_03']      = XZ_Yl_Yq_03_fracInAllNodes
        XZ_Yl_Yq_0033_fracInAllNodes     = [(XZ_Yl_Yq_0033 == 0).sum(), (XZ_Yl_Yq_0033 == 1).sum(), (XZ_Yl_Yq_0033 == 2).sum()]
        XZ_selectedDict['Yl_Yq_0033']    = XZ_Yl_Yq_0033_fracInAllNodes
        XZ_Yl_03_Yq_01_fracInAllNodes    = [(XZ_Yl_03_Yq_01 == 0).sum(), (XZ_Yl_03_Yq_01 == 1).sum(), (XZ_Yl_03_Yq_01 == 2).sum()]
        XZ_selectedDict['Yl_03_Yq_01']   = XZ_Yl_03_Yq_01_fracInAllNodes
        XZ_Yl_0033_Yq_01_fracInAllNodes  = [(XZ_Yl_0033_Yq_01 == 0).sum(), (XZ_Yl_0033_Yq_01 == 1).sum(), (XZ_Yl_0033_Yq_01 == 2).sum()]
        XZ_selectedDict['Yl_0033_Yq_01'] = XZ_Yl_0033_Yq_01_fracInAllNodes
        
        HZ_Yl_Yq_01_fracInAllNodes       = [(HZ_Yl_Yq_01 == 0).sum(), (HZ_Yl_Yq_01 == 1).sum(), (HZ_Yl_Yq_01 == 2).sum()]
        HZ_selectedDict['Yl_Yq_01']      = HZ_Yl_Yq_01_fracInAllNodes
        HZ_Yl_Yq_03_fracInAllNodes       = [(HZ_Yl_Yq_03 == 0).sum(), (HZ_Yl_Yq_03 == 1).sum(), (HZ_Yl_Yq_03 == 2).sum()]
        HZ_selectedDict['Yl_Yq_03']      = HZ_Yl_Yq_03_fracInAllNodes
        HZ_Yl_Yq_0033_fracInAllNodes     = [(HZ_Yl_Yq_0033 == 0).sum(), (HZ_Yl_Yq_0033 == 1).sum(), (HZ_Yl_Yq_0033 == 2).sum()]
        HZ_selectedDict['Yl_Yq_0033']    = HZ_Yl_Yq_0033_fracInAllNodes
        HZ_Yl_03_Yq_01_fracInAllNodes    = [(HZ_Yl_03_Yq_01 == 0).sum(), (HZ_Yl_03_Yq_01 == 1).sum(), (HZ_Yl_03_Yq_01 == 2).sum()]
        HZ_selectedDict['Yl_03_Yq_01']   = HZ_Yl_03_Yq_01_fracInAllNodes
        HZ_Yl_0033_Yq_01_fracInAllNodes  = [(HZ_Yl_0033_Yq_01 == 0).sum(), (HZ_Yl_0033_Yq_01 == 1).sum(), (HZ_Yl_0033_Yq_01 == 2).sum()]
        HZ_selectedDict['Yl_0033_Yq_01'] = HZ_Yl_0033_Yq_01_fracInAllNodes


        #print('XZ events in 3 nodes : for all couplings : {}'.format(XZ_selectedDict))
        #print('HZ events in 3 nodes : for all couplings : {}'.format(HZ_selectedDict))
        '''
        bkgWZsSelected    = [[(diffClass == 0).sum(), (diffClass == 1).sum(), (diffClass == 2).sum()] for diffClass in bkgWZsPredClass]
        restBkgsSelected  = [[(diffClass == 0).sum(), (diffClass == 1).sum(), (diffClass == 2).sum()] for diffClass in restBkgsPredClass]
        #print('sig1-sig2-bkg like events in WZTo3LNu: {}'.format(bkgWZsSelected[0]))
        #print('sig1-sig2-bkg like events in RestOfTheBkgs: {}'.format(restBkgsSelected))
        lumiScaling = lambda xsec, lumi, nev : xsec*lumi/nev
        
        nWZsInSig1Node = 0
        nWZsInSig2Node = 0
        nWZsInWZNode   = 0
        nRestBkgsInSig1Node = 0
        nRestBkgsInSig2Node = 0
        nRestBkgsInWZNode   = 0
        for i, item in enumerate(bkgWZsSelected):
            nWZsInSig1Node += item[0]*lumiScaling(WZxsecs[i],lumi,WZnEvts[i])
            nWZsInSig2Node += item[1]*lumiScaling(WZxsecs[i],lumi,WZnEvts[i])
            nWZsInWZNode   += item[2]*lumiScaling(WZxsecs[i],lumi,WZnEvts[i])
        for i, item in enumerate(restBkgsSelected):
            nRestBkgsInSig1Node += item[0]*lumiScaling(RestBKGxsecs[i],lumi,RestBKGnEvts[i])
            nRestBkgsInSig2Node += item[1]*lumiScaling(RestBKGxsecs[i],lumi,RestBKGnEvts[i])
            nRestBkgsInWZNode   += item[2]*lumiScaling(RestBKGxsecs[i],lumi,RestBKGnEvts[i]) 

        #logging.info('nWZsInSig1Node : {}, nWZsInSig2Node : {}, nWZsInWZNode : {}, nRestBkgsInSig1Node : {}, nRestBkgsInSig2Node: {}, nRestBkgsInWZNode : {}'.\
        #             format('%.4f'%nWZsInSig1Node, '%.4f'%nWZsInSig2Node, '%.4f'%nWZsInWZNode, '%.4f'%nRestBkgsInSig1Node, '%.4f'%nRestBkgsInSig2Node, '%.4f'%nRestBkgsInWZNode))

        sig1Node_signifs = []
        sig2Node_signifs = []
        combined_signifs = []
        sig1Node_signifs_Log = []
        sig2Node_signifs_Log = []
        combined_signifs_Log = []

        for i, item in enumerate(couplings) :
            logging.info('Coupling : {}'.format(item))
            XZnEventsInAllNodes = XZ_selectedDict.get(str(item))
            HZnEventsInAllNodes = HZ_selectedDict.get(str(item))
            logging.info('Before LumiScaling :: nXZEvents : {}'.format(XZnEventsInAllNodes[0]+XZnEventsInAllNodes[1]+XZnEventsInAllNodes[2]))
            logging.info('Before LumiScaling :: nXZeventsInSig1Node : {}, xsec : {}, nEventsProduced : {}'.format(XZnEventsInAllNodes[0], allXZxsecs[i], allXZnEvts[i]))
            logging.info('Before LumiScaling :: nXZeventsInSig2Node : {}, xsec : {}, nEventsProduced : {}'.format(XZnEventsInAllNodes[1], allXZxsecs[i], allXZnEvts[i]))
            logging.info('Before LumiScaling :: nHZEvents : {}'.format(HZnEventsInAllNodes[0]+HZnEventsInAllNodes[1]+HZnEventsInAllNodes[2]))
            logging.info('Before LumiScaling :: nHZeventsInSig1Node : {}, xsec : {}, nEventsProduced : {}'.format(HZnEventsInAllNodes[0], allH2Zxsecs[i], allH2ZnEvts[i]))
            logging.info('Before LumiScaling :: nHZeventsInSig2Node : {}, xsec : {}, nEventsProduced : {}'.format(HZnEventsInAllNodes[1], allH2Zxsecs[i], allH2ZnEvts[i]))

            XZinSig1Node = XZnEventsInAllNodes[0]*lumiScaling(allXZxsecs[i], lumi, allXZnEvts[i])
            XZinSig2Node = XZnEventsInAllNodes[1]*lumiScaling(allXZxsecs[i], lumi, allXZnEvts[i])

            HZinSig1Node = HZnEventsInAllNodes[0]*lumiScaling(allH2Zxsecs[i], lumi, allH2ZnEvts[i])
            HZinSig2Node = HZnEventsInAllNodes[1]*lumiScaling(allH2Zxsecs[i], lumi, allH2ZnEvts[i])

            logging.info('After LumiScaling  :: nXZsInSig1Node : {}, nXZsInSig2Node : {}, nHZInSig1Node : {}, nHZInSig2Node: {}'.\
                         format('%.2f'%XZinSig1Node, '%.2f'%XZinSig2Node, '%.2f'%HZinSig1Node, '%.2f'%HZinSig2Node))

            
            sig1Node_signifs.append(estimateSignificance(XZinSig1Node+HZinSig1Node, nWZsInSig1Node+nRestBkgsInSig1Node, isLog=False))
            sig2Node_signifs.append(estimateSignificance(XZinSig2Node+HZinSig2Node, nWZsInSig2Node+nRestBkgsInSig2Node, isLog=False))
            combined_signifs.append(estimateSignificance(XZinSig1Node+HZinSig1Node+XZinSig2Node+HZinSig2Node,
                                                         nWZsInSig2Node+nRestBkgsInSig2Node+nWZsInSig1Node+nRestBkgsInSig1Node, isLog=False))
            sig1Node_signifs_Log.append(estimateSignificance(XZinSig1Node+HZinSig1Node, nWZsInSig1Node+nRestBkgsInSig1Node, isLog=True))
            sig2Node_signifs_Log.append(estimateSignificance(XZinSig2Node+HZinSig2Node, nWZsInSig2Node+nRestBkgsInSig2Node, isLog=True))
            combined_signifs_Log.append(estimateSignificance(XZinSig1Node+HZinSig1Node+XZinSig2Node+HZinSig2Node,
                                                             nWZsInSig2Node+nRestBkgsInSig2Node+nWZsInSig1Node+nRestBkgsInSig1Node, isLog=True))
            
        logging.info("Couplings ===> {}".format([item for item in couplings]))
        logging.info('sig-1 node || significance ==> {}'.format(['%.2f'%item for item in sig1Node_signifs]))
        logging.info('sig-2 node || significance ==> {}'.format(['%.2f'%item for item in sig2Node_signifs]))
        logging.info('sig-1 + sig-2 nodes || significance ==> {}'.format(['%.2f'%item for item in combined_signifs]))

        logging.info('sig-1 node || significance [Log formula] ==> {}'.format(['%.2f'%item for item in sig1Node_signifs_Log]))
        logging.info('sig-2 node || significance [Log formula] ==> {}'.format(['%.2f'%item for item in sig2Node_signifs_Log]))
        logging.info('sig-1 + sig-2 nodes || significance [Log formula]==> {}'.format(['%.2f'%item for item in combined_signifs_Log]))
            
        
        '''
        # More compact way
        allXZsPredClass   = [model.predict_classes(item) for item in allXZsScaled]
        allH2ZsPredClass  = [model.predict_classes(item) for item in allH2ZsScaled]
        bkgWZsPredClass   = [model.predict_classes(item) for item in WZsScaled]
        restBkgsPredClass = [model.predict_classes(item) for item in RestBKGsScaled]
        
        allXZsSelected    = [[(diffCoups == 0).sum(), (diffCoups == 1).sum(), (diffCoups == 2).sum()] for diffCoups in allXZsPredClass]
        allH2ZsSelected   = [[(diffCoups == 0).sum(), (diffCoups == 1).sum(), (diffCoups == 2).sum()] for diffCoups in allH2ZsPredClass]
        bkgWZsSelected    = [[(diffCoups == 0).sum(), (diffCoups == 1).sum(), (diffCoups == 2).sum()] for diffCoups in bkgWZsPredClass]
        restBkgsSelected  = [[(diffCoups == 0).sum(), (diffCoups == 1).sum(), (diffCoups == 2).sum()] for diffCoups in restBkgsPredClass]

        logging.info('Evaluating : allXZs {} allHZs {} WZs {} RestBkgs {}'.format(len(allXZsSelected), len(allH2ZsSelected), len(bkgWZsSelected), len(restBkgsSelected)))
        logging.info('Evaluating : XZxsecs {} HZxsecs {} WZxsecs {} Restxsecs {}'.format(len(allXZxsecs), len(allH2Zxsecs), len(WZxsecs), len(RestBKGxsecs)))

        nWZsInSig1Node = 0
        nWZsInSig2Node = 0
        nRestBkgsInSig1Node = 0
        nRestBkgsInSig2Node = 0
        lumiScaling = lambda xsec, lumi, nev : xsec*lumi/nev
        for i, item in enumerate(bkgWZsSelected):
            nWZsInSig1Node += item[0]*lumiScaling(WZxsecs[i],lumi,WZnEvts[i])
            nWZsInSig2Node += item[1]*lumiScaling(WZxsecs[i],lumi,WZnEvts[i])
        for i, item in enumerate(restBkgsSelected):
            nRestBkgsInSig1Node += item[0]*lumiScaling(RestBKGxsecs[i],lumi,RestBKGnEvts[i])
            nRestBkgsInSig2Node += item[1]*lumiScaling(RestBKGxsecs[i],lumi,RestBKGnEvts[i]) 

        #logging.info('%.4f'%nWZsInSig1Node, '%.4f'%nWZsInSig2Node, '%.4f'%nRestBkgsInSig1Node, '%.4f'%nRestBkgsInSig2Node)

        sig1Node_signifs = []
        sig2Node_signifs = []
        combined_signifs = []
        sig1Node_signifs_Log = []
        sig2Node_signifs_Log = []
        combined_signifs_Log = []
        for i, item in enumerate(allXZsSelected):
            sig1InSig1Node = item[0]*lumiScaling(allXZxsecs[i],lumi,allXZnEvts[i])
            sig1InSig2Node = item[1]*lumiScaling(allXZxsecs[i],lumi,allXZnEvts[i])
            sig2InSig1Node = allH2ZsSelected[i][0]*lumiScaling(allH2Zxsecs[i],lumi,allH2ZnEvts[i])
            sig2InSig2Node = allH2ZsSelected[i][1]*lumiScaling(allH2Zxsecs[i],lumi,allH2ZnEvts[i])

            sig1Node_signifs.append(estimateSignificance(sig1InSig1Node+sig2InSig1Node, nWZsInSig1Node+nRestBkgsInSig1Node, isLog=False))
            sig2Node_signifs.append(estimateSignificance(sig1InSig2Node+sig2InSig2Node, nWZsInSig2Node+nRestBkgsInSig2Node, isLog=False))
            combined_signifs.append(estimateSignificance(sig1InSig2Node+sig2InSig2Node+sig1InSig1Node+sig2InSig1Node,
                                                         nWZsInSig2Node+nRestBkgsInSig2Node+nWZsInSig1Node+nRestBkgsInSig1Node, isLog=False))
            sig1Node_signifs_Log.append(estimateSignificance(sig1InSig1Node+sig2InSig1Node, nWZsInSig1Node+nRestBkgsInSig1Node, isLog=True))
            sig2Node_signifs_Log.append(estimateSignificance(sig1InSig2Node+sig2InSig2Node, nWZsInSig2Node+nRestBkgsInSig2Node, isLog=True))
            combined_signifs_Log.append(estimateSignificance(sig1InSig2Node+sig2InSig2Node+sig1InSig1Node+sig2InSig1Node,
                                                             nWZsInSig2Node+nRestBkgsInSig2Node+nWZsInSig1Node+nRestBkgsInSig1Node, isLog=True))
            
        logging.info('sig-1 node || significance ==> {}'.format(['%.2f'%item for item in sig1Node_signifs]))
        logging.info('sig-2 node || significance ==> {}'.format(['%.2f'%item for item in sig2Node_signifs]))
        logging.info('sig-1 + sig-2 nodes || significance ==> {}'.format(['%.2f'%item for item in combined_signifs]))

        logging.info('sig-1 node || significance [Log formula] ==> {}'.format(['%.2f'%item for item in sig1Node_signifs_Log]))
        logging.info('sig-2 node || significance [Log formula] ==> {}'.format(['%.2f'%item for item in sig2Node_signifs_Log]))
        logging.info('sig-1 + sig-2 nodes || significance [Log formula]==> {}'.format(['%.2f'%item for item in combined_signifs_Log]))
        
        '''
        
if __name__ == "__main__":
    main()
    
