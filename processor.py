import os
import sys
import datetime
import glob
import json
import pickle
import pandas as pd
import numpy as np
import random
import scipy
#################################################################################################################################
#Utils

def getInt(inStr, default):
    try:
        val = int(inStr)
    except:
        val = default
    return val

def getFloat(inStr, default):
    try:
        val = float(inStr)
    except:
        val = default
    return val

def jsonTotxt(data,dataPath,fname):
    for arNum in sorted(data.keys()):
        for date in sorted(data[arNum].keys()):
            fl = open(dataPath + '/%s.txt'%fname,'a')
            fl.write('%d'%arNum)
            fl.write('\t%s'%date.strftime('%Y-%m-%dT%H:%M:%S'))
            valArr = data[arNum][date]
            for val in valArr:
                fl.write('\t%0.6e'%val)
            fl.write('\n')
            fl.close()
    return
    
def txtTojson(dataPath,fname):
    with open(dataPath + '/%s.txt'%fname,'r') as fl:
        txtData = fl.readlines()
    txtData = [x.strip() for x in txtData]
    data = {}
    for entry in txtData:
        splits = entry.split('\t')
        arNum = int(splits[0])
        date = datetime.datetime.strptime(splits[1],'%Y-%m-%dT%H:%M:%S')
        valArr = map(float,splits[2:])
        if arNum not in data:
            data[arNum] = {}
        data[arNum][date] = valArr
    return data

def sharpProcessor(dataPath,fname): #Extracts valid SHARP data from raw data
    '''
    sharp keywords = ['DATE','DATE__OBS','HARPNUM','USFLUX','AREA_ACR','TOTUSJZ','TOTUSJH','TOTPOT','ABSNJZH','SAVNCPP',
                      'R_VALUE','MEANPOT','SHRGT45']
    lorentz keywords = ['DATE','DATE__OBS','HARPNUM','TOTBSQ','TOTFZ']
    '''
    if not os.path.exists(dataPath + '/endPoints.txt'):
        #Load first and last AR observations and maximum observed areas
        jsonFiles = glob.glob(dataPath + '/sharpDates/*.json')
        jsonFiles = sorted(jsonFiles)
        endPoints = {}
        maxAreas = {}
        hmiRes = 0.380 #Mm/Pixel 
        for k,f in enumerate(jsonFiles):
            jsonFile = json.load(open(f))
            sharpCount = jsonFile['count']
            if sharpCount > 0:
                jsonFile = jsonFile['keywords']
                for i in xrange(sharpCount):
                    arNum = int(jsonFile[1]['values'][i])
                    date = datetime.datetime.strptime(jsonFile[0]['values'][i][:-4],'%Y-%m-%dT%H:%M:%S')
                    rem = (date.minute*60 + date.second) % 720

                    if rem!= 0:
                        date = date + datetime.timedelta(seconds = 720 - rem)
                    arArea = jsonFile[2]['values'][i]
                    if arArea != 'MISSING':
                        arArea = float(arArea)
                        if not np.isnan(arArea):
                            if arNum not in maxAreas:
                                maxAreas[arNum] = []
                            maxAreas[arNum].append(arArea)
                    if arNum not in endPoints:
                        endPoints[arNum] = []
                    endPoints[arNum].append(date)
        for arNum in endPoints:
            endPoints[arNum] = sorted(endPoints[arNum])
            endDates = [endPoints[arNum][0],endPoints[arNum][-1]]
            endPoints[arNum] = endDates
        for arNum in maxAreas:
            maxAreas[arNum] = max(maxAreas[arNum])* hmiRes * hmiRes
        pickle.dump(endPoints,open(dataPath + '/endPoints.txt','w'))
        pickle.dump(maxAreas,open(dataPath + '/maxAreas.txt','w'))
        print 'Downloaded first and last AR observations and AR maximum observed area.'
        
    if os.path.exists(dataPath + '/%s.txt'%fname):
        return txtTojson(dataPath,fname)
    else:    
        jsonFiles = glob.glob(dataPath + '/raw-sharp/*.json')
        jsonFiles = sorted(jsonFiles)
        for i in xrange(len(jsonFiles)):
            jsonFiles[i] = jsonFiles[i].split('/')[-1]
            
        data = {}
        for k,f in enumerate(jsonFiles):
            sys.stdout.write('\rPreprocessing SHARPs data %0.2f%%'%(float(k+1)*100/len(jsonFiles)))
            sys.stdout.flush()
            sharpJson = json.load(open(dataPath + '/raw-sharp/' + f))
            lorentzJson = json.load(open(dataPath + '/raw-lorentz/' + f))
            sharpCount = sharpJson['count']
            lorentzCount = lorentzJson['count']
            sharpJson = sharpJson['keywords']
            lorentzJson = lorentzJson['keywords']
            indices = []
            if sharpCount != lorentzCount:
                sharpTuples = zip(sharpJson[2]['values'],sharpJson[1]['values'])
                lorentzTuples = zip(lorentzJson[2]['values'],lorentzJson[1]['values']) 
                commons = set(sharpTuples).intersection(lorentzTuples)
                sharpIndices = [sharpTuples.index(x) for x in commons]
                lorentzIndices = [lorentzTuples.index(x) for x in commons]
                assert len(sharpIndices) == len(lorentzIndices)
                for i in xrange(len(sharpIndices)):
                    indices.append((sharpIndices[i],lorentzIndices[i]))
            else:
               for i in xrange(sharpCount):
                   indices.append((i,i))
                    
            for (i,j) in indices:
                arNum = int(sharpJson[2]['values'][i])
                date = datetime.datetime.strptime(sharpJson[1]['values'][i][:-4],'%Y-%m-%dT%H:%M:%S')
                rem = (date.minute*60 + date.second) % 720 
                if rem!= 0:
                    date = date + datetime.timedelta(seconds = 720 - rem)
                sharpCalc = datetime.datetime.strptime(sharpJson[0]['values'][i][:-1],'%Y-%m-%dT%H:%M:%S') 
                lorentzCalc = datetime.datetime.strptime(lorentzJson[0]['values'][j][:-1],'%Y-%m-%dT%H:%M:%S') 
                tmpArr = []
                flag = 1
                for w in xrange(3,13):
                    if sharpJson[w]['values'][i] == 'MISSING':
                        flag = 0 
                        break
                    else:
                        val = getFloat(sharpJson[w]['values'][i], 0.0)
                        tmpArr.append(val)
                for w in xrange(3,5):
                    if lorentzJson[w]['values'][j] == 'MISSING':
                        flag = 0
                        break
                    else:
                        val = getFloat(lorentzJson[w]['values'][j], 0.0)
                        tmpArr.append(val)
                tmpArr.append(sharpCalc)
                tmpArr.append(lorentzCalc)
                if flag:
                    if arNum not in data:
                        data[arNum] = {}
                    if date not in data[arNum]:
                        data[arNum][date] = tmpArr
                    else:
                        if sharpCalc > data[arNum][date][-2]:
                            data[arNum][date][:10] = tmpArr[:10]
                            data[arNum][date][12] = sharpCalc
                        if  lorentzCalc > data[arNum][date][-1]:
                            data[arNum][date][10:12] = tmpArr[10:12]
                            data[arNum][date][13] = lorentzCalc
        for arNum in data:
            for date in data[arNum]:
                data[arNum][date] = data[arNum][date][:-2]
        jsonTotxt(data,dataPath,fname)
        
        return data
    
    
def noaa_to_harp(mapper, noaa):
    idx = mapper[mapper['NOAA_ARS'].str.contains(str(int(noaa)))]
    return None if idx.empty else idx.HARPNUM.values[0]   

def flareReader(dataPath,fname):  #Reads flare information from HEK dataset
    if os.path.exists(dataPath + '/%s.txt'%fname):
        with open(dataPath + '/%s.txt'%fname,'r') as fl:
            txtData = fl.readlines()
        txtData = [x.strip() for x in txtData]
        data = {}
        for entry in txtData:
            splits = entry.split('\t')
            arNum = int(splits[0])
            data[arNum] = []
            for item in splits[1:]:
                date = datetime.datetime.strptime(item.split(',')[0],'%Y-%m-%dT%H:%M:%S')
                val = float(item.split(',')[1])
                data[arNum].append([date,val])
    else:
        from sunpy.net import hek
        from sunpy.time import TimeRange
        import re
        
    	mapper = pd.read_csv(dataPath + '/all_harps_with_noaa_ars.txt',sep=' ')

    	start_date = '2010-05-01'
    	end_date = '2016-04-30' 
  
    	timerange = TimeRange(start_date, end_date)
    	goes_class_filter = 'M1'
    	# use HEK module to search for GOES events
    	client = hek.HEKClient()
    	event_type = 'FL'
    	tstart = timerange.start
    	tend = timerange.end

    	result = client.search(hek.attrs.Time(tstart, tend),
    	                      hek.attrs.EventType(event_type),
    	                      hek.attrs.FL.GOESCls > goes_class_filter,
    	                      hek.attrs.OBS.Observatory == 'GOES')

    	data = {}
    	for r in result:
    	    noaa = getInt(r['ar_noaanum'],-1)
    	    if noaa < 10000:
    	        noaa += 10000
    	    arNum = noaa_to_harp(mapper, noaa)
    	    if arNum:
                if arNum not in data:
                    data[arNum] = [] 
    	        date = datetime.datetime.strptime(r['event_peaktime'],'%Y-%m-%dT%H:%M:%S')
    	        rem = (date.minute*60 + date.second) % 720 
    	        #Rounding off timestamp to nearest 12 minute multiple
                if rem != 0:
    	            date = date + datetime.timedelta(seconds = 720 - rem)
                    
    	        if r['fl_goescls'][0] == 'M':  
                    flVal = 1
    	            is_duplicate = 0
    	            for i in xrange(len(data[arNum])):
    	                diff = date - data[arNum][i][0]
    	                diff = abs(diff.days*24.0*60.0 + float(diff.seconds)/60.0)
    	                if diff <= 12.0:
    	                    is_duplicate = 1
    	            if is_duplicate == 0:
    	                data[arNum].append([date,flVal])
                        
    	        elif r['fl_goescls'][0] == 'X':  
                    flVal = 2
    	            is_duplicate = 0
    	            for i in xrange(len(data[arNum])):
    	                diff = date - data[arNum][i][0]
    	                diff = abs(diff.days*24.0*60.0 + float(diff.seconds)/60.0)
    	                if diff <= 12.0:
    	                    is_duplicate = 1
    	            if is_duplicate == 0:
    	                data[arNum].append([date,flVal])
                 
        for arNum in sorted(data.keys()):
            data[arNum] = sorted(data[arNum]) 
            with open(dataPath + '/%s.txt'%fname,'a') as fl:
                fl.write('%d'%arNum)
                for item in data[arNum]:
                    fl.write('\t%s,%0.3f'%(item[0].strftime('%Y-%m-%dT%H:%M:%S'),item[1]))
                fl.write('\n')
    print 'Downloaded M- and X-class flare data.'
    return data

#############################################################################################################################
#Identifying flaring and non-flaring ARs over cutoff area 25 Mm^2 
USFLUX = 0
AREA = 1
TOTUSJZ = 2 
TOTUSJH = 3
TOTPOT = 4
ABSNJZH = 5
SAVNCPP = 6
R_VALUE = 7
MEANPOT = 8
SHRGT45 = 9
TOTBSQ = 10
TOTFZ = 11  
xInd = [USFLUX,AREA,TOTUSJZ,TOTUSJH,TOTPOT,ABSNJZH,SAVNCPP,R_VALUE,MEANPOT,SHRGT45,TOTBSQ,TOTFZ]
xSym = ['USFLUX','AREA','TOTUSJZ','TOTUSJH','TOTPOT','ABSNJZH','SAVNCPP','R_VALUE','MEANPOT','SHRGT45','TNameQ','TOTFZ']

def isTrain(lastTS):
    if lastTS.year <= 2013:
        return 1
    else:
        return 0

def isTest(lastTS):
    if lastTS < datetime.datetime.strptime('2016-04-13','%Y-%m-%d'):
        return 1
    else:
        return 0

def loadFirstObLongitudes(dataPath):
    filePath = dataPath + '/longitudes.txt'
    with open(filePath,'r') as fl:
        txtData = fl.readlines()
    txtData = [x.strip() for x in txtData]
    data = {}
    for item in txtData:
        splits = item.split('\t')
        if len(splits):
            data[int(splits[0])] = float(splits[1])
    return data

def loadEndPoints(dataPath):
    filePath = dataPath + '/endPoints.txt'
    with open(filePath,'r') as fl:
        txtData = fl.readlines()
    txtData = [x.strip() for x in txtData]
    data = {}
    for item in txtData:
        splits= item.split('\t')
        data[int(splits[0])] = [datetime.datetime.strptime(splits[1],'%Y-%m-%dT%H:%M:%S'),
                                datetime.datetime.strptime(splits[2],'%Y-%m-%dT%H:%M:%S')]
    return data

def loadMaxareas(dataPath):
    filePath = dataPath + '/maxAreas.txt'
    hmiRes = 0.380 #Mm/Pixel
    if os.path.exists(filePath):  
        with open(filePath,'r') as fl:
            txtData = fl.readlines()
        txtData = [x.strip() for x in txtData]
        data = {}
        for item in txtData:
            splits= item.split('\t')
            data[int(splits[0])] = float(splits[1])
    
    return data

def categorizeARs(dataPath,sharpName,flareName,cutOffArea =25):
    
    print 'Loading data files..'
    sharpData = sharpProcessor(dataPath, sharpName)
    flareData = flareReader(dataPath, flareName)
    longitude = loadFirstObLongitudes(dataPath)
    endPoints = pickle.load(open(dataPath + '/endPoints.txt','r'))
    maxArea = pickle.load(open(dataPath + '/maxAreas.txt','r'))
    
    # Assembling flaring and nonflaring ARs HARP numbers
    print 'Splitting Train, Test and Emerging ARs..'
    flaring_ARs_train = []
    nonflaring_ARs_train = []
    
    flaring_ARs_test = []
    nonflaring_ARs_test = []
    
    emerging_flaring_ARs = []
    emerging_nonflaring_ARs = []

    for arNum in sharpData:
        if maxArea[arNum] > cutOffArea:
            lastTS = endPoints[arNum][1]
            firstTS = endPoints[arNum][0]
            if arNum in flareData:
                if isTrain(lastTS):
                    if abs(longitude[arNum]) <= 60:
                        emerging_flaring_ARs.append(arNum)
                    else:
                        flaring_ARs_train.append(arNum)
                elif isTest(lastTS):
                    if abs(longitude[arNum]) <= 60:
                        emerging_flaring_ARs.append(arNum)
                    else:
                        flaring_ARs_test.append(arNum)
            else:
                if isTrain(lastTS):
                    if abs(longitude[arNum]) <= 60:
                        emerging_nonflaring_ARs.append(arNum)
                    else:
                        nonflaring_ARs_train.append(arNum)
                elif isTest(lastTS):
                    if abs(longitude[arNum]) <= 60:
                        emerging_nonflaring_ARs.append(arNum)
                    else:
                        nonflaring_ARs_test.append(arNum)
             
    np.savetxt(dataPath + '/Test_Flaring_ARs.txt',flaring_ARs_test,fmt='%d')
    np.savetxt(dataPath + '/Test_NonFlaring_ARs.txt',nonflaring_ARs_test,fmt='%d')
    np.savetxt(dataPath + '/Train_Flaring_ARs.txt',flaring_ARs_train,fmt='%d')
    np.savetxt(dataPath + '/Train_NonFlaring_ARs.txt',nonflaring_ARs_train,fmt='%d')
    np.savetxt(dataPath + '/Emerging_Flaring_ARs.txt',emerging_flaring_ARs,fmt='%d')
    np.savetxt(dataPath + '/Emerging_NonFlaring_ARs.txt',emerging_nonflaring_ARs,fmt='%d')
    
    TestFlaresM = 0
    TestFlaresX = 0
    for arNum in flaring_ARs_test:
        for item in flareData[arNum]:
            if item[1] == 2:
                TestFlaresX += 1
            elif item[1] == 1:
                TestFlaresM += 1
    TrainFlaresM = 0
    TrainFlaresX = 0
    for arNum in flaring_ARs_train:
        for item in flareData[arNum]:
            if item[1] == 2:
                TrainFlaresX += 1
            elif item[1] == 1:
                TrainFlaresM += 1
    EFlaresM = 0
    EFlaresX = 0
    for arNum in emerging_flaring_ARs:
        for item in flareData[arNum]:
            if item[1] == 2:
                EFlaresX += 1
            elif item[1] == 1:
                EFlaresM += 1
    
    print 'Test Data: #Flaring ARs = %d, #Nonflaring ARs = %d, #M-flares=%d, #X-flares = %d'% \
           (len(flaring_ARs_test),len(nonflaring_ARs_test),TestFlaresM,TestFlaresX)
    print 'Train Data: #Flaring ARs = %d, #Nonflaring ARs = %d, #M-flares=%d, #X-flares = %d'% \
           (len(flaring_ARs_train),len(nonflaring_ARs_train),TrainFlaresM,TrainFlaresX)
    print 'Emerging ARs Data: #Flaring ARs = %d, #Nonflaring ARs = %d, #M-flares=%d, #X-flares = %d'% \
           (len(emerging_flaring_ARs),len(emerging_nonflaring_ARs),EFlaresM,EFlaresX)
    print 'Done.'
    return
    
################################################################################################################################
#Collecting AR observations for training with one observation per time history
def getNormParas(dataPath,sharpName):
    #Calculating parameters to standardize SHARP feature vectors
    if os.path.exists(dataPath + '/means.txt'):
        means = np.loadtxt(dataPath + '/means.txt')
        stds =  np.loadtxt(dataPath + '/stds.txt')     
    else:
        sharpData = sharpProcessor(dataPath, sharpName)
        trainARs = np.loadtxt(dataPath + '/Train_Flaring_ARs.txt',dtype=np.int).tolist()
        trainARs += np.loadtxt(dataPath + '/Train_NonFlaring_ARs.txt',dtype=np.int).tolist()
        means = []
        stds = []
        for i in xInd:
            tmpArr = []
            for arNum in trainARs:
                for date in sharpData[arNum]:
                    tmpArr.append(sharpData[arNum][date][i])
            means.append(np.mean(tmpArr))
            stds.append(np.std(tmpArr))
        means = np.array(means)
        stds = np.array(stds)
        np.savetxt(dataPath + '/means.txt',means)
        np.savetxt(dataPath + '/stds.txt',stds)
    return means,stds

def getFlareDates(ARFlareData):
    #timestamps of M- and X-class flare dates of ARs
    dates = []
    for item in ARFlareData:
        dates.append(item[0])
    return sorted(dates)

def getTrainingData(dataPath, sharpName, flareName, NCV=10):
    sharpData = sharpProcessor(dataPath, sharpName)
    flareData = flareReader(dataPath, flareName)
    endPoints = pickle.load(open(dataPath + '/endPoints.txt','r'))
    
    flaring_ARs_train = np.loadtxt(dataPath + '/Train_Flaring_ARs.txt',dtype=np.int).tolist()
    nonflaring_ARs_train = np.loadtxt(dataPath + '/Train_NonFlaring_ARs.txt',dtype=np.int).tolist()
    
    means, stds = getNormParas(dataPath, sharpName)
    
    TW = 72 # Hours
     
    trainObs = {}   #Collecting observations to be included for training for each training AR
    
    for arNum in flaring_ARs_train:
        flaredates = getFlareDates(flareData[arNum])
        firstTS = endPoints[arNum][0]
        lastTS = endPoints[arNum][1]
        
        date_pre = flaredates[0] - datetime.timedelta(seconds = TW*3600.0)
        date_post = flaredates[-1] + datetime.timedelta(seconds = TW*3600.0)
        if date_pre < firstTS:
           date_pre = firstTS
        if date_post > lastTS:
           date_post = lastTS
       
        #Excluding flares outside sharp observation window
        for i in xrange(len(flaredates)):
            if flaredates[i] > date_pre:
                break
        flaredates = flaredates[i:]
    
        for i in xrange(len(flaredates)-1,-1,1):
            if flaredates[i] < date_post:
                break
        flaredates = flaredates[:i+1]
       
        flaredates = [date_pre] + flaredates + [date_post]
    
        trainObs[arNum] = []
        for i in xrange(1,len(flaredates)-1):
            date = flaredates[i] - datetime.timedelta(seconds = TW*3600)
            to_date = flaredates[i] +  datetime.timedelta(seconds = TW*3600)
            if date < flaredates[i-1]:
                date = flaredates[i-1] + datetime.timedelta(seconds = 720)
            if to_date > flaredates[i+1]:
                to_date = flaredates[i+1]
    
            good_dates = []
            while date < to_date:
                if date in sharpData[arNum]:
                    good_dates.append(date)
                date += datetime.timedelta(seconds = 720)
            i = 0
            while i<len(good_dates):
                trainObs[arNum].append(good_dates[i])
                if abs(len(good_dates)/2 - i) <= 30:
                    i += 8  #corresponding to approximately 96 mins
                else:
                    i += 72 #Corresponding to approximately 864 mins
        trainObs[arNum] = list(set(trainObs[arNum]))
    
    for arNum in nonflaring_ARs_train:
        firstTS = endPoints[arNum][0]
        lastTS = endPoints[arNum][1]
        obsTS = sorted(sharpData[arNum].keys())
        mid_date = obsTS[len(obsTS)/2] 
        
        #Considering average lifetime of an AR of approximately 12 days
        date_pre = mid_date - datetime.timedelta(seconds = 120.0*3600.0)
        date_post = mid_date + datetime.timedelta(seconds = 120.0*3600.0)
        if date_pre < firstTS:
           date_pre = firstTS
        if date_post > lastTS:
           date_post = lastTS
        for i in xrange(len(obsTS)):
            if obsTS[i] >= date_pre:
                break
        obsTS = obsTS[i:]
        for i in xrange(len(obsTS)-1,-1,-1):
            if obsTS[i] <= date_post:
                break
        obsTS = obsTS[:i+1]
        good_dates = []
        for date in obsTS:
            if date in sharpData[arNum]:
                good_dates.append(date)
        trainObs[arNum] = []
        for i in xrange(0,len(good_dates),75):  #Selecting an observation approximately every 900 mins
            trainObs[arNum].append(good_dates[i])
        trainObs[arNum] = list(set(trainObs[arNum]))
    
    #Creating cross-validation training sets
    sectF = len(flaring_ARs_train)/NCV
    sectNF = len(nonflaring_ARs_train)/NCV
    random.shuffle(flaring_ARs_train)
    random.shuffle(nonflaring_ARs_train) 
    for ver in xrange(NCV):
        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        for no in xrange(0,ver*sectF):
            arNum = flaring_ARs_train[no] 
            for date in trainObs[arNum]:
                xtrain.append(sharpData[arNum][date][:len(xInd)])
                ytrain.append(1)
        for no in xrange((ver+1)*sectF,len(flaring_ARs_train)):
            arNum = flaring_ARs_train[no] 
            for date in trainObs[arNum]:
                xtrain.append(sharpData[arNum][date][:len(xInd)])
                ytrain.append(1)
        for no in xrange(ver*sectF,(ver+1)*sectF):
            arNum = flaring_ARs_train[no] 
            for date in trainObs[arNum]:
                xtest.append(sharpData[arNum][date][:len(xInd)])
                ytest.append(1)
        for no in xrange(0,ver*sectNF):
            arNum = nonflaring_ARs_train[no] 
            for date in trainObs[arNum]:
                xtrain.append(sharpData[arNum][date][:len(xInd)])
                ytrain.append(0)
        for no in xrange((ver+1)*sectNF,len(nonflaring_ARs_train)):
            arNum = nonflaring_ARs_train[no] 
            for date in trainObs[arNum]:
                xtrain.append(sharpData[arNum][date][:len(xInd)])
                ytrain.append(0)
        for no in xrange(ver*sectNF, (ver+1)*sectNF):
            arNum = nonflaring_ARs_train[no] 
            for date in trainObs[arNum]:
                xtest.append(sharpData[arNum][date][:len(xInd)])
                ytest.append(0)
        
        xtrain = np.array(xtrain, dtype=np.float)
        xtest = np.array(xtest, dtype=np.float)
        ytrain = np.array(ytrain, dtype=np.int)
        ytest = np.array(ytest, dtype=np.int)
        
        for i in xrange(len(xInd)):
            xtrain[:,i] -= means[i]
            xtrain[:,i] /= stds[i]
            xtest[:,i] -= means[i]
            xtest[:,i] /= stds[i]
           
        trainDataPath = dataPath + '/training'
        if not os.path.exists(trainDataPath):
            os.makedirs(trainDataPath)
      
        with open(trainDataPath + '/stats.txt', 'a') as fl:
            fl.write('%d\ttest positives = %d\ttest negatives = %d\ttrain positives = %d\ttrain negative = %d\n'%(ver,(ytest == 1).sum(),(ytest == 0).sum(), (ytrain == 1).sum(), (ytrain == 0).sum()))
            
        assert len(ytrain) == len(xtrain)
        assert len(ytest) == len(xtest)
    
        xtrain.dump(trainDataPath + '/xtrain%d.dat'%(ver)) 
        xtest.dump(trainDataPath + '/xtest%d.dat'%(ver)) 
        ytrain.dump(trainDataPath + '/ytrain%d.dat'%(ver)) 
        ytest.dump(trainDataPath + '/ytest%d.dat'%(ver))

    return
##############################################################################################################################
# pre-post flare observations for training/unseen data by aligning all flare data

def dumpAlignedPrePostObs(arNum,obS,dates,storePath):
    TW = 144 #hours of observation: 72 hours pre-flare and 72 hours post flare
    NOBS = TW*5
    Nfeat = len(xInd) + 1
    obsTS = sorted(obS.keys())
    firstTS = obsTS[0]
    lastTS = obsTS[-1]
    for i in xrange(len(dates)):
        if dates[i] > firstTS:
            break
    dates = dates[i:]
    for i in xrange(len(dates)-1, -1, -1):
        if dates[i] < lastTS:
            break
    dates = dates[:i+1]
    dates = [firstTS] + dates + [lastTS]
 
    count = 0 #number of aligned pre-post flare segments for the AR
    for i in xrange(1,len(dates)-1): 
        date = dates[i] 
        diff = dates[i] - dates[i-1]
        X = np.zeros((NOBS, Nfeat),dtype=np.float)
                 
        #Pre flare observations
        diff = diff.total_seconds()/3600.0
        if diff < 72*(1+(i>1)): #Check if difference between flare and previous flare is less than 144 hours, except for first flare
            to_date = dates[i] - datetime.timedelta(seconds = int(diff*5/(1+(i>1)))*720) 
        else:
            to_date = dates[i] - datetime.timedelta(seconds = 72*3600) 
        counter = 72*5 - 1
        while date >= to_date and counter >= 0:
            if date in obS:
                X[counter][:Nfeat-1] = np.array(obS[date][:Nfeat-1])
                #X[counter][:Nfeat-1] -= means
                #X[counter][:Nfeat-1] /= stds
                X[counter][Nfeat-1] = 1.0
            counter -= 1
            date -= datetime.timedelta(seconds = 720)
 
        #post flare
        date = dates[i] + datetime.timedelta(seconds = 720) 
        diff = dates[i+1] - dates[i]
        diff = diff.total_seconds()/3600.0
        if diff < 72*(1+(i<len(dates)-2)): #check difference between flare and next flare is less than 144 hours except last one
            to_date = dates[i] + datetime.timedelta(seconds = int(diff*5/(1+(i<len(dates)-2)))*720)
        else:
            to_date = dates[i] + datetime.timedelta(seconds = 72*3600) 
        counter = 72*5
        while date <= to_date and counter < 144*5:
            if date in obS:
                X[counter][:Nfeat-1] = np.array(obS[date][:Nfeat-1])
                #X[counter][:Nfeat-1] -= means
                #X[counter][:Nfeat-1] /= stds
                X[counter][Nfeat-1] = 1.0
            counter += 1
            date += datetime.timedelta(seconds = 720)
 
        if X[:,Nfeat-1].sum() > 0.5*NOBS: #Store only when more that half of the total observations are available for each flare
            X.dump(storePath + '/%d_%d.dat'%(arNum,count))
        count += 1 
    return 

def getFlareQuietObs(obS, dates, ARLastOb):
    #Samples temporally separated from flares by more than 72 hours.
    #We consider such observations post first flare on flaring ARs.
    #We collect time series on continous observations on flaring ARs of 72 hours.
    flareQuietObs = []
    for i in xrange(len(dates)-1, -1, -1):
        if dates[i] < ARLastOb:
            break
    dates = dates[:i+1]
    dates = dates + [ARLastOb]

    for i in xrange(0,len(dates)-1):
        date = dates[i] + datetime.timedelta(seconds = 72*3600.0)
        endDate = dates[i+1] - datetime.timedelta(seconds = 72*3600.0)
        while ((endDate - date).days*24.0 + float((endDate - date).seconds)/3600.0) >= 72:
            counter = 0
            while counter < 72*5:
                if date in obS:
                    flareQuietObs.append(date) 
                counter += 1
                date += datetime.timedelta(seconds = 720)
    return flareQuietObs

def countSamples(dataPath):

    resultPath = 'Results/counts'
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
   
    flaringStorePath = dataPath + '/aligned/flaring'
    nonflaringStorePath = dataPath + '/nonflaring'

    #count quiet samples
    with open(resultPath + '/quiet','w') as fl:
        data = np.load(flaringStorePath + '/quiet_train.dat')
        fl.write('train\t%d\n'%len(data))
        data = np.load(flaringStorePath + '/quiet_test.dat')
        fl.write('test\t%d\n'%len(data))

    #count nonflaring samples
    with open(resultPath + '/nonflaring','w') as fl:
        data = np.load(nonflaringStorePath + '/train.dat')
        fl.write('train\t%d\n'%len(data))
        data = np.load(nonflaringStorePath + '/test.dat')
        fl.write('test\t%d\n'%len(data))
        data = np.load(nonflaringStorePath + '/emerging.dat')
        fl.write('emerging\t%d\n'%len(data))

    #count train pre post flaring samples
    files = glob.glob(flaringStorePath + '/train/*.dat')
    avgObsSpan = 30 #two hours * 5 obervations per hour
    counts = np.zeros((144*5/avgObsSpan),dtype=np.int)
    for f in files:
        data = np.load(f)
        j = 0
        for i in xrange(0,len(data),avgObsSpan):
            counts[j] += int(data[i:i+avgObsSpan,-1].sum() > 0)
            j += 1
    counts.dump(resultPath + '/train.dat') 

    #count test pre post flaring samples
    files = glob.glob(flaringStorePath + '/test/*.dat')
    avgObsSpan = 30 #two hours * 5 obervations per hour
    counts = np.zeros((144*5/avgObsSpan),dtype=np.int)
    for f in files:
        data = np.load(f)
        j = 0
        for i in xrange(0,len(data),avgObsSpan):
            counts[j] += int(data[i:i+avgObsSpan,-1].sum() > 0)
            j += 1
    counts.dump(resultPath + '/test.dat') 

    #count emerging  pre first flaring samples
    files = glob.glob(flaringStorePath + '/emerging/*.dat')
    avgObsSpan = 30 #two hours * 5 obervations per hour
    counts = np.zeros((72*5/avgObsSpan),dtype=np.int)
    for f in files:
        data = np.load(f)
        j = 0
        for i in xrange(0,len(data),avgObsSpan):
            counts[j] += int(data[i:i+avgObsSpan,-1].sum() > 0)
            j += 1
    counts.dump(resultPath + '/emerging.dat') 

    return

def getAlignedSamples(dataPath, sharpName, flareName):
    
    sharpData = sharpProcessor(dataPath, sharpName)
    flareData = flareReader(dataPath, flareName)
    endPoints = pickle.load(open(dataPath + '/endPoints.txt','r'))
    
    flaring_ARs_train = np.loadtxt(dataPath + '/Train_Flaring_ARs.txt',dtype=np.int).tolist()
    nonflaring_ARs_train = np.loadtxt(dataPath + '/Train_NonFlaring_ARs.txt',dtype=np.int).tolist()
    flaring_ARs_test = np.loadtxt(dataPath + '/Test_Flaring_ARs.txt',dtype=np.int).tolist()
    nonflaring_ARs_test = np.loadtxt(dataPath + '/Test_NonFlaring_ARs.txt',dtype=np.int).tolist()
    emerging_flaring_ARs = np.loadtxt(dataPath + '/Emerging_Flaring_ARs.txt',dtype=np.int).tolist()
    emerging_nonflaring_ARs = np.loadtxt(dataPath + '/Emerging_NonFlaring_ARs.txt',dtype=np.int).tolist()
      
    #All emerging ARs are also included in test data
    flaring_ARs_test += emerging_flaring_ARs
    nonflaring_ARs_test += emerging_nonflaring_ARs
    
    Nfeat = len(xInd) + 1  #features + flag to indicate whether the observation is missing

    flaringStorePath = dataPath + '/aligned/flaring'
    nonflaringStorePath = dataPath + '/nonflaring'
    if not os.path.exists(flaringStorePath):
        os.makedirs(flaringStorePath)
    if not os.path.exists(nonflaringStorePath):
        os.makedirs(nonflaringStorePath)

    # Pre first flare (72 hours) for emerging flaring ARs
    print 'Processing Emerging ARs data'
    if not os.path.exists(flaringStorePath + '/emerging'):
        os.makedirs(flaringStorePath + '/emerging')
    TW = 72
    NOBS = TW*5
    for arNum in emerging_flaring_ARs:
        dates = getFlareDates(flareData[arNum])
        obS = sorted(sharpData[arNum].keys())
        firstTS = obS[0]
        lastTS = obS[-1]
        date = dates[0] 
        X = np.zeros((NOBS, Nfeat),dtype=np.float)
        counter = NOBS - 1
        while date >= firstTS and counter >= 0:
            if date in sharpData[arNum]:
                X[counter][:-1] = np.array(sharpData[arNum][date][:Nfeat-1])
                X[counter][-1] = 1.0
            counter -= 1
            date -= datetime.timedelta(seconds = 720)
        if X[:,Nfeat-1].sum() > 0.0: #Store if at least one valid observation is available
            X.dump(flaringStorePath + '/emerging/%d_0.dat'%(arNum))

    #collecting all observations from emerging nonflaring ARs         
    X = [] 
    for arNum in emerging_nonflaring_ARs:
        for date in sharpData[arNum]:
            X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(nonflaringStorePath + '/emerging.dat')

    print 'Processing Pre-Post Flaring ARs data'
    #PrePost Flare Observations from training and test data as well as quiet samples which are temporally separated from flares by at least 72 hours
    testStorePath = flaringStorePath + '/test'
    trainStorePath = flaringStorePath + '/train'
    if not os.path.exists(testStorePath):
        os.makedirs(testStorePath)
        os.makedirs(trainStorePath)

    X = [] #collecting all observations from training flaring ARs separated from flares by more than 72 hours 
    for arNum in flaring_ARs_train:
        dumpAlignedPrePostObs(arNum,sharpData[arNum],getFlareDates(flareData[arNum]),trainStorePath)
        quietObs = getFlareQuietObs(sharpData[arNum].keys(),getFlareDates(flareData[arNum]),endPoints[arNum][1])
        for date in quietObs:
            X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(flaringStorePath + '/quiet_train.dat')
 
    X = [] #collecting all observations from test flaring ARs separated from flares by more than 72 hours 
    for arNum in flaring_ARs_test:
        dumpAlignedPrePostObs(arNum,sharpData[arNum],getFlareDates(flareData[arNum]),testStorePath)
        quietObs = getFlareQuietObs(sharpData[arNum].keys(),getFlareDates(flareData[arNum]),endPoints[arNum][1])
        for date in quietObs:
            X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(flaringStorePath + '/quiet_test.dat')
                     
    print 'Processing  nonflaring ARs data'
    X = [] #collecting all observations from training nonflaring ARs
    #We consider observations from middle 72 hours of the observation span for a non-flaring AR.
    for arNum in nonflaring_ARs_train:
        obDates = sorted(sharpData[arNum].keys())
        mindate = obDates[len(obDates)/2] - datetime.timedelta(seconds=36*3600)
        maxdate = obDates[len(obDates)/2] + datetime.timedelta(seconds=36*3600)
        for date in sharpData[arNum]:
            if date <= maxdate and date >= mindate:
                X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(nonflaringStorePath + '/train.dat')

    X = [] #collecting all observations from test nonflaring ARs        
    for arNum in nonflaring_ARs_test:
        obDates = sorted(sharpData[arNum].keys())
        mindate = obDates[len(obDates)/2] - datetime.timedelta(seconds=36*3600)
        maxdate = obDates[len(obDates)/2] + datetime.timedelta(seconds=36*3600)
        for date in sharpData[arNum]:
            if date <= maxdate and date >= mindate:
                X.append(sharpData[arNum][date][:Nfeat-1])
    X = np.array(X,dtype=np.float)
    X.dump(nonflaringStorePath + '/test.dat')

    print 'Counting Samples'
    countSamples(dataPath) 
    print 'Done'
    return

def getFeatureCorrelations(dataPath,sharpName):
    #Return all standardized features from SHARP data to obtain correlation between different features.
    if os.path.exists('Results/Corr.dat'):
        return np.load('Results/Corr.dat')
    else:
        xOrder = [USFLUX,AREA,TOTUSJZ,TOTUSJH,TOTPOT,TOTBSQ,ABSNJZH,SAVNCPP,R_VALUE,MEANPOT,SHRGT45,TOTFZ]
        nFeatures = len(xOrder)
        sharpData = sharpProcessor(dataPath, sharpName)
        
        flaring_ARs_train = np.loadtxt(dataPath + '/Train_Flaring_ARs.txt',dtype=np.int).tolist()
        nonflaring_ARs_train = np.loadtxt(dataPath + '/Train_NonFlaring_ARs.txt',dtype=np.int).tolist()
        flaring_ARs_test = np.loadtxt(dataPath + '/Test_Flaring_ARs.txt',dtype=np.int).tolist()
        nonflaring_ARs_test = np.loadtxt(dataPath + '/Test_NonFlaring_ARs.txt',dtype=np.int).tolist()
        emerging_flaring_ARs = np.loadtxt(dataPath + '/Emerging_Flaring_ARs.txt',dtype=np.int).tolist()
        emerging_nonflaring_ARs = np.loadtxt(dataPath + '/Emerging_NonFlaring_ARs.txt',dtype=np.int).tolist()
       
        allARs = flaring_ARs_train + nonflaring_ARs_train
        allARs += flaring_ARs_test + nonflaring_ARs_test
        allARs += emerging_flaring_ARs + emerging_nonflaring_ARs
        
        X = []
        for arNum in allARs:
            for date in sharpData[arNum]:
                X.append(sharpData[arNum][date][:nFeatures])
        X = np.array(X)
        X[:,:]=X[:,xOrder]
        Covr = np.zeros((nFeatures,nFeatures),dtype=np.float)
        for i in xrange(nFeatures):
            for j in xrange(nFeatures):
                Covr[i,j], pval = scipy.stats.pearsonr(X[:,i],X[:,j])
                Covr[i,j] = abs(Covr[i,j])
        Covr /= np.max(Covr)
        Covr.dump('Results/Corr.dat')
        return  Covr
 
###################################################################################################################################################
