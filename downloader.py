#Download SHARP keywords from series hmi.sharp_cea_720s and Lorentz force data series cgem.Lorentz  
# available at http://jsoc.stanford.edu
import datetime
import os
import time
#import urllib2
import requests
import json
def downloadJson(outpath,url):
    if not os.path.exists(outpath):
        print 'Downloading from %s ...' % url
        response = requests.get(url).json()
        with open(outpath, 'w') as fp:
            json.dump(response,fp)
        return 1
    else:
        print 'File already downloaded %s ...' % outpath
        return 0 

def downloader(startDate, endDate, downloadPath):
    
    sharpKeys = ['DATE','DATE__OBS','HARPNUM','USFLUX','AREA_ACR','TOTUSJZ','TOTUSJH','TOTPOT','ABSNJZH','SAVNCPP',
                 'R_VALUE','MEANPOT','SHRGT45']
    lorentzKeys = ['DATE','DATE__OBS','HARPNUM','TOTBSQ','TOTFZ']
    sharpPath = downloadPath + '/raw-sharp/'
    lorentzPath = downloadPath + '/raw-lorentz/'
    if not os.path.exists(sharpPath):
        os.makedirs(sharpPath)
    if not os.path.exists(lorentzPath):
        os.makedirs(lorentzPath)
    baseUrl = 'http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?' 
    sharpSeries = 'hmi.sharp_cea_720s'
    lorentzSeries = 'cgem.lorentz'
    date = datetime.datetime.strptime(startDate, '%Y-%m-%dT%H:%M:%S')
    endDate = datetime.datetime.strptime(endDate,'%Y-%m-%dT%H:%M:%S')
    endDate -= datetime.timedelta(days=7)
    while date <= endDate:
        sharpStr = 'ds=%s[][%s/7d@12m]'%(sharpSeries,date.strftime('%Y-%m-%dT%H:%M:%S'))
        lorentzStr ='ds=%s[][%s/7d@12m]'%(lorentzSeries,date.strftime('%Y-%m-%dT%H:%M:%S'))
        condnStr = '[? (abs(OBS_VR)< 3500) and (QUALITY<65536) ?]'
        sharpkeyStr = '&op=rs_list&key=%s'%(','.join(sharpKeys))
        lorentzkeyStr = '&op=rs_list&key=%s'%(','.join(lorentzKeys))
        sharpUrl = baseUrl + sharpStr + condnStr + sharpkeyStr
        lorentzUrl = baseUrl + lorentzStr + condnStr + lorentzkeyStr
        outfile = date.strftime('%Y-%m-%d.json')
        dnSt = downloadJson(sharpPath + outfile,sharpUrl)
        dnSt += downloadJson(lorentzPath + outfile,lorentzUrl)
        if dnSt != 0:
            print 'Downloaded. Sleeping before next query...'
            time.sleep(1)
        date += datetime.timedelta(days=7)   
    return 

def getEndPoints(startDate, endDate, downloadPath):
    #Downloads timestamps of first and last AR observation irrespective of Quality
    sharpKeys = ['DATE__OBS','HARPNUM','AREA_ACR']
    sharpPath = downloadPath + '/sharpDates/'
    if not os.path.exists(sharpPath):
        os.makedirs(sharpPath)
    baseUrl = 'http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?' 
    sharpSeries = 'hmi.sharp_cea_720s'
    date = datetime.datetime.strptime(startDate, '%Y-%m-%dT%H:%M:%S')
    endDate = datetime.datetime.strptime(endDate,'%Y-%m-%dT%H:%M:%S')
    endDate -= datetime.timedelta(days=7)
    while date <= endDate:
        sharpStr = 'ds=%s[][%s/7d@12m]'%(sharpSeries,date.strftime('%Y-%m-%dT%H:%M:%S'))
        sharpkeyStr = '&op=rs_list&key=%s'%(','.join(sharpKeys))
        sharpUrl = baseUrl + sharpStr + sharpkeyStr
        print sharpUrl
        outfile = date.strftime('%Y-%m-%d.json')
        dnSt = downloadJson(sharpPath + outfile,sharpUrl)
        if dnSt != 0:
            print 'Downloaded. Sleeping before next query...'
            time.sleep(1)
        date += datetime.timedelta(days=7)
    return 
