import sys
import os
import re
import string
import numpy as np
from numpy import *
from scipy import linalg as SLA
from pysnptools.snpreader import Bed
import shutil

print '###Read the parameter card file###'
if len(sys.argv) != 2:
	print 'Please input the paramter card file'
	exit()


paraFile = sys.argv[1]
f = open(paraFile, 'r')
line = f.readline()

while line:
	key = re.match('\$(\S+)',line)
	if key:
		if key.group(1) == 'bedFile':
			line = f.readline()
			bedFile = line.strip()
		elif key.group(1) == 'pheFile':
			line = f.readline()
			pheFile = line.strip()
		elif key.group(1) == 'fixOrder':
			line = f.readline()
			fixOrder = int(line.strip())
		elif key.group(1) == 'snpSet':
			arr = line.split()
			snpStart = int(arr[1])
			snpEnd = int(arr[2])
		elif key.group(1) == 'var':
			arr = line.split()
			addOrder = int(arr[1])
			perOrder = int(arr[2])
			addVar = np.zeros((int(arr[1]) + 1, int(arr[1]) + 1))
			perVar = np.zeros((int(arr[2]) + 1, int(arr[2]) + 1))
			line = f.readline()
			while line:
				arr = line.split()
				if len(arr) != 4:
					line = f.readline()
					continue
				i = int(arr[1]) - 1
				j = int(arr[2]) - 1
				if arr[0] == '1':
					addVar[i][j] = addVar[j][i] = float(arr[3])
				elif arr[0] == '2':
					perVar[i][j] = perVar[j][i] = float(arr[3])
				elif arr[0] =='3':
					resVar = float(arr[3])
				line = f.readline()
	line = f.readline()

f.close()

print 'Prefix of plink binary file: ' + bedFile
print 'Phenotype file: ' + pheFile
print 'Polynomial order for fixed regression: ',fixOrder
print 'start and end snp: ', snpStart, snpEnd
print 'Addtive covariance: '
print addVar
print 'Permanent covariance : '
print perVar
print 'Residual variance: ' + str(resVar)

print '\n###Rearrange the phenotype file###'
f = open(pheFile, 'r')
line = f.readline()
testday = []
testNum = 0
idLst = []
while line:
	testNum += 1
	arr = line.split()
	testday.append(float(arr[-2]))
	line = f.readline()
	idLst.append(arr[0])

f.close()

timeMax = max(testday)
timeMin = min(testday)

print 'the max and min time point:',timeMax,timeMin

##define legendre polynomial function
def factorials(n):
	if n==0:
		return 1
	else:
		k = 1
		for i in range(1,n+1):
			k = k*i
		return k

def legVal(timeMin, timeMax, timeP, order):
	
	timeMin = float(timeMin)
	timeMax = float(timeMax)
	timeP = float(timeP)
	order = int(order)
	
	timen = 2*(timeP - timeMin)/(timeMax - timeMin) - 1
	c = int(order/2)
	
	j = order
	p = 0
	for r in range(0,c+1):
		p += np.sqrt((2*j+1.0)/2.0) * pow(0.5, j) * (pow(-1, r) * factorials(2*j-2*r) / (factorials(r) * factorials(j - r) * factorials(j - 2*r))) * pow(timen,j-2*r)
	return p


orderMax = max([fixOrder, addOrder, perOrder])


f = open(pheFile, 'r')
line = f.readline()
arr = line.split()
intNum = len(arr) - 2
reNum = [0]*intNum
reDict = {}
for i in range(0, intNum):
	reDict[i] = {}

tempFile = pheFile + '.temp'
fout = open(tempFile,'w')

while line:
	arr = line.split()
	for i in range(0,intNum):
		if arr[i] in reDict[i]:
			arr[i] = reDict[i][arr[i]]
		else:
			reNum[i] += 1
			reDict[i][arr[i]] = str(reNum[i])
			arr[i] = reDict[i][arr[i]]
		stri = arr[i] + '\t'
		fout.write(stri)
	
	timeP = float(arr[-2])
	for i in range(0, orderMax+1):
		val = legVal(timeMin, timeMax, timeP, i)
		stri = str(val) + '\t'
		fout.write(stri)
	
	stri = arr[-1] + '\n'
	fout.write(stri)
	
	line = f.readline()

nID = reNum[0]
fixDegree = reNum[:]

f.close()
fout.close()



print '\n###Prepare the weighted kinship matrix'
print 'the numter of record and individuals ', testNum, nID

#design matrix
zDesi = np.zeros([testNum,nID*(orderMax+1)])

fin = open(tempFile, 'r')
line = fin.readline()
i = 0
while line:
	arr = line.split()
	k = int(arr[0])-1
	for j in range(0, orderMax+1):
		zDesi[i, k*(orderMax+1)+j] = float(arr[j-orderMax-2])
	i += 1
	line =fin.readline()


fin.close()


#additive part
addFile = bedFile + '.addGmat.grm'

kin = np.loadtxt(addFile)

covMat = np.kron(kin,addVar)


if orderMax == addOrder:
	zTemp = zDesi[:,:]
else:
	subBool = [True]*(addOrder+1)
	subBool.extend([False]*(orderMax - addOrder))
	subBool = np.array(subBool*nID)
	zTemp = zDesi[:,subBool]

wghtKin = np.dot(np.dot(zTemp, covMat), zTemp.T)


#permanent environmental part

kin = np.diag([1.0]*nID)
covMat = np.kron(kin,perVar)


if orderMax == perOrder:
	zTemp = zDesi[:,:]
else:
	subBool = [True]*(perOrder+1)
	subBool.extend([False]*(orderMax - perOrder))
	subBool = np.array(subBool*nID)
	zTemp = zDesi[:,subBool]

wghtKin += np.dot(np.dot(zTemp, covMat), zTemp.T)


#eigen decomposition
w, v = SLA.eigh(wghtKin)

resDiag = w + np.array([resVar]*testNum)


print '\n###Prepare the fixed effect part'

#design matrix for fix effect
fixNum = fixDegree[1]*(fixOrder+1)
for i in range(2, len(fixDegree)):
	fixNum += fixDegree[i] - 1


fixDesi = np.zeros([testNum,fixNum])
pheno = []
snpReg = np.zeros([testNum, fixOrder+1])

fin = open(tempFile, 'r')
line = fin.readline()
i = 0
while line:
	arr = line.split()
	k = int(arr[1]) - 1
	for j in range(0, fixOrder+1):
		fixDesi[i, k*(fixOrder+1)+j] = float(arr[j-orderMax-2])
		snpReg[i, j] = float(arr[j-orderMax-2])
	
	for j in range(2, len(fixDegree)):
		k = int(arr[j])
		bef = fixDegree[1]*(fixOrder+1)
		if j != 2:
			bef += fixDegree[j-1] - 1
		if(k != 1):
			fixDesi[i, bef+k-2] = 1
	
	pheno.append(float(arr[-1]))
	
	i += 1
	line =fin.readline()

fin.close()


resDiagInv = 1.0/resDiag
fixDesi = np.dot(v.T, fixDesi)
pheno = np.dot(v.T, pheno)

effRight = resDiagInv * pheno

snp = Bed(bedFile,count_A1 = False)
snp = snp.read()



reFile = pheFile + '.' + str(snpStart) + '-' + str(snpEnd) + '.Addres'
fout = open(reFile, 'w')

for k in range(snpStart - 1, snpEnd):
	
	dct = dict(zip(snp.iid[:,1], snp.val[:,k]))
	snpEachAdd = np.array([0.0]*testNum)
	for i in range(0,len(idLst)):
		snpEachAdd[i] = dct[idLst[i]]
	
	snpReAdd = snpReg.T * snpEachAdd
	
	snpReAll = np.dot(v.T, snpReAdd.T)
	
	xDesi = np.hstack((fixDesi, snpReAll))
	
	xTemp = xDesi.T*resDiagInv
	xDesiSelf = np.dot(xTemp, xDesi)
	
	xDesiSelfInv = np.linalg.inv(xDesiSelf)
	xDesiSelfInv = np.array(xDesiSelfInv)
	fixEff = np.dot(xDesi.T, effRight)
	fixEff = np.dot(xDesiSelfInv, fixEff)
	
	#test add
	covTest = xDesiSelfInv[(-fixOrder-1):, (-fixOrder-1):]
	effTest = fixEff[(-fixOrder-1):]
	
	covTestInv = np.linalg.inv(covTest)
	chiVal = np.dot(covTestInv, effTest)
	chiValAdd = np.dot(effTest, chiVal)

	
	stri = snp.sid[k] + '\t' + str(chiValAdd) + '\n'
	fout.write(stri)

fout.close()
