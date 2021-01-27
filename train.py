import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import torch as T 
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sys import argv, exit
import copy
import PIL
import time
from sklearn.metrics import confusion_matrix




def Model(device, numClasses):
	model = models.resnet18(pretrained=True)

	model.fc = nn.Sequential(nn.Linear(512, 256), 
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(256, numClasses),
		nn.LogSoftmax(dim=1))

	model.to(device)


	return model

def getTransform():
	customTransform = transforms.Compose([transforms.RandomRotation((-270,270)),
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

	return customTransform


def saveLosses(trainLosses, testLosses):
	xAxis = range(1, len(trainLosses) + 1)
	
	plt.plot(xAxis, trainLosses, label="Train Loss")
	plt.plot(xAxis, testLosses, label="Test Loss")
	plt.legend()
	plt.savefig("losses.png")
	plt.clf()


def train(model, device, optimizer, criterion, dataloaders, weightName, epochs,scheduler):
	miniBatch = 0
	runningLoss = 0.0
	printMiniBatch = 100
	bestAcc = 0.0
	bestWts = copy.deepcopy(model.state_dict())
	trainLosses, testLosses = [], []

	startTime = time.time()
	for epoch in range(epochs):
		for inputs, labels in dataloaders['test']:
			miniBatch += 1
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			logps = model.forward(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()
			runningLoss += loss.item()

			if(miniBatch%printMiniBatch == 0):
				testLoss, testAcc = evaluate(model, device, dataloaders['test'], criterion)

				trainLoss = runningLoss/printMiniBatch 
				trainLosses.append(trainLoss)
				testLosses.append(testLoss)
				saveLosses(trainLosses, testLosses)

				print("Epoch: {}/{}, Minibatch: {}/{}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}"
					.format(
					epoch+1,
					epochs,
					miniBatch,
					epochs*len(dataloaders['train']),
					trainLoss, 
					testLoss,
					testAcc))
				
				runningLoss = 0.0
				
				model.train()

				if(testAcc > bestAcc):
					bestAcc = testAcc
					bestWts = copy.deepcopy(model.state_dict())
					T.save(bestWts, weightName)

		epochWeight = "epoch{}.pth".format(epoch+1)
		bestWts = copy.deepcopy(model.state_dict())
		T.save(bestWts, epochWeight)

		scheduler.step()
	endTime = time.time()

	print("Training completed in {:.4f} seconds".format(endTime - startTime))
	
	# model.load_state_dict(bestWts)
	# del model, optimizer, scheduler, loss, outputs ## <<<<---- HERE
 #    T.cuda.empty_cache() ## <<<<---- AND HERE
 #    T.cuda.synchronize()
 #    show_gpu(f'{i}: GPU memory usage after clearing cache:')
	# # T.save(bestWts, weightName)


	return model

def evaluate(model, device, testloader, criterion):
	testLoss = 0
	testAcc = 0

	model.eval()
	with T.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.to(device), labels.to(device)
			logps = model.forward(inputs)
			batchLoss = criterion(logps, labels)
			testLoss += batchLoss.item()

			ps = T.exp(logps)
			topP, topClass = ps.topk(1, dim=1)
			equals = topClass == labels.view(*topClass.shape)
			testAcc += T.mean(equals.type(T.FloatTensor)).item()

	testAcc = 100 * testAcc/len(testloader)
	testLoss = testLoss/len(testloader)

	return testLoss, testAcc

def predictImage(img, model, device,testloader):
	
	testTransform = getTransform()

	model.eval()
	with T.no_grad():
		imgTensor = testTransform(img)
		imgTensor = imgTensor.unsqueeze_(0)
		imgTensor = imgTensor.to(device)	
		predict = model(imgTensor)
		index = predict.data.cpu().numpy().argmax()


# calculating confusion matrix
		# for i, data in enumerate(testloader, 0):
		# 	# get the inputs
		# 	t_image, mask = data
		# 	t_image, mask = Variable(t_image.to(device)), Variable(mask.to(device))
		
		# 	output = model(t_image)
		# 	pred = torch.exp(output)
		# 	conf_matrix = confusion_matrix(pred, mask)
		# 	print(conf_matrix)


	return index, T.exp(predict).data.cpu().numpy()

	
def evalImages(dataDir, model, device, classNames,testloader):
	classFolder = classNames[0]
	imgFiles = os.listdir(dataDir+classFolder)

	correctCount = 0

	for i, imgFile in enumerate(imgFiles):
		try:
			img = PIL.Image.open(os.path.join(dataDir, classFolder, imgFile))
		except IOError:
			continue

		index, probs = predictImage(img, model, device,testloader)
		# print("{}. Image belongs to class: {} | Probabilities: {}".format(
		# 	i, classNames[index], probs))

		# plt.imshow(np.asarray(img))
		# plt.show()

		if(classNames[index] == classFolder):
			correctCount += 1

	print("Accuracy for {} class is: {:.4f} | Correct Prediction: {} | Total Images: {} ".format(
		classFolder, correctCount*100/len(imgFiles),
		correctCount,
		len(imgFiles)))



if __name__ == '__main__':
	root_dir = 'tiny-imagenet-200/'
	# dataDir = 'tiny-imagenet-200/train'
	# dataDir2 = 'tiny-imagenet-200/test/images'
	num_workers = {'train' : 4,'val'   : 0,'test'  : 0}
	weightName = "tinp.pth"
	image_transform = {
	'train': transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]),
	'test':  transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]),
	'val':  transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])}



	image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x), image_transform[x]) 
                  for x in ['train', 'val','test']}
	dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
                  for x in ['train', 'val', 'test']}
	
	device = T.device("cpu")
	numClasses=200
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
	# input_dim = 224*224*3
	# hidden_dim = 300
	# output_dim = 200
	epochs = 10
	classNames = image_datasets['train'].classes


	model = Model(device, numClasses)


	criterion = nn.NLLLoss()
	# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	expLrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	model = train(model, device, optimizer, criterion,dataloaders, weightName, epochs,expLrScheduler)
	# model=train_model("64",model, dataloaders, dataset_sizes, criterion, optimizer, epochs, scheduler=expLrScheduler)

	model.load_state_dict(T.load(weightName))
	testLoss, testAcc = evaluate(model, device, dataloaders['test'], criterion)
	print("Final Accuracy: {:.4f} and Loss: {:.4f}".format(testAcc, testLoss))

	evalImages(dataDir, model, device, classNames,dataloaders['test'])

