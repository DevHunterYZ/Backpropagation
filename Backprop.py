inputs=[[1.0,1.0],[0.0,1.0],[1.0,0.0]]
outputs=[[1.0],[0.0],[0.0]]

people={}
people.update({'murat':[1.0,0.0]})
people.update({'ali':[0.0,1.0]})
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
net = buildNetwork(2, 3, 1, bias=True)
ds=SupervisedDataSet(2,1)
for i,j in zip(inputs, outputs):
  ds.addSample(tuple(i),tuple(j))
print(ds)

back=BackpropTrainer(net,ds) # eğitim algoritması
for epoch in range(1000):
    print(back.train())
for person, features in people.items():
  compute=net.activate(features)
  prop=compute[0]
  print(person,' is ',prop,'% likely to buy')
