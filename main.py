import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets

	

if __name__ == '__main__':

	#解析命令行参数
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	
	#读取配置文件
	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	#获取数据集
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	#得到服务器对象
	server = Server(conf, eval_datasets)
	clients = []
	
	#得到客户端对象，构成一个列表
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n")
	for e in range(conf["global_epochs"]):

		#随机选择k个客户端
		candidates = random.sample(clients, conf["k"])
		
		#初始化权重累加器
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		
		#得到差异字典更新权重累加器
		for c in candidates:
			diff = c.local_train(server.global_model)
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		#模型聚合
		server.model_aggregate(weight_accumulator)
		
		#模型评估
		acc, loss = server.model_eval()
		
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
				
			
		
		
	
		
		
	