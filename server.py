import models, torch

class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		#配置文件
		self.conf = conf 
		
		#全局模型
		self.global_model = models.get_model(self.conf["model_name"]) 
		
		#测试集加载器
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	#模型聚合
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			#存储每一层的参数更新的词典
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			#更新全局模型的参数，加上上述字典
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)

	#模型评估	
	def model_eval(self):

		#设置成评估模式，防止参数更新
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0

		#以下为这个数据集的一些必要操作
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 

			#加上张量的第一个维度的大小
			dataset_size += data.size()[0]
			
			#放到gpu上
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			
			#得到全局的输出
			output = self.global_model(data)
			
			#整体损失率
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			
			#计算正确率
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		#计算总体正确率和平均每一批损失率
		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l