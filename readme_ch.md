

# INTRODUCTION
	Embdded_SSD_Mobilenet

# RESULT_SHOW

# Installation
	Tensorflow-gpu==1.10.0&Tensorflow-gpu==1.12.0
	opencv-python==3.4.0
	pillow
	matplotlib
	
	1、数据准备
		imges 存放图片；
		ammitations存放标签；
		
	2、环境配置
	label_map_person.txt中写入id，name（id=多少个类就写那个id，name=具体的类型名称）。
	creat_name.py生成训练数据集；
	
	进入slim执行以下命令编译安装环境
		python setup.py build
		python setup.py install
		
	进入根目录下同样执行以上操作
		python setup.py build
		python setup.py install
		
	3、进入object_detection目录
		修改create_tf_record.py中146、149、162、167行生成如下文件
		train.record
		val.record
		
	4、进入根目录下找到embedded_ssd_mobilenet_v1_coco.config中9、141、146、156行。
		9行类别参数
		141行batch_size参数
		146行学习率
		156行是否迁移学习（如果采用从头训练需要注释掉这段代码）
		170行训练数据train.record路径
		172行写入label_map_person.pbtxt
		182行写入验证的val.record路径
		184行写入验证的label_map_person.pbtxt
		
# Training	
	1、进入object_detection目录
		修改训练train.sh路径内容
		
# Testing
	1、生成网络结构和参数固定化融合生成pb,进入object_detection目录
		export_inference_graph.sh
		
	2、进入object_detection目录修改27、28、37、63行为自己对应文件路径进行测试
	

# Citation
	《SSD》
	《mobilenetv1》
	《pruning》
	《Weight Factorization》
	《Weight Sharing》
	《Quantization》
	 参数修剪和共享（parameter pruning and sharing）
	 低秩因子分解（low-rank factorization）
	 转移/紧凑卷积滤波器（transferred/compact convolutional filters）
	 知识蒸馏（knowledge distillation）
	
# 项目目录
	annotations 存放xml标签
	images      存放训练图片
	weights		
		input_w  存放生成的ckpt、enents、pipline
		output_w 存放export输出的pb文件
	test	      存放验证数据集
	result        存放输出结果图
	