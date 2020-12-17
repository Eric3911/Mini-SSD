# Embedded target detection based on mobilenet SSD
  1. The performance of this model for embedded design in small object detection will be better, such as Northwest University of technology aircraft data set (sparse small target).
  
  2. In the detection of the new aspect ratio, the coarse and large resolution channel is used to add attention, and the performance of the sample layer target is improved greatly (multi-scale: playground, aircraft, aisle).
  
  3. In our model, we first use the deconvolution structure for multi-source rough high-resolution, and the sampling layer is faster.
  
  4. Up sampling adopts label smoothing and training initialization weight, resulting in loss transformation modification, which makes frame drawing more accurate。	

## VOC Dataset
   voc2007 format
	
## Training
	Tensorflow-gpu==1.10.0&Tensorflow-gpu==1.12.0
	cuda =8/9
	opencv-python==3.4.0
	pillow
	matplotlib

	1、datsets
		imges ，imgs files；
		ammitations，xml files；

	2、label_map_person.txt ；id，name.
	creat_name.py

	slim files
		python setup.py build
		python setup.py install

		python setup.py build
		python setup.py install

	3、object_detection
		create_tf_record.py，146、149、162、167 rows
		train.record
		val.record

	4、embedded_ssd_mobilenet_v1_coco.config中9、141、146、156、172、182、184 rows。
		
	5、object_detection
		train.sh
		
	6、reference resources readme_ch.md
	
## Evaluation	
	python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./rfcn_resnet101_coco.config 
	--trained_checkpoint_prefix ./models/train/model.ckpt-5000 --output_directory ./fine_tuned_model
  
## Citing mini_SSD
  This project realizes high-speed detection of mobile terminal from the comprehensive optimization method of engineering hardware. I declare that this model is modified by referring to a sub model of Google's object detection API model and integrating mobilnetv1 and slim methods. It is mainly for an engineering product. We realize the use of embedded system, taking into account the calculation balance burden and thread synchronization between chip modules We have made the corresponding design.
  
  At present, there are still some small problems in speed and accuracy. I hope to communicate with you to solve this problem and achieve the goal of 95fps. The biggest advantage of the model is that the weight volume is about 100 / 1 of the ordinary model when the speed is relatively superior. For ordinary problems, we can use this model to realize the deployment of the model within 5m, and the weight problem is generally within 2m.
  
  i7 CPU on a single PC 300 ms, 1200*960 size pictures. I will complete other relevant parameters. Recently, we have been busy with our own measurement. The accuracy and speed of this target can be further optimized. I will tell you the details in the second edition. Open source will be used to solve the problem of deployment on ARMLinux with a large model and to produce your own idea. In the process of using the model, we need to compile tensorflow source code and deploy TF on related arm.
  
  If you are not familiar with Google's target detection API, please strictly follow my tutorial, otherwise many problems will lead to compilation problems between various components.

![result](https://github.com/Eric3911/miniDetection/blob/master/oilplot_pr.png)
![result](https://github.com/Eric3911/miniDetection/blob/master/oiltank_155.jpg)
![result](https://github.com/Eric3911/mini_SSD/blob/master/001.jpg)
