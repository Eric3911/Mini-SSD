嵌入式目标检测

  这个项目从工程硬件综合优化的方法来实现移动端高速检测，声明一下这个模型是我参考谷歌的objectdetectionAPI模型的一个子模型和slim方法改过来测试的，主要
是为了一个工程产品我们实现嵌入式的使用，考虑到芯片模组之间的计算均衡负担和线程同步问题我们做了相应的设计。

  目前速度和准确率还存在小问题。希望能与大家交流解决这个问题实现95FPS目标，模型最大的优点在速度相对优势的情况下权重体积是普通模型的100/1分之一左右，对
普通问题我们采用该模型一般可以控制在5M内实现模型的部署，权重问题一般在2M以内。

  i7CPU的单机PC上测300ms，1200*960尺寸图片。其他相关参数我会补全。最近比较忙大家先自己测测，这个目标的准确率和速度相对可以进一步优化。我会在第二版为
大家讲细节，先开源出来解决大家面临模型很大在ARMLinux上的部署问题和产品化自己的idea。模型在使用过程中需要对TensorFlow的源码进行编译和相关ARM上的TF部
署。
如果你对谷歌的目标检测API使用不熟悉请严格按照我的教程，否则问题很多会导致各种组件之间的编译问题。刚需条件：tensorflow=1.10.0  cuda =8/9训练都可以。



![识别结果](https://github.com/Eric3911/miniDetection/blob/master/oilplot_pr.png)