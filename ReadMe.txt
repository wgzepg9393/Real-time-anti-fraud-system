在这个代码移植的过程中，遇到了很多的问题，我花了近一天的时间终于调通了。
问题①代码移植后，openCV的人脸检测函数detectmultiscale函数检测的人脸个数很大很大，这显然不对。
      问题出在了链接库上，在Debug版本下，要选择opencv_world310d.lib。
      链接https://blog.csdn.net/cracent/article/details/51165332

问题②人脸检测函数detectMultiScale的范围选择（30,30）,如果太大，那么检测的人脸的个数也会出错，甚至检测不到人脸。

从Qt移植到VS上的最大的问题就是运行时间的问题，在Qt上运行时间很少，但是在VS上运行时间需要很长很长，主要是Dlib提取人脸代码那块儿用时较长，具体原因不明。