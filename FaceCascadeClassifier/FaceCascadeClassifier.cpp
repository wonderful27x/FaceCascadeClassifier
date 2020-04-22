// FaceCascadeClassifier.cpp: 定义应用程序的入口点。
//

#include "FaceCascadeClassifier.h"

#include <direct.h>
#include <io.h>
#include <fstream>
#include <sstream>


//本项目利用级联分类器训练人脸识别模型，此项目主要做训练前的样本和数据准备，训练需要使用命令行
//什么是级联分类器，对于这个概念我的理解任然不太清楚，
//级联分类器大概就是为了弥补单个分类器的缺点设计的一种全新的分类思想，将多及分类器联起来，
//由上往下，分类越来越严格，越来越准确，这样即便是单个分类不太准确的弱分类器，通过这种方式级联n级后，
//也能得出较准确的结果，然而这单个分类器他们使用的是什么分类方法（svm？人工神经网络？），
//我没有找到答案，这就是我理解不清晰的地方，但是从网上的一些例子来看，级联分类器能做的事很多，
//分辨是否是人类（接下来我们要做的），分辨是非是特朗普的脸，分辨是否是个足球等等

//关于级联分类器，上面的理解似乎是不太正确，要理解级联分类器先了解一些概念
//Boosting：Boosting是一个算法框架，目的是把若干个分类器整合为一个分类器的方法，
//Boosting可以将多个弱分类器整合成一个较强的分类器，Adaboost是Boosting的实现
//级联分类器由多级强分类器级联而成（并不是上面说的弱分类器），而每一级强分类器可以由Adaboost算法将多个弱分类器组合而成，
//在使用haar特征人脸识别中，每一个haar特征就可以看成一个弱分类器，构成一级强分类器的弱分类器的数量就等于这一级分类器haar特征的数量，
//在级联分类器中还有识别率和错误率的概念，识别率指的是能够将正确的物体（人脸）识别出来的程度，而错误率指的是把其他物体当成正确物体（人脸）的程度，
//例如一共有200个样本，正、负样本各100个，我们能够把正样本全部识别出来，此时的识别率为100%，但同时有50个负样本也被当成了正样本，则此时的错误率为50%。
//识别率和错误率具有相同的变化趋势和程度，即提高识别率的同时，错误率也会显著提高，这里需要好好理解一下。
//下面的博客给出了详细的分析，对后面的参数设置有很大的帮助
//https://blog.csdn.net/zhaocj/article/details/54015501

//https://www.jianshu.com/p/c3120e8301aa
//https://blog.csdn.net/qq_32742009/article/details/81392651
//https://blog.csdn.net/cyf15238622067/article/details/86542682


//训练需要一些列样本。样本分两类：负样本和正样本。负样本是指不包括物体的图像。
//正样本是待检测的物体的图像。负样本必须手工准备，正样本使用 opencv_createsamples 创建。
//正、负样本个数比列大约为 1： 3
//下面的地址给出了详细的训练流程
//http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/user_guide/ug_traincascade.html

//文件跟目录
#define FILE_DIR "E:/VisualStudio/FILE/cascadeClassifier/samples/face"

//正样本图片目录
#define POS_FILE "/pos"
//正样本描述文件
#define POS_DSCRP "posDescription.txt"
//由描述文件生成的训练数据
#define POS_VEC "posTrainData.vec"

//负样本图片路径
#define NEG_FILE "/neg"
//负样本描述文件
#define NEG_DSCRP "negDescription.txt"

//模型保存路径
#define MODEL_FILE "E:/VisualStudio/FILE/cascadeClassifier/model/face"

#define POS_NUMBER 100
#define NEG_NUMBER 300

void dynamicUnFaceCheck();
void collectSamples(Mat frame, Rect face, int number);
void dynamicFaceCheck();
void crateNegDscrp(string negFile, string negDescription);
void cratePosDscrp(string posFile, string posDescription);
void createPosTrainData();
void train();


//级联分类器：用于训练和检测；
CascadeClassifier faceCascadeClassifier;
//摄像头
VideoCapture capture;
bool collect = false;

int main()
{
	//人脸追踪（获取人脸图片-正样本，这里是利用opencv简单获取一些人脸图片，实际情况下应该通过各种途径获取大量的样本）
	dynamicFaceCheck();
	cout << "回车开始采集负样本，务必避开人脸！" << endl;
	while (waitKey(1) != 13);
	//利用摄像头获取负样本图片
	dynamicUnFaceCheck();
	//创建正样本描述
	cratePosDscrp(POS_FILE, POS_DSCRP);
	//创建负样本描述
	crateNegDscrp(NEG_FILE, NEG_DSCRP);
	//创建正样本训练数据
	createPosTrainData();
	//训练级联分类器
	train();
	cout << "Hello CMake." << endl;
	return 0;
}

//训练级联分类器
//使用opencv_traincascade命令训练，
//以下并不是真正的函数，而是一些命令
//这些参数有的不太好理解，需要分析源码
//https://blog.csdn.net/zhaocj/article/details/54015501
void train() {
	//string vecFile = "E:/VisualStudio/FILE/cascadeClassifier/samples/face/posTrainData.vec";
	//string bgFile = "E:/VisualStudio/FILE/cascadeClassifier/samples/face/negDescription.txt";

	//opencv_traincascade -data MODEL_FILE -vec vecFile -bg bgFile -numPos 95 -numNeg 300 -numStages 15 -featureType LBP -w 24 -h 24 -precalcValBufSize 3000 -precalcIdxBufSize 1000 -minHitRate 0.999 -maxFalseAlarmRate 0.25 -maxWeakCount 150
	//opencv_traincascade -data E:/VisualStudio/FILE/cascadeClassifier/model/face -vec posTrainData.vec -bg negDescription.txt -numPos 95 -numNeg 300 -numStages 15 -featureType LBP -w 24 -h 24 -precalcValBufSize 3000 -precalcIdxBufSize 1000 -minHitRate 0.999 -maxFalseAlarmRate 0.25 -maxWeakCount 150
	//opencv_traincascade -data E:/VisualStudio/FILE/cascadeClassifier/model/face -vec posTrainData.vec -bg negDescription.txt -numPos 95 -numNeg 300 -numStages 15 -featureType LBP -w 24 -h 24
	
	//- data：模型保存路径
	//- vec：正样本数据
	//- bg：负样本描述
	//- numPos：每级分类器训练时所用的正样本数目
	//- numNeg：每级分类器训练时所用的负样本数目，可以大于 - bg 指定的图片数目
	//- numStages：训练的分类器的级数
	//- featureType：特征的类型： HAAR - 类Haar特征； LBP - 局部纹理模式特征。
	//- w：样本宽
	//- h：样本高
	//-precalcValBufSize：缓存大小，用于存储预先计算的特征值(feature values)，单位为MB。
	//-precalcIdxBufSize：缓存大小，用于存储预先计算的特征索引(feature indices)，单位为MB。内存越大，训练时间越短。
	//-minHitRate：分类器的每一级希望得到的最小检测率。
	//-maxFalseAlarmRate：分类器的每一级希望得到的最大误检率。
	//-maxWeakCount：每一级中的弱分类器的最大数目。The boosted classifier (stage) will have so many weak trees (<=maxWeakCount), as needed to achieve the given -maxFalseAlarmRate.
}

//创建正样本训练数据
//正样本由 opencv_createsamples命令通过正样本描述生成
//以下并不是真正的函数，而是一些命令
void createPosTrainData() {
	//	string vecFile = "E:/VisualStudio/FILE/cascadeClassifier/samples/face/posTrainData.vec";
	//	string info = "E:/VisualStudio/FILE/cascadeClassifier/samples/face/posDescription.txt";

	//	opencv_createsamples -vec vecFile -info info -w 24 -h 24 -num POS_NUMBER;
	//	opencv_createsamples -vec posTrainData.vec -info posDescription.txt -w 24 -h 24 -num 100;

	//	- vec：生成文件路径
	//	- info：文件描述
	//	- w：宽
	//	- h：高
	//	- num：生成的数量

	//	可以查看生成的vec样本
	//	opencv_createsamples -vec vecFile -w 24 -h 24;
	//  opencv_createsamples -vec E:/VisualStudio/FILE/cascadeClassifier/samples/face/posTrainData.vec -w 24 -h 24
}

//创建负样本的描述文件
//negFile: 负样本目录，存放负样本图片以及将要生成的描述文件
//negDescription：描述文件名
void crateNegDscrp(string negFile, string negDescription) {
	//描述文件
	string description;
	//图片路径
	string filePath = "";
	filePath = filePath.append(FILE_DIR);
	filePath = filePath.append(negFile);
	filePath.append("/*.png");
	//用于存储各种文件信息
	struct _finddata_t fileInfo;
	//返回值： 
	//如果查找成功的话，将返回一个long型的唯一的查找用的句柄（就是一个唯一编号）。这个句柄将在_findnext函数中被使用。若失败，则返回-1。 
	//参数： 
	//filespec：标明文件的字符串，可支持通配符。比如：*.c，则表示当前文件夹下的所有后缀为C的文件。 
	//fileinfo ：这里就是用来存放文件信息的结构体的指针。这个结构体必须在调用此函数前声明，不过不用初始化，只要分配了内存空间就可以了。 
	//函数成功后，函数会把找到的文件的信息放入这个结构体中。
	//调用此函数后fileInfo保存的是指定路径下的第一个文件，如a1.png
	auto file_handler = _findfirst(filePath.c_str(), &fileInfo);
	//文件的格式是固定的，详细请看前面的链接
	while (file_handler != -1) {
		//相对路径
		description.append("neg/");
		//文件名
		description.append(fileInfo.name);
		//换行
		description.append("\n");
		//寻找下一个文件
		//调用此函数后file_info保存的是指定路径下的下一个文件，如a2.png
		if (_findnext(file_handler, &fileInfo) != 0) break;
	}
	//删除最后的"\n"
	description.pop_back();
	//cout << description << endl;
	//关闭
	_findclose(file_handler);

	//写入文件
	ofstream outfile;
	//文件路径
	string outfilePath = "";
	outfilePath = outfilePath.append(FILE_DIR);
	outfilePath = outfilePath.append("/");
	outfilePath = outfilePath.append(negDescription);
	//删除文件重写
	outfile.open(outfilePath, ios_base::out | ios_base::trunc);
	stringstream descriptionStream;
	descriptionStream << description;
	outfile << descriptionStream.str();
	outfile.close();

	cout << "负样本描述创建完成" << endl;
}

//创建正样本的描述文件
//posFile: 正样本目录，存放正样本图片以及将要生成的描述文件
//posDescription：描述文件名
void cratePosDscrp(string posFile, string posDescription) {
	//描述文件
	vector<string> descriptions;
	//图片路径
	string filePath = "";
	filePath = filePath.append(FILE_DIR);
	filePath = filePath.append(posFile);
	filePath.append("/*.png");
	//用于存储各种文件信息
	struct _finddata_t fileInfo;
	//返回值： 
	//如果查找成功的话，将返回一个long型的唯一的查找用的句柄（就是一个唯一编号）。这个句柄将在_findnext函数中被使用。若失败，则返回-1。 
	//参数： 
	//filespec：标明文件的字符串，可支持通配符。比如：*.c，则表示当前文件夹下的所有后缀为C的文件。 
	//fileinfo ：这里就是用来存放文件信息的结构体的指针。这个结构体必须在调用此函数前声明，不过不用初始化，只要分配了内存空间就可以了。 
	//函数成功后，函数会把找到的文件的信息放入这个结构体中。
	//调用此函数后fileInfo保存的是指定路径下的第一个文件，如a1.png
	auto file_handler = _findfirst(filePath.c_str(), &fileInfo);
	//获取图片宽高
	int width = 0;
	int height = 0;
	if(file_handler != -1){
		string imagePath = "";
		imagePath.append(FILE_DIR);
		imagePath.append(posFile);
		imagePath.append("/");
		imagePath.append(fileInfo.name);
		Mat mat = imread(imagePath);
		width = mat.cols;
		height = mat.rows;
	}
	while (file_handler != -1) {
		//文件的格式是固定的，详细请看前面的链接
		string description;
		//相对路径
		description.append("pos/");
		//文件名
		description.append(fileInfo.name);
		//图片中人脸的数量
		description.append(" 1");
		//图片中人脸的起始坐标
		description.append(" 0 0 ");
		//图片中人脸的宽
		description.append(to_string(width));
		//空格
		description.append(" ");
		//图片中人脸的高
		description.append(to_string(height));
		//保存到集合中
		descriptions.push_back(description);
		//寻找下一个文件
		//调用此函数后file_info保存的是指定路径下的下一个文件，如a2.png
		if (_findnext(file_handler, &fileInfo) != 0) break;
	}
	//cout << description << endl;
	//关闭
	_findclose(file_handler);

	//写入文件
	ofstream outfile;
	//文件路径
	string outfilePath = "";
	outfilePath = outfilePath.append(FILE_DIR);
	outfilePath = outfilePath.append("/");
	outfilePath = outfilePath.append(posDescription);
	//删除文件重写
	outfile.open(outfilePath, ios_base::out | ios_base::trunc);
	for (int i = 0; i < descriptions.size(); i++) {
		stringstream descriptionStream;
		descriptionStream << descriptions[i];
		if (i == descriptions.size() - 1) {
			outfile << descriptionStream.str();
		}
		else {
			outfile << descriptionStream.str() << endl;
		}
	}

	outfile.close();
	
	cout << "正样本描述创建完成" << endl;
}

//动态人脸检测，这种方式适用于视频检测
//动态检测需要用到适配器
void dynamicFaceCheck() {
	//人脸模型路径
	//两个种方式都可以
	const string path = "E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	const char* pathP = path.c_str();
	//创建主适配器
	Ptr<CascadeClassifier> mainCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(mainCascadeClassifier);
	//创建追踪检测适配器
	Ptr<CascadeClassifier> trackerCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(trackerCascadeClassifier);
	//创建追踪器
	Ptr<DetectionBasedTracker> detectorTracker;
	DetectionBasedTracker::Parameters detectorParams;
	detectorTracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, detectorParams);
	//启动追踪器
	detectorTracker->run();

	//打开摄像头
	capture.open(0);
	//opencv中用Mat矩阵表示图像，
	//frame用于保存原始图像，opencv彩色图像默认使用bgr格式
	Mat frame;
	//灰度图像
	Mat gray;
	collect = true;
	while (collect) {
		capture >> frame;
		if (frame.empty()) {
			cout << "图像采集失败！";
			continue;
		}
		//转成灰度图
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//直方图均衡化，增强对比度
		equalizeHist(gray, gray);
		//检测向量
		vector<Rect> faces;
		//人脸检测处理
		detectorTracker->process(gray);
		//获取检测结果
		detectorTracker->getObjects(faces);
		//画人脸框
		//vs特有的for循环
		for each (Rect face in faces)
		{
			//在原始图上画一个红色的框
			rectangle(frame, face, Scalar(0, 0, 255));

			//保存人脸图片
			collectSamples(frame, face, POS_NUMBER);

		}
		//显示图像
		imshow("人脸检测", frame);
		//30ms刷新一次，如果27：ESC按下则break；
		if (waitKey(30) == 27) {
			break;
		}
	}

	detectorTracker->stop();
	cout << "人脸正样本采集完成" << endl;
}

//采集人脸训练正样本
void collectSamples(Mat frame, Rect face, int number) {
	static int count = 0;
	if (count == number) {
		collect = false;
		return;
	}
	count++;
	Mat sample;
	//从frame中把人脸face抠出来保存到sample中
	frame(face).copyTo(sample);
	//归一化大小
	resize(sample, sample, Size(24, 24));
	//转成灰度图
	cvtColor(sample, sample, COLOR_BGR2GRAY);
	//生成路径
	char p[100];
	string file = "";
	file.append(FILE_DIR);
	file.append(POS_FILE);
	sprintf(p, "%s/face_%d.png", file.c_str(), count);
	//保存样本图片
	imwrite(p, sample);
}

//采集负样本
void dynamicUnFaceCheck() {
	int count = 0;
	Mat frame;
	Mat sample;
	while (1) {
		capture >> frame;
		if (frame.empty()) {
			cout << "图像采集失败！";
			continue;
		}
		if (count == NEG_NUMBER) {
			break;
		}
		count++;
		//归一化大小，负样本稍大一些
		resize(frame, sample, Size(30, 30));
		//转成灰度图
		cvtColor(sample, sample, COLOR_BGR2GRAY);
		//生成路径
		char p[100];
		string file = "";
		file.append(FILE_DIR);
		file.append(NEG_FILE);
		sprintf(p, "%s/face_%d.png", file.c_str(), count);
		//保存样本图片
		imwrite(p, sample);
		//显示图像
		imshow("负样本采集", frame);
		//30ms刷新一次，如果27：ESC按下则break；
		if (waitKey(30) == 27) {
			break;
		}
	}

	cout << "负样本采集完成" << endl;
}
















/* #include "FaceCascadeClassifier.h"

#include <direct.h>
#include <io.h>
#include <fstream>
#include <sstream>


//本项目利用级联分类器训练人脸识别模型，此项目主要做训练前的样本和数据准备，训练需要使用命令行
//什么是级联分类器，对于这个概念我的理解任然不太清楚，
//级联分类器大概就是为了弥补单个分类器的缺点设计的一种全新的分类思想，将多及分类器联起来，
//由上往下，分类越来越严格，越来越准确，这样即便是单个分类不太准确的弱分类器，通过这种方式级联n级后，
//也能得出较准确的结果，然而这单个分类器他们使用的是什么分类方法（svm？人工神经网络？），
//我没有找到答案，这就是我理解不清晰的地方，但是从网上的一些例子来看，级联分类器能做的事很多，
//分辨是否是人类（接下来我们要做的），分辨是非是特朗普的脸，分辨是否是个足球等等
//https://www.jianshu.com/p/c3120e8301aa
//https://blog.csdn.net/qq_32742009/article/details/81392651


//训练需要一些列样本。样本分两类：负样本和正样本。负样本是指不包括物体的图像。
//正样本是待检测的物体的图像。负样本必须手工准备，正样本使用 opencv_createsamples 创建。
//正、负样本个数比列大约为 1： 3
//下面的地址给出了详细的训练流程
//http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/user_guide/ug_traincascade.html

//正样本
#define POS_FILE "E:/VisualStudio/FILE/cascadeClassifier/samples/face/pos"
#define POS_DSCRP "posDescription.txt"
//负样本
#define NEG_FILE "E:/VisualStudio/FILE/cascadeClassifier/samples/face/neg"
#define NEG_DSCRP "negDescription.txt"

#define POS_NUMBER 10
#define NEG_NUMBER 30

void dynamicUnFaceCheck();
void collectSamples(Mat frame, Rect face, int number);
void dynamicFaceCheck();
void crateNegDscrp(string negFile, string negDescription);
void cratePosDscrp(string posFile, string posDescription);


//级联分类器：用于训练和检测；
CascadeClassifier faceCascadeClassifier;
//摄像头
VideoCapture capture;
bool collect = false;

int main()
{
	//人脸追踪（获取人脸图片-正样本，这里是利用opencv简单获取一些人脸图片，实际情况下应该通过各种途径获取大量的样本）
	dynamicFaceCheck();
	cout << "回车开始采集负样本，务必避开人脸！" << endl;
	while (waitKey(1) != 13);
	//利用摄像头获取负样本图片
	dynamicUnFaceCheck();
	//创建正样本描述
	cratePosDscrp(POS_FILE, POS_DSCRP);
	//创建负样本描述
	crateNegDscrp(NEG_FILE, NEG_DSCRP);
	cout << "Hello CMake." << endl;
	return 0;
}

//创建负样本的描述文件
//negFile: 负样本目录，存放负样本图片以及将要生成的描述文件
//negDescription：描述文件名
void crateNegDscrp(string negFile, string negDescription) {
	//描述文件
	string description;
	//图片路径
	string filePath = "";
	filePath = filePath.append(negFile);
	filePath.append("/*.png");
	//用于存储各种文件信息
	struct _finddata_t fileInfo;
	//返回值： 
	//如果查找成功的话，将返回一个long型的唯一的查找用的句柄（就是一个唯一编号）。这个句柄将在_findnext函数中被使用。若失败，则返回-1。 
	//参数： 
	//filespec：标明文件的字符串，可支持通配符。比如：*.c，则表示当前文件夹下的所有后缀为C的文件。 
	//fileinfo ：这里就是用来存放文件信息的结构体的指针。这个结构体必须在调用此函数前声明，不过不用初始化，只要分配了内存空间就可以了。 
	//函数成功后，函数会把找到的文件的信息放入这个结构体中。
	//调用此函数后fileInfo保存的是指定路径下的第一个文件，如a1.png
	auto file_handler = _findfirst(filePath.c_str(), &fileInfo);
	while (file_handler != -1) {
		//相对路径
		description.append("neg/");
		//文件名
		description.append(fileInfo.name);
		//换行
		description.append("\n");
		//寻找下一个文件
		//调用此函数后file_info保存的是指定路径下的下一个文件，如a2.png
		if (_findnext(file_handler, &fileInfo) != 0) break;
	}
	//删除最后的"\n"
	description.pop_back();
	//cout << description << endl;
	//关闭
	_findclose(file_handler);

	//写入文件
	ofstream outfile;
	//文件路径
	string outfilePath = "";
	outfilePath = outfilePath.append(negFile);
	outfilePath = outfilePath.append("/");
	outfilePath = outfilePath.append(negDescription);
	//删除文件重写
	outfile.open(outfilePath, ios_base::out | ios_base::trunc);
	stringstream descriptionStream;
	descriptionStream << description;
	outfile << descriptionStream.str();
	outfile.close();

	cout << "负样本描述创建完成" << endl;
}

//创建正样本的描述文件
//posFile: 正样本目录，存放正样本图片以及将要生成的描述文件
//posDescription：描述文件名
void cratePosDscrp(string posFile, string posDescription) {
	cout << "正样本描述创建完成" << endl;
}

//动态人脸检测，这种方式适用于视频检测
//动态检测需要用到适配器
void dynamicFaceCheck() {
	//人脸模型路径
	//两个种方式都可以
	const string path = "E:/OPENCV/opencv4.1.2/INSTALL/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	const char* pathP = path.c_str();
	//创建主适配器
	Ptr<CascadeClassifier> mainCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(mainCascadeClassifier);
	//创建追踪检测适配器
	Ptr<CascadeClassifier> trackerCascadeClassifier = makePtr<CascadeClassifier>(pathP);
	Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(trackerCascadeClassifier);
	//创建追踪器
	Ptr<DetectionBasedTracker> detectorTracker;
	DetectionBasedTracker::Parameters detectorParams;
	detectorTracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, detectorParams);
	//启动追踪器
	detectorTracker->run();

	//打开摄像头
	capture.open(0);
	//opencv中用Mat矩阵表示图像，
	//frame用于保存原始图像，opencv彩色图像默认使用bgr格式
	Mat frame;
	//灰度图像
	Mat gray;
	collect = true;
	while (collect) {
		capture >> frame;
		if (frame.empty()) {
			cout << "图像采集失败！";
			continue;
		}
		//转成灰度图
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		//直方图均衡化，增强对比度
		equalizeHist(gray, gray);
		//检测向量
		vector<Rect> faces;
		//人脸检测处理
		detectorTracker->process(gray);
		//获取检测结果
		detectorTracker->getObjects(faces);
		//画人脸框
		//vs特有的for循环
		for each (Rect face in faces)
		{
			//在原始图上画一个红色的框
			rectangle(frame, face, Scalar(0, 0, 255));

			//保存人脸图片
			collectSamples(frame, face, POS_NUMBER);

		}
		//显示图像
		imshow("人脸检测", frame);
		//30ms刷新一次，如果27：ESC按下则break；
		if (waitKey(30) == 27) {
			break;
		}
	}

	detectorTracker->stop();
	cout << "人脸正样本采集完成" << endl;
}

//采集人脸训练正样本
void collectSamples(Mat frame, Rect face, int number) {
	static int count = 0;
	if (count == number) {
		collect = false;
		return;
	}
	count++;
	Mat sample;
	//从frame中把人脸face抠出来保存到sample中
	frame(face).copyTo(sample);
	//归一化大小
	resize(sample, sample, Size(24, 24));
	//转成灰度图
	cvtColor(sample, sample, COLOR_BGR2GRAY);
	//生成路径
	char p[100];
	string file = POS_FILE;
	sprintf(p, "%s/face_%d.png", file.c_str(), count);
	//保存样本图片
	imwrite(p, sample);
}

//采集负样本
void dynamicUnFaceCheck() {
	int count = 0;
	Mat frame;
	Mat sample;
	while (1) {
		capture >> frame;
		if (frame.empty()) {
			cout << "图像采集失败！";
			continue;
		}
		if (count == NEG_NUMBER) {
			break;
		}
		count++;
		//归一化大小，负样本稍大一些
		resize(frame, sample, Size(30, 30));
		//转成灰度图
		cvtColor(sample, sample, COLOR_BGR2GRAY);
		//生成路径
		char p[100];
		string file = NEG_FILE;
		sprintf(p, "%s/face_%d.png", file.c_str(), count);
		//保存样本图片
		imwrite(p, sample);
		//显示图像
		imshow("负样本采集", frame);
		//30ms刷新一次，如果27：ESC按下则break；
		if (waitKey(30) == 27) {
			break;
		}
	}

	cout << "负样本采集完成" << endl;
} */
