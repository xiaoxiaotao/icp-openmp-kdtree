#include "icp.h"


PointCloudT::Ptr ICP::data2cloud(const Eigen::MatrixXd data){

	int size = data.cols();
	pcl::PointXYZ temp;
	PointCloudT temp_point;
	temp_point.resize(size);
#pragma omp parallel for
	for(int i=0;i<size;i++){
		temp_point[i].x = data(0,i);
		temp_point[i].y = data(1,i);
		temp_point[i].z = data(2,i);
	}

	return temp_point.makeShared();

}

double ICP::FindClosest(const Eigen::MatrixXd  cloud_target,	const Eigen::MatrixXd cloud_source,  Eigen::MatrixXd &ConQ)
{


	PointCloudT::Ptr cloud_target_ = data2cloud(cloud_target);
	PointCloudT::Ptr cloud_source_ = data2cloud(cloud_source);
	double error = 0.0;
	std::vector<double> error_vec;
	error_vec.resize(cloud_source_->size());
	kdtree_search.setInputCloud(cloud_target_);
#pragma omp parallel for
	for(int i=0;i<cloud_source_->size();i++){
		pcl::PointXYZ pointSel = (*cloud_source_)[i];
		std::vector<int> pointSearchInd;//ËÑË÷×îœüµãµÄid
		std::vector<float> pointSearchSqDis;//×îœüµãµÄŸàÀë
		kdtree_search.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
		pcl::PointXYZ temp_point = (*cloud_source_)[pointSearchInd.back()];
		Eigen::Vector3d v(temp_point.x,temp_point.y,temp_point.z);
		ConQ.col(i) = v;
		//error += pointSearchSqDis.back();
		error_vec[i] =  pointSearchSqDis.back();

	}

	error = std::accumulate(error_vec.begin(),error_vec.end(),0);
	
	return error;
}

Eigen::Matrix4d ICP::GetTransform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd ConQ)
{
	Eigen::initParallel();
	int nsize = ConP.cols();
	Eigen::VectorXd MeanP = ConP.rowwise().mean();
	Eigen::VectorXd MeanQ = ConQ.rowwise().mean();
	//cout << MeanP<< MeanQ<<endl;
	Eigen::MatrixXd ReP = ConP.colwise() - MeanP;
	Eigen::MatrixXd ReQ  = ConQ.colwise() - MeanQ;
	//Eigen::MatrixXd H = ReQ*(ReP.transpose());
	Eigen::MatrixXd H = ReP*(ReQ.transpose());
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	float det = (U * V.transpose()).determinant();
	Eigen::Vector3d diagVec(1.0, 1.0, det);//¶ÔœÇŸØÕó
	Eigen::MatrixXd R = V * diagVec.asDiagonal() * U.transpose();//È¥³ýŸµÃæŸØÕó
	//Eigen::MatrixXd R = H*((ReP*(ReP.transpose())).inverse());
	Eigen::VectorXd T = MeanQ - R*MeanP;

	Eigen::MatrixXd Transmatrix = Eigen::Matrix4d::Identity();
	Transmatrix.block(0, 0, 3, 3) = R; 
	Transmatrix.block(0, 3, 3, 1) = T;
	//cout << Transmatrix << endl;
	return Transmatrix;
}

Eigen::MatrixXd ICP::Transform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd Transmatrix)
{
	Eigen::initParallel();
	Eigen::MatrixXd R = Transmatrix.block(0, 0, 3, 3);
	Eigen::VectorXd T = Transmatrix.block(0, 3, 3, 1);
	
	Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
	return NewP;
}

Eigen::MatrixXd ICP::cloud2data(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	int nsize = cloud->size();
	Eigen::MatrixXd Q(3, nsize);
	Eigen::initParallel();
	omp_set_num_threads(8);
#pragma omp parallel for
	for (int i = 0; i < nsize; i++) {
			Q(0, i) = cloud->points[i].x;
			Q(1, i) = cloud->points[i].y;
			Q(2, i) = cloud->points[i].z;
		}
	return Q;
}

void ICP::icp(const PointCloudT::Ptr cloud_target,
	const PointCloudT::Ptr cloud_source,
	const Iter_para Iter, Eigen::Matrix4d &transformation_matrix)
{
	int nP = cloud_target->size();
	int nQ = cloud_source->size();
	Eigen::MatrixXd P = cloud2data(cloud_target);
	Eigen::MatrixXd Q = cloud2data(cloud_source);

	
	int i = 1;
	Eigen::MatrixXd ConP = P;
	Eigen::MatrixXd ConQ = Q;
	pcl::console::TicToc time;
	//time.tic();
	while(i<Iter.Maxiterate)
	{
		time.tic();
		double error=FindClosest(ConP, Q, ConQ);
		//std::cout <<  " FindClosest use time: " << time.toc() << " ms" << std::endl;
		//time.tic();
		transformation_matrix = GetTransform(ConP, ConQ);// 
		std::cout <<  " GetTransform use time: " << time.toc() << " ms" << std::endl;
		std::cout <<  " error: " << error<< std::endl;

		ConP = Transform(ConP, transformation_matrix);
		if (abs(error)<Iter.ControlN*Iter.acceptrate*Iter.threshold)
		//if(error/float(ConQ.cols())<0.01)
		{
			break;
		}
		i++;
		
	}
	transformation_matrix = GetTransform(P,ConP);
	std::cout<<transformation_matrix.matrix()<<std::endl;
}

void ICP::print4x4Matrix(const Eigen::Matrix4d & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

