
#include "icp.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h> 


PointCloudT::Ptr ReadFile(std::string FileName)
{

	PointCloudT::Ptr cloud(new PointCloudT);
	if (pcl::io::loadPCDFile(FileName, *cloud) < 0)
	{
		PCL_ERROR("Error loading cloud %s.\n", FileName);
	}
	return cloud;
}


int main()
{
	
	PointCloudT::Ptr cloud_in(new PointCloudT);  // Original point cloud
	PointCloudT::Ptr cloud_tr(new PointCloudT);  // Transformed point cloud
	PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud
	PointCloudT::Ptr cloud_filtered (new  PointCloudT);

	pcl::io::loadPCDFile("/home/tao/learning/icp_openmp_kdtree/data/cloud0.pcd", *cloud_in);

	pcl::VoxelGrid<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud_in);
	filter.setLeafSize(0.5f, 0.5f, 0.5f);
	filter.filter(*cloud_filtered);
	cout << "point size::" << cloud_filtered->size() << endl;
	

	// Defining a rotation matrix and translation vector
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d matrix_icp = Eigen::Matrix4d::Identity();

	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	double theta = M_PI / 8;  // The angle of rotation in radians
	transformation_matrix(0, 0) = cos(theta);
	transformation_matrix(0, 1) = -sin(theta);
	transformation_matrix(1, 0) = sin(theta);
	transformation_matrix(1, 1) = cos(theta);
	transformation_matrix(2, 3) = 4;
	transformation_matrix(1, 3) = 1.2;
	transformation_matrix(0, 3) = 0.6;
	pcl::transformPointCloud(*cloud_filtered, *cloud_icp, transformation_matrix);



	*cloud_tr = *cloud_icp;  // We backup cloud_icp into cloud_tr for later use

	//icp algorithm
	pcl::console::TicToc time;
	time.tic();
	Iter_para iter{ cloud_filtered->size(),100,0.01,0.8 };//
	ICP icp;

	icp.print4x4Matrix(transformation_matrix);
	icp.icp( cloud_filtered, cloud_icp, iter, matrix_icp);

	std::cout <<  " ICP use time: " << time.toc() << " ms" << std::endl;

    return 0;
}