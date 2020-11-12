#pragma once
#include <string>
#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/time.h>   // TicToc
#include <omp.h>
#include <numeric>
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct Iter_para //Interation paraments
{
	int ControlN;
	int Maxiterate;
	double threshold;
	double acceptrate;

};

class ICP{


public:
    

    ICP(){}

    PointCloudT::Ptr data2cloud(const Eigen::MatrixXd data);

    double FindClosest(const Eigen::MatrixXd  cloud_target,	const Eigen::MatrixXd cloud_source,  Eigen::MatrixXd &ConQ);

    Eigen::Matrix4d GetTransform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd ConQ);

    Eigen::MatrixXd Transform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd Transmatrix);

    Eigen::MatrixXd cloud2data(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void icp(const PointCloudT::Ptr cloud_target,const PointCloudT::Ptr cloud_source,
	                                                    const Iter_para Iter, Eigen::Matrix4d &transformation_matrix);

    void print4x4Matrix(const Eigen::Matrix4d & matrix);



public:

   pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_search;
   Eigen::Matrix4d transformation_matrix;


};
