/*
In this file, I read a point cloud data, generate the collision map and run some fmm algorithms on it.
*/
#include <iostream>
#include "stdio.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Eigen>
#include <math.h>
#include <random>
#include <string>

#include "trajectory_generator.h"
#include "bezier_base.h"
#include "data_type.h"
#include "utils.h"
#include "a_star.h"
#include "backward.hpp"

#include "quadrotor_msgs/PositionCommand.h"
#include "quadrotor_msgs/PolynomialTrajectory.h"

using namespace std;
using namespace Eigen;
using namespace sdf_tools;

pcl::PointCloud<pcl::PointXYZ> cloudMap;
sensor_msgs::PointCloud2 globalMap_pcd;
sensor_msgs::PointCloud2 localMap_pcd;
ros::Publisher _local_map_pub;
ros::Publisher _all_map_pub;

CollisionMapGrid * collision_map       = new CollisionMapGrid();
COLLISION_CELL _free_cell(0.0);
COLLISION_CELL _obst_cell(1.0);

CollisionMapGrid::DistanceField EDT;
bool EDT_inited = false;
FMGrid3D grid_fmm;
bool grid_fmm_inited = false;


double _MAX_Vel, _MAX_Acc;
int _max_x_id, _max_y_id, _max_z_id;
Vector3d _map_origin;
double _x_size, _y_size, _z_size, _resolution, _inv_resolution;
Vector3d _start_pt, _end_pt, _start_vel, _start_acc;
double _pt_max_x, _pt_min_x, _pt_max_y, _pt_min_y, _pt_max_z, _pt_min_z;
int _max_inflate_iter = 100, _step_length = 2;

std::string _pc_fnm;
bool   _is_use_fm, _is_proj_cube, _is_limit_vel, _is_limit_acc;
double _vis_traj_width;

quadrotor_msgs::PolynomialTrajectory _traj;
bool _has_traj, _is_emerg;

// ros related
ros::Publisher _fm_path_vis_pub, _local_map_vis_pub, _inf_map_vis_pub, _corridor_vis_pub, _traj_vis_pub, _grid_path_vis_pub, _nodes_vis_pub, _traj_pub, _checkTraj_vis_pub, _stopTraj_vis_pub;

void sortPath(vector<Vector3d> & path_coord, vector<double> & time);
void timeAllocation(vector<Cube> & corridor, vector<double> time);
void timeAllocation(vector<Cube> & corridor);
vector<Cube> corridorGenerationSmart(vector<Vector3d> path_coord, vector<double> time);
double velMapping(double d, double max_v);
Cube generateCube( Vector3d pt) ;
pair<Cube, bool> inflateCube(Cube cube, Cube lstcube);

// for visualization of start and goal
static  string mesh_resource;
ros::Publisher meshPub1;
visualization_msgs::Marker meshROS1;
ros::Publisher meshPub2;
visualization_msgs::Marker meshROS2;

void ReadMapDataset() 
{
   // In this function, I read one downloaded dataset and show it here... It may be messy.
   // char fnm[] = "/home/motion/OTT/conference_scaled.bin";
   //char fnm[] = "/home/motion/Downloads/pcd.dat";
   FILE *fp = fopen(_pc_fnm.c_str(), "rb");
   // get size
   //fseek(fp, 0L, SEEK_END);
   //long sz = ftell(fp);
   double tmp[3];
   pcl::PointXYZ pt_random;
   while(!feof(fp)) {
      fread(tmp, sizeof(double), 3, fp);
      pt_random.x = tmp[0];
      pt_random.y = tmp[1];
      pt_random.z = tmp[2];
      cloudMap.points.push_back( pt_random );
   }
   cloudMap.width = cloudMap.points.size();
   cloudMap.height = 1;
   cloudMap.is_dense = true;
   cloudMap.header.frame_id = "world";

   ROS_WARN("Finished read map dataset of size %d", cloudMap.size());   
   
   // with the point cloud, now I have to generate the collision data
   collision_map->RestMap();
   for(int i = 0; i < (int)cloudMap.points.size(); i++)
   {   
      pcl::PointXYZ inf_pt = cloudMap.points[i];
      Vector3d addPt(inf_pt.x, inf_pt.y, inf_pt.z);
      collision_map->Set3d(addPt, _obst_cell);
   }
}

visualization_msgs::MarkerArray path_vis; 
void visPath(vector<Vector3d> path)
{
    for(auto & mk: path_vis.markers) 
        mk.action = visualization_msgs::Marker::DELETE;

    _fm_path_vis_pub.publish(path_vis);
    path_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.ns = "b_traj/fast_marching_path";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.a = 0.6;
    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;

    int idx = 0;
    for(int i = 0; i < int(path.size()); i++)
    {
        mk.id = idx;

        mk.pose.position.x = path[i](0); 
        mk.pose.position.y = path[i](1); 
        mk.pose.position.z = path[i](2);  

        mk.scale.x = _resolution;
        mk.scale.y = _resolution;
        mk.scale.z = _resolution;

        idx ++;
        path_vis.markers.push_back(mk);
    }

    _fm_path_vis_pub.publish(path_vis);
}

visualization_msgs::MarkerArray cube_vis;
void visCorridor(vector<Cube> corridor)
{   
    ROS_INFO("Entering corridor visualization");
    for(auto & mk: cube_vis.markers) 
        mk.action = visualization_msgs::Marker::DELETE;
    
    _corridor_vis_pub.publish(cube_vis);

    cube_vis.markers.clear();

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.ns = "corridor";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;

    mk.color.a = 0.4;
    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 1.0;

    int idx = 0;
    for(int i = 0; i < int(corridor.size()); i++)
    {   
        mk.id = idx;

        mk.pose.position.x = (corridor[i].vertex(0, 0) + corridor[i].vertex(3, 0) ) / 2.0; 
        mk.pose.position.y = (corridor[i].vertex(0, 1) + corridor[i].vertex(1, 1) ) / 2.0; 

        if(_is_proj_cube)
            mk.pose.position.z = 0.0; 
        else
            mk.pose.position.z = (corridor[i].vertex(0, 2) + corridor[i].vertex(4, 2) ) / 2.0; 

        mk.scale.x = (corridor[i].vertex(0, 0) - corridor[i].vertex(3, 0) );
        mk.scale.y = (corridor[i].vertex(1, 1) - corridor[i].vertex(0, 1) );

        if(_is_proj_cube)
            mk.scale.z = 0.05; 
        else
            mk.scale.z = (corridor[i].vertex(0, 2) - corridor[i].vertex(4, 2) );

        idx ++;
        cube_vis.markers.push_back(mk);
    }

    _corridor_vis_pub.publish(cube_vis);
}

void find_path_fmm(){
   if(!EDT_inited) {
      ros::Time time_1 = ros::Time::now();
      EDT_inited = true;
      float oob_value = INFINITY;
      EDT = collision_map->ExtractDistanceField(oob_value);
      ros::Time time_2 = ros::Time::now();
      ROS_WARN("time in generate EDT is %f", (time_2 - time_1).toSec());
   }

   unsigned int idx;
   double max_vel = _MAX_Vel * 0.75; 
   vector<unsigned int> obs;            
   Vector3d pt;
   vector<int64_t> pt_idx;
   double flow_vel;

   unsigned int size_x = (unsigned int)(_max_x_id);
   unsigned int size_y = (unsigned int)(_max_y_id);
   unsigned int size_z = (unsigned int)(_max_z_id);

   if(!grid_fmm_inited) {
      grid_fmm_inited = true;
      Coord3D dimsize {size_x, size_y, size_z};
      grid_fmm.resize(dimsize);

      for(unsigned int k = 0; k < size_z; k++)
      {
         for(unsigned int j = 0; j < size_y; j++)
         {
             for(unsigned int i = 0; i < size_x; i++)
             {
                 idx = k * size_y * size_x + j * size_x + i;
                 pt << (i + 0.5) * _resolution + _map_origin(0), 
                       (j + 0.5) * _resolution + _map_origin(1), 
                       (k + 0.5) * _resolution + _map_origin(2);

                  Vector3i index = collision_map->LocationToGridIndex(pt);
                  double d = sqrt(EDT.GetImmutable(index).first.distance_square) * _resolution;
                  flow_vel = velMapping(d, max_vel);

                 if( k == 0 || k == (size_z - 1) || j == 0 || j == (size_y - 1) || i == 0 || i == (size_x - 1) )
                     flow_vel = 0.0;

                 grid_fmm[idx].setOccupancy(flow_vel);
                 if (grid_fmm[idx].isOccupied())
                     obs.push_back(idx);
             }
         }
      }
      grid_fmm.setOccupiedCells(std::move(obs));
      grid_fmm.setLeafSize(_resolution);
   }

    vector<Cube> corridor;
   Vector3d startIdx3d = (_start_pt - _map_origin) * _inv_resolution; 
   Vector3d endIdx3d   = (_end_pt   - _map_origin) * _inv_resolution;
   ROS_INFO("Init point is {%f %f %f}\n", startIdx3d[0], startIdx3d[1], startIdx3d[2]);
   ROS_INFO("End point is {%f %f %f}\n", endIdx3d[0], endIdx3d[1], endIdx3d[2]);

   Coord3D goal_point = {(unsigned int)startIdx3d[0], (unsigned int)startIdx3d[1], (unsigned int)startIdx3d[2]};
   Coord3D init_point = {(unsigned int)endIdx3d[0],   (unsigned int)endIdx3d[1],   (unsigned int)endIdx3d[2]}; 

   unsigned int startIdx;
   vector<unsigned int> startIndices;
   grid_fmm.coord2idx(init_point, startIdx);


   startIndices.push_back(startIdx);

   unsigned int goalIdx;
   grid_fmm.coord2idx(goal_point, goalIdx);
   double _init_occupacy = grid_fmm[goalIdx].getOccupancy();
   grid_fmm[goalIdx].setOccupancy(max_vel);     

   Solver<FMGrid3D>* fm_solver = new FMMStar<FMGrid3D>("FMM*_Dist", TIME); // LSM, FMM

   fm_solver->setEnvironment(&grid_fmm);

   fm_solver->setInitialAndGoalPoints(startIndices, goalIdx);

   ros::Time time_bef_fm = ros::Time::now();
   if(fm_solver->compute(max_vel) == -1)
   {
      ROS_WARN("[Fast Marching Node] No path can be found");
      _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
      // _traj_pub.publish(_traj);
      _has_traj = false;
      grid_fmm[goalIdx].setOccupancy(_init_occupacy); 
      return;
   }
   ros::Time time_aft_fm = ros::Time::now();
   ROS_WARN("[Fast Marching Node] Time in Fast Marching computing is %f", (time_aft_fm - time_bef_fm).toSec() );

   Path3D path3D;
   vector<double> path_vels, time;
   GradientDescent< FMGrid3D > grad3D;
   grid_fmm.coord2idx(goal_point, goalIdx);

   if(grad3D.gradient_descent(grid_fmm, goalIdx, path3D, path_vels, time) == -1)
   {
      ROS_WARN("[Fast Marching Node] FMM failed, valid path not exists");
      if(_has_traj && _is_emerg)
      {
          _traj.action = quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE;
          _traj_pub.publish(_traj);
          _has_traj = false;
      }
      grid_fmm[goalIdx].setOccupancy(_init_occupacy);
      return;
   }
   grid_fmm[goalIdx].setOccupancy(_init_occupacy);

   vector<Vector3d> path_coord;
   path_coord.push_back(_start_pt);

   double coord_x, coord_y, coord_z;
   for( int i = 0; i < (int)path3D.size(); i++)
   {
      coord_x = max(min( (path3D[i][0]+0.5) * _resolution + _map_origin(0), _x_size), -_x_size);
      coord_y = max(min( (path3D[i][1]+0.5) * _resolution + _map_origin(1), _y_size), -_y_size);
      coord_z = max(min( (path3D[i][2]+0.5) * _resolution, _z_size), 0.0);

      Vector3d pt(coord_x, coord_y, coord_z);
      path_coord.push_back(pt);
   }
   visPath(path_coord);

   ros::Time time_bef_corridor = ros::Time::now();    
   sortPath(path_coord, time);
   corridor = corridorGenerationSmart(path_coord, time);
   ros::Time time_aft_corridor = ros::Time::now();
   ROS_WARN("Time consume in corridor generation is %f", (time_aft_corridor - time_bef_corridor).toSec());

   timeAllocation(corridor, time);
    for(int i = 0; i < (int)corridor.size(); i++) {
        ROS_WARN("Corridor %d time %15.10f %f %f %f %f %f %f", i, corridor[i].t, 
            corridor[i].box[0].first, corridor[i].box[0].second, 
            corridor[i].box[1].first, corridor[i].box[1].second,
            corridor[i].box[2].first, corridor[i].box[2].second);
    }
   visCorridor(corridor);
   delete fm_solver;
}


void sortPath(vector<Vector3d> & path_coord, vector<double> & time)
{   
    vector<Vector3d> path_tmp;
    vector<double> time_tmp;

    for (int i = 0; i < (int)path_coord.size(); i += 1)
    {
        if( i )
            if( std::isinf(time[i]) || time[i] == 0.0 || time[i] == time[i-1] )
                continue;

        if( (path_coord[i] - _end_pt).norm() < 0.2)
            break;

        path_tmp.push_back(path_coord[i]);
        time_tmp.push_back(time[i]);
    }
    path_coord = path_tmp;
    time       = time_tmp;
}   

void timeAllocation(vector<Cube> & corridor, vector<double> time)
{   
    vector<double> tmp_time;

    for(int i  = 0; i < (int)corridor.size() - 1; i++)
    {   
        double duration  = (corridor[i].t - corridor[i+1].t);
        tmp_time.push_back(duration);
    }    
    double lst_time = corridor.back().t;
    tmp_time.push_back(lst_time);

    vector<Vector3d> points;
    points.push_back (_start_pt);
    for(int i = 1; i < (int)corridor.size(); i++)
        points.push_back(corridor[i].center);

    points.push_back (_end_pt);

    double _Vel = _MAX_Vel * 0.6;
    double _Acc = _MAX_Acc * 0.6;

    Eigen::Vector3d initv = _start_vel;
    for(int i = 0; i < (int)points.size() - 1; i++)
    {
        double dtxyz;

        Eigen::Vector3d p0   = points[i];    
        Eigen::Vector3d p1   = points[i + 1];
        Eigen::Vector3d d    = p1 - p0;            
        Eigen::Vector3d v0(0.0, 0.0, 0.0);        
        
        if( i == 0) v0 = initv;

        double D    = d.norm();                   
        double V0   = v0.dot(d / D);              
        double aV0  = fabs(V0);                   

        double acct = (_Vel - V0) / _Acc * ((_Vel > V0)?1:-1); 
        double accd = V0 * acct + (_Acc * acct * acct / 2) * ((_Vel > V0)?1:-1);
        double dcct = _Vel / _Acc;                                              
        double dccd = _Acc * dcct * dcct / 2;                                   

        if (D < aV0 * aV0 / (2 * _Acc))
        {                 
            double t1 = (V0 < 0)?2.0 * aV0 / _Acc:0.0;
            double t2 = aV0 / _Acc;
            dtxyz     = t1 + t2;                 
        }
        else if (D < accd + dccd)
        {
            double t1 = (V0 < 0)?2.0 * aV0 / _Acc:0.0;
            double t2 = (-aV0 + sqrt(aV0 * aV0 + _Acc * D - aV0 * aV0 / 2)) / _Acc;
            double t3 = (aV0 + _Acc * t2) / _Acc;
            dtxyz     = t1 + t2 + t3;    
        }
        else
        {
            double t1 = acct;                              
            double t2 = (D - accd - dccd) / _Vel;
            double t3 = dcct;
            dtxyz     = t1 + t2 + t3;
        }

        if(dtxyz < tmp_time[i] * 0.5)
            tmp_time[i] = dtxyz; // if FM given time in this segment is rediculous long, use the new value
    }

    for(int i = 0; i < (int)corridor.size(); i++)
        corridor[i].t = tmp_time[i];
}

vector<Cube> corridorGenerationSmart(vector<Vector3d> path_coord, vector<double> time)
{   
    vector<Cube> cubeList;
    Vector3d pt;

    Cube lstcube;
    double accumt = 0;
    //for(int i = 0; i < time.size() - 1; i++)
    //    time[i] = time[i] - time[i + 1];

    for (int i = 0; i < (int)path_coord.size(); i += 1)
    {
        pt = path_coord[i];
        accumt += time[i];
        // ROS_WARN("Path coordinates %f %f %f time %f", pt(0), pt(1), pt(2), time[i]);
        if(cubeList.size() > 0) {
            Cube &tail = cubeList.back();
            // if point is within the cube
            bool is_within = true;
            for(int j = 0; j < 3; j++) {
                if(pt(j) >= tail.box[j].first && pt(j) <= tail.box[j].second)
                    continue;
                is_within = false;
                cubeList.back().t = time[i - 1];
                //ROS_WARN("Time input is %f", cubeList.back().t);
                break;
            }
            if(is_within)
                continue;
        }

        Cube cube = generateCube(pt);
        auto result = inflateCube(cube, lstcube);

        if(result.second == false)
            continue;

        cube = result.first;
        
        lstcube = cube;
        // cube.t = time[i];
        cubeList.push_back(cube);
    }
    cubeList.back().t = time.back();
    // ROS_INFO("Corridor length %d", cubeList.size());
    return cubeList;
}

double velMapping(double d, double max_v)
{   
    double vel;

    if( d <= 0.25)
        vel = 2.0 * d * d;
    else if(d > 0.25 && d <= 0.75)
        vel = 1.5 * d - 0.25;
    else if(d > 0.75 && d <= 1.0)
        vel = - 2.0 * (d - 1.0) * (d - 1.0) + 1;  
    else
        vel = 1.0;

    return vel * max_v;
}

void rcvStartCallback(const geometry_msgs::PoseStamped & msg)
{    
    ROS_WARN("[Odom Generator] arbitraily change the start");
    _start_pt(0) = msg.pose.position.x;
    _start_pt(1) = msg.pose.position.y;
    _start_pt(2) = msg.pose.position.z;
    _start_vel.setZero();
    _start_acc.setZero();

    meshROS1.header.frame_id = "world";
    meshROS1.header.stamp = ros::Time::now(); 
    meshROS1.ns = "mesh";
    meshROS1.id = 0;
    meshROS1.type = visualization_msgs::Marker::MESH_RESOURCE;
    meshROS1.action = visualization_msgs::Marker::ADD;
    meshROS1.pose.position.x = msg.pose.position.x;
    meshROS1.pose.position.y = msg.pose.position.y;
    meshROS1.pose.position.z = msg.pose.position.z;
    double scale = 2;
    meshROS1.scale.x = scale;
    meshROS1.scale.y = scale;
    meshROS1.scale.z = scale;
    meshROS1.color.a = 1;
    meshROS1.color.r = 1;
    meshROS1.color.g = 0;
    meshROS1.color.b = 0;
    meshROS1.mesh_resource = mesh_resource;
    meshPub1.publish(meshROS1);                                                  

    find_path_fmm();  // whenever start is changed
}

void rcvGoalCallback(const geometry_msgs::PoseStamped & msg)
{    
    ROS_WARN("[Odom Generator] arbitraily change the start");
    _end_pt(0) = msg.pose.position.x;
    _end_pt(1) = msg.pose.position.y;
    _end_pt(2) = msg.pose.position.z;

    meshROS2.header.frame_id = "world";
    meshROS2.header.stamp = ros::Time::now(); 
    meshROS2.ns = "mesh";
    meshROS2.id = 0;
    meshROS2.type = visualization_msgs::Marker::MESH_RESOURCE;
    meshROS2.action = visualization_msgs::Marker::ADD;
    meshROS2.pose.position.x = msg.pose.position.x;
    meshROS2.pose.position.y = msg.pose.position.y;
    meshROS2.pose.position.z = msg.pose.position.z;
    double scale = 2;
    meshROS2.scale.x = scale;
    meshROS2.scale.y = scale;
    meshROS2.scale.z = scale;
    meshROS2.color.a = 1;
    meshROS2.color.r = 0;
    meshROS2.color.g = 0;
    meshROS2.color.b = 0;
    meshROS2.mesh_resource = mesh_resource;
    meshPub2.publish(meshROS2);                                                  

    find_path_fmm(); // whenever goal is changed
}

void pubSensedPoints()
{ 

   /* Modification by Gao so the map is never changed */
   //pcl::toROSMsg(cloudMap, globalMap_pcd);
   static bool inited = false;
   if(!inited) {
        pcl::toROSMsg(cloudMap, localMap_pcd);
        inited = true;
   }
   //localMap_pcd.header.frame_id = "world";
   //_all_map_pub.publish(globalMap_pcd);
   _local_map_pub.publish(localMap_pcd);
   return;
}

int main (int argc, char** argv) {
   ros::init (argc, argv, "field_server");
   // read file name from launch file...
   ros::NodeHandle n( "~" );
   std::string default_str;
   n.param("pcdfnm", _pc_fnm, default_str);                  
   
   ros::Subscriber _start_sub = n.subscribe("/start", 1, rcvStartCallback);
   ros::Subscriber _goal_sub = n.subscribe("/goal", 1, rcvGoalCallback);

    _local_map_pub = n.advertise<sensor_msgs::PointCloud2>("random_forest", 1);                      
    _all_map_pub   = n.advertise<sensor_msgs::PointCloud2>("all_map", 1);                      

    _traj_vis_pub      = n.advertise<visualization_msgs::Marker>("trajectory_vis", 1);    
    _corridor_vis_pub  = n.advertise<visualization_msgs::MarkerArray>("corridor_vis", 1);
    _fm_path_vis_pub   = n.advertise<visualization_msgs::MarkerArray>("path_vis", 1);

    meshPub1   = n.advertise<visualization_msgs::Marker>("start_robot",               100, true);  
    meshPub2   = n.advertise<visualization_msgs::Marker>("goal_robot",               100, true);  
   
   n.param("mesh_resource", mesh_resource, std::string("package://odom_visualization/meshes/hummingbird.mesh"));

   n.param("init_state_x", _start_pt(0),       0.0);
   n.param("init_state_y", _start_pt(1),       0.0);
   n.param("init_state_z", _start_pt(2),       0.0);
   n.param("goal_state_x", _end_pt(0),       0.0);
   n.param("goal_state_y", _end_pt(1),       0.0);
   n.param("goal_state_z", _end_pt(2),       0.0);

   n.param("x_size",  _x_size, 50.0);
   n.param("y_size",  _y_size, 50.0);
   n.param("z_size",  _z_size, 5.0 );
   n.param("resolution",  _resolution, 0.2);

    n.param("vis_traj_width", _vis_traj_width, 0.15);
    n.param("is_proj_cube",   _is_proj_cube, true);

    n.param("planning/max_vel",       _MAX_Vel,  1.0);
    n.param("planning/max_acc",       _MAX_Acc,  1.0);

   _map_origin << -_x_size / 2.0, -_y_size / 2.0, 0.0;
    _pt_max_x = + _x_size / 2.0;
    _pt_min_x = - _x_size / 2.0;
    _pt_max_y = + _y_size / 2.0;
    _pt_min_y = - _y_size / 2.0; 
    _pt_max_z = + _z_size;
    _pt_min_z = 0.0;
   _inv_resolution = 1.0 / _resolution;
   _max_x_id = (int)(_x_size * _inv_resolution);
   _max_y_id = (int)(_y_size * _inv_resolution);
   _max_z_id = (int)(_z_size * _inv_resolution);
   
   Translation3d origin_translation( _map_origin(0), _map_origin(1), 0.0);
   Quaterniond origin_rotation(1.0, 0.0, 0.0, 0.0);
   Affine3d origin_transform = origin_translation * origin_rotation;
   collision_map = new CollisionMapGrid(origin_transform, "world", _resolution, _x_size, _y_size, _z_size, _free_cell);

   ReadMapDataset();

   double lrate = 1;
   n.param("loop_rate",   lrate, 1.);

   ros::Rate loop_rate(lrate);

   while (ros::ok())
   {
     pubSensedPoints();
     ros::spinOnce();
     loop_rate.sleep();
   }
}

bool isContains(Cube cube1, Cube cube2)
{   
    if( cube1.vertex(0, 0) >= cube2.vertex(0, 0) && cube1.vertex(0, 1) <= cube2.vertex(0, 1) && cube1.vertex(0, 2) >= cube2.vertex(0, 2) &&
        cube1.vertex(6, 0) <= cube2.vertex(6, 0) && cube1.vertex(6, 1) >= cube2.vertex(6, 1) && cube1.vertex(6, 2) <= cube2.vertex(6, 2)  )
        return true;
    else
        return false; 
}

pair<Cube, bool> inflateCube(Cube cube, Cube lstcube)
{   
    Cube cubeMax = cube;

    // Inflate sequence: left, right, front, back, below, above                                                                                
    MatrixXi vertex_idx(8, 3);
    for (int i = 0; i < 8; i++)
    { 
        double coord_x = max(min(cube.vertex(i, 0), _pt_max_x), _pt_min_x);
        double coord_y = max(min(cube.vertex(i, 1), _pt_max_y), _pt_min_y);
        double coord_z = max(min(cube.vertex(i, 2), _pt_max_z), _pt_min_z);
        Vector3d coord(coord_x, coord_y, coord_z);

        Vector3i pt_idx = collision_map->LocationToGridIndex(coord);

        if( collision_map->Get( (int64_t)pt_idx(0), (int64_t)pt_idx(1), (int64_t)pt_idx(2) ).first.occupancy > 0.5 )
        {       
            ROS_ERROR("[Planning Node] path has node in obstacles !");
            return make_pair(cubeMax, false);
        }
        
        vertex_idx.row(i) = pt_idx;
    }

    int id_x, id_y, id_z;

    /*
               P4------------P3 
               /|           /|              ^
              / |          / |              | z
            P1--|---------P2 |              |
             |  P8--------|--p7             |
             | /          | /               /--------> y
             |/           |/               /  
            P5------------P6              / x
    */           

    // Y- now is the left side : (p1 -- p4 -- p8 -- p5) face sweep
    // ############################################################################################################
    bool collide;

    MatrixXi vertex_idx_lst = vertex_idx;

    int iter = 0;
    while(iter < _max_inflate_iter)
    {   
        collide  = false; 
        int y_lo = max(0, vertex_idx(0, 1) - _step_length);
        int y_up = min(_max_y_id, vertex_idx(1, 1) + _step_length);

        for(id_y = vertex_idx(0, 1); id_y >= y_lo; id_y-- )
        {   
            if( collide == true) 
                break;
            
            for(id_x = vertex_idx(0, 0); id_x >= vertex_idx(3, 0); id_x-- )
            {    
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(0, 2); id_z >= vertex_idx(4, 2); id_z-- )
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(0, 1) = min(id_y+2, vertex_idx(0, 1));
            vertex_idx(3, 1) = min(id_y+2, vertex_idx(3, 1));
            vertex_idx(7, 1) = min(id_y+2, vertex_idx(7, 1));
            vertex_idx(4, 1) = min(id_y+2, vertex_idx(4, 1));
        }
        else
            vertex_idx(0, 1) = vertex_idx(3, 1) = vertex_idx(7, 1) = vertex_idx(4, 1) = id_y + 1;
        
        // Y+ now is the right side : (p2 -- p3 -- p7 -- p6) face
        // ############################################################################################################
        collide = false;
        for(id_y = vertex_idx(1, 1); id_y <= y_up; id_y++ )
        {   
            if( collide == true) 
                break;
            
            for(id_x = vertex_idx(1, 0); id_x >= vertex_idx(2, 0); id_x-- )
            {
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(1, 2); id_z >= vertex_idx(5, 2); id_z-- )
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(1, 1) = max(id_y-2, vertex_idx(1, 1));
            vertex_idx(2, 1) = max(id_y-2, vertex_idx(2, 1));
            vertex_idx(6, 1) = max(id_y-2, vertex_idx(6, 1));
            vertex_idx(5, 1) = max(id_y-2, vertex_idx(5, 1));
        }
        else
            vertex_idx(1, 1) = vertex_idx(2, 1) = vertex_idx(6, 1) = vertex_idx(5, 1) = id_y - 1;

        // X + now is the front side : (p1 -- p2 -- p6 -- p5) face
        // ############################################################################################################
        int x_lo = max(0, vertex_idx(3, 0) - _step_length);
        int x_up = min(_max_x_id, vertex_idx(0, 0) + _step_length);

        collide = false;
        for(id_x = vertex_idx(0, 0); id_x <= x_up; id_x++ )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(0, 1); id_y <= vertex_idx(1, 1); id_y++ )
            {
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(0, 2); id_z >= vertex_idx(4, 2); id_z-- )
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(0, 0) = max(id_x-2, vertex_idx(0, 0)); 
            vertex_idx(1, 0) = max(id_x-2, vertex_idx(1, 0)); 
            vertex_idx(5, 0) = max(id_x-2, vertex_idx(5, 0)); 
            vertex_idx(4, 0) = max(id_x-2, vertex_idx(4, 0)); 
        }
        else
            vertex_idx(0, 0) = vertex_idx(1, 0) = vertex_idx(5, 0) = vertex_idx(4, 0) = id_x - 1;    

        // X- now is the back side : (p4 -- p3 -- p7 -- p8) face
        // ############################################################################################################
        collide = false;
        for(id_x = vertex_idx(3, 0); id_x >= x_lo; id_x-- )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(3, 1); id_y <= vertex_idx(2, 1); id_y++ )
            {
                if( collide == true) 
                    break;

                for(id_z = vertex_idx(3, 2); id_z >= vertex_idx(7, 2); id_z-- )
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(3, 0) = min(id_x+2, vertex_idx(3, 0)); 
            vertex_idx(2, 0) = min(id_x+2, vertex_idx(2, 0)); 
            vertex_idx(6, 0) = min(id_x+2, vertex_idx(6, 0)); 
            vertex_idx(7, 0) = min(id_x+2, vertex_idx(7, 0)); 
        }
        else
            vertex_idx(3, 0) = vertex_idx(2, 0) = vertex_idx(6, 0) = vertex_idx(7, 0) = id_x + 1;

        // Z+ now is the above side : (p1 -- p2 -- p3 -- p4) face
        // ############################################################################################################
        collide = false;
        int z_lo = max(0, vertex_idx(4, 2) - _step_length);
        int z_up = min(_max_z_id, vertex_idx(0, 2) + _step_length);
        for(id_z = vertex_idx(0, 2); id_z <= z_up; id_z++ )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(0, 1); id_y <= vertex_idx(1, 1); id_y++ )
            {
                if( collide == true) 
                    break;

                for(id_x = vertex_idx(0, 0); id_x >= vertex_idx(3, 0); id_x-- )
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(0, 2) = max(id_z-2, vertex_idx(0, 2));
            vertex_idx(1, 2) = max(id_z-2, vertex_idx(1, 2));
            vertex_idx(2, 2) = max(id_z-2, vertex_idx(2, 2));
            vertex_idx(3, 2) = max(id_z-2, vertex_idx(3, 2));
        }
        vertex_idx(0, 2) = vertex_idx(1, 2) = vertex_idx(2, 2) = vertex_idx(3, 2) = id_z - 1;

        // now is the below side : (p5 -- p6 -- p7 -- p8) face
        // ############################################################################################################
        collide = false;
        for(id_z = vertex_idx(4, 2); id_z >= z_lo; id_z-- )
        {   
            if( collide == true) 
                break;
            
            for(id_y = vertex_idx(4, 1); id_y <= vertex_idx(5, 1); id_y++ )
            {
                if( collide == true) 
                    break;

                for(id_x = vertex_idx(4, 0); id_x >= vertex_idx(7, 0); id_x-- )
                {
                    double occupy = collision_map->Get( (int64_t)id_x, (int64_t)id_y, (int64_t)id_z).first.occupancy;    
                    if(occupy > 0.5) // the voxel is occupied
                    {   
                        collide = true;
                        break;
                    }
                }
            }
        }

        if(collide)
        {
            vertex_idx(4, 2) = min(id_z+2, vertex_idx(4, 2));
            vertex_idx(5, 2) = min(id_z+2, vertex_idx(5, 2));
            vertex_idx(6, 2) = min(id_z+2, vertex_idx(6, 2));
            vertex_idx(7, 2) = min(id_z+2, vertex_idx(7, 2));
        }
        else
            vertex_idx(4, 2) = vertex_idx(5, 2) = vertex_idx(6, 2) = vertex_idx(7, 2) = id_z + 1;

        if(vertex_idx_lst == vertex_idx)
            break;

        vertex_idx_lst = vertex_idx;

        MatrixXd vertex_coord(8, 3);
        for(int i = 0; i < 8; i++)
        {   
            int index_x = max(min(vertex_idx(i, 0), _max_x_id - 1), 0);
            int index_y = max(min(vertex_idx(i, 1), _max_y_id - 1), 0);
            int index_z = max(min(vertex_idx(i, 2), _max_z_id - 1), 0);

            Vector3i index(index_x, index_y, index_z);
            Vector3d pos = collision_map->GridIndexToLocation(index);
            vertex_coord.row(i) = pos;
        }

        cubeMax.setVertex(vertex_coord, _resolution);
        if( isContains(lstcube, cubeMax))        
            return make_pair(lstcube, false);

        iter ++;
    }

    return make_pair(cubeMax, true);
}
Cube generateCube( Vector3d pt) 
{   
/*
           P4------------P3 
           /|           /|              ^
          / |          / |              | z
        P1--|---------P2 |              |
         |  P8--------|--p7             |
         | /          | /               /--------> y
         |/           |/               /  
        P5------------P6              / x
*/       
    Cube cube;
    
    pt(0) = max(min(pt(0), _pt_max_x), _pt_min_x);
    pt(1) = max(min(pt(1), _pt_max_y), _pt_min_y);
    pt(2) = max(min(pt(2), _pt_max_z), _pt_min_z);

    Vector3i pc_index = collision_map->LocationToGridIndex(pt);    
    Vector3d pc_coord = collision_map->GridIndexToLocation(pc_index);

    cube.center = pc_coord;
    double x_u = pc_coord(0);
    double x_l = pc_coord(0);
    
    double y_u = pc_coord(1);
    double y_l = pc_coord(1);
    
    double z_u = pc_coord(2);
    double z_l = pc_coord(2);

    cube.vertex.row(0) = Vector3d(x_u, y_l, z_u);  
    cube.vertex.row(1) = Vector3d(x_u, y_u, z_u);  
    cube.vertex.row(2) = Vector3d(x_l, y_u, z_u);  
    cube.vertex.row(3) = Vector3d(x_l, y_l, z_u);  

    cube.vertex.row(4) = Vector3d(x_u, y_l, z_l);  
    cube.vertex.row(5) = Vector3d(x_u, y_u, z_l);  
    cube.vertex.row(6) = Vector3d(x_l, y_u, z_l);  
    cube.vertex.row(7) = Vector3d(x_l, y_l, z_l);  

    return cube;
}
