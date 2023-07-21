//
// Created by xiang on 2022/7/14.
//

#include "ndt_3d.h"
#include "common/math_utils.h"

#include <glog/logging.h>
#include <Eigen/SVD>
#include <execution>
#include <fstream>
// #include <nlohmann/json.hpp>
#include "json.hpp"

using json = nlohmann::json;

namespace sad {

void Ndt3d::BuildVoxels() {
    assert(target_ != nullptr);
    assert(target_->empty() == false);
    grids_.clear();

    /// 分配体素
    std::vector<size_t> index(target_->size());
    std::for_each(index.begin(), index.end(), [idx = 0](size_t& i) mutable { i = idx++; });

    std::for_each(index.begin(), index.end(), [this](const size_t& idx) {
        auto pt = ToVec3d(target_->points[idx]);
        auto key = (pt * options_.inv_voxel_size_).cast<int>();
        if (grids_.find(key) == grids_.end()) {
            grids_.insert({key, {idx}});
        } else {
            grids_[key].idx_.emplace_back(idx);
        }
    });

    /// 计算每个体素中的均值和协方差
    std::for_each(std::execution::par_unseq, grids_.begin(), grids_.end(), [this](auto& v) {
        if (v.second.idx_.size() > options_.min_pts_in_voxel_) {
            // 要求至少有３个点
            math::ComputeMeanAndCov(v.second.idx_, v.second.mu_, v.second.sigma_,
                                    [this](const size_t& idx) { return ToVec3d(target_->points[idx]); });
            // SVD 检查最大与最小奇异值，限制最小奇异值

            Eigen::JacobiSVD svd(v.second.sigma_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vec3d lambda = svd.singularValues();
            if (lambda[1] < lambda[0] * 1e-3) {
                lambda[1] = lambda[0] * 1e-3;
            }

            if (lambda[2] < lambda[0] * 1e-3) {
                lambda[2] = lambda[0] * 1e-3;
            }

            Mat3d inv_lambda = Vec3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2]).asDiagonal();

            // v.second.info_ = (v.second.sigma_ + Mat3d::Identity() * 1e-3).inverse();  // 避免出nan
            v.second.info_ = svd.matrixV() * inv_lambda * svd.matrixU().transpose();
        }
    });

    /// 删除点数不够的
    for (auto iter = grids_.begin(); iter != grids_.end();) {
        if (iter->second.idx_.size() > options_.min_pts_in_voxel_) {
            iter++;
        } else {
            iter = grids_.erase(iter);
        }
    }
}

bool Ndt3d::AlignNdt(SE3& init_pose) {
    LOG(INFO) << "aligning with ndt";
    assert(grids_.empty() == false);

    SE3 pose = init_pose;
    if (options_.remove_centroid_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
        LOG(INFO) << "init trans set to " << pose.translation().transpose();
    }

    // 对点的索引，预先生成
    int num_residual_per_point = 1;
    if (options_.nearby_type_ == NearbyType::NEARBY6) {
        num_residual_per_point = 7;
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // 我们来写一些并发代码
    int total_size = index.size() * num_residual_per_point;

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        std::vector<bool> effect_pts(total_size, false);
        std::vector<Eigen::Matrix<double, 3, 6>> jacobians(total_size);
        std::vector<Vec3d> errors(total_size);
        std::vector<Mat3d> infos(total_size);

        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q

            // 计算qs所在的栅格以及它的最近邻栅格
            Vec3i key = (qs * options_.inv_voxel_size_).cast<int>();

            for (int i = 0; i < nearby_grids_.size(); ++i) {
                auto key_off = key + nearby_grids_[i];
                auto it = grids_.find(key_off);
                int real_idx = idx * num_residual_per_point + i;
                if (it != grids_.end()) {
                    auto& v = it->second;  // voxel
                    Vec3d e = qs - v.mu_;

                    // check chi2 th
                    double res = e.transpose() * v.info_ * e;
                    if (std::isnan(res) || res > options_.res_outlier_th_) {
                        effect_pts[real_idx] = false;
                        continue;
                    }

                    // build residual
                    Eigen::Matrix<double, 3, 6> J;
                    J.block<3, 3>(0, 0) = -pose.so3().matrix() * SO3::hat(q);
                    J.block<3, 3>(0, 3) = Mat3d::Identity();

                    jacobians[real_idx] = J;
                    errors[real_idx] = e;
                    infos[real_idx] = v.info_;
                    effect_pts[real_idx] = true;
                } else {
                    effect_pts[real_idx] = false;
                }
            }
        });

        // calc matching score
        matching_score_ = 0;
        int count=0;
        // first of nearby_grids_ is origin
        for (int i = 0; i < total_size; i += nearby_grids_.size()) {
            if (effect_pts[i] == false) {
                continue;
            }
            auto e = errors[i];
            auto info = infos[i];

            double x = e.transpose() * info * e;
            double p = exp(-x/2);
            matching_score_ += p;
            count++;
        }
#if 0
        LOG(INFO) << "valid count " << count;
        if (count) {
            matching_score_/=count;
        }
#endif

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;

        Mat6d H = Mat6d::Zero();
        Vec6d err = Vec6d::Zero();

        for (int idx = 0; idx < effect_pts.size(); ++idx) {
            if (!effect_pts[idx]) {
                continue;
            }

            total_res += errors[idx].transpose() * infos[idx] * errors[idx];
            // chi2.emplace_back(errors[idx].transpose() * infos[idx] * errors[idx]);
            effective_num++;

            H += jacobians[idx].transpose() * infos[idx] * jacobians[idx];
            err += -jacobians[idx].transpose() * infos[idx] * errors[idx];
        }

        if (effective_num < options_.min_effective_pts_) {
            LOG(WARNING) << "effective num too small: " << effective_num;
            return false;
        }

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        LOG(INFO) << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                  << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm()
                  << ", dx: " << dx.transpose();

        // std::sort(chi2.begin(), chi2.end());
        // LOG(INFO) << "chi2 med: " << chi2[chi2.size() / 2] << ", .7: " << chi2[chi2.size() * 0.7]
        //           << ", .9: " << chi2[chi2.size() * 0.9] << ", max: " << chi2.back();

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            LOG(INFO) << "iter " << iter << " pose error: " << pose_error;
        }

        if (dx.norm() < options_.eps_) {
            LOG(INFO) << "converged, dx = " << dx.transpose();
            break;
        }
    }

    init_pose = pose;
    return true;
}

double Ndt3d::GetMatchingScore()
{
    return matching_score_;
}

bool Ndt3d::SaveToFile(std::string&& path)
{
    json j;

    j["option"] = {
        { "max_iteration", options_.max_iteration_},
        { "voxel_size", options_.voxel_size_},
        // { "inv_voxel_size", options_.inv_voxel_size_},
        { "min_effective_pts", options_.min_effective_pts_},
        { "min_pts_in_voxel", options_.min_pts_in_voxel_},
        { "eps", options_.eps_},
        { "res_outlier_th", options_.res_outlier_th_},
        { "remove_centroid", options_.remove_centroid_},
        { "nearby_type", options_.nearby_type_},
    };

    json j_grids = json::array();
    for (auto & grid : grids_) {
        auto& key = grid.first;
        auto& voxel_data = grid.second;

        std::vector<double> mu(voxel_data.mu_.data(),
                voxel_data.mu_.data()+voxel_data.mu_.size());
        std::vector<double> sigma(voxel_data.sigma_.data(),
                voxel_data.sigma_.data()+voxel_data.sigma_.size());
        std::vector<double> info(voxel_data.info_.data(),
                voxel_data.info_.data()+voxel_data.info_.size());
        json j_grid = {
            {"key", json::array({key[0], key[1], key[2]})},
            {"voxel_data", {
                {"idx", voxel_data.idx_},
                {"mu", mu},
                {"sigma", sigma},
                {"info", info},
            }},
        };

        j_grids.push_back(j_grid);
    }

    json j_points = json::array();
    for (auto &pt : target_->points) {
        json j_point = json::array({
            pt.x, pt.y, pt.z, pt.intensity
        });
        j_points.push_back(j_point);
    }

    j["grids"] = j_grids;
    j["target_center"] = json::array({
        target_center_[0],
        target_center_[1],
        target_center_[2],
    });
    j["target"] = j_points;

#if 1
    std::ofstream o(path + ".json");
    o << std::setw(4) << j << std::endl;
    o.close();
#endif

    std::vector<std::uint8_t> v_cbor = json::to_cbor(j);

    std::ofstream o_cbor(path, std::ios::out | std::ios::binary | std::ios::trunc);
    o_cbor.write((char *)v_cbor.data(), v_cbor.size());
    o_cbor.close();

    return true;
}

bool Ndt3d::LoadFromFile(std::string&& path)
{
    std::ifstream instream(path, std::ios::in | std::ios::binary);
    if (instream.is_open()) {
        LOG(ERROR) << "can not open " << path;
        return false;
    }

    std::istreambuf_iterator<char> fs_begin(instream), end;
    std::vector<std::uint8_t> data(fs_begin, end);

    if (!data.size()) {
        LOG(ERROR) << "file empty " << path;
        return false;
    }

    grids_.clear();
    CloudPtr cloud(new PointCloudType);
    Options option;

    json j = json::from_cbor(data);

    return true;
}

void Ndt3d::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    }
}

}  // namespace sad