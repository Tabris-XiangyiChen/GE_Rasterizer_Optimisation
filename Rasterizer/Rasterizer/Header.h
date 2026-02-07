//#pragma once
//
//
//// 优化的渲染流程
////void renderScene_SoA_Optimized(Renderer& renderer, ThreadPool2& pool,
////    Mesh_SoA& mesh, matrix& camera, Light& light, float ka, float kd) {
////    const int vCount = (int)mesh.positions_x.size();
////    const int tileCount = tiles.size();
//
////    // 1. 并行变换顶点数据（多线程）
////    int chunkSize = (vCount + pool.getThreadCount() - 1) / pool.getThreadCount();
////    Mesh_SoA_Transformed transformed_mesh;
////    transformed_mesh.triangles = mesh.triangles;
////    transformed_mesh.ka = ka;
////    transformed_mesh.kd = kd;
//
////    //pool.beginFrame();
////    for (int chunk_start = 0; chunk_start < vCount; chunk_start += chunkSize) {
////        int chunk_end = std::min(chunk_start + chunkSize, vCount);
//
////        pool.enqueue([&, chunk_start, chunk_end]() {
////            // 每个线程处理一部分顶点
////            matrix p = renderer.perspective * camera * mesh.world;
////            transformed_mesh.transformBatchSIMD(p, mesh.world, mesh,
////                chunk_start, chunk_end - chunk_start);
////            });
////    }
////    //pool.waitFrame();
////    pool.waitAll();
//
////    // 2. 三角形筛选到瓦片（多线程）
////    std::vector<std::vector<int>> triangle_buckets(tileCount);
//
////    //pool.beginFrame();
////    const int tri_chunk = 64;
////    for (int chunk_start = 0; chunk_start < transformed_mesh.triangles.size(); chunk_start += tri_chunk) {
////        int chunk_end = std::min(chunk_start + tri_chunk, (int)transformed_mesh.triangles.size());
//
////        pool.enqueue([&, chunk_start, chunk_end]() {
////            for (int t = chunk_start; t < chunk_end; ++t) {
////                triIndices& idx = transformed_mesh.triangles[t];
//
////                // 从SOA数据获取包围盒
////                float min_x = std::min({
////                    transformed_mesh.transformed_positions_x[idx.v[0]],
////                    transformed_mesh.transformed_positions_x[idx.v[1]],
////                    transformed_mesh.transformed_positions_x[idx.v[2]]
////                    });
////                float max_x = std::max({
////                    transformed_mesh.transformed_positions_x[idx.v[0]],
////                    transformed_mesh.transformed_positions_x[idx.v[1]],
////                    transformed_mesh.transformed_positions_x[idx.v[2]]
////                    });
////                float min_y = std::min({
////                    transformed_mesh.transformed_positions_y[idx.v[0]],
////                    transformed_mesh.transformed_positions_y[idx.v[1]],
////                    transformed_mesh.transformed_positions_y[idx.v[2]]
////                    });
////                float max_y = std::max({
////                    transformed_mesh.transformed_positions_y[idx.v[0]],
////                    transformed_mesh.transformed_positions_y[idx.v[1]],
////                    transformed_mesh.transformed_positions_y[idx.v[2]]
////                    });
//
////                // 确定覆盖的瓦片
////                int min_tx = std::max(0, (int)min_x / TILE_SIZE);
////                int max_tx = std::min(tiles_x - 1, (int)max_x / TILE_SIZE);
////                int min_ty = std::max(0, (int)min_y / TILE_SIZE);
////                int max_ty = std::min(tiles_y - 1, (int)max_y / TILE_SIZE);
//
////                // 添加到瓦片桶
////                for (int ty = min_ty; ty <= max_ty; ++ty) {
////                    for (int tx = min_tx; tx <= max_tx; ++tx) {
////                        triangle_buckets[ty * tiles_x + tx].push_back(t);
////                    }
////                }
////            }
////            });
////    }
////    //pool.waitFrame();
////    pool.waitAll();
//
////    // 3. 并行瓦片渲染
////    //pool.beginFrame();
////    for (size_t tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
////        if (!triangle_buckets[tile_idx].empty()) {
////            pool.enqueue([&, tile_idx]() {
////                Tile& tile = tiles[tile_idx];
////                std::fill(tile.zbuffer.begin(), tile.zbuffer.end(),
////                    std::numeric_limits<float>::max());
////                std::fill(tile.colors.begin(), tile.colors.end(), 0);
//
////                for (int tri_idx : triangle_buckets[tile_idx]) {
////                    //renderTriangleFromSOA(transformed_mesh, light, ka, kd, tri_idx, tile);
////                    //renderTriangleFromSOA(transformed_mesh, light, ka, kd, tri_idx, tile);
////                }
////                });
////        }
////    }
////    //pool.waitFrame();
////    pool.waitAll();
//
////    // 4. 合并瓦片
////    mergeTilesToCanvas(renderer);
////}
//
//void renderScene(Renderer& renderer, Mesh_SoA& mesh, matrix& camera, Light& light, float ka, float kd) {
//    // 第一阶段：三角形筛选和分发
//    std::vector<std::vector<int>> triangle_buckets(tiles.size());
//
//    pool.beginFrame();
//
//    // 并行筛选三角形到瓦片
//    const int triangle_chunk = 64;  // 每批处理的三角形数量
//    for (int chunk_start = 0; chunk_start < mesh.triangles.size(); chunk_start += triangle_chunk) {
//        int chunk_end = std::min(chunk_start + triangle_chunk, (int)mesh.triangles.size());
//
//        pool.enqueue([&, chunk_start, chunk_end]() {
//            for (int t = chunk_start; t < chunk_end; ++t) {
//                // 获取三角形包围盒
//                //int idx0 = mesh.indices[t * 3];
//                //int idx1 = mesh.indices[t * 3 + 1];
//                //int idx2 = mesh.indices[t * 3 + 2];
//                triIndices& idx = mesh.triangles[t];
//
//                // 计算屏幕空间包围盒
//                //vec2D minV = { std::min({mesh.positions_x[idx[0]], mesh.positions_x[idx1], mesh.positions_x[idx2]}),
//                //               std::min({mesh.positions_y[idx0], mesh.positions_y[idx1], mesh.positions_y[idx2]}) };
//                //vec2D maxV = { std::max({mesh.positions_x[idx0], mesh.positions_x[idx1], mesh.positions_x[idx2]}),
//                //               std::max({mesh.positions_y[idx0], mesh.positions_y[idx1], mesh.positions_y[idx2]}) };
//                int mixX = std::min({ mesh.positions_x[idx.v[0]], mesh.positions_x[idx.v[1]], mesh.positions_x[idx.v[2]] });
//                int mixY = std::min({ mesh.positions_y[idx.v[0]], mesh.positions_y[idx.v[1]], mesh.positions_y[idx.v[2]] });
//                int maxX = std::max({ mesh.positions_x[idx.v[0]], mesh.positions_x[idx.v[1]], mesh.positions_x[idx.v[2]] });
//                int maxY = std::max({ mesh.positions_y[idx.v[0]], mesh.positions_y[idx.v[1]], mesh.positions_y[idx.v[2]] });
//
//
//                // 确定覆盖的瓦片
//                int min_tile_x = std::max(0, (int)mixX / TILE_SIZE);
//                int max_tile_x = std::min(tiles_x - 1, (int)maxX / TILE_SIZE);
//                int min_tile_y = std::max(0, (int)mixY / TILE_SIZE);
//                int max_tile_y = std::min(tiles_y - 1, (int)maxY / TILE_SIZE);
//
//                // 添加到对应的瓦片桶
//                for (int ty = min_tile_y; ty <= max_tile_y; ++ty) {
//                    for (int tx = min_tile_x; tx <= max_tile_x; ++tx) {
//                        triangle_buckets[ty * tiles_x + tx].push_back(t);
//                    }
//                }
//            }
//            });
//    }
//
//    // 等待筛选完成
//    pool.waitFrame();
//
//    // 第二阶段：并行瓦片渲染
//    std::vector<std::future<void>> futures;
//
//    pool.beginFrame();
//    for (size_t tile_idx = 0; tile_idx < tiles.size(); ++tile_idx) {
//        Tile& tile = tiles[tile_idx];
//
//        //if (!triangle_buckets[tile_idx].empty()) {
//        //    futures.push_back(pool.enqueue([&, tile_idx]() {
//        //        renderTile(mesh, light, ka, kd, tile_idx, triangle_buckets[tile_idx]);
//        //        }));
//        //}
//        if (!triangle_buckets[tile_idx].empty()) {
//            pool.enqueue([&, tile_idx]() {
//                renderTile(mesh, light, ka, kd, tile_idx, triangle_buckets[tile_idx]);
//                });
//        }
//    }
//
//    // 等待所有瓦片渲染完成
//    //for (auto& f : futures) {
//    //    f.wait();
//    //}
//    pool.waitFrame();
//
//    // 第三阶段：合并结果到主画布
//    mergeTilesToCanvas(renderer);
//}
//
//
///*
//void renderTriangleInTile_imcomplete(Mesh_SoA& mesh, Light& light, float ka, float kd,
//    int triangle_idx, Tile& tile) {
//    // 加载顶点数据（SOA -> SIMD寄存器）
//    //int idx0 = mesh.indices[triangle_idx * 3];
//    //int idx1 = mesh.indices[triangle_idx * 3 + 1];
//    //int idx2 = mesh.indices[triangle_idx * 3 + 2];
//    triIndices& idx = mesh.triangles[triangle_idx];
//
//    // 加载到标量变量
//    float v0_x = mesh.positions_x[idx.v[0]];
//    float v0_y = mesh.positions_y[idx.v[0]];
//    float v0_z = mesh.positions_z[idx.v[0]];
//    float v0_w = mesh.positions_w[idx.v[0]];
//    float v1_x = mesh.positions_x[idx.v[1]];
//    float v1_y = mesh.positions_y[idx.v[1]];
//    float v1_z = mesh.positions_z[idx.v[1]];
//    float v1_w = mesh.positions_w[idx.v[1]];
//    float v2_x = mesh.positions_x[idx.v[2]];
//    float v2_y = mesh.positions_y[idx.v[2]];
//    float v2_z = mesh.positions_z[idx.v[2]];
//    float v2_w = mesh.positions_w[idx.v[2]];
//
//    vec4 e0 = makeEdge0(v1_x, v1_y, v2_x, v2_y);
//    vec4 e1 = makeEdge0(v2_x, v2_y, v0_x, v0_y);
//    vec4 e2 = makeEdge0(v0_x, v0_y, v1_x, v1_y);
//
//    // ... 加载其他属性
//
//    // 转换到瓦片局部坐标
//    float local_min_x = std::max(0.0f, v0_x - tile.x);
//    float local_max_x = std::min((float)tile.width, v0_x - tile.x);
//    float local_min_y = std::max(0.0f, v0_y - tile.y);
//    float local_max_y = std::min((float)tile.height, v0_y - tile.y);
//    // ...
//
//    // 使用SIMD进行光栅化（类似你现有的draw_AVX2_Optimized3，但调整到瓦片坐标）
//    // 处理瓦片内的像素
//    for (int y = 0; y < tile.height; ++y) {
//        for (int x = 0; x < tile.width; x += 8) {
//            // SIMD计算边缘方程
//            __m256 x_coords = _mm256_setr_ps(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
//            __m256 y_coord = _mm256_set1_ps(y);
//
//            // 计算w0, w1, w2
//            __m256 w0 = _mm256_fmadd_ps(_mm256_set1_ps(e0.x), x_coords,
//                _mm256_fmadd_ps(_mm256_set1_ps(e0.y), y_coord,
//                    _mm256_set1_ps(e0.z)));
//            __m256 w1 = _mm256_fmadd_ps(_mm256_set1_ps(e1.x), x_coords,
//                _mm256_fmadd_ps(_mm256_set1_ps(e1.y), y_coord,
//                    _mm256_set1_ps(e1.z)));
//            __m256 w2 = _mm256_fmadd_ps(_mm256_set1_ps(e2.x), x_coords,
//                _mm256_fmadd_ps(_mm256_set1_ps(e2.y), y_coord,
//                    _mm256_set1_ps(e2.z)));
//
//            // 深度测试
//            // 光照计算
//            // 写入瓦片缓冲区
//
//            int buffer_idx = (y * tile.width + x) * 3;
//            tile.colors[buffer_idx] = r;
//            tile.colors[buffer_idx + 1] = g;
//            tile.colors[buffer_idx + 2] = b;
//            tile.zbuffer[y * tile.width + x] = z;
//        }
//    }
//}
//*/

/*        // 瓦片渲染函数
        void renderTile(Mesh_SoA& mesh, Light& light, float ka, float kd,
            int tile_idx, const std::vector<int>& triangles_in_tile) 
        {
            Tile& tile = tiles[tile_idx];

            // 初始化瓦片缓冲区
            std::fill(tile.zbuffer.begin(), tile.zbuffer.end(), std::numeric_limits<float>::max());
            //std::fill(tile.colors.begin(), tile.colors.end(), 0);

            // 渲染瓦片内的所有三角形
            for (int tri_idx : triangles_in_tile) {
                renderTriangleInTile(mesh, light, ka, kd, tri_idx, tile);
            }

            tile.completed = true;
        }

        void renderTriangleInTile(Mesh_SoA& mesh, Light& light, float ka, float kd,
            int triangle_idx, Tile& tile) {
            // 加载顶点数据（SOA -> SIMD寄存器）
            triIndices& idx = mesh.triangles[triangle_idx];

            // 加载顶点位置（屏幕空间）
            float v0_x = mesh.positions_x[idx.v[0]];
            float v0_y = mesh.positions_y[idx.v[0]];
            float v0_z = mesh.positions_z[idx.v[0]];
            float v0_w = mesh.positions_w[idx.v[0]];

            float v1_x = mesh.positions_x[idx.v[1]];
            float v1_y = mesh.positions_y[idx.v[1]];
            float v1_z = mesh.positions_z[idx.v[1]];
            float v1_w = mesh.positions_w[idx.v[1]];

            float v2_x = mesh.positions_x[idx.v[2]];
            float v2_y = mesh.positions_y[idx.v[2]];
            float v2_z = mesh.positions_z[idx.v[2]];
            float v2_w = mesh.positions_w[idx.v[2]];

            // 计算三角形面积（用于重心坐标插值）
            vec4 e0 = makeEdge0(v1_x, v1_y, v2_x, v2_y);
            vec4 e1 = makeEdge0(v2_x, v2_y, v0_x, v0_y);
            vec4 e2 = makeEdge0(v0_x, v0_y, v1_x, v1_y);

            float area = v0_x * e0.x + v0_y * e0.y + e0.z;
            if (area < 1e-6f) return;  // 面积太小，忽略

            float invArea = 1.0f / area;

            // 加载法线数据（假设已经在视图空间）
            float n0_x = mesh.normals_x[idx.v[0]];
            float n0_y = mesh.normals_y[idx.v[0]];
            float n0_z = mesh.normals_z[idx.v[0]];

            float n1_x = mesh.normals_x[idx.v[1]];
            float n1_y = mesh.normals_y[idx.v[1]];
            float n1_z = mesh.normals_z[idx.v[1]];

            float n2_x = mesh.normals_x[idx.v[2]];
            float n2_y = mesh.normals_y[idx.v[2]];
            float n2_z = mesh.normals_z[idx.v[2]];

            // 加载颜色数据
            float c0_r = mesh.colors_r[idx.v[0]];
            float c0_g = mesh.colors_g[idx.v[0]];
            float c0_b = mesh.colors_b[idx.v[0]];

            float c1_r = mesh.colors_r[idx.v[1]];
            float c1_g = mesh.colors_g[idx.v[1]];
            float c1_b = mesh.colors_b[idx.v[1]];

            float c2_r = mesh.colors_r[idx.v[2]];
            float c2_g = mesh.colors_g[idx.v[2]];
            float c2_b = mesh.colors_b[idx.v[2]];

            // 计算插值增量
            float dz_dx = (v0_z * e0.x + v1_z * e1.x + v2_z * e2.x) * invArea;
            float dz_dy = (v0_z * e0.y + v1_z * e1.y + v2_z * e2.y) * invArea;

            float dn_dx_x = (n0_x * e0.x + n1_x * e1.x + n2_x * e2.x) * invArea;
            float dn_dx_y = (n0_y * e0.x + n1_y * e1.x + n2_y * e2.x) * invArea;
            float dn_dx_z = (n0_z * e0.x + n1_z * e1.x + n2_z * e2.x) * invArea;

            float dn_dy_x = (n0_x * e0.y + n1_x * e1.y + n2_x * e2.y) * invArea;
            float dn_dy_y = (n0_y * e0.y + n1_y * e1.y + n2_y * e2.y) * invArea;
            float dn_dy_z = (n0_z * e0.y + n1_z * e1.y + n2_z * e2.y) * invArea;

            float dc_dx_r = (c0_r * e0.x + c1_r * e1.x + c2_r * e2.x) * invArea;
            float dc_dx_g = (c0_g * e0.x + c1_g * e1.x + c2_g * e2.x) * invArea;
            float dc_dx_b = (c0_b * e0.x + c1_b * e1.x + c2_b * e2.x) * invArea;

            float dc_dy_r = (c0_r * e0.y + c1_r * e1.y + c2_r * e2.y) * invArea;
            float dc_dy_g = (c0_g * e0.y + c1_g * e1.y + c2_g * e2.y) * invArea;
            float dc_dy_b = (c0_b * e0.y + c1_b * e1.y + c2_b * e2.y) * invArea;

            // 计算三角形在瓦片空间的包围盒
            float min_x = std::max((float)tile.x, std::min({ v0_x, v1_x, v2_x }));
            float max_x = std::min((float)(tile.x + tile.width), std::max({ v0_x, v1_x, v2_x }));
            float min_y = std::max((float)tile.y, std::min({ v0_y, v1_y, v2_y }));
            float max_y = std::min((float)(tile.y + tile.height), std::max({ v0_y, v1_y, v2_y }));

            //if (min_x > max_x || min_y > max_y) return;
            if (min_x > max_x || min_y > max_y ||
                max_x < tile.x || min_x > tile.x + tile.width - 1 ||
                max_y < tile.y || min_y > tile.y + tile.height - 1) {
                return;
            }

            // 转换为瓦片局部坐标
            int tile_min_x = (int)std::floor(min_x) - tile.x;
            int tile_max_x = (int)std::ceil(max_x) - tile.x;
            int tile_min_y = (int)std::floor(min_y) - tile.y;
            int tile_max_y = (int)std::ceil(max_y) - tile.y;

            // 裁剪到瓦片边界
            tile_min_x = std::max(0, tile_min_x);
            tile_max_x = std::min(tile.width - 1, tile_max_x);
            tile_min_y = std::max(0, tile_min_y);
            tile_max_y = std::min(tile.height - 1, tile_max_y);

            if (tile_min_x > tile_max_x || tile_min_y > tile_max_y) return;

            // --- AVX2常量 ---
            const __m256 zero = _mm256_setzero_ps();
            const __m256 lane = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
            const __m256 _255 = _mm256_set1_ps(255.0f);
            const __m256 _001 = _mm256_set1_ps(0.001f);
            const __m256 _1 = _mm256_set1_ps(1.0f);

            // 预计算增量
            __m256 w0_lane = _mm256_mul_ps(_mm256_set1_ps(e0.x), lane);
            __m256 w1_lane = _mm256_mul_ps(_mm256_set1_ps(e1.x), lane);
            __m256 w2_lane = _mm256_mul_ps(_mm256_set1_ps(e2.x), lane);

            __m256 z_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lane);
            __m256 nx_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_x), lane);
            __m256 ny_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_y), lane);
            __m256 nz_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_z), lane);
            __m256 cr_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_r), lane);
            __m256 cg_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_g), lane);
            __m256 cb_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_b), lane);

            __m256 w0_step8 = _mm256_set1_ps(e0.x * 8.0f);
            __m256 w1_step8 = _mm256_set1_ps(e1.x * 8.0f);
            __m256 w2_step8 = _mm256_set1_ps(e2.x * 8.0f);
            __m256 z_step8 = _mm256_set1_ps(dz_dx * 8.0f);
            __m256 nx_step8 = _mm256_set1_ps(dn_dx_x * 8.0f);
            __m256 ny_step8 = _mm256_set1_ps(dn_dx_y * 8.0f);
            __m256 nz_step8 = _mm256_set1_ps(dn_dx_z * 8.0f);
            __m256 cr_step8 = _mm256_set1_ps(dc_dx_r * 8.0f);
            __m256 cg_step8 = _mm256_set1_ps(dc_dx_g * 8.0f);
            __m256 cb_step8 = _mm256_set1_ps(dc_dx_b * 8.0f);

            // 光照常量
            __m256 Lr = _mm256_set1_ps(light.L.r);
            __m256 Lg = _mm256_set1_ps(light.L.g);
            __m256 Lb = _mm256_set1_ps(light.L.b);
            __m256 L_omega_i_x = _mm256_set1_ps(light.omega_i.x);
            __m256 L_omega_i_y = _mm256_set1_ps(light.omega_i.y);
            __m256 L_omega_i_z = _mm256_set1_ps(light.omega_i.z);
            __m256 kd_vec = _mm256_set1_ps(kd);
            __m256 ka_vec = _mm256_set1_ps(ka);
            __m256 amb_r_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.r), ka_vec);
            __m256 amb_g_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.g), ka_vec);
            __m256 amb_b_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.b), ka_vec);

            // 行循环
            for (int y = tile_min_y; y <= tile_max_y; ++y) {
                float global_y = (float)(tile.y + y);

                // 计算当前行的基础值
                float w0_row = e0.x * (tile.x + tile_min_x) + e0.y * global_y + e0.z;
                float w1_row = e1.x * (tile.x + tile_min_x) + e1.y * global_y + e1.z;
                float w2_row = e2.x * (tile.x + tile_min_x) + e2.y * global_y + e2.z;

                float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
                float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
                float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
                float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
                float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
                float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
                float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

                __m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
                __m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
                __m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);
                __m256 zv = _mm256_add_ps(_mm256_set1_ps(z_row), z_lane);
                __m256 nx = _mm256_add_ps(_mm256_set1_ps(nx_row), nx_lane);
                __m256 ny = _mm256_add_ps(_mm256_set1_ps(ny_row), ny_lane);
                __m256 nz = _mm256_add_ps(_mm256_set1_ps(nz_row), nz_lane);
                __m256 cr = _mm256_add_ps(_mm256_set1_ps(cr_row), cr_lane);
                __m256 cg = _mm256_add_ps(_mm256_set1_ps(cg_row), cg_lane);
                __m256 cb = _mm256_add_ps(_mm256_set1_ps(cb_row), cb_lane);

                // 像素循环（8像素一组）
                for (int x = tile_min_x; x <= tile_max_x - 7; x += 8) {
                    // 确保 x 在有效范围内
                    if (x + 7 >= tile.width) break;
                    // 检查像素是否在三角形内
                    __m256 inside = _mm256_and_ps(
                        _mm256_and_ps(
                            _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
                            _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)
                        ),
                        _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ)
                    );

                    int mask = _mm256_movemask_ps(inside);
                    if (mask == 0) {
                        w0v = _mm256_add_ps(w0v, w0_step8);
                        w1v = _mm256_add_ps(w1v, w1_step8);
                        w2v = _mm256_add_ps(w2v, w2_step8);
                        zv = _mm256_add_ps(zv, z_step8);
                        nx = _mm256_add_ps(nx, nx_step8);
                        ny = _mm256_add_ps(ny, ny_step8);
                        nz = _mm256_add_ps(nz, nz_step8);
                        cr = _mm256_add_ps(cr, cr_step8);
                        cg = _mm256_add_ps(cg, cg_step8);
                        cb = _mm256_add_ps(cb, cb_step8);
                        continue;
                    }

                    // 加载瓦片深度缓冲区
                    alignas(32) float zbuf_tmp[8];
                    for (int i = 0; i < 8; ++i) {
                        int pixel_x = x + i;
                        if (pixel_x < tile.width) {
                            zbuf_tmp[i] = tile.zbuffer[y * tile.width + pixel_x];
                        }
                        else {
                            zbuf_tmp[i] = std::numeric_limits<float>::max();
                        }
                    }
                    __m256 zbuf = _mm256_load_ps(zbuf_tmp);

                    // 深度测试
                    __m256 depth_ok = _mm256_and_ps(
                        _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
                        _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ)
                    );

                    __m256 final_mask = _mm256_and_ps(inside, depth_ok);
                    int final_mask_bits = _mm256_movemask_ps(final_mask);

                    if (final_mask_bits == 0) {
                        w0v = _mm256_add_ps(w0v, w0_step8);
                        w1v = _mm256_add_ps(w1v, w1_step8);
                        w2v = _mm256_add_ps(w2v, w2_step8);
                        zv = _mm256_add_ps(zv, z_step8);
                        nx = _mm256_add_ps(nx, nx_step8);
                        ny = _mm256_add_ps(ny, ny_step8);
                        nz = _mm256_add_ps(nz, nz_step8);
                        cr = _mm256_add_ps(cr, cr_step8);
                        cg = _mm256_add_ps(cg, cg_step8);
                        cb = _mm256_add_ps(cb, cb_step8);
                        continue;
                    }

                    // 归一化法线（只对有掩码的像素）
                    __m256 len = _mm256_sqrt_ps(_mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                        _mm256_mul_ps(nz, nz)
                    ));

                    // 避免除零（为掩码外的像素设置len=1）
                    __m256 safe_len = _mm256_blendv_ps(_mm256_set1_ps(1.0f), len, final_mask);
                    nx = _mm256_div_ps(nx, safe_len);
                    ny = _mm256_div_ps(ny, safe_len);
                    nz = _mm256_div_ps(nz, safe_len);

                    // 光照计算
                    __m256 dot = _mm256_max_ps(_mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_mul_ps(nx, L_omega_i_x),
                            _mm256_mul_ps(ny, L_omega_i_y)
                        ),
                        _mm256_mul_ps(nz, L_omega_i_z)
                    ), zero);

                    // 颜色计算
                    __m256 r = _mm256_add_ps(
                        _mm256_mul_ps(_mm256_mul_ps(cr, kd_vec), _mm256_mul_ps(Lr, dot)),
                        amb_r_ka
                    );

                    __m256 g = _mm256_add_ps(
                        _mm256_mul_ps(_mm256_mul_ps(cg, kd_vec), _mm256_mul_ps(Lg, dot)),
                        amb_g_ka
                    );

                    __m256 b = _mm256_add_ps(
                        _mm256_mul_ps(_mm256_mul_ps(cb, kd_vec), _mm256_mul_ps(Lb, dot)),
                        amb_b_ka
                    );

                    // 裁剪颜色到[0,1]
                    r = _mm256_min_ps(_mm256_max_ps(r, zero), _1);
                    g = _mm256_min_ps(_mm256_max_ps(g, zero), _1);
                    b = _mm256_min_ps(_mm256_max_ps(b, zero), _1);

                    // 转换为8位
                    r = _mm256_mul_ps(r, _255);
                    g = _mm256_mul_ps(g, _255);
                    b = _mm256_mul_ps(b, _255);

                    // 存储结果
                    alignas(32) float rr[8], gg[8], bb[8], zz[8];
                    _mm256_store_ps(rr, r);
                    _mm256_store_ps(gg, g);
                    _mm256_store_ps(bb, b);
                    _mm256_store_ps(zz, zv);

                    // 使用掩码写入瓦片缓冲区
                    int m = final_mask_bits;
                    while (m) {
                        int i = _tzcnt_u32(m);
                        m &= m - 1;

                        int pixel_x = x + i;
                        if (pixel_x < tile.width) {
                            int color_idx = (y * tile.width + pixel_x) * 3;
                            tile.colors[color_idx] = (unsigned char)rr[i];
                            tile.colors[color_idx + 1] = (unsigned char)gg[i];
                            tile.colors[color_idx + 2] = (unsigned char)bb[i];
                            tile.zbuffer[y * tile.width + pixel_x] = zz[i];
                        }
                    }

                    // 增量到下一个8像素组
                    w0v = _mm256_add_ps(w0v, w0_step8);
                    w1v = _mm256_add_ps(w1v, w1_step8);
                    w2v = _mm256_add_ps(w2v, w2_step8);
                    zv = _mm256_add_ps(zv, z_step8);
                    nx = _mm256_add_ps(nx, nx_step8);
                    ny = _mm256_add_ps(ny, ny_step8);
                    nz = _mm256_add_ps(nz, nz_step8);
                    cr = _mm256_add_ps(cr, cr_step8);
                    cg = _mm256_add_ps(cg, cg_step8);
                    cb = _mm256_add_ps(cb, cb_step8);
                }

                // 处理尾数像素（少于8个）
                int tail_start = tile_min_x + ((tile_max_x - tile_min_x + 1) / 8) * 8;
                for (int x = tail_start; x <= tile_max_x; ++x) {
                    float w0 = e0.x * (tile.x + x) + e0.y * global_y + e0.z;
                    float w1 = e1.x * (tile.x + x) + e1.y * global_y + e1.z;
                    float w2 = e2.x * (tile.x + x) + e2.y * global_y + e2.z;

                    if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;

                    float alpha = w0 * invArea;
                    float beta = w1 * invArea;
                    float gamma = w2 * invArea;

                    float z = v0_z * alpha + v1_z * beta + v2_z * gamma;
                    if (z < 0.001f || z >= tile.zbuffer[y * tile.width + x]) continue;

                    // 法线插值和归一化
                    float nx_val = n0_x * alpha + n1_x * beta + n2_x * gamma;
                    float ny_val = n0_y * alpha + n1_y * beta + n2_y * gamma;
                    float nz_val = n0_z * alpha + n1_z * beta + n2_z * gamma;

                    float inv_len = 1.0f / sqrtf(nx_val * nx_val + ny_val * ny_val + nz_val * nz_val + 1e-8f);
                    nx_val *= inv_len;
                    ny_val *= inv_len;
                    nz_val *= inv_len;

                    // 颜色插值
                    float cr_val = c0_r * alpha + c1_r * beta + c2_r * gamma;
                    float cg_val = c0_g * alpha + c1_g * beta + c2_g * gamma;
                    float cb_val = c0_b * alpha + c1_b * beta + c2_b * gamma;

                    // 光照计算
                    float dot = std::max(nx_val * light.omega_i.x + ny_val * light.omega_i.y +
                        nz_val * light.omega_i.z, 0.0f);

                    float out_r = (cr_val * kd) * (light.L.r * dot) + (light.ambient.r * ka);
                    float out_g = (cg_val * kd) * (light.L.g * dot) + (light.ambient.g * ka);
                    float out_b = (cb_val * kd) * (light.L.b * dot) + (light.ambient.b * ka);

                    // 裁剪和转换
                    out_r = std::max(0.0f, std::min(1.0f, out_r));
                    out_g = std::max(0.0f, std::min(1.0f, out_g));
                    out_b = std::max(0.0f, std::min(1.0f, out_b));

                    // 写入瓦片缓冲区
                    int color_idx = (y * tile.width + x) * 3;
                    tile.colors[color_idx] = (unsigned char)(out_r * 255.0f);
                    tile.colors[color_idx + 1] = (unsigned char)(out_g * 255.0f);
                    tile.colors[color_idx + 2] = (unsigned char)(out_b * 255.0f);
                    tile.zbuffer[y * tile.width + x] = z;
                }
            }
        }*/


void renderTriangleFromSOA_InTile(Renderer& renderer, const Mesh_SoA_Transformed& mesh, Light& light, float ka, float kd,
    int triangle_idx, Tile& tile) {
    // 加载顶点数据（从变换后的SOA数据）
    const triIndices& idx = mesh.triangles[triangle_idx];

    // 加载变换后的顶点位置（屏幕空间）
    float v0_x = mesh.transformed_positions_x[idx.v[0]];
    float v1_x = mesh.transformed_positions_x[idx.v[1]];
    float v2_x = mesh.transformed_positions_x[idx.v[2]];

    float v0_y = mesh.transformed_positions_y[idx.v[0]];
    float v1_y = mesh.transformed_positions_y[idx.v[1]];
    float v2_y = mesh.transformed_positions_y[idx.v[2]];

    float v0_z = mesh.transformed_positions_z[idx.v[0]];
    float v1_z = mesh.transformed_positions_z[idx.v[1]];
    float v2_z = mesh.transformed_positions_z[idx.v[2]];

    // 检查有效性
    if (!isfinite(v0_x) || !isfinite(v0_y) || !isfinite(v0_z) ||
        !isfinite(v1_x) || !isfinite(v1_y) || !isfinite(v1_z) ||
        !isfinite(v2_x) || !isfinite(v2_y) || !isfinite(v2_z)) {
        return;
    }

    // 计算三角形面积（用于重心坐标插值）
    vec4 e0 = makeEdge0(v1_x, v1_y, v2_x, v2_y);
    vec4 e1 = makeEdge0(v2_x, v2_y, v0_x, v0_y);
    vec4 e2 = makeEdge0(v0_x, v0_y, v1_x, v1_y);

    float area = v0_x * e0.x + v0_y * e0.y + e0.z;
    if (area < 1e-6f) return;  // 面积太小，忽略

    float invArea = 1.0f / area;

    // 加载变换后的法线数据
    //const float* norm_x = mesh.transformed_normals_x.data();
    //const float* norm_y = mesh.transformed_normals_y.data();
    //const float* norm_z = mesh.transformed_normals_z.data();

    float n0_x = mesh.transformed_normals_x[idx.v[0]];
    float n0_y = mesh.transformed_normals_y[idx.v[0]];
    float n0_z = mesh.transformed_normals_z[idx.v[0]];

    float n1_x = mesh.transformed_normals_x[idx.v[1]];
    float n1_y = mesh.transformed_normals_y[idx.v[1]];
    float n1_z = mesh.transformed_normals_z[idx.v[1]];

    float n2_x = mesh.transformed_normals_x[idx.v[2]];
    float n2_y = mesh.transformed_normals_y[idx.v[2]];
    float n2_z = mesh.transformed_normals_z[idx.v[2]];

    // 加载颜色数据
    //const float* col_r = mesh.colors_r.data();
    //const float* col_g = mesh.colors_g.data();
    //const float* col_b = mesh.colors_b.data();

    float c0_r = mesh.colors_r[idx.v[0]];
    float c0_g = mesh.colors_g[idx.v[0]];
    float c0_b = mesh.colors_b[idx.v[0]];

    float c1_r = mesh.colors_r[idx.v[1]];
    float c1_g = mesh.colors_g[idx.v[1]];
    float c1_b = mesh.colors_b[idx.v[1]];

    float c2_r = mesh.colors_r[idx.v[2]];
    float c2_g = mesh.colors_g[idx.v[2]];
    float c2_b = mesh.colors_b[idx.v[2]];


    // 计算插值增量
    float dz_dx = (v0_z * e0.x + v1_z * e1.x + v2_z * e2.x) * invArea;
    float dz_dy = (v0_z * e0.y + v1_z * e1.y + v2_z * e2.y) * invArea;

    float dn_dx_x = (n0_x * e0.x + n1_x * e1.x + n2_x * e2.x) * invArea;
    float dn_dx_y = (n0_y * e0.x + n1_y * e1.x + n2_y * e2.x) * invArea;
    float dn_dx_z = (n0_z * e0.x + n1_z * e1.x + n2_z * e2.x) * invArea;

    float dn_dy_x = (n0_x * e0.y + n1_x * e1.y + n2_x * e2.y) * invArea;
    float dn_dy_y = (n0_y * e0.y + n1_y * e1.y + n2_y * e2.y) * invArea;
    float dn_dy_z = (n0_z * e0.y + n1_z * e1.y + n2_z * e2.y) * invArea;

    float dc_dx_r = (c0_r * e0.x + c1_r * e1.x + c2_r * e2.x) * invArea;
    float dc_dx_g = (c0_g * e0.x + c1_g * e1.x + c2_g * e2.x) * invArea;
    float dc_dx_b = (c0_b * e0.x + c1_b * e1.x + c2_b * e2.x) * invArea;

    float dc_dy_r = (c0_r * e0.y + c1_r * e1.y + c2_r * e2.y) * invArea;
    float dc_dy_g = (c0_g * e0.y + c1_g * e1.y + c2_g * e2.y) * invArea;
    float dc_dy_b = (c0_b * e0.y + c1_b * e1.y + c2_b * e2.y) * invArea;

    // 计算三角形在瓦片空间的包围盒
    float min_x = std::max((float)tile.x, std::min({ v0_x, v1_x, v2_x }));
    float max_x = std::min((float)(tile.x + tile.width), std::max({ v0_x, v1_x, v2_x }));
    float min_y = std::max((float)tile.y, std::min({ v0_y, v1_y, v2_y }));
    float max_y = std::min((float)(tile.y + tile.height), std::max({ v0_y, v1_y, v2_y }));

    if (min_x > max_x || min_y > max_y) return;

    // 转换为瓦片局部坐标
    int tile_min_x = (int)std::floor(min_x) - tile.x;
    int tile_max_x = (int)std::ceil(max_x) - tile.x;
    int tile_min_y = (int)std::floor(min_y) - tile.y;
    int tile_max_y = (int)std::ceil(max_y) - tile.y;

    // 裁剪到瓦片边界
    tile_min_x = std::max(0, tile_min_x);
    tile_max_x = std::min(tile.width - 1, tile_max_x);
    tile_min_y = std::max(0, tile_min_y);
    tile_max_y = std::min(tile.height - 1, tile_max_y);

    if (tile_min_x > tile_max_x || tile_min_y > tile_max_y) return;

    // --- AVX2常量 ---
    const __m256 zero = _mm256_setzero_ps();
    const __m256 lane = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    const __m256 _255 = _mm256_set1_ps(255.0f);
    const __m256 _001 = _mm256_set1_ps(0.001f);
    const __m256 _1 = _mm256_set1_ps(1.0f);

    // 预计算增量
    __m256 w0_lane = _mm256_mul_ps(_mm256_set1_ps(e0.x), lane);
    __m256 w1_lane = _mm256_mul_ps(_mm256_set1_ps(e1.x), lane);
    __m256 w2_lane = _mm256_mul_ps(_mm256_set1_ps(e2.x), lane);

    __m256 z_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lane);
    __m256 nx_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_x), lane);
    __m256 ny_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_y), lane);
    __m256 nz_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_z), lane);
    __m256 cr_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_r), lane);
    __m256 cg_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_g), lane);
    __m256 cb_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_b), lane);

    __m256 w0_step8 = _mm256_set1_ps(e0.x * 8.0f);
    __m256 w1_step8 = _mm256_set1_ps(e1.x * 8.0f);
    __m256 w2_step8 = _mm256_set1_ps(e2.x * 8.0f);
    __m256 z_step8 = _mm256_set1_ps(dz_dx * 8.0f);
    __m256 nx_step8 = _mm256_set1_ps(dn_dx_x * 8.0f);
    __m256 ny_step8 = _mm256_set1_ps(dn_dx_y * 8.0f);
    __m256 nz_step8 = _mm256_set1_ps(dn_dx_z * 8.0f);
    __m256 cr_step8 = _mm256_set1_ps(dc_dx_r * 8.0f);
    __m256 cg_step8 = _mm256_set1_ps(dc_dx_g * 8.0f);
    __m256 cb_step8 = _mm256_set1_ps(dc_dx_b * 8.0f);

    // 光照常量
    __m256 Lr = _mm256_set1_ps(light.L.r);
    __m256 Lg = _mm256_set1_ps(light.L.g);
    __m256 Lb = _mm256_set1_ps(light.L.b);
    __m256 L_omega_i_x = _mm256_set1_ps(light.omega_i.x);
    __m256 L_omega_i_y = _mm256_set1_ps(light.omega_i.y);
    __m256 L_omega_i_z = _mm256_set1_ps(light.omega_i.z);
    __m256 kd_step8 = _mm256_set1_ps(kd);
    __m256 ka_step8 = _mm256_set1_ps(ka);
    __m256 ambinet_r_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.r), ka_step8);
    __m256 ambinet_g_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.g), ka_step8);
    __m256 ambinet_b_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.b), ka_step8);

    // row init value
    const float start_x = (float)(tile.x + tile_min_x);
    float global_y0 = (float)(tile.y + tile_min_y);

    // w 在 (start_x, tile_min_y)
    float w0_row = e0.x * start_x + e0.y * global_y0 + e0.z;
    float w1_row = e1.x * start_x + e1.y * global_y0 + e1.z;
    float w2_row = e2.x * start_x + e2.y * global_y0 + e2.z;

    // 插值初值
    float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
    float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
    float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
    float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
    float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
    float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
    float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

    // 行循环
    for (int y = tile_min_y; y <= tile_max_y; ++y) {
        //float global_y = (float)(tile.y + y);

        //// 计算当前行的基础值
        //float w0_row = e0.x * (tile.x + tile_min_x) + e0.y * global_y + e0.z;
        //float w1_row = e1.x * (tile.x + tile_min_x) + e1.y * global_y + e1.z;
        //float w2_row = e2.x * (tile.x + tile_min_x) + e2.y * global_y + e2.z;

        //float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
        //float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
        //float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
        //float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
        //float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
        //float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
        //float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

        __m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
        __m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
        __m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);
        __m256 zv = _mm256_add_ps(_mm256_set1_ps(z_row), z_lane);
        __m256 nx = _mm256_add_ps(_mm256_set1_ps(nx_row), nx_lane);
        __m256 ny = _mm256_add_ps(_mm256_set1_ps(ny_row), ny_lane);
        __m256 nz = _mm256_add_ps(_mm256_set1_ps(nz_row), nz_lane);
        __m256 cr = _mm256_add_ps(_mm256_set1_ps(cr_row), cr_lane);
        __m256 cg = _mm256_add_ps(_mm256_set1_ps(cg_row), cg_lane);
        __m256 cb = _mm256_add_ps(_mm256_set1_ps(cb_row), cb_lane);

        int x = tile_min_x;
        // 像素循环（8像素一组）
        for (; x <= tile_max_x - 7; x += 8) {
            // 检查像素是否在三角形内
            //__m256 inside = _mm256_and_ps(
            //    _mm256_and_ps(
            //        _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
            //        _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)
            //    ),
            //    _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ)
            //);

            //int mask = _mm256_movemask_ps(inside);
            //if (mask == 0) {
            //    w0v = _mm256_add_ps(w0v, w0_step8);
            //    w1v = _mm256_add_ps(w1v, w1_step8);
            //    w2v = _mm256_add_ps(w2v, w2_step8);
            //    zv = _mm256_add_ps(zv, z_step8);
            //    nx = _mm256_add_ps(nx, nx_step8);
            //    ny = _mm256_add_ps(ny, ny_step8);
            //    nz = _mm256_add_ps(nz, nz_step8);
            //    cr = _mm256_add_ps(cr, cr_step8);
            //    cg = _mm256_add_ps(cg, cg_step8);
            //    cb = _mm256_add_ps(cb, cb_step8);
            //    continue;
            //}

            //// 加载瓦片深度缓冲区
            //// TODO
            ////alignas(32) float zbuf_tmp[8];
            ////for (int i = 0; i < 8; ++i) {
            ////    int pixel_x = x + i;
            ////    if (pixel_x < tile.width) {
            ////        zbuf_tmp[i] = tile.zbuffer[y * tile.width + pixel_x];
            ////    }
            ////    else {
            ////        //zbuf_tmp[i] = std::numeric_limits<float>::max();
            ////        zbuf_tmp[i] = 1.0f;
            ////    }
            ////}
            ////__m256 zbuf = _mm256_load_ps(zbuf_tmp);
            __m256 zbuf = _mm256_load_ps(&tile.zbuffer[y * tile.width + x]);

            //// 深度测试
            ////__m256 depth_ok = _mm256_and_ps(
            ////    _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
            ////    _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ)
            ////);
            //__m256 depth_ok = _mm256_and_ps(
            //    _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
            //    _mm256_cmp_ps(zbuf, zv, _CMP_GT_OQ)
            //);

            //__m256 final_mask = _mm256_and_ps(inside, depth_ok);
            __m256 final_mask =
                _mm256_and_ps(
                    _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
                    _mm256_and_ps(
                        _mm256_cmp_ps(zbuf, zv, _CMP_GT_OQ),
                        _mm256_and_ps(
                            _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
                            _mm256_and_ps(
                                _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ),
                                _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ)
                            )
                        )
                    )
                );

            int final_mask_bits = _mm256_movemask_ps(final_mask);

            if (final_mask_bits == 0) {
                w0v = _mm256_add_ps(w0v, w0_step8);
                w1v = _mm256_add_ps(w1v, w1_step8);
                w2v = _mm256_add_ps(w2v, w2_step8);
                zv = _mm256_add_ps(zv, z_step8);
                nx = _mm256_add_ps(nx, nx_step8);
                ny = _mm256_add_ps(ny, ny_step8);
                nz = _mm256_add_ps(nz, nz_step8);
                cr = _mm256_add_ps(cr, cr_step8);
                cg = _mm256_add_ps(cg, cg_step8);
                cb = _mm256_add_ps(cb, cb_step8);
                continue;
            }

            // 归一化法线
            __m256 len = _mm256_sqrt_ps(_mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                _mm256_mul_ps(nz, nz)
            ));

            __m256 safe_len = _mm256_blendv_ps(_mm256_set1_ps(1.0f), len, final_mask);
            nx = _mm256_div_ps(nx, safe_len);
            ny = _mm256_div_ps(ny, safe_len);
            nz = _mm256_div_ps(nz, safe_len);

            // 光照计算
            __m256 dot = _mm256_max_ps(_mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(nx, L_omega_i_x),
                    _mm256_mul_ps(ny, L_omega_i_y)
                ),
                _mm256_mul_ps(nz, L_omega_i_z)
            ), zero);

            // 颜色计算
            __m256 r = _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8), _mm256_mul_ps(Lr, dot)),
                ambinet_r_ka
            );

            __m256 g = _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8), _mm256_mul_ps(Lg, dot)),
                ambinet_g_ka
            );

            __m256 b = _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8), _mm256_mul_ps(Lb, dot)),
                ambinet_b_ka
            );

            // 裁剪颜色到[0,1]
            r = _mm256_min_ps(_mm256_max_ps(r, zero), _1);
            g = _mm256_min_ps(_mm256_max_ps(g, zero), _1);
            b = _mm256_min_ps(_mm256_max_ps(b, zero), _1);

            // 转换为8位
            r = _mm256_mul_ps(r, _255);
            g = _mm256_mul_ps(g, _255);
            b = _mm256_mul_ps(b, _255);

            // 存储结果
            alignas(32) float rr[8], gg[8], bb[8], zz[8];
            _mm256_store_ps(rr, r);
            _mm256_store_ps(gg, g);
            _mm256_store_ps(bb, b);
            _mm256_store_ps(zz, zv);

            // 使用掩码写入瓦片缓冲区
            int m = final_mask_bits;
            while (m) {
                int i = _tzcnt_u32(m);
                m &= m - 1;

                int pixel_x = x + i;
                if (pixel_x < tile.width) {
                    //int color_idx = (y * tile.width + pixel_x) * 3;
                    //tile.colors[color_idx] = (unsigned char)rr[i];
                    //tile.colors[color_idx + 1] = (unsigned char)gg[i];
                    //tile.colors[color_idx + 2] = (unsigned char)bb[i];
                    //tile.zbuffer[y * tile.width + pixel_x] = zz[i];
                    renderer.canvas.draw(
                        tile.x + x + i, tile.y + y,
                        (unsigned char)(rr[i]),
                        (unsigned char)(gg[i]),
                        (unsigned char)(bb[i]));
                }
            }
            //TODO
            __m256i storeMask = _mm256_castps_si256(final_mask);
            _mm256_maskstore_ps(&tile.zbuffer[y * tile.width + x], storeMask, zv);

            // 增量到下一个8像素组
            w0v = _mm256_add_ps(w0v, w0_step8);
            w1v = _mm256_add_ps(w1v, w1_step8);
            w2v = _mm256_add_ps(w2v, w2_step8);
            zv = _mm256_add_ps(zv, z_step8);
            nx = _mm256_add_ps(nx, nx_step8);
            ny = _mm256_add_ps(ny, ny_step8);
            nz = _mm256_add_ps(nz, nz_step8);
            cr = _mm256_add_ps(cr, cr_step8);
            cg = _mm256_add_ps(cg, cg_step8);
            cb = _mm256_add_ps(cb, cb_step8);
        }

        // 处理尾数像素（少于8个）
        //int tail_start = tile_min_x + ((tile_max_x - tile_min_x + 1) / 8) * 8;
        //for (int x = tail_start; x <= tile_max_x; ++x) {
        //    float w0 = e0.x * (tile.x + x) + e0.y * global_y + e0.z;
        //    float w1 = e1.x * (tile.x + x) + e1.y * global_y + e1.z;
        //    float w2 = e2.x * (tile.x + x) + e2.y * global_y + e2.z;

        //    if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;

        //    float alpha = w0 * invArea;
        //    float beta = w1 * invArea;
        //    float gamma = w2 * invArea;

        //    float z = v0_z * alpha + v1_z * beta + v2_z * gamma;
        //    if (z < 0.001f || z >= tile.zbuffer[y * tile.width + x]) continue;

        //    // 法线插值和归一化
        //    float nx_val = n0_x * alpha + n1_x * beta + n2_x * gamma;
        //    float ny_val = n0_y * alpha + n1_y * beta + n2_y * gamma;
        //    float nz_val = n0_z * alpha + n1_z * beta + n2_z * gamma;

        //    float inv_len = 1.0f / sqrtf(nx_val * nx_val + ny_val * ny_val + nz_val * nz_val + 1e-8f);
        //    nx_val *= inv_len;
        //    ny_val *= inv_len;
        //    nz_val *= inv_len;

        //    // 颜色插值
        //    float cr_val = c0_r * alpha + c1_r * beta + c2_r * gamma;
        //    float cg_val = c0_g * alpha + c1_g * beta + c2_g * gamma;
        //    float cb_val = c0_b * alpha + c1_b * beta + c2_b * gamma;

        //    // 光照计算
        //    float dot = std::max(nx_val * light.omega_i.x + ny_val * light.omega_i.y +
        //        nz_val * light.omega_i.z, 0.0f);

        //    float out_r = (cr_val * kd) * (light.L.r * dot) + (light.ambient.r * ka);
        //    float out_g = (cg_val * kd) * (light.L.g * dot) + (light.ambient.g * ka);
        //    float out_b = (cb_val * kd) * (light.L.b * dot) + (light.ambient.b * ka);

        //    // 裁剪和转换
        //    out_r = std::max(0.0f, std::min(1.0f, out_r));
        //    out_g = std::max(0.0f, std::min(1.0f, out_g));
        //    out_b = std::max(0.0f, std::min(1.0f, out_b));

        //    // 写入瓦片缓冲区
        //    int color_idx = (y * tile.width + x) * 3;
        //    tile.colors[color_idx] = (unsigned char)(out_r * 255.0f);
        //    tile.colors[color_idx + 1] = (unsigned char)(out_g * 255.0f);
        //    tile.colors[color_idx + 2] = (unsigned char)(out_b * 255.0f);
        //    tile.zbuffer[y * tile.width + x] = z;
        //}

        int tail_count = tile_max_x - x + 1;
        if (tail_count != 0) {
            __m256i lane_i = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 tailMask =
                _mm256_castsi256_ps(
                    _mm256_cmpgt_epi32(
                        _mm256_set1_epi32(tail_count),
                        lane_i
                    )
                );

            __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
            __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
            __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
            __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
            inside = _mm256_and_ps(inside, w2v_zero);

            inside = _mm256_and_ps(inside, tailMask);

            int mask = _mm256_movemask_ps(inside);
            if (mask != 0)
            {
                //__m256 zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));

                alignas(32) float zb[8];
                for (int i = 0; i < 8; ++i)
                {
                    int px = x + i;
                    zb[i] = (px <= tile_max_x)
                        ? tile.zbuffer(px, y)
                        : std::numeric_limits<float>::infinity();
                }
                __m256 zbuf = _mm256_load_ps(zb);

                __m256 zv_001 = _mm256_cmp_ps(zv, _001, _CMP_GE_OQ);
                __m256 zbuf_zv = _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ);
                __m256 depth_ok = _mm256_and_ps(zv_001, zbuf_zv);

                __m256 final_mask = _mm256_and_ps(inside, depth_ok);
                int final_mask_bits = _mm256_movemask_ps(_mm256_and_ps(inside, depth_ok));


                if (final_mask_bits != 0)
                {
                    // normalize normal
                    __m256 len = _mm256_sqrt_ps(
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                            _mm256_mul_ps(nz, nz)));

                    nx = _mm256_div_ps(nx, len);
                    ny = _mm256_div_ps(ny, len);
                    nz = _mm256_div_ps(nz, len);

                    __m256 dot =
                        _mm256_max_ps(
                            _mm256_add_ps(
                                _mm256_add_ps(_mm256_mul_ps(nx, L_omega_i_x),
                                    _mm256_mul_ps(ny, L_omega_i_y)),
                                _mm256_mul_ps(nz, L_omega_i_z)),
                            zero);

                    __m256 r =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8),
                                _mm256_mul_ps(Lr, dot)),
                            ambinet_r_ka);

                    __m256 g =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8),
                                _mm256_mul_ps(Lg, dot)),
                            ambinet_g_ka);

                    __m256 b =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8),
                                _mm256_mul_ps(Lb, dot)),
                            ambinet_b_ka);

                    r = _mm256_mul_ps(r, _255);
                    g = _mm256_mul_ps(g, _255);
                    b = _mm256_mul_ps(b, _255);


                    alignas(32) float rr[8], gg[8], bb[8], zz[8];
                    _mm256_store_ps(rr, r);
                    _mm256_store_ps(gg, g);
                    _mm256_store_ps(bb, b);
                    _mm256_store_ps(zz, zv);

                    //for (int i = 0; i < tail_count; i++) {
                    //    renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]),
                    //        (unsigned char)(gg[i]),
                    //        (unsigned char)(bb[i]));
                    //    renderer.zbuffer(x + i, y) = zz[i];
                    //}
                    int m = final_mask_bits;
                    while (m)
                    {
                        int i = _tzcnt_u32(m);
                        m &= m - 1;
                        //int px = x + i;
                        //if (px <= tile.width)
                        //{
                        //    renderer.canvas.draw(px, y,
                        //        (unsigned char)rr[i],
                        //        (unsigned char)gg[i],
                        //        (unsigned char)bb[i]);
                        //    renderer.zbuffer(px, y) = zz[i];
                        //}
                        int pixel_x = x + i;
                        if (pixel_x < tile.width) {
                            //int color_idx = (y * tile.width + pixel_x) * 3;
                            //tile.colors[color_idx] = (unsigned char)rr[i];
                            //tile.colors[color_idx + 1] = (unsigned char)gg[i];
                            //tile.colors[color_idx + 2] = (unsigned char)bb[i];
                            renderer.canvas.draw(
                                tile.x + x + i, tile.y + y,
                                (unsigned char)(rr[i]),
                                (unsigned char)(gg[i]),
                                (unsigned char)(bb[i]));
                            //tile.zbuffer[y * tile.width + pixel_x] = zz[i];
                        }
                    }
                    __m256i storeMask = _mm256_castps_si256(final_mask);
                    _mm256_maskstore_ps(&tile.zbuffer[y * tile.width + x], storeMask, zv);
                }
            }
        }

        w0_row += e0.y;
        w1_row += e1.y;
        w2_row += e2.y;

        z_row += dz_dy;
        nx_row += dn_dy_x;
        ny_row += dn_dy_y;
        nz_row += dn_dy_z;
        cr_row += dc_dy_r;
        cg_row += dc_dy_g;
        cb_row += dc_dy_b;
    }
}

// multiple thread
void render5(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L)
{
    // Combine perspective, camera, and world transformations
    matrix p = renderer.perspective * camera * mesh->world;

    const int triCount = (int)mesh->triangles.size();

    static std::mutex logMtx;

    renderer.pool.beginFrame();
    int batchSize = 32;

    for (unsigned int triIdx = 0; triIdx < triCount; ++triIdx) {

        renderer.pool.enqueue([triIdx, &renderer, &mesh, &p, &L]() {

            triIndices& ind = mesh->triangles[triIdx];

            Vertex tvert[3];

            for (unsigned int i = 0; i < 3; i++)
            {
                tvert[i].p = p * mesh->vertices[ind.v[i]].p;
                tvert[i].p.divideW();

                tvert[i].normal =
                    mesh->world * mesh->vertices[ind.v[i]].normal;
                tvert[i].normal.normalise();

                tvert[i].p[0] =
                    (tvert[i].p[0] + 1.f) * 0.5f *
                    (float)renderer.canvas.getWidth();

                tvert[i].p[1] =
                    (tvert[i].p[1] + 1.f) * 0.5f *
                    (float)renderer.canvas.getHeight();

                tvert[i].p[1] =
                    renderer.canvas.getHeight() - tvert[i].p[1];

                tvert[i].rgb = mesh->vertices[ind.v[i]].rgb;
            }

            // clip
            if (!(fabs(tvert[0].p[2]) > 1.0f || fabs(tvert[1].p[2]) > 1.0f || fabs(tvert[2].p[2]) > 1.0f))
            {
                triangle tri(tvert[0], tvert[1], tvert[2]);
                tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
                //tri.draw_SSE(renderer, L, mesh->ka, mesh->kd);
                //tri.draw(renderer, L, mesh->ka, mesh->kd);
            }
            });
    }
    renderer.pool.waitFrame();

}
// multiple thread with transform all vertex first
void render6(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L)
{
    // Combine perspective, camera, and world transformations
    matrix p = renderer.perspective * camera * mesh->world;
    L.omega_i.normalise();

    const int vCount = (int)mesh->vertices.size();
    const int triCount = (int)mesh->triangles.size();

    // Vertex transform cache
    std::vector<Vertex> vsCache(vCount);
    // Transform vertexs first
    for (int i = 0; i < vCount; ++i)
    {
        Vertex& out = vsCache[i];
        const Vertex& in = mesh->vertices[i];

        // clip space
        out.p = p * in.p;
        out.p.divideW();

        // viewport transform
        out.p[0] = (out.p[0] + 1.f) * 0.5f * renderer.canvas.getWidth();
        out.p[1] = (out.p[1] + 1.f) * 0.5f * renderer.canvas.getHeight();
        out.p[1] = renderer.canvas.getHeight() - out.p[1];

        // normal (world space)
        out.normal = mesh->world * in.normal;
        out.normal.normalise();

        out.rgb = in.rgb;
    }

    renderer.pool.beginFrame();

    for (unsigned int triIdx = 0; triIdx < triCount; ++triIdx)
    {
        renderer.pool.enqueue([triIdx, &renderer, &mesh, &vsCache, &L]() {

            triIndices& ind = mesh->triangles[triIdx];

            Vertex tvert[3];

            for (int i = 0; i < 3; ++i)
            {
                const Vertex& v = vsCache[ind.v[i]];

                tvert[i].p = v.p;
                tvert[i].normal = v.normal;
                tvert[i].rgb = v.rgb;
            }

            // clip
            if (!(fabs(tvert[0].p[2]) > 1.0f || fabs(tvert[1].p[2]) > 1.0f || fabs(tvert[2].p[2]) > 1.0f))
            {
                triangle tri(tvert[0], tvert[1], tvert[2]);
                tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
                //tri.draw_SSE(renderer, L, mesh->ka, mesh->kd);
                //tri.draw(renderer, L, mesh->ka, mesh->kd);
            }
            });
    }
    renderer.pool.waitFrame();

}
// 32 batch per thread
void render61(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L)
{
    // Combine perspective, camera, and world transformations
    matrix p = renderer.perspective * camera * mesh->world;

    const int vCount = (int)mesh->vertices.size();
    const int triCount = (int)mesh->triangles.size();
    constexpr int TRI_BATCH = 32;

    // Vertex transform cache
    std::vector<Vertex> vsCache(vCount);
    // Transform vertexs first
    for (int i = 0; i < vCount; ++i)
    {
        Vertex& out = vsCache[i];
        const Vertex& in = mesh->vertices[i];

        // clip space
        out.p = p * in.p;
        out.p.divideW();

        // viewport transform
        out.p[0] = (out.p[0] + 1.f) * 0.5f * renderer.canvas.getWidth();
        out.p[1] = (out.p[1] + 1.f) * 0.5f * renderer.canvas.getHeight();
        out.p[1] = renderer.canvas.getHeight() - out.p[1];

        // normal (world space)
        out.normal = mesh->world * in.normal;
        out.normal.normalise();

        out.rgb = in.rgb;
    }

    renderer.pool.beginFrame();

    for (int triIdx = 0; triIdx < triCount; triIdx += TRI_BATCH)
    {
        int start = triIdx;
        int end = std::min(triIdx + TRI_BATCH, triCount);

        renderer.pool.enqueue([start, end, &renderer, &mesh, &vsCache, &L]() {

            for (int triIdx = start; triIdx < end; ++triIdx)
            {
                triIndices& ind = mesh->triangles[triIdx];
                Vertex tvert[3];
                for (int i = 0; i < 3; ++i)
                {
                    const Vertex& v = vsCache[ind.v[i]];
                    tvert[i].p = v.p;
                    tvert[i].normal = v.normal;
                    tvert[i].rgb = v.rgb;
                }
                // clip
                if (!(fabs(tvert[0].p[2]) > 1.0f || fabs(tvert[1].p[2]) > 1.0f || fabs(tvert[2].p[2]) > 1.0f))
                {
                    triangle tri(tvert[0], tvert[1], tvert[2]);
                    tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
                    //tri.draw_SSE(renderer, L, mesh->ka, mesh->kd);
                    //tri.draw(renderer, L, mesh->ka, mesh->kd);
                }
            }
            //for (int triIdx = start; triIdx < end; ++triIdx){
            //    triIndices& ind = mesh->triangles[triIdx];
            //    // clip
            //    if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.0f || fabs(vsCache[ind.v[1]].p[2]) > 1.0f || fabs(vsCache[ind.v[2]].p[2]) > 1.0f))
            //    {
            //        triangle tri(vsCache[ind.v[0]], vsCache[ind.v[1]], vsCache[ind.v[2]]);
            //        //tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
            //        //tri.draw_SSE(renderer, L, mesh->ka, mesh->kd);
            //        //tri.draw(renderer, L, mesh->ka, mesh->kd);
            //    }
            //}
            });
    }
    renderer.pool.waitFrame();

}


// create thread per frame 
void render2(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L)
{
    // Combine perspective, camera, and world transformations
    matrix p = renderer.perspective * camera * mesh->world;

    const int triCount = (int)mesh->triangles.size();

    // atomic triangle index
    std::atomic<int> nextTri{ 0 };

    // thread count
    //const unsigned int threadCount = std::max(1u, std::thread::hardware_concurrency());
    const unsigned int threadCount = 11;

    unsigned int per_thread_work_num = triCount / threadCount;

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    // launch threads
    for (unsigned int t = 0; t < threadCount; ++t)
    {
        workers.emplace_back([&, t]() {

            while (true)
            {
                int triIdx = nextTri.fetch_add(1, std::memory_order_relaxed);
                if (triIdx >= triCount)
                    break;

                // ================================
                // ↓↓↓ 原来 for(triIndices& ind) 的内容
                // ================================
                triIndices& ind = mesh->triangles[triIdx];

                Vertex tvert[3];

                for (unsigned int i = 0; i < 3; i++)
                {
                    tvert[i].p = p * mesh->vertices[ind.v[i]].p;
                    tvert[i].p.divideW();

                    tvert[i].normal =
                        mesh->world * mesh->vertices[ind.v[i]].normal;
                    tvert[i].normal.normalise();

                    tvert[i].p[0] =
                        (tvert[i].p[0] + 1.f) * 0.5f *
                        (float)renderer.canvas.getWidth();

                    tvert[i].p[1] =
                        (tvert[i].p[1] + 1.f) * 0.5f *
                        (float)renderer.canvas.getHeight();

                    tvert[i].p[1] =
                        renderer.canvas.getHeight() - tvert[i].p[1];

                    tvert[i].rgb = mesh->vertices[ind.v[i]].rgb;
                }

                // clip
                if (fabs(tvert[0].p[2]) > 1.0f ||
                    fabs(tvert[1].p[2]) > 1.0f ||
                    fabs(tvert[2].p[2]) > 1.0f)
                    continue;

                triangle tri(tvert[0], tvert[1], tvert[2]);
                tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
                //tri.draw(renderer, L, mesh->ka, mesh->kd);
            }
            });
    }

    // join
    for (auto& th : workers)
        th.join();
}

void render21(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L)
{
    // Combine perspective, camera, and world transformations
    matrix p = renderer.perspective * camera * mesh->world;

    const int triCount = (int)mesh->triangles.size();

    // atomic triangle index
    std::atomic<int> nextTri{ 0 };

    // thread count
    //const unsigned int threadCount = std::max(1u, std::thread::hardware_concurrency());
    const unsigned int threadCount = 11;

    unsigned int per_thread_work_num = triCount / threadCount;

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    // launch threads
    for (unsigned int t = 0; t < threadCount; ++t)
    {
        workers.emplace_back([&, t]() {

            unsigned int start = t * per_thread_work_num;
            unsigned int end = std::min(start + per_thread_work_num, (unsigned int)triCount);


            //int triIdx = nextTri.fetch_add(1, std::memory_order_relaxed);
            //if (triIdx >= triCount)
            //    break;
            for (unsigned int triIdx = start; triIdx < end; ++triIdx)
            {
                triIndices& ind = mesh->triangles[triIdx];

                Vertex tvert[3];

                for (unsigned int i = 0; i < 3; i++)
                {
                    tvert[i].p = p * mesh->vertices[ind.v[i]].p;
                    tvert[i].p.divideW();

                    tvert[i].normal =
                        mesh->world * mesh->vertices[ind.v[i]].normal;
                    tvert[i].normal.normalise();

                    tvert[i].p[0] =
                        (tvert[i].p[0] + 1.f) * 0.5f *
                        (float)renderer.canvas.getWidth();

                    tvert[i].p[1] =
                        (tvert[i].p[1] + 1.f) * 0.5f *
                        (float)renderer.canvas.getHeight();

                    tvert[i].p[1] =
                        renderer.canvas.getHeight() - tvert[i].p[1];

                    tvert[i].rgb = mesh->vertices[ind.v[i]].rgb;
                }

                // clip
                if (fabs(tvert[0].p[2]) > 1.0f ||
                    fabs(tvert[1].p[2]) > 1.0f ||
                    fabs(tvert[2].p[2]) > 1.0f)
                    continue;

                triangle tri(tvert[0], tvert[1], tvert[2]);
                //tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
                //tri.draw(renderer, L, mesh->ka, mesh->kd);
            }

            });
    }

    // join
    for (auto& th : workers)
        th.join();

    for (unsigned int tr = threadCount * per_thread_work_num; tr < mesh->triangles.size(); ++tr) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices
        triIndices& ind = mesh->triangles[tr];
        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2]);
        //tri.draw_AVX2(renderer, L, mesh->ka, mesh->kd);
        //tri.draw_LEE1(renderer, L, mesh->ka, mesh->kd);
        //tri.draw_LEE2(renderer, L, mesh->ka, mesh->kd);
        //tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}


/*
void render_transfirst_SoA_SIMD_AVX2(Renderer& renderer, Mesh_SoA* mesh, matrix& camera, Light& L) {
    // Combine matrices
    matrix p = renderer.perspective * camera * mesh->world;

    L.omega_i.normalise();

    const int vCount = (int)mesh->px.size();
    const int triCount = (int)mesh->triangles.size();

    // Vertex cache
    std::vector<Vertex> vsCache(vCount);

    int i = 0;
    for (; i + 7 < vCount; i += 8) {
        // Output buffers for 8 vertices
        float outX[8], outY[8], outZ[8], outW[8];

        // Batch matrix * vertex multiplication
        matrix::batchMul(p.data,
            &mesh->px[i], &mesh->py[i], &mesh->pz[i], &mesh->pw[i],
            outX, outY, outZ, outW);

        // Perspective divide and viewport transform
        __m256 vx = _mm256_loadu_ps(outX);
        __m256 vy = _mm256_loadu_ps(outY);
        __m256 vz = _mm256_loadu_ps(outZ);
        __m256 vw = _mm256_loadu_ps(outW);

        // Divide x/y/z by w
        __m256 invW = _mm256_div_ps(_mm256_set1_ps(1.f), vw);
        vx = _mm256_mul_ps(vx, invW);
        vy = _mm256_mul_ps(vy, invW);
        vz = _mm256_mul_ps(vz, invW);

        // Viewport transform
        const float width = float(renderer.canvas.getWidth());
        const float height = float(renderer.canvas.getHeight());
        __m256 halfWidth = _mm256_set1_ps(0.5f * width);
        __m256 halfHeight = _mm256_set1_ps(0.5f * height);
        __m256 one = _mm256_set1_ps(1.f);

        vx = _mm256_mul_ps(_mm256_add_ps(vx, one), halfWidth);
        vy = _mm256_mul_ps(_mm256_add_ps(vy, one), halfHeight);
        vy = _mm256_sub_ps(_mm256_set1_ps(height), vy); // flip y

        // Store back to Vertex cache
        float tx[8], ty[8], tz[8];
        _mm256_storeu_ps(tx, vx);
        _mm256_storeu_ps(ty, vy);
        _mm256_storeu_ps(tz, vz);

        for (int j = 0; j < 8; ++j) {
            Vertex& out = vsCache[i + j];
            out.p = vec4(tx[j], ty[j], tz[j], 1.f);

            // Normal transform (标量, 因为矩阵 * vec4 很小)
            vec4 normal(mesh->nx[i + j], mesh->ny[i + j], mesh->nz[i + j], 0.f);
            out.normal = mesh->world * normal;
            out.normal.normalise();

            out.rgb.set(mesh->cr[i + j], mesh->cg[i + j], mesh->cb[i + j]);
        }
    }

    // Handle remaining vertices
    for (; i < vCount; ++i) {
        Vertex& out = vsCache[i];

        vec4 pos(mesh->px[i], mesh->py[i], mesh->pz[i], mesh->pw[i]);
        out.p = p * pos;
        out.p.divideW();

        out.p[0] = (out.p[0] + 1.f) * 0.5f * renderer.canvas.getWidth();
        out.p[1] = (out.p[1] + 1.f) * 0.5f * renderer.canvas.getHeight();
        out.p[1] = float(renderer.canvas.getHeight()) - out.p[1];

        vec4 normal(mesh->nx[i], mesh->ny[i], mesh->nz[i], 0.f);
        out.normal = mesh->world * normal;
        out.normal.normalise();

        out.rgb.set(mesh->cr[i], mesh->cg[i], mesh->cb[i]);
    }

    // Draw triangles
    for (int triIdx = 0; triIdx < triCount; ++triIdx) {
        triIndices& ind = mesh->triangles[triIdx];

        if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.0f ||
            fabs(vsCache[ind.v[1]].p[2]) > 1.0f ||
            fabs(vsCache[ind.v[2]].p[2]) > 1.0f)) {

            triangle tri(vsCache[ind.v[0]], vsCache[ind.v[1]], vsCache[ind.v[2]]);
            tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
        }
    }
}
*/
void render_transfirst_SoA_SIMD_AVX2(Renderer& renderer, Mesh_SoA* mesh, matrix& camera, Light& L) {
    // Combine matrices
    matrix p = renderer.perspective * camera * mesh->world;

    L.omega_i.normalise();

    const int vCount = (int)mesh->positions_x.size();
    const int triCount = (int)mesh->triangles.size();

    // Vertex cache
    std::vector<Vertex> vsCache(vCount);

    int i = 0;
    for (; i + 7 < vCount; i += 8) {
        // 1️⃣ 顶点坐标批量矩阵乘法
        float outX[8], outY[8], outZ[8], outW[8];
        matrix::batchMul(p.data,
            &mesh->positions_x[i], &mesh->positions_y[i], &mesh->positions_z[i], &mesh->positions_w[i],
            outX, outY, outZ, outW);

        // 2️⃣ AVX2 透视除法 + 视口变换
        __m256 vx = _mm256_loadu_ps(outX);
        __m256 vy = _mm256_loadu_ps(outY);
        __m256 vz = _mm256_loadu_ps(outZ);
        __m256 vw = _mm256_loadu_ps(outW);

        __m256 invW = _mm256_div_ps(_mm256_set1_ps(1.f), vw);
        vx = _mm256_mul_ps(vx, invW);
        vy = _mm256_mul_ps(vy, invW);
        vz = _mm256_mul_ps(vz, invW);

        const float width = float(renderer.canvas.getWidth());
        const float height = float(renderer.canvas.getHeight());
        __m256 halfWidth = _mm256_set1_ps(0.5f * width);
        __m256 halfHeight = _mm256_set1_ps(0.5f * height);
        __m256 one = _mm256_set1_ps(1.f);

        vx = _mm256_mul_ps(_mm256_add_ps(vx, one), halfWidth);
        vy = _mm256_mul_ps(_mm256_add_ps(vy, one), halfHeight);
        vy = _mm256_sub_ps(_mm256_set1_ps(height), vy); // flip y

        float tx[8], ty[8], tz[8];
        _mm256_storeu_ps(tx, vx);
        _mm256_storeu_ps(ty, vy);
        _mm256_storeu_ps(tz, vz);

        // 3️⃣ 法线批量变换 + normalize
        __m256 nx = _mm256_loadu_ps(&mesh->normals_x[i]);
        __m256 ny = _mm256_loadu_ps(&mesh->normals_y[i]);
        __m256 nz = _mm256_loadu_ps(&mesh->normals_z[i]);

        // world matrix rows
        __m256 row0 = _mm256_set_m128(
            _mm_load_ps(&mesh->world.data.a[12]), _mm_load_ps(&mesh->world.data.a[0])
        );
        __m256 row1 = _mm256_set_m128(
            _mm_load_ps(&mesh->world.data.a[13]), _mm_load_ps(&mesh->world.data.a[4])
        );
        __m256 row2 = _mm256_set_m128(
            _mm_load_ps(&mesh->world.data.a[14]), _mm_load_ps(&mesh->world.data.a[8])
        );

        __m256 txN = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[0]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[1]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[2]), nz)
        );
        __m256 tyN = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[4]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[5]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[6]), nz)
        );
        __m256 tzN = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[8]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[9]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[10]), nz)
        );

        // normalize
        __m256 lenSq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(txN, txN),
            _mm256_mul_ps(tyN, tyN)),
            _mm256_mul_ps(tzN, tzN));
        __m256 invLen = _mm256_rsqrt_ps(lenSq); // approx reciprocal sqrt
        txN = _mm256_mul_ps(txN, invLen);
        tyN = _mm256_mul_ps(tyN, invLen);
        tzN = _mm256_mul_ps(tzN, invLen);

        float nxOut[8], nyOut[8], nzOut[8];
        _mm256_storeu_ps(nxOut, txN);
        _mm256_storeu_ps(nyOut, tyN);
        _mm256_storeu_ps(nzOut, tzN);

        // 4️⃣ 存回 Vertex cache
        for (int j = 0; j < 8; ++j) {
            Vertex& out = vsCache[i + j];
            out.p = vec4(tx[j], ty[j], tz[j], 1.f);
            out.normal = vec4(nxOut[j], nyOut[j], nzOut[j], 0.f);
            out.rgb.set(mesh->colors_r[i + j], mesh->colors_g[i + j], mesh->colors_b[i + j]);
        }
    }

    // 处理剩余顶点
    for (; i < vCount; ++i) {
        Vertex& out = vsCache[i];

        vec4 pos(mesh->positions_x[i], mesh->positions_y[i], mesh->positions_z[i], mesh->positions_w[i]);
        out.p = p * pos;
        out.p.divideW();
        out.p[0] = (out.p[0] + 1.f) * 0.5f * renderer.canvas.getWidth();
        out.p[1] = (out.p[1] + 1.f) * 0.5f * renderer.canvas.getHeight();
        out.p[1] = float(renderer.canvas.getHeight()) - out.p[1];

        vec4 normal(mesh->normals_x[i], mesh->normals_y[i], mesh->normals_z[i], 0.f);
        out.normal = mesh->world * normal;
        out.normal.normalise();

        out.rgb.set(mesh->colors_r[i], mesh->colors_g[i], mesh->colors_b[i]);
    }

    // 绘制三角形
    for (int triIdx = 0; triIdx < triCount; ++triIdx) {
        triIndices& ind = mesh->triangles[triIdx];

        if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.0f ||
            fabs(vsCache[ind.v[1]].p[2]) > 1.0f ||
            fabs(vsCache[ind.v[2]].p[2]) > 1.0f)) {

            triangle tri(vsCache[ind.v[0]], vsCache[ind.v[1]], vsCache[ind.v[2]]);
            tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
        }
    }
}



void transformBatchSIMD2(const matrix& mvp, const matrix& world,
    const Mesh_SoA& source, int start_idx, int count,
    float canvas_width, float canvas_height) {
    // 边界检查
    int total_vertices = static_cast<int>(source.positions_x.size());
    if (start_idx < 0 || start_idx >= total_vertices || count <= 0) {
        return;
    }

    int end_idx = std::min(start_idx + count, total_vertices);
    int actual_count = end_idx - start_idx;

    // 确保输出向量有足够空间
    ensureCapacity(start_idx + actual_count);

    // 如果是第一次调用，复制三角形和材质属性
    if (start_idx == 0 && actual_count == total_vertices) {
        triangles = source.triangles;
        ka = source.ka;
        kd = source.kd;
    }

    // 获取源数据指针
    const float* px = source.positions_x.data() + start_idx;
    const float* py = source.positions_y.data() + start_idx;
    const float* pz = source.positions_z.data() + start_idx;
    const float* pw = source.positions_w.data() + start_idx;

    const float* nx = source.normals_x.data() + start_idx;
    const float* ny = source.normals_y.data() + start_idx;
    const float* nz = source.normals_z.data() + start_idx;

    const float* cr = source.colors_r.data() + start_idx;
    const float* cg = source.colors_g.data() + start_idx;
    const float* cb = source.colors_b.data() + start_idx;

    // 获取目标数据指针
    float* tpx = transformed_positions_x.data() + start_idx;
    float* tpy = transformed_positions_y.data() + start_idx;
    float* tpz = transformed_positions_z.data() + start_idx;
    float* tpw = transformed_positions_w.data() + start_idx;

    float* tnx = transformed_normals_x.data() + start_idx;
    float* tny = transformed_normals_y.data() + start_idx;
    float* tnz = transformed_normals_z.data() + start_idx;

    float* out_cr = colors_r.data() + start_idx;
    float* out_cg = colors_g.data() + start_idx;
    float* out_cb = colors_b.data() + start_idx;

    // 每次处理8个顶点
    const int simd_width = 8;
    int i = 0;

    // 临时存储数组（用于batchMul的输出）
    alignas(32) float temp_x[8], temp_y[8], temp_z[8], temp_w[8];
    alignas(32) float temp_nx[8], temp_ny[8], temp_nz[8];

    for (; i <= actual_count - simd_width; i += simd_width) {
        // 1️⃣ 顶点坐标批量矩阵乘法 (使用 batchMul 方法)
        matrix::batchMul(mvp.data,
            px + i, py + i, pz + i, pw + i,
            temp_x, temp_y, temp_z, temp_w);

        // 2️⃣ AVX2 透视除法 + 视口变换
        __m256 vx = _mm256_load_ps(temp_x);
        __m256 vy = _mm256_load_ps(temp_y);
        __m256 vz = _mm256_load_ps(temp_z);
        __m256 vw = _mm256_load_ps(temp_w);

        // 透视除法
        __m256 invW = _mm256_div_ps(_mm256_set1_ps(1.0f), vw);
        vx = _mm256_mul_ps(vx, invW);
        vy = _mm256_mul_ps(vy, invW);
        vz = _mm256_mul_ps(vz, invW);

        // 视口变换
        const __m256 width_vec = _mm256_set1_ps(canvas_width);
        const __m256 height_vec = _mm256_set1_ps(canvas_height);
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);

        // screen_x = (clip_x + 1.0) * 0.5 * width
        vx = _mm256_mul_ps(_mm256_add_ps(vx, one), _mm256_mul_ps(half, width_vec));

        // screen_y = height - ((clip_y + 1.0) * 0.5 * height)
        vy = _mm256_mul_ps(_mm256_add_ps(vy, one), _mm256_mul_ps(half, height_vec));
        vy = _mm256_sub_ps(height_vec, vy); // flip y

        // 存储变换后的顶点位置
        _mm256_storeu_ps(tpx + i, vx);
        _mm256_storeu_ps(tpy + i, vy);
        _mm256_storeu_ps(tpz + i, vz);
        _mm256_storeu_ps(tpw + i, vw);

        // 3️⃣ 法线批量变换 + normalize
        // 加载原始法线
        __m256 nx_vec = _mm256_loadu_ps(nx + i);
        __m256 ny_vec = _mm256_loadu_ps(ny + i);
        __m256 nz_vec = _mm256_loadu_ps(nz + i);

        // 法线变换：只使用世界矩阵的3x3部分（假设法线w=0）
        // txN = world[0]*nx + world[1]*ny + world[2]*nz
        // tyN = world[4]*nx + world[5]*ny + world[6]*nz
        // tzN = world[8]*nx + world[9]*ny + world[10]*nz

        __m256 txN = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(world.data.a[0]), nx_vec),
                _mm256_mul_ps(_mm256_set1_ps(world.data.a[1]), ny_vec)),
            _mm256_mul_ps(_mm256_set1_ps(world.data.a[2]), nz_vec)
        );

        __m256 tyN = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(world.data.a[4]), nx_vec),
                _mm256_mul_ps(_mm256_set1_ps(world.data.a[5]), ny_vec)),
            _mm256_mul_ps(_mm256_set1_ps(world.data.a[6]), nz_vec)
        );

        __m256 tzN = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(world.data.a[8]), nx_vec),
                _mm256_mul_ps(_mm256_set1_ps(world.data.a[9]), ny_vec)),
            _mm256_mul_ps(_mm256_set1_ps(world.data.a[10]), nz_vec)
        );

        // 归一化
        __m256 lenSq = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(txN, txN),
                _mm256_mul_ps(tyN, tyN)),
            _mm256_mul_ps(tzN, tzN)
        );

        // 使用 rsqrt 进行近似倒数平方根，然后进行一次牛顿迭代提高精度
        __m256 invLen = _mm256_rsqrt_ps(lenSq);

        // 牛顿迭代：x_{n+1} = x_n * (1.5 - 0.5 * a * x_n^2)
        __m256 three_halfs = _mm256_set1_ps(1.5f);
        __m256 half_vec = _mm256_set1_ps(0.5f);
        __m256 x2 = _mm256_mul_ps(invLen, invLen);
        __m256 a = _mm256_mul_ps(lenSq, x2);
        a = _mm256_sub_ps(three_halfs, _mm256_mul_ps(half_vec, a));
        invLen = _mm256_mul_ps(invLen, a);

        // 应用归一化
        txN = _mm256_mul_ps(txN, invLen);
        tyN = _mm256_mul_ps(tyN, invLen);
        tzN = _mm256_mul_ps(tzN, invLen);

        // 存储变换后的法线
        _mm256_storeu_ps(tnx + i, txN);
        _mm256_storeu_ps(tny + i, tyN);
        _mm256_storeu_ps(tnz + i, tzN);

        // 4️⃣ 复制颜色数据
        __m256 color_r = _mm256_loadu_ps(cr + i);
        __m256 color_g = _mm256_loadu_ps(cg + i);
        __m256 color_b = _mm256_loadu_ps(cb + i);

        _mm256_storeu_ps(out_cr + i, color_r);
        _mm256_storeu_ps(out_cg + i, color_g);
        _mm256_storeu_ps(out_cb + i, color_b);
    }

    // 处理剩余顶点（非SIMD路径）
    for (; i < actual_count; ++i) {
        int idx = start_idx + i;
        int src_idx = idx;

        // 顶点位置变换
        vec4 pos(source.positions_x[src_idx],
            source.positions_y[src_idx],
            source.positions_z[src_idx],
            source.positions_w[src_idx]);

        vec4 clip_pos = mvp * pos;
        clip_pos.divideW();

        // 视口变换
        float screen_x = (clip_pos[0] + 1.0f) * 0.5f * canvas_width;
        float screen_y = (clip_pos[1] + 1.0f) * 0.5f * canvas_height;
        screen_y = canvas_height - screen_y;  // Y轴翻转

        transformed_positions_x[idx] = screen_x;
        transformed_positions_y[idx] = screen_y;
        transformed_positions_z[idx] = clip_pos[2];
        transformed_positions_w[idx] = clip_pos[3];

        // 法线变换
        vec4 normal(source.normals_x[src_idx],
            source.normals_y[src_idx],
            source.normals_z[src_idx],
            0.0f);

        normal = world * normal;
        normal.normalise();

        transformed_normals_x[idx] = normal[0];
        transformed_normals_y[idx] = normal[1];
        transformed_normals_z[idx] = normal[2];

        // 颜色复制
        colors_r[idx] = source.colors_r[src_idx];
        colors_g[idx] = source.colors_g[src_idx];
        colors_b[idx] = source.colors_b[src_idx];
    }
}

/*
void transformBatchSIMD(const matrix& p, const matrix& world,
    const Mesh_SoA& source, int start_idx, int count) {
    const int simd_count = (count + 7) / 8 * 8; // 对齐到8的倍数

     //确保有足够空间
    if (transformed_positions_x.size() < start_idx + simd_count) {
        transformed_positions_x.resize(start_idx + simd_count);
        transformed_positions_y.resize(start_idx + simd_count);
        transformed_positions_z.resize(start_idx + simd_count);
        transformed_positions_w.resize(start_idx + simd_count);
        transformed_normals_x.resize(start_idx + simd_count);
        transformed_normals_y.resize(start_idx + simd_count);
        transformed_normals_z.resize(start_idx + simd_count);
    }

    // 获取画布尺寸（需要从外部传入）
    const float width = 800.0f; // 应该从renderer获取
    const float height = 600.0f;
    const __m256 halfWidth = _mm256_set1_ps(0.5f * width);
    const __m256 halfHeight = _mm256_set1_ps(0.5f * height);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 canvasHeight = _mm256_set1_ps(height);

    // 批量变换顶点
    for (int i = 0; i < count; i += 8) {
        int actual_count = std::min(8, count - i);
        int store_idx = start_idx + i;

        // 加载源数据
        __m256 vx = loadPartial(&source.positions_x[start_idx + i], actual_count);
        __m256 vy = loadPartial(&source.positions_y[start_idx + i], actual_count);
        __m256 vz = loadPartial(&source.positions_z[start_idx + i], actual_count);
        __m256 vw = loadPartial(&source.positions_w[start_idx + i], actual_count);

        // 矩阵变换
        __m256 outX = _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[0]), vx,
            _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[1]), vy,
                _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[2]), vz,
                    _mm256_mul_ps(_mm256_set1_ps(p.data.a[3]), vw))));

        __m256 outY = _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[4]), vx,
            _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[5]), vy,
                _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[6]), vz,
                    _mm256_mul_ps(_mm256_set1_ps(p.data.a[7]), vw))));

        __m256 outZ = _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[8]), vx,
            _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[9]), vy,
                _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[10]), vz,
                    _mm256_mul_ps(_mm256_set1_ps(p.data.a[11]), vw))));

        __m256 outW = _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[12]), vx,
            _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[13]), vy,
                _mm256_fmadd_ps(_mm256_set1_ps(p.data.a[14]), vz,
                    _mm256_mul_ps(_mm256_set1_ps(p.data.a[15]), vw))));

        // 透视除法（使用更精确的倒数）
        __m256 invW = _mm256_rcp_ps(outW);
        // Newton-Raphson refinement for better precision
        invW = _mm256_mul_ps(invW, _mm256_sub_ps(_mm256_set1_ps(2.0f),
            _mm256_mul_ps(outW, invW)));

        outX = _mm256_mul_ps(outX, invW);
        outY = _mm256_mul_ps(outY, invW);
        outZ = _mm256_mul_ps(outZ, invW);

        // 视口变换
        outX = _mm256_fmadd_ps(_mm256_add_ps(outX, one), halfWidth, zero);
        outY = _mm256_fmadd_ps(_mm256_sub_ps(one, outY), halfHeight, zero); // Y翻转

        // 存储变换后的位置
        storePartial(&transformed_positions_x[store_idx], outX, actual_count);
        storePartial(&transformed_positions_y[store_idx], outY, actual_count);
        storePartial(&transformed_positions_z[store_idx], outZ, actual_count);
        storePartial(&transformed_positions_w[store_idx], outW, actual_count);

        // 变换法线（使用世界矩阵的3x3部分）
        __m256 nx = loadPartial(&source.normals_x[start_idx + i], actual_count);
        __m256 ny = loadPartial(&source.normals_y[start_idx + i], actual_count);
        __m256 nz = loadPartial(&source.normals_z[start_idx + i], actual_count);

        __m256 outNX = _mm256_fmadd_ps(_mm256_set1_ps(world.data.a[0]), nx,
            _mm256_fmadd_ps(_mm256_set1_ps(world.data.a[1]), ny,
                _mm256_mul_ps(_mm256_set1_ps(world.data.a[2]), nz)));

        __m256 outNY = _mm256_fmadd_ps(_mm256_set1_ps(world.data.a[4]), nx,
            _mm256_fmadd_ps(_mm256_set1_ps(world.data.a[5]), ny,
                _mm256_mul_ps(_mm256_set1_ps(world.data.a[6]), nz)));

        __m256 outNZ = _mm256_fmadd_ps(_mm256_set1_ps(world.data.a[8]), nx,
            _mm256_fmadd_ps(_mm256_set1_ps(world.data.a[9]), ny,
                _mm256_mul_ps(_mm256_set1_ps(world.data.a[10]), nz)));

        // 归一化
        __m256 lenSq = _mm256_fmadd_ps(outNX, outNX,
            _mm256_fmadd_ps(outNY, outNY,
                _mm256_mul_ps(outNZ, outNZ)));

        __m256 invLen = _mm256_rsqrt_ps(lenSq);
        // Newton-Raphson refinement
        invLen = _mm256_mul_ps(invLen, _mm256_sub_ps(_mm256_set1_ps(1.5f),
            _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_mul_ps(lenSq, _mm256_mul_ps(invLen, invLen)))));

        outNX = _mm256_mul_ps(outNX, invLen);
        outNY = _mm256_mul_ps(outNY, invLen);
        outNZ = _mm256_mul_ps(outNZ, invLen);

        // 存储变换后的法线
        storePartial(&transformed_normals_x[store_idx], outNX, actual_count);
        storePartial(&transformed_normals_y[store_idx], outNY, actual_count);
        storePartial(&transformed_normals_z[store_idx], outNZ, actual_count);
    }

    //// 复制颜色数据
    //if (colors_r.size() < start_idx + count) {
    //    colors_r.resize(start_idx + count);
    //    colors_g.resize(start_idx + count);
    //    colors_b.resize(start_idx + count);
    //}

    std::copy(source.colors_r.begin() + start_idx,
        source.colors_r.begin() + start_idx + count,
        colors_r.begin() + start_idx);
    std::copy(source.colors_g.begin() + start_idx,
        source.colors_g.begin() + start_idx + count,
        colors_g.begin() + start_idx);
    std::copy(source.colors_b.begin() + start_idx,
        source.colors_b.begin() + start_idx + count,
        colors_b.begin() + start_idx);
}
*/



// 这个函数基本就是你的renderTriangleInTile，但使用transformed_mesh
void renderTriangleFromSOA_InTile(Renderer& renderer, const Mesh_SoA_Transformed& mesh, Light& light, float ka, float kd,
    int triangle_idx, Tile& tile) {
    // 加载顶点数据（从变换后的SOA数据）
    const triIndices& idx = mesh.triangles[triangle_idx];

    // 加载变换后的顶点位置（屏幕空间）
    float v0_x = mesh.transformed_positions_x[idx.v[0]];
    float v1_x = mesh.transformed_positions_x[idx.v[1]];
    float v2_x = mesh.transformed_positions_x[idx.v[2]];

    float v0_y = mesh.transformed_positions_y[idx.v[0]];
    float v1_y = mesh.transformed_positions_y[idx.v[1]];
    float v2_y = mesh.transformed_positions_y[idx.v[2]];

    float v0_z = mesh.transformed_positions_z[idx.v[0]];
    float v1_z = mesh.transformed_positions_z[idx.v[1]];
    float v2_z = mesh.transformed_positions_z[idx.v[2]];

    // 检查有效性
    if (!isfinite(v0_x) || !isfinite(v0_y) || !isfinite(v0_z) ||
        !isfinite(v1_x) || !isfinite(v1_y) || !isfinite(v1_z) ||
        !isfinite(v2_x) || !isfinite(v2_y) || !isfinite(v2_z)) {
        return;
    }

    // 计算三角形面积（用于重心坐标插值）
    vec4 e0 = makeEdge0(v1_x, v1_y, v2_x, v2_y);
    vec4 e1 = makeEdge0(v2_x, v2_y, v0_x, v0_y);
    vec4 e2 = makeEdge0(v0_x, v0_y, v1_x, v1_y);

    float area = v0_x * e0.x + v0_y * e0.y + e0.z;
    if (area < 1e-6f) return;  // 面积太小，忽略

    float invArea = 1.0f / area;

    // 加载变换后的法线数据
    //const float* norm_x = mesh.transformed_normals_x.data();
    //const float* norm_y = mesh.transformed_normals_y.data();
    //const float* norm_z = mesh.transformed_normals_z.data();

    float n0_x = mesh.transformed_normals_x[idx.v[0]];
    float n0_y = mesh.transformed_normals_y[idx.v[0]];
    float n0_z = mesh.transformed_normals_z[idx.v[0]];

    float n1_x = mesh.transformed_normals_x[idx.v[1]];
    float n1_y = mesh.transformed_normals_y[idx.v[1]];
    float n1_z = mesh.transformed_normals_z[idx.v[1]];

    float n2_x = mesh.transformed_normals_x[idx.v[2]];
    float n2_y = mesh.transformed_normals_y[idx.v[2]];
    float n2_z = mesh.transformed_normals_z[idx.v[2]];

    // 加载颜色数据
    //const float* col_r = mesh.colors_r.data();
    //const float* col_g = mesh.colors_g.data();
    //const float* col_b = mesh.colors_b.data();

    float c0_r = mesh.colors_r[idx.v[0]];
    float c0_g = mesh.colors_g[idx.v[0]];
    float c0_b = mesh.colors_b[idx.v[0]];

    float c1_r = mesh.colors_r[idx.v[1]];
    float c1_g = mesh.colors_g[idx.v[1]];
    float c1_b = mesh.colors_b[idx.v[1]];

    float c2_r = mesh.colors_r[idx.v[2]];
    float c2_g = mesh.colors_g[idx.v[2]];
    float c2_b = mesh.colors_b[idx.v[2]];


    // 计算插值增量
    float dz_dx = (v0_z * e0.x + v1_z * e1.x + v2_z * e2.x) * invArea;
    float dz_dy = (v0_z * e0.y + v1_z * e1.y + v2_z * e2.y) * invArea;

    float dn_dx_x = (n0_x * e0.x + n1_x * e1.x + n2_x * e2.x) * invArea;
    float dn_dx_y = (n0_y * e0.x + n1_y * e1.x + n2_y * e2.x) * invArea;
    float dn_dx_z = (n0_z * e0.x + n1_z * e1.x + n2_z * e2.x) * invArea;

    float dn_dy_x = (n0_x * e0.y + n1_x * e1.y + n2_x * e2.y) * invArea;
    float dn_dy_y = (n0_y * e0.y + n1_y * e1.y + n2_y * e2.y) * invArea;
    float dn_dy_z = (n0_z * e0.y + n1_z * e1.y + n2_z * e2.y) * invArea;

    float dc_dx_r = (c0_r * e0.x + c1_r * e1.x + c2_r * e2.x) * invArea;
    float dc_dx_g = (c0_g * e0.x + c1_g * e1.x + c2_g * e2.x) * invArea;
    float dc_dx_b = (c0_b * e0.x + c1_b * e1.x + c2_b * e2.x) * invArea;

    float dc_dy_r = (c0_r * e0.y + c1_r * e1.y + c2_r * e2.y) * invArea;
    float dc_dy_g = (c0_g * e0.y + c1_g * e1.y + c2_g * e2.y) * invArea;
    float dc_dy_b = (c0_b * e0.y + c1_b * e1.y + c2_b * e2.y) * invArea;

    // 计算三角形在瓦片空间的包围盒
    float min_x = std::max((float)tile.x, std::min({ v0_x, v1_x, v2_x }));
    float max_x = std::min((float)(tile.x + tile.width), std::max({ v0_x, v1_x, v2_x }));
    float min_y = std::max((float)tile.y, std::min({ v0_y, v1_y, v2_y }));
    float max_y = std::min((float)(tile.y + tile.height), std::max({ v0_y, v1_y, v2_y }));

    if (min_x > max_x || min_y > max_y) return;

    // 转换为瓦片局部坐标
    int tile_min_x = (int)std::floor(min_x) - tile.x;
    int tile_max_x = (int)std::ceil(max_x) - tile.x;
    int tile_min_y = (int)std::floor(min_y) - tile.y;
    int tile_max_y = (int)std::ceil(max_y) - tile.y;

    // 裁剪到瓦片边界
    tile_min_x = std::max(0, tile_min_x);
    tile_max_x = std::min(tile.width - 1, tile_max_x);
    tile_min_y = std::max(0, tile_min_y);
    tile_max_y = std::min(tile.height - 1, tile_max_y);

    if (tile_min_x > tile_max_x || tile_min_y > tile_max_y) return;

    // --- AVX2常量 ---
    const __m256 zero = _mm256_setzero_ps();
    const __m256 lane = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    const __m256 _255 = _mm256_set1_ps(255.0f);
    const __m256 _001 = _mm256_set1_ps(0.001f);
    const __m256 _1 = _mm256_set1_ps(1.0f);

    // 预计算增量
    __m256 w0_lane = _mm256_mul_ps(_mm256_set1_ps(e0.x), lane);
    __m256 w1_lane = _mm256_mul_ps(_mm256_set1_ps(e1.x), lane);
    __m256 w2_lane = _mm256_mul_ps(_mm256_set1_ps(e2.x), lane);

    __m256 z_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lane);
    __m256 nx_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_x), lane);
    __m256 ny_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_y), lane);
    __m256 nz_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_z), lane);
    __m256 cr_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_r), lane);
    __m256 cg_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_g), lane);
    __m256 cb_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_b), lane);

    __m256 w0_step8 = _mm256_set1_ps(e0.x * 8.0f);
    __m256 w1_step8 = _mm256_set1_ps(e1.x * 8.0f);
    __m256 w2_step8 = _mm256_set1_ps(e2.x * 8.0f);
    __m256 z_step8 = _mm256_set1_ps(dz_dx * 8.0f);
    __m256 nx_step8 = _mm256_set1_ps(dn_dx_x * 8.0f);
    __m256 ny_step8 = _mm256_set1_ps(dn_dx_y * 8.0f);
    __m256 nz_step8 = _mm256_set1_ps(dn_dx_z * 8.0f);
    __m256 cr_step8 = _mm256_set1_ps(dc_dx_r * 8.0f);
    __m256 cg_step8 = _mm256_set1_ps(dc_dx_g * 8.0f);
    __m256 cb_step8 = _mm256_set1_ps(dc_dx_b * 8.0f);

    // 光照常量
    __m256 Lr = _mm256_set1_ps(light.L.r);
    __m256 Lg = _mm256_set1_ps(light.L.g);
    __m256 Lb = _mm256_set1_ps(light.L.b);
    __m256 L_omega_i_x = _mm256_set1_ps(light.omega_i.x);
    __m256 L_omega_i_y = _mm256_set1_ps(light.omega_i.y);
    __m256 L_omega_i_z = _mm256_set1_ps(light.omega_i.z);
    __m256 kd_step8 = _mm256_set1_ps(kd);
    __m256 ka_step8 = _mm256_set1_ps(ka);
    __m256 ambinet_r_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.r), ka_step8);
    __m256 ambinet_g_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.g), ka_step8);
    __m256 ambinet_b_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.b), ka_step8);

    // row init value
    const float start_x = (float)(tile.x + tile_min_x);
    float global_y0 = (float)(tile.y + tile_min_y);

    // w 在 (start_x, tile_min_y)
    float w0_row = e0.x * start_x + e0.y * global_y0 + e0.z;
    float w1_row = e1.x * start_x + e1.y * global_y0 + e1.z;
    float w2_row = e2.x * start_x + e2.y * global_y0 + e2.z;

    // 插值初值
    float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
    float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
    float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
    float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
    float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
    float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
    float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

    // 行循环
    for (int y = tile_min_y; y <= tile_max_y; ++y) {
        //float global_y = (float)(tile.y + y);

        //// 计算当前行的基础值
        //float w0_row = e0.x * (tile.x + tile_min_x) + e0.y * global_y + e0.z;
        //float w1_row = e1.x * (tile.x + tile_min_x) + e1.y * global_y + e1.z;
        //float w2_row = e2.x * (tile.x + tile_min_x) + e2.y * global_y + e2.z;

        //float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
        //float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
        //float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
        //float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
        //float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
        //float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
        //float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

        __m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
        __m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
        __m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);
        __m256 zv = _mm256_add_ps(_mm256_set1_ps(z_row), z_lane);
        __m256 nx = _mm256_add_ps(_mm256_set1_ps(nx_row), nx_lane);
        __m256 ny = _mm256_add_ps(_mm256_set1_ps(ny_row), ny_lane);
        __m256 nz = _mm256_add_ps(_mm256_set1_ps(nz_row), nz_lane);
        __m256 cr = _mm256_add_ps(_mm256_set1_ps(cr_row), cr_lane);
        __m256 cg = _mm256_add_ps(_mm256_set1_ps(cg_row), cg_lane);
        __m256 cb = _mm256_add_ps(_mm256_set1_ps(cb_row), cb_lane);

        int x = tile_min_x;
        // 像素循环（8像素一组）
        for (; x <= tile_max_x - 7; x += 8) {
            // 检查像素是否在三角形内
            //__m256 inside = _mm256_and_ps(
            //    _mm256_and_ps(
            //        _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
            //        _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)
            //    ),
            //    _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ)
            //);

            //int mask = _mm256_movemask_ps(inside);
            //if (mask == 0) {
            //    w0v = _mm256_add_ps(w0v, w0_step8);
            //    w1v = _mm256_add_ps(w1v, w1_step8);
            //    w2v = _mm256_add_ps(w2v, w2_step8);
            //    zv = _mm256_add_ps(zv, z_step8);
            //    nx = _mm256_add_ps(nx, nx_step8);
            //    ny = _mm256_add_ps(ny, ny_step8);
            //    nz = _mm256_add_ps(nz, nz_step8);
            //    cr = _mm256_add_ps(cr, cr_step8);
            //    cg = _mm256_add_ps(cg, cg_step8);
            //    cb = _mm256_add_ps(cb, cb_step8);
            //    continue;
            //}

            //// 加载瓦片深度缓冲区
            //// TODO
            ////alignas(32) float zbuf_tmp[8];
            ////for (int i = 0; i < 8; ++i) {
            ////    int pixel_x = x + i;
            ////    if (pixel_x < tile.width) {
            ////        zbuf_tmp[i] = tile.zbuffer[y * tile.width + pixel_x];
            ////    }
            ////    else {
            ////        //zbuf_tmp[i] = std::numeric_limits<float>::max();
            ////        zbuf_tmp[i] = 1.0f;
            ////    }
            ////}
            ////__m256 zbuf = _mm256_load_ps(zbuf_tmp);
            __m256 zbuf = _mm256_load_ps(&tile.zbuffer[y * tile.width + x]);

            //// 深度测试
            ////__m256 depth_ok = _mm256_and_ps(
            ////    _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
            ////    _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ)
            ////);
            //__m256 depth_ok = _mm256_and_ps(
            //    _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
            //    _mm256_cmp_ps(zbuf, zv, _CMP_GT_OQ)
            //);

            //__m256 final_mask = _mm256_and_ps(inside, depth_ok);
            __m256 final_mask =
                _mm256_and_ps(
                    _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
                    _mm256_and_ps(
                        _mm256_cmp_ps(zbuf, zv, _CMP_GT_OQ),
                        _mm256_and_ps(
                            _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
                            _mm256_and_ps(
                                _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ),
                                _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ)
                            )
                        )
                    )
                );

            int final_mask_bits = _mm256_movemask_ps(final_mask);

            if (final_mask_bits == 0) {
                w0v = _mm256_add_ps(w0v, w0_step8);
                w1v = _mm256_add_ps(w1v, w1_step8);
                w2v = _mm256_add_ps(w2v, w2_step8);
                zv = _mm256_add_ps(zv, z_step8);
                nx = _mm256_add_ps(nx, nx_step8);
                ny = _mm256_add_ps(ny, ny_step8);
                nz = _mm256_add_ps(nz, nz_step8);
                cr = _mm256_add_ps(cr, cr_step8);
                cg = _mm256_add_ps(cg, cg_step8);
                cb = _mm256_add_ps(cb, cb_step8);
                continue;
            }

            // 归一化法线
            __m256 len = _mm256_sqrt_ps(_mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                _mm256_mul_ps(nz, nz)
            ));

            __m256 safe_len = _mm256_blendv_ps(_mm256_set1_ps(1.0f), len, final_mask);
            nx = _mm256_div_ps(nx, safe_len);
            ny = _mm256_div_ps(ny, safe_len);
            nz = _mm256_div_ps(nz, safe_len);

            // 光照计算
            __m256 dot = _mm256_max_ps(_mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(nx, L_omega_i_x),
                    _mm256_mul_ps(ny, L_omega_i_y)
                ),
                _mm256_mul_ps(nz, L_omega_i_z)
            ), zero);

            // 颜色计算
            __m256 r = _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8), _mm256_mul_ps(Lr, dot)),
                ambinet_r_ka
            );

            __m256 g = _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8), _mm256_mul_ps(Lg, dot)),
                ambinet_g_ka
            );

            __m256 b = _mm256_add_ps(
                _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8), _mm256_mul_ps(Lb, dot)),
                ambinet_b_ka
            );

            // 裁剪颜色到[0,1]
            r = _mm256_min_ps(_mm256_max_ps(r, zero), _1);
            g = _mm256_min_ps(_mm256_max_ps(g, zero), _1);
            b = _mm256_min_ps(_mm256_max_ps(b, zero), _1);

            // 转换为8位
            r = _mm256_mul_ps(r, _255);
            g = _mm256_mul_ps(g, _255);
            b = _mm256_mul_ps(b, _255);

            // 存储结果
            alignas(32) float rr[8], gg[8], bb[8], zz[8];
            _mm256_store_ps(rr, r);
            _mm256_store_ps(gg, g);
            _mm256_store_ps(bb, b);
            _mm256_store_ps(zz, zv);

            // 使用掩码写入瓦片缓冲区
            int m = final_mask_bits;
            while (m) {
                int i = _tzcnt_u32(m);
                m &= m - 1;

                int pixel_x = x + i;
                if (pixel_x < tile.width) {
                    //int color_idx = (y * tile.width + pixel_x) * 3;
                    //tile.colors[color_idx] = (unsigned char)rr[i];
                    //tile.colors[color_idx + 1] = (unsigned char)gg[i];
                    //tile.colors[color_idx + 2] = (unsigned char)bb[i];
                    //tile.zbuffer[y * tile.width + pixel_x] = zz[i];
                    renderer.canvas.draw(
                        tile.x + x + i, tile.y + y,
                        (unsigned char)(rr[i]),
                        (unsigned char)(gg[i]),
                        (unsigned char)(bb[i]));
                }
            }
            //TODO
            __m256i storeMask = _mm256_castps_si256(final_mask);
            _mm256_maskstore_ps(&tile.zbuffer[y * tile.width + x], storeMask, zv);

            // 增量到下一个8像素组
            w0v = _mm256_add_ps(w0v, w0_step8);
            w1v = _mm256_add_ps(w1v, w1_step8);
            w2v = _mm256_add_ps(w2v, w2_step8);
            zv = _mm256_add_ps(zv, z_step8);
            nx = _mm256_add_ps(nx, nx_step8);
            ny = _mm256_add_ps(ny, ny_step8);
            nz = _mm256_add_ps(nz, nz_step8);
            cr = _mm256_add_ps(cr, cr_step8);
            cg = _mm256_add_ps(cg, cg_step8);
            cb = _mm256_add_ps(cb, cb_step8);
        }

        // 处理尾数像素（少于8个）
        //int tail_start = tile_min_x + ((tile_max_x - tile_min_x + 1) / 8) * 8;
        //for (int x = tail_start; x <= tile_max_x; ++x) {
        //    float w0 = e0.x * (tile.x + x) + e0.y * global_y + e0.z;
        //    float w1 = e1.x * (tile.x + x) + e1.y * global_y + e1.z;
        //    float w2 = e2.x * (tile.x + x) + e2.y * global_y + e2.z;

        //    if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;

        //    float alpha = w0 * invArea;
        //    float beta = w1 * invArea;
        //    float gamma = w2 * invArea;

        //    float z = v0_z * alpha + v1_z * beta + v2_z * gamma;
        //    if (z < 0.001f || z >= tile.zbuffer[y * tile.width + x]) continue;

        //    // 法线插值和归一化
        //    float nx_val = n0_x * alpha + n1_x * beta + n2_x * gamma;
        //    float ny_val = n0_y * alpha + n1_y * beta + n2_y * gamma;
        //    float nz_val = n0_z * alpha + n1_z * beta + n2_z * gamma;

        //    float inv_len = 1.0f / sqrtf(nx_val * nx_val + ny_val * ny_val + nz_val * nz_val + 1e-8f);
        //    nx_val *= inv_len;
        //    ny_val *= inv_len;
        //    nz_val *= inv_len;

        //    // 颜色插值
        //    float cr_val = c0_r * alpha + c1_r * beta + c2_r * gamma;
        //    float cg_val = c0_g * alpha + c1_g * beta + c2_g * gamma;
        //    float cb_val = c0_b * alpha + c1_b * beta + c2_b * gamma;

        //    // 光照计算
        //    float dot = std::max(nx_val * light.omega_i.x + ny_val * light.omega_i.y +
        //        nz_val * light.omega_i.z, 0.0f);

        //    float out_r = (cr_val * kd) * (light.L.r * dot) + (light.ambient.r * ka);
        //    float out_g = (cg_val * kd) * (light.L.g * dot) + (light.ambient.g * ka);
        //    float out_b = (cb_val * kd) * (light.L.b * dot) + (light.ambient.b * ka);

        //    // 裁剪和转换
        //    out_r = std::max(0.0f, std::min(1.0f, out_r));
        //    out_g = std::max(0.0f, std::min(1.0f, out_g));
        //    out_b = std::max(0.0f, std::min(1.0f, out_b));

        //    // 写入瓦片缓冲区
        //    int color_idx = (y * tile.width + x) * 3;
        //    tile.colors[color_idx] = (unsigned char)(out_r * 255.0f);
        //    tile.colors[color_idx + 1] = (unsigned char)(out_g * 255.0f);
        //    tile.colors[color_idx + 2] = (unsigned char)(out_b * 255.0f);
        //    tile.zbuffer[y * tile.width + x] = z;
        //}

        int tail_count = tile_max_x - x + 1;
        if (tail_count != 0) {
            __m256i lane_i = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 tailMask =
                _mm256_castsi256_ps(
                    _mm256_cmpgt_epi32(
                        _mm256_set1_epi32(tail_count),
                        lane_i
                    )
                );

            __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
            __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
            __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
            __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
            inside = _mm256_and_ps(inside, w2v_zero);

            inside = _mm256_and_ps(inside, tailMask);

            int mask = _mm256_movemask_ps(inside);
            if (mask != 0)
            {
                //__m256 zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));

                alignas(32) float zb[8];
                for (int i = 0; i < 8; ++i)
                {
                    int px = x + i;
                    zb[i] = (px <= tile_max_x)
                        ? tile.zbuffer(px, y)
                        : std::numeric_limits<float>::infinity();
                }
                __m256 zbuf = _mm256_load_ps(zb);

                __m256 zv_001 = _mm256_cmp_ps(zv, _001, _CMP_GE_OQ);
                __m256 zbuf_zv = _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ);
                __m256 depth_ok = _mm256_and_ps(zv_001, zbuf_zv);

                __m256 final_mask = _mm256_and_ps(inside, depth_ok);
                int final_mask_bits = _mm256_movemask_ps(_mm256_and_ps(inside, depth_ok));


                if (final_mask_bits != 0)
                {
                    // normalize normal
                    __m256 len = _mm256_sqrt_ps(
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                            _mm256_mul_ps(nz, nz)));

                    nx = _mm256_div_ps(nx, len);
                    ny = _mm256_div_ps(ny, len);
                    nz = _mm256_div_ps(nz, len);

                    __m256 dot =
                        _mm256_max_ps(
                            _mm256_add_ps(
                                _mm256_add_ps(_mm256_mul_ps(nx, L_omega_i_x),
                                    _mm256_mul_ps(ny, L_omega_i_y)),
                                _mm256_mul_ps(nz, L_omega_i_z)),
                            zero);

                    __m256 r =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8),
                                _mm256_mul_ps(Lr, dot)),
                            ambinet_r_ka);

                    __m256 g =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8),
                                _mm256_mul_ps(Lg, dot)),
                            ambinet_g_ka);

                    __m256 b =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8),
                                _mm256_mul_ps(Lb, dot)),
                            ambinet_b_ka);

                    r = _mm256_mul_ps(r, _255);
                    g = _mm256_mul_ps(g, _255);
                    b = _mm256_mul_ps(b, _255);


                    alignas(32) float rr[8], gg[8], bb[8], zz[8];
                    _mm256_store_ps(rr, r);
                    _mm256_store_ps(gg, g);
                    _mm256_store_ps(bb, b);
                    _mm256_store_ps(zz, zv);

                    //for (int i = 0; i < tail_count; i++) {
                    //    renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]),
                    //        (unsigned char)(gg[i]),
                    //        (unsigned char)(bb[i]));
                    //    renderer.zbuffer(x + i, y) = zz[i];
                    //}
                    int m = final_mask_bits;
                    while (m)
                    {
                        int i = _tzcnt_u32(m);
                        m &= m - 1;
                        //int px = x + i;
                        //if (px <= tile.width)
                        //{
                        //    renderer.canvas.draw(px, y,
                        //        (unsigned char)rr[i],
                        //        (unsigned char)gg[i],
                        //        (unsigned char)bb[i]);
                        //    renderer.zbuffer(px, y) = zz[i];
                        //}
                        int pixel_x = x + i;
                        if (pixel_x < tile.width) {
                            //int color_idx = (y * tile.width + pixel_x) * 3;
                            //tile.colors[color_idx] = (unsigned char)rr[i];
                            //tile.colors[color_idx + 1] = (unsigned char)gg[i];
                            //tile.colors[color_idx + 2] = (unsigned char)bb[i];
                            renderer.canvas.draw(
                                tile.x + x + i, tile.y + y,
                                (unsigned char)(rr[i]),
                                (unsigned char)(gg[i]),
                                (unsigned char)(bb[i]));
                            //tile.zbuffer[y * tile.width + pixel_x] = zz[i];
                        }
                    }
                    __m256i storeMask = _mm256_castps_si256(final_mask);
                    _mm256_maskstore_ps(&tile.zbuffer[y * tile.width + x], storeMask, zv);
                }
            }
        }

        w0_row += e0.y;
        w1_row += e1.y;
        w2_row += e2.y;

        z_row += dz_dy;
        nx_row += dn_dy_x;
        ny_row += dn_dy_y;
        nz_row += dn_dy_z;
        cr_row += dc_dy_r;
        cg_row += dc_dy_g;
        cb_row += dc_dy_b;
    }
}
void renderTriangleFromSOA_InTile2(Renderer& renderer, const Mesh_SoA_Transformed& mesh, Light& light, float ka, float kd,
    int triangle_idx, Tile& tile) {
    // 加载顶点数据（从变换后的SOA数据）
    const triIndices& idx = mesh.triangles[triangle_idx];

    // 加载变换后的顶点位置（屏幕空间）
    float v0_x = mesh.transformed_positions_x[idx.v[0]];
    float v1_x = mesh.transformed_positions_x[idx.v[1]];
    float v2_x = mesh.transformed_positions_x[idx.v[2]];

    float v0_y = mesh.transformed_positions_y[idx.v[0]];
    float v1_y = mesh.transformed_positions_y[idx.v[1]];
    float v2_y = mesh.transformed_positions_y[idx.v[2]];

    float v0_z = mesh.transformed_positions_z[idx.v[0]];
    float v1_z = mesh.transformed_positions_z[idx.v[1]];
    float v2_z = mesh.transformed_positions_z[idx.v[2]];

    // 检查有效性
    if (!isfinite(v0_x) || !isfinite(v0_y) || !isfinite(v0_z) ||
        !isfinite(v1_x) || !isfinite(v1_y) || !isfinite(v1_z) ||
        !isfinite(v2_x) || !isfinite(v2_y) || !isfinite(v2_z)) {
        return;
    }

    // 计算三角形面积（用于重心坐标插值）
    vec4 e0 = makeEdge0(v1_x, v1_y, v2_x, v2_y);
    vec4 e1 = makeEdge0(v2_x, v2_y, v0_x, v0_y);
    vec4 e2 = makeEdge0(v0_x, v0_y, v1_x, v1_y);

    float area = v0_x * e0.x + v0_y * e0.y + e0.z;
    if (area < 1e-6f) return;  // 面积太小，忽略

    float invArea = 1.0f / area;

    // 加载变换后的法线数据
    //const float* norm_x = mesh.transformed_normals_x.data();
    //const float* norm_y = mesh.transformed_normals_y.data();
    //const float* norm_z = mesh.transformed_normals_z.data();

    float n0_x = mesh.transformed_normals_x[idx.v[0]];
    float n0_y = mesh.transformed_normals_y[idx.v[0]];
    float n0_z = mesh.transformed_normals_z[idx.v[0]];

    float n1_x = mesh.transformed_normals_x[idx.v[1]];
    float n1_y = mesh.transformed_normals_y[idx.v[1]];
    float n1_z = mesh.transformed_normals_z[idx.v[1]];

    float n2_x = mesh.transformed_normals_x[idx.v[2]];
    float n2_y = mesh.transformed_normals_y[idx.v[2]];
    float n2_z = mesh.transformed_normals_z[idx.v[2]];

    // 加载颜色数据
    //const float* col_r = mesh.colors_r.data();
    //const float* col_g = mesh.colors_g.data();
    //const float* col_b = mesh.colors_b.data();

    float c0_r = mesh.colors_r[idx.v[0]];
    float c0_g = mesh.colors_g[idx.v[0]];
    float c0_b = mesh.colors_b[idx.v[0]];

    float c1_r = mesh.colors_r[idx.v[1]];
    float c1_g = mesh.colors_g[idx.v[1]];
    float c1_b = mesh.colors_b[idx.v[1]];

    float c2_r = mesh.colors_r[idx.v[2]];
    float c2_g = mesh.colors_g[idx.v[2]];
    float c2_b = mesh.colors_b[idx.v[2]];


    // 计算插值增量
    float dz_dx = (v0_z * e0.x + v1_z * e1.x + v2_z * e2.x) * invArea;
    float dz_dy = (v0_z * e0.y + v1_z * e1.y + v2_z * e2.y) * invArea;

    float dn_dx_x = (n0_x * e0.x + n1_x * e1.x + n2_x * e2.x) * invArea;
    float dn_dx_y = (n0_y * e0.x + n1_y * e1.x + n2_y * e2.x) * invArea;
    float dn_dx_z = (n0_z * e0.x + n1_z * e1.x + n2_z * e2.x) * invArea;

    float dn_dy_x = (n0_x * e0.y + n1_x * e1.y + n2_x * e2.y) * invArea;
    float dn_dy_y = (n0_y * e0.y + n1_y * e1.y + n2_y * e2.y) * invArea;
    float dn_dy_z = (n0_z * e0.y + n1_z * e1.y + n2_z * e2.y) * invArea;

    float dc_dx_r = (c0_r * e0.x + c1_r * e1.x + c2_r * e2.x) * invArea;
    float dc_dx_g = (c0_g * e0.x + c1_g * e1.x + c2_g * e2.x) * invArea;
    float dc_dx_b = (c0_b * e0.x + c1_b * e1.x + c2_b * e2.x) * invArea;

    float dc_dy_r = (c0_r * e0.y + c1_r * e1.y + c2_r * e2.y) * invArea;
    float dc_dy_g = (c0_g * e0.y + c1_g * e1.y + c2_g * e2.y) * invArea;
    float dc_dy_b = (c0_b * e0.y + c1_b * e1.y + c2_b * e2.y) * invArea;

    // 计算三角形在瓦片空间的包围盒
    float min_x = std::max((float)tile.x, std::min({ v0_x, v1_x, v2_x }));
    float max_x = std::min((float)(tile.x + tile.width), std::max({ v0_x, v1_x, v2_x }));
    float min_y = std::max((float)tile.y, std::min({ v0_y, v1_y, v2_y }));
    float max_y = std::min((float)(tile.y + tile.height), std::max({ v0_y, v1_y, v2_y }));

    if (min_x > max_x || min_y > max_y) return;

    // 转换为瓦片局部坐标
    int tile_min_x = (int)std::floor(min_x) - tile.x;
    int tile_max_x = (int)std::ceil(max_x) - tile.x;
    int tile_min_y = (int)std::floor(min_y) - tile.y;
    int tile_max_y = (int)std::ceil(max_y) - tile.y;

    // 裁剪到瓦片边界
    tile_min_x = std::max(0, tile_min_x);
    tile_max_x = std::min(tile.width - 1, tile_max_x);
    tile_min_y = std::max(0, tile_min_y);
    tile_max_y = std::min(tile.height - 1, tile_max_y);

    if (tile_min_x > tile_max_x || tile_min_y > tile_max_y) return;

    // --- AVX2常量 ---
    const __m256 zero = _mm256_setzero_ps();
    const __m256 lane = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    const __m256 _255 = _mm256_set1_ps(255.0f);
    const __m256 _001 = _mm256_set1_ps(0.001f);
    const __m256 _1 = _mm256_set1_ps(1.0f);

    // 预计算增量
    __m256 w0_lane = _mm256_mul_ps(_mm256_set1_ps(e0.x), lane);
    __m256 w1_lane = _mm256_mul_ps(_mm256_set1_ps(e1.x), lane);
    __m256 w2_lane = _mm256_mul_ps(_mm256_set1_ps(e2.x), lane);

    __m256 z_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lane);
    __m256 nx_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_x), lane);
    __m256 ny_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_y), lane);
    __m256 nz_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx_z), lane);
    __m256 cr_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_r), lane);
    __m256 cg_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_g), lane);
    __m256 cb_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx_b), lane);

    __m256 w0_step8 = _mm256_set1_ps(e0.x * 8.0f);
    __m256 w1_step8 = _mm256_set1_ps(e1.x * 8.0f);
    __m256 w2_step8 = _mm256_set1_ps(e2.x * 8.0f);
    __m256 z_step8 = _mm256_set1_ps(dz_dx * 8.0f);
    __m256 nx_step8 = _mm256_set1_ps(dn_dx_x * 8.0f);
    __m256 ny_step8 = _mm256_set1_ps(dn_dx_y * 8.0f);
    __m256 nz_step8 = _mm256_set1_ps(dn_dx_z * 8.0f);
    __m256 cr_step8 = _mm256_set1_ps(dc_dx_r * 8.0f);
    __m256 cg_step8 = _mm256_set1_ps(dc_dx_g * 8.0f);
    __m256 cb_step8 = _mm256_set1_ps(dc_dx_b * 8.0f);

    // 光照常量
    __m256 Lr = _mm256_set1_ps(light.L.r);
    __m256 Lg = _mm256_set1_ps(light.L.g);
    __m256 Lb = _mm256_set1_ps(light.L.b);
    __m256 L_omega_i_x = _mm256_set1_ps(light.omega_i.x);
    __m256 L_omega_i_y = _mm256_set1_ps(light.omega_i.y);
    __m256 L_omega_i_z = _mm256_set1_ps(light.omega_i.z);
    __m256 kd_step8 = _mm256_set1_ps(kd);
    __m256 ka_step8 = _mm256_set1_ps(ka);
    __m256 ambinet_r_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.r), ka_step8);
    __m256 ambinet_g_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.g), ka_step8);
    __m256 ambinet_b_ka = _mm256_mul_ps(_mm256_set1_ps(light.ambient.b), ka_step8);

    // row init value
    const float start_x = (float)(tile.x + tile_min_x);
    float global_y0 = (float)(tile.y + tile_min_y);

    // w 在 (start_x, tile_min_y)
    float w0_row = e0.x * start_x + e0.y * global_y0 + e0.z;
    float w1_row = e1.x * start_x + e1.y * global_y0 + e1.z;
    float w2_row = e2.x * start_x + e2.y * global_y0 + e2.z;

    // 插值初值
    float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
    float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
    float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
    float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
    float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
    float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
    float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

    // 行循环
    for (int y = min_y; y <= max_y; ++y)
    {
        __m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
        __m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
        __m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);
        __m256 zv = _mm256_add_ps(_mm256_set1_ps(z_row), z_lane);
        __m256 nx = _mm256_add_ps(_mm256_set1_ps(nx_row), nx_lane);
        __m256 ny = _mm256_add_ps(_mm256_set1_ps(ny_row), ny_lane);
        __m256 nz = _mm256_add_ps(_mm256_set1_ps(nz_row), nz_lane);
        __m256 cr = _mm256_add_ps(_mm256_set1_ps(cr_row), cr_lane);
        __m256 cg = _mm256_add_ps(_mm256_set1_ps(cg_row), cg_lane);
        __m256 cb = _mm256_add_ps(_mm256_set1_ps(cb_row), cb_lane);

        int x = min_x;

        for (; x <= max_x - 8; x += 8)
        {

            //__m256 inside =
            //    _mm256_and_ps(
            //        _mm256_and_ps(_mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
            //            _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)),
            //        _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ));

            __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
            __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
            __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
            __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
            inside = _mm256_and_ps(inside, w2v_zero);

            //__m256 inside0 = tl0 ? _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ) : _mm256_cmp_ps(w0v, zero, _CMP_GT_OQ);
            //__m256 inside1 = tl1 ? _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ) : _mm256_cmp_ps(w1v, zero, _CMP_GT_OQ);
            //__m256 inside2 = tl2 ? _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ) : _mm256_cmp_ps(w2v, zero, _CMP_GT_OQ);
            //__m256 inside = _mm256_and_ps(_mm256_and_ps(inside0, inside1), inside2);

            int mask = _mm256_movemask_ps(inside);
            if (mask == 0)
            {
                w0v = _mm256_add_ps(w0v, w0_step8);
                w1v = _mm256_add_ps(w1v, w1_step8);
                w2v = _mm256_add_ps(w2v, w2_step8);
                zv = _mm256_add_ps(zv, z_step8);
                nx = _mm256_add_ps(nx, nx_step8);
                ny = _mm256_add_ps(ny, ny_step8);
                nz = _mm256_add_ps(nz, nz_step8);
                cr = _mm256_add_ps(cr, cr_step8);
                cg = _mm256_add_ps(cg, cg_step8);
                cb = _mm256_add_ps(cb, cb_step8);
                continue;
            }

            __m256 zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));
            //__m256 depth_ok =
            //    _mm256_and_ps(
            //        _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
            //        _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ));

            __m256 zv_001 = _mm256_cmp_ps(zv, _001, _CMP_GE_OQ);
            __m256 zbuf_zv = _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ);
            __m256 depth_ok = _mm256_and_ps(zv_001, zbuf_zv);

            int final_mask = _mm256_movemask_ps(_mm256_and_ps(inside, depth_ok));


            if (final_mask == 0)
            {
                w0v = _mm256_add_ps(w0v, w0_step8);
                w1v = _mm256_add_ps(w1v, w1_step8);
                w2v = _mm256_add_ps(w2v, w2_step8);
                zv = _mm256_add_ps(zv, z_step8);
                nx = _mm256_add_ps(nx, nx_step8);
                ny = _mm256_add_ps(ny, ny_step8);
                nz = _mm256_add_ps(nz, nz_step8);
                cr = _mm256_add_ps(cr, cr_step8);
                cg = _mm256_add_ps(cg, cg_step8);
                cb = _mm256_add_ps(cb, cb_step8);
                continue;
            }
            // normalize normal
            __m256 len = _mm256_sqrt_ps(
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                    _mm256_mul_ps(nz, nz)));

            //__m256 xmul = _mm256_mul_ps(nx, nx);
            //__m256 ymul = _mm256_mul_ps(ny, ny);
            //__m256 zmul = _mm256_mul_ps(nz, nz);
            //__m256 len = _mm256_add_ps(xmul, ymul);
            //len = _mm256_add_ps(len, zmul);
            //len = _mm256_sqrt_ps(len);

            //__m256 valid = _mm256_castsi256_ps(_mm256_set1_epi32(final_mask));
            //len = _mm256_blendv_ps(_mm256_set1_ps(1.f), len, valid);

            nx = _mm256_div_ps(nx, len);
            ny = _mm256_div_ps(ny, len);
            nz = _mm256_div_ps(nz, len);

            __m256 dot =
                _mm256_max_ps(
                    _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(nx, L_omega_i_x),
                            _mm256_mul_ps(ny, L_omega_i_y)),
                        _mm256_mul_ps(nz, L_omega_i_z)),
                    zero);

            //__m256 ndotl_x = _mm256_mul_ps(nx, L_omega_i_x);
            //__m256 ndotl_y = _mm256_mul_ps(ny, L_omega_i_y);
            //__m256 ndotl_z = _mm256_mul_ps(nz, L_omega_i_z);
            //__m256 dot = _mm256_add_ps(ndotl_x, ndotl_y);
            //dot = _mm256_add_ps(dot, ndotl_z);
            //dot = _mm256_max_ps(dot, zero);

            //cr = _mm256_min_ps(cr, _1);
            //cg = _mm256_min_ps(cg, _1);
            //cb = _mm256_min_ps(cb, _1);

            __m256 r =
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8),
                        _mm256_mul_ps(Lr, dot)),
                    ambinet_r_ka);

            //__m256 r = _mm256_mul_ps(cr, kd_step8);
            //r = _mm256_mul_ps(r, Lr);
            //r = _mm256_mul_ps(r, dot);
            //r = _mm256_add_ps(r, ambinet_r_ka);
            //r = _mm256_mul_ps(r, _255);

            __m256 g =
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8),
                        _mm256_mul_ps(Lg, dot)),
                    ambinet_g_ka);

            //__m256 g = _mm256_mul_ps(cg, kd_step8);
            //g = _mm256_mul_ps(g, Lg);
            //g = _mm256_mul_ps(g, dot);
            //g = _mm256_add_ps(g, ambinet_g_ka);
            //g = _mm256_mul_ps(g, _255);

            __m256 b =
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8),
                        _mm256_mul_ps(Lb, dot)),
                    ambinet_b_ka);

            //__m256 b = _mm256_mul_ps(cb, kd_step8);
            //b = _mm256_mul_ps(b, Lb);
            //b = _mm256_mul_ps(b, dot);
            //b = _mm256_add_ps(b, ambinet_b_ka);
            //b = _mm256_mul_ps(b, _255);

            r = _mm256_mul_ps(r, _255);
            g = _mm256_mul_ps(g, _255);
            b = _mm256_mul_ps(b, _255);

            if (final_mask == 0xFF)
            {
                alignas(32) float rr[8], gg[8], bb[8], zz[8];
                _mm256_store_ps(rr, r);
                _mm256_store_ps(gg, g);
                _mm256_store_ps(bb, b);
                //_mm256_store_ps(zz, zv);
                _mm256_store_ps(&renderer.zbuffer(x, y), zv);

                //for (int i = 0; i < 8; i+= 4)
                //{
                //    //if (final_mask & (1 << i))
                //    //{
                //        //renderer.canvas.draw(
                //        //    x + i, y,
                //        //    (unsigned char)(rr[i] * 255.f),
                //        //    (unsigned char)(gg[i] * 255.f),
                //        //    (unsigned char)(bb[i] * 255.f)
                //        //);
                //        renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]), (unsigned char)(gg[i]), (unsigned char)(bb[i]));
                //        renderer.canvas.draw(x+ 1 + i, y, (unsigned char)(rr[i+1]), (unsigned char)(gg[i+1]), (unsigned char)(bb[i+1]));
                //        renderer.canvas.draw(x + 2+ i, y, (unsigned char)(rr[i + 2]), (unsigned char)(gg[i+2]), (unsigned char)(bb[i+2]));
                //        renderer.canvas.draw(x + 3 +i, y, (unsigned char)(rr[i+3]), (unsigned char)(gg[i+3]), (unsigned char)(bb[i+3]));
                //        //renderer.zbuffer(x + i, y) = zz[i];
                //    //}
                //}
                //for (int i = 0; i < 8; ++i)
                //{
                //    //if (final_mask & (1 << i))
                //    //{
                //        //renderer.canvas.draw(
                //        //    x + i, y,
                //        //    (unsigned char)(rr[i] * 255.f),
                //        //    (unsigned char)(gg[i] * 255.f),
                //        //    (unsigned char)(bb[i] * 255.f)
                //        //);
                //        renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]), (unsigned char)(gg[i]), (unsigned char)(bb[i]));
                //        //renderer.zbuffer(x + i, y) = zz[i];
                //    //}
                //}
                //renderer.canvas.draw(x    , y, (unsigned char)(rr[0]), (unsigned char)(gg[0]), (unsigned char)(bb[0]));
                //renderer.canvas.draw(x + 1, y, (unsigned char)(rr[1]), (unsigned char)(gg[1]), (unsigned char)(bb[1]));
                //renderer.canvas.draw(x + 2, y, (unsigned char)(rr[2]), (unsigned char)(gg[2]), (unsigned char)(bb[2]));
                //renderer.canvas.draw(x + 3, y, (unsigned char)(rr[3]), (unsigned char)(gg[3]), (unsigned char)(bb[3]));
                //renderer.canvas.draw(x + 4, y, (unsigned char)(rr[4]), (unsigned char)(gg[4]), (unsigned char)(bb[4]));
                //renderer.canvas.draw(x + 5, y, (unsigned char)(rr[5]), (unsigned char)(gg[5]), (unsigned char)(bb[5]));
                //renderer.canvas.draw(x + 6, y, (unsigned char)(rr[6]), (unsigned char)(gg[6]), (unsigned char)(bb[6]));
                //renderer.canvas.draw(x + 7, y, (unsigned char)(rr[7]), (unsigned char)(gg[7]), (unsigned char)(bb[7]));
                renderer.canvas.draw(x, y, rr, gg, bb);
            }
            else
            {
                alignas(32) float rr[8], gg[8], bb[8], zz[8];
                _mm256_store_ps(rr, r);
                _mm256_store_ps(gg, g);
                _mm256_store_ps(bb, b);
                _mm256_store_ps(zz, zv);

                int m = final_mask;
                while (m)
                {
                    // count trailing zeros
                    int i = _tzcnt_u32(m);
                    m &= m - 1;
                    //renderer.canvas.draw(x + i, y,
                    //    (unsigned char)(rr[i] * 255),
                    //    (unsigned char)(gg[i] * 255),
                    //    (unsigned char)(bb[i] * 255));
                    renderer.canvas.draw(
                        x + i, y,
                        (unsigned char)(rr[i]),
                        (unsigned char)(gg[i]),
                        (unsigned char)(bb[i])
                    );
                    renderer.zbuffer(x + i, y) = zz[i];
                }

            }

            w0v = _mm256_add_ps(w0v, w0_step8);
            w1v = _mm256_add_ps(w1v, w1_step8);
            w2v = _mm256_add_ps(w2v, w2_step8);
            zv = _mm256_add_ps(zv, z_step8);
            nx = _mm256_add_ps(nx, nx_step8);
            ny = _mm256_add_ps(ny, ny_step8);
            nz = _mm256_add_ps(nz, nz_step8);
            cr = _mm256_add_ps(cr, cr_step8);
            cg = _mm256_add_ps(cg, cg_step8);
            cb = _mm256_add_ps(cb, cb_step8);
        }
        /*
        for (; x <= maxX; ++x)
        {
            // === 1. Edge test ===
            //float w0 = e0.A * (x + 0.5f) + e0.B * (y + 0.5f) + e0.C;
            //float w1 = e1.A * (x + 0.5f) + e1.B * (y + 0.5f) + e1.C;
            //float w2 = e2.A * (x + 0.5f) + e2.B * (y + 0.5f) + e2.C;
            float w0 = e0.x * x + e0.y * y + e0.z;
            float w1 = e1.x * x + e1.y * y + e1.z;
            float w2 = e2.x * x + e2.y * y + e2.z;

            if (w0 < 0.f || w1 < 0.f || w2 < 0.f)
                continue;

            // === 2. Barycentric interpolation ===
            float alpha = w0 * invArea;
            float beta = w1 * invArea;
            float gamma = w2 * invArea;

            float z = interpolate(alpha, beta, gamma, v[0].p[2], v[1].p[2], v[2].p[2]);

            if (z < 0.001f || z >= renderer.zbuffer(x, y))
                continue;

            // === 3. Normal ===
            vec4 n = interpolate(alpha, beta, gamma, v[0].normal, v[1].normal, v[2].normal);

            n.normalise();

            colour c = interpolate(alpha, beta, gamma, v[0].rgb, v[1].rgb, v[2].rgb);
            c.clampColour();

            // === 5. Lighting ===
            float dot = std::max(vec4::dot(n, L.omega_i), 0.f);

            colour out =(c * kd) * (L.L * dot) +(L.ambient * ka);

            out.clampColour();

            unsigned char r, g, b;
            out.toRGB(r, g, b);

            // === 6. Write ===
            renderer.canvas.draw(x, y, r ,g, b);
            renderer.zbuffer(x, y) = z;
        }
        */

        //尾部处理部分：
        //int tail_start = maxX - maxX % 8 - 1;
        int tail_count = max_x - x + 1;
        if (tail_count != 0) {
            __m256i lane_i = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 tailMask =
                _mm256_castsi256_ps(
                    _mm256_cmpgt_epi32(
                        _mm256_set1_epi32(tail_count),
                        lane_i
                    )
                );

            __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
            __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
            __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
            __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
            inside = _mm256_and_ps(inside, w2v_zero);

            inside = _mm256_and_ps(inside, tailMask);

            int mask = _mm256_movemask_ps(inside);
            if (mask != 0)
            {
                //__m256 zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));

                alignas(32) float zb[8];
                for (int i = 0; i < 8; ++i)
                {
                    int px = x + i;
                    zb[i] = (px <= max_x)
                        ? renderer.zbuffer(px, y)
                        : std::numeric_limits<float>::infinity();
                }
                __m256 zbuf = _mm256_load_ps(zb);

                __m256 zv_001 = _mm256_cmp_ps(zv, _001, _CMP_GE_OQ);
                __m256 zbuf_zv = _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ);
                __m256 depth_ok = _mm256_and_ps(zv_001, zbuf_zv);

                int final_mask = _mm256_movemask_ps(_mm256_and_ps(inside, depth_ok));


                if (final_mask != 0)
                {
                    // normalize normal
                    __m256 len = _mm256_sqrt_ps(
                        _mm256_add_ps(
                            _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                            _mm256_mul_ps(nz, nz)));

                    nx = _mm256_div_ps(nx, len);
                    ny = _mm256_div_ps(ny, len);
                    nz = _mm256_div_ps(nz, len);

                    __m256 dot =
                        _mm256_max_ps(
                            _mm256_add_ps(
                                _mm256_add_ps(_mm256_mul_ps(nx, L_omega_i_x),
                                    _mm256_mul_ps(ny, L_omega_i_y)),
                                _mm256_mul_ps(nz, L_omega_i_z)),
                            zero);

                    __m256 r =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8),
                                _mm256_mul_ps(Lr, dot)),
                            ambinet_r_ka);

                    __m256 g =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8),
                                _mm256_mul_ps(Lg, dot)),
                            ambinet_g_ka);

                    __m256 b =
                        _mm256_add_ps(
                            _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8),
                                _mm256_mul_ps(Lb, dot)),
                            ambinet_b_ka);

                    r = _mm256_mul_ps(r, _255);
                    g = _mm256_mul_ps(g, _255);
                    b = _mm256_mul_ps(b, _255);


                    alignas(32) float rr[8], gg[8], bb[8], zz[8];
                    _mm256_store_ps(rr, r);
                    _mm256_store_ps(gg, g);
                    _mm256_store_ps(bb, b);
                    _mm256_store_ps(zz, zv);

                    //for (int i = 0; i < tail_count; i++) {
                    //    renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]),
                    //        (unsigned char)(gg[i]),
                    //        (unsigned char)(bb[i]));
                    //    renderer.zbuffer(x + i, y) = zz[i];
                    //}
                    int m = final_mask;
                    while (m)
                    {
                        int i = _tzcnt_u32(m);
                        m &= m - 1;
                        int px = x + i;
                        if (px <= max_x)
                        {
                            renderer.canvas.draw(px, y,
                                (unsigned char)rr[i],
                                (unsigned char)gg[i],
                                (unsigned char)bb[i]);
                            renderer.zbuffer(px, y) = zz[i];
                        }
                    }
                }
            }
        }

        w0_row += e0.y;
        w1_row += e1.y;
        w2_row += e2.y;

        z_row += dz_dy;
        nx_row += dn_dy_x;
        ny_row += dn_dy_y;
        nz_row += dn_dy_z;
        cr_row += dc_dy_r;
        cg_row += dc_dy_g;
        cb_row += dc_dy_b;
    }
}


//// 合并瓦片到主画布
//void mergeTilesToCanvas(Renderer& renderer) {
//    // 并行合并瓦片
//    for (int tile_idx = 0; tile_idx < tiles.size(); ++tile_idx) {
//        pool.enqueue([this,&renderer, tile_idx]() {
//            Tile& tile = tiles[tile_idx];

//            for (int y = 0; y < tile.height; ++y) {
//                for (int x = 0; x < tile.width; ++x) {
//                    int global_x = tile.x + x;
//                    int global_y = tile.y + y;
//                    int tile_pixel_idx = (y * tile.width + x) * 3;

//                    // 直接复制，因为瓦片内已经做了深度测试
//                    renderer.canvas.draw(global_x, global_y,
//                        tile.colors[tile_pixel_idx],
//                        tile.colors[tile_pixel_idx + 1],
//                        tile.colors[tile_pixel_idx + 2]);
//                    renderer.zbuffer(global_x, global_y) = tile.zbuffer[y * tile.width + x];
//                }
//            }
//            });
//    }

//    pool.waitFrame();
//}
//void mergeTilesToCanvas_ST(Renderer& renderer) {
//    for (int tile_idx = 0; tile_idx < tiles.size(); ++tile_idx) {
//        Tile& tile = tiles[tile_idx];

//        for (int y = 0; y < tile.height; ++y) {
//            for (int x = 0; x < tile.width; ++x) {
//                int global_x = tile.x + x;
//                int global_y = tile.y + y;
//                int tile_pixel_idx = (y * tile.width + x) * 3;

//                // 直接复制，因为瓦片内已经做了深度测试
//                renderer.canvas.draw(global_x, global_y,
//                    tile.colors[tile_pixel_idx],
//                    tile.colors[tile_pixel_idx + 1],
//                    tile.colors[tile_pixel_idx + 2]);
//                renderer.zbuffer(global_x, global_y) = tile.zbuffer[y * tile.width + x];
//            }
//        }

//    }

//}


void transformBatchSIMD3(const matrix& p, const matrix& world,
    const Mesh_SoA& mesh, int start_idx, int count,
    float canvas_width, float canvas_height) {
    const int vCount = (int)mesh.positions_x.size();
    const int triCount = (int)mesh.triangles.size();

    const float width = float(canvas_width);
    const float height = float(canvas_height);
    const __m256 halfWidth = _mm256_set1_ps(0.5f * width);
    const __m256 halfHeight = _mm256_set1_ps(0.5f * height);
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 canvasHeight = _mm256_set1_ps(height);

    const float* px = mesh.positions_x.data() + start_idx;
    const float* py = mesh.positions_y.data() + start_idx;
    const float* pz = mesh.positions_z.data() + start_idx;
    const float* pw = mesh.positions_w.data() + start_idx;

    const float* nx = mesh.normals_x.data() + start_idx;
    const float* ny = mesh.normals_y.data() + start_idx;
    const float* nz = mesh.normals_z.data() + start_idx;

    const float* cr = mesh.colors_r.data() + start_idx;
    const float* cg = mesh.colors_g.data() + start_idx;
    const float* cb = mesh.colors_b.data() + start_idx;

    // 获取目标数据指针
    float* tpx = transformed_positions_x.data() + start_idx;
    float* tpy = transformed_positions_y.data() + start_idx;
    float* tpz = transformed_positions_z.data() + start_idx;
    float* tpw = transformed_positions_w.data() + start_idx;

    float* tnx = transformed_normals_x.data() + start_idx;
    float* tny = transformed_normals_y.data() + start_idx;
    float* tnz = transformed_normals_z.data() + start_idx;

    float* out_cr = colors_r.data() + start_idx;
    float* out_cg = colors_g.data() + start_idx;
    float* out_cb = colors_b.data() + start_idx;

    int i = 0;
    for (; i + 7 < vCount; i += 8) {
        // -------------------------------
        // 1️⃣ 顶点矩阵变换 (batchMul 内联)
        // -------------------------------
        __m256 vx = _mm256_load_ps(&mesh.positions_x[i]);
        __m256 vy = _mm256_load_ps(&mesh.positions_y[i]);
        __m256 vz = _mm256_load_ps(&mesh.positions_z[i]);
        __m256 vw = _mm256_load_ps(&mesh.positions_w[i]);

        __m256 outX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[0]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[1]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[2]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[3]), vw))
        );
        __m256 outY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[4]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[5]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[6]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[7]), vw))
        );
        __m256 outZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[8]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[9]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[10]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[11]), vw))
        );
        __m256 outW = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[12]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[13]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[14]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[15]), vw))
        );

        // -------------------------------
        // 2️⃣ 透视除法 + 视口变换
        // -------------------------------
        __m256 invW = _mm256_rcp_ps(outW); // fast approx
        // Newton-Raphson refine for better precision
        invW = _mm256_mul_ps(invW, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(outW, invW)));

        outX = _mm256_mul_ps(outX, invW);
        outY = _mm256_mul_ps(outY, invW);
        outZ = _mm256_mul_ps(outZ, invW);

        // viewport transform
        outX = _mm256_mul_ps(_mm256_add_ps(outX, one), halfWidth);
        outY = _mm256_mul_ps(_mm256_add_ps(outY, one), halfHeight);
        outY = _mm256_sub_ps(canvasHeight, outY); // flip y

        // -------------------------------
        // 3️⃣ 法线矩阵变换 + normalize
        // -------------------------------
        __m256 nx = _mm256_load_ps(&mesh.normals_x[i]);
        __m256 ny = _mm256_load_ps(&mesh.normals_y[i]);
        __m256 nz = _mm256_load_ps(&mesh.normals_z[i]);

        __m256 outNX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[0]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[1]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[2]), nz)
        );
        __m256 outNY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[4]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[5]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[6]), nz)
        );
        __m256 outNZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[8]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[9]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh.world.data.a[10]), nz)
        );

        // normalize
        __m256 lenSq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(outNX, outNX),
            _mm256_mul_ps(outNY, outNY)),
            _mm256_mul_ps(outNZ, outNZ));
        __m256 invLen = _mm256_rsqrt_ps(lenSq);
        invLen = _mm256_mul_ps(invLen, _mm256_sub_ps(_mm256_set1_ps(1.5f),
            _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_mul_ps(lenSq, _mm256_mul_ps(invLen, invLen)))));
        outNX = _mm256_mul_ps(outNX, invLen);
        outNY = _mm256_mul_ps(outNY, invLen);
        outNZ = _mm256_mul_ps(outNZ, invLen);

        _mm256_storeu_ps(transformed_positions_x.data() + start_idx + i, outX);
        _mm256_storeu_ps(transformed_positions_y.data() + start_idx + i, outY);
        _mm256_storeu_ps(transformed_positions_z.data() + start_idx + i, outZ);
        _mm256_storeu_ps(transformed_normals_x.data() + start_idx + i, outNX);
        _mm256_storeu_ps(transformed_normals_y.data() + start_idx + i, outNY);
        _mm256_storeu_ps(transformed_normals_z.data() + start_idx + i, outNZ);

        __m256 color_r = _mm256_loadu_ps(cr + i);
        __m256 color_g = _mm256_loadu_ps(cg + i);
        __m256 color_b = _mm256_loadu_ps(cb + i);

        _mm256_storeu_ps(out_cr + i, color_r);
        _mm256_storeu_ps(out_cg + i, color_g);
        _mm256_storeu_ps(out_cb + i, color_b);

        // -------------------------------
        // 4️⃣ 存入 Vertex cache
        // -------------------------------
        //for (int j = 0; j < 8; ++j) {
        //    Vertex& out = vsCache[i + j];
        //    out.p = vec4(((float*)&outX)[j], ((float*)&outY)[j], ((float*)&outZ)[j], 1.f);
        //    out.normal = vec4(((float*)&outNX)[j], ((float*)&outNY)[j], ((float*)&outNZ)[j], 0.f);
        //    out.rgb.set(mesh->colors_r[i + j], mesh->colors_g[i + j], mesh->colors_b[i + j]);
        //}
    }

    // 处理剩余顶点（非SIMD路径）
    for (; i < vCount; ++i) {
        int idx = start_idx + i;
        int src_idx = idx;

        // 顶点位置变换
        vec4 pos(mesh.positions_x[src_idx],
            mesh.positions_y[src_idx],
            mesh.positions_z[src_idx],
            mesh.positions_w[src_idx]);

        vec4 clip_pos = p * pos;
        clip_pos.divideW();

        // 视口变换
        float screen_x = (clip_pos[0] + 1.0f) * 0.5f * canvas_width;
        float screen_y = (clip_pos[1] + 1.0f) * 0.5f * canvas_height;
        screen_y = canvas_height - screen_y;  // Y轴翻转

        transformed_positions_x[idx] = screen_x;
        transformed_positions_y[idx] = screen_y;
        transformed_positions_z[idx] = clip_pos[2];
        transformed_positions_w[idx] = clip_pos[3];

        // 法线变换
        vec4 normal(mesh.normals_x[src_idx],
            mesh.normals_y[src_idx],
            mesh.normals_z[src_idx],
            0.0f);

        normal = world * normal;
        normal.normalise();

        transformed_normals_x[idx] = normal[0];
        transformed_normals_y[idx] = normal[1];
        transformed_normals_z[idx] = normal[2];

        // 颜色复制
        colors_r[idx] = mesh.colors_r[src_idx];
        colors_g[idx] = mesh.colors_g[src_idx];
        colors_b[idx] = mesh.colors_b[src_idx];
    }
}

void render_transfirst_SoA_AVX2_Optimized_MT(Renderer& renderer, Mesh_SoA* mesh, matrix& camera, Light& L, ThreadPool2& pool)
{
    // =====================================================
    // 1️⃣ Vertex transform（你现有代码，完全不动）
    // =====================================================
    matrix p = renderer.perspective * camera * mesh->world;
    L.omega_i.normalise();

    const int vCount = (int)mesh->positions_x.size();
    const int triCount = (int)mesh->triangles.size();

    std::vector<Vertex> vsCache(vCount);

    const float width = float(renderer.canvas.getWidth());
    const float height = float(renderer.canvas.getHeight());
    const __m256 halfWidth = _mm256_set1_ps(0.5f * width);
    const __m256 halfHeight = _mm256_set1_ps(0.5f * height);
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 canvasHeight = _mm256_set1_ps(height);

    int i = 0;
    for (; i + 7 < vCount; i += 8) {
        // -------------------------------
        // 1️⃣ 顶点矩阵变换 (batchMul 内联)
        // -------------------------------
        __m256 vx = _mm256_load_ps(&mesh->positions_x[i]);
        __m256 vy = _mm256_load_ps(&mesh->positions_y[i]);
        __m256 vz = _mm256_load_ps(&mesh->positions_z[i]);
        __m256 vw = _mm256_load_ps(&mesh->positions_w[i]);

        __m256 outX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[0]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[1]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[2]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[3]), vw))
        );
        __m256 outY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[4]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[5]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[6]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[7]), vw))
        );
        __m256 outZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[8]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[9]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[10]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[11]), vw))
        );
        __m256 outW = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[12]), vx),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[13]), vy)),
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(p.data.a[14]), vz),
                _mm256_mul_ps(_mm256_set1_ps(p.data.a[15]), vw))
        );

        // -------------------------------
        // 2️⃣ 透视除法 + 视口变换
        // -------------------------------
        __m256 invW = _mm256_rcp_ps(outW); // fast approx
        // Newton-Raphson refine for better precision
        invW = _mm256_mul_ps(invW, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(outW, invW)));

        outX = _mm256_mul_ps(outX, invW);
        outY = _mm256_mul_ps(outY, invW);
        outZ = _mm256_mul_ps(outZ, invW);

        // viewport transform
        outX = _mm256_mul_ps(_mm256_add_ps(outX, one), halfWidth);
        outY = _mm256_mul_ps(_mm256_add_ps(outY, one), halfHeight);
        outY = _mm256_sub_ps(canvasHeight, outY); // flip y

        // -------------------------------
        // 3️⃣ 法线矩阵变换 + normalize
        // -------------------------------
        __m256 nx = _mm256_load_ps(&mesh->normals_x[i]);
        __m256 ny = _mm256_load_ps(&mesh->normals_y[i]);
        __m256 nz = _mm256_load_ps(&mesh->normals_z[i]);

        __m256 outNX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[0]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[1]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[2]), nz)
        );
        __m256 outNY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[4]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[5]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[6]), nz)
        );
        __m256 outNZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[8]), nx),
                _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[9]), ny)),
            _mm256_mul_ps(_mm256_set1_ps(mesh->world.data.a[10]), nz)
        );

        // normalize
        __m256 lenSq = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(outNX, outNX),
            _mm256_mul_ps(outNY, outNY)),
            _mm256_mul_ps(outNZ, outNZ));
        __m256 invLen = _mm256_rsqrt_ps(lenSq);
        invLen = _mm256_mul_ps(invLen, _mm256_sub_ps(_mm256_set1_ps(1.5f),
            _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_mul_ps(lenSq, _mm256_mul_ps(invLen, invLen)))));
        outNX = _mm256_mul_ps(outNX, invLen);
        outNY = _mm256_mul_ps(outNY, invLen);
        outNZ = _mm256_mul_ps(outNZ, invLen);

        // -------------------------------
        // 4️⃣ 存入 Vertex cache
        // -------------------------------
        for (int j = 0; j < 8; ++j) {
            Vertex& out = vsCache[i + j];
            out.p = vec4(((float*)&outX)[j], ((float*)&outY)[j], ((float*)&outZ)[j], 1.f);
            out.normal = vec4(((float*)&outNX)[j], ((float*)&outNY)[j], ((float*)&outNZ)[j], 0.f);
            out.rgb.set(mesh->colors_r[i + j], mesh->colors_g[i + j], mesh->colors_b[i + j]);
        }
    }

    // 处理剩余顶点
    for (; i < vCount; ++i) {
        Vertex& out = vsCache[i];
        vec4 pos(mesh->positions_x[i], mesh->positions_y[i], mesh->positions_z[i], mesh->positions_w[i]);
        out.p = p * pos;
        out.p.divideW();
        out.p[0] = (out.p[0] + 1.f) * 0.5f * width;
        out.p[1] = (out.p[1] + 1.f) * 0.5f * height;
        out.p[1] = height - out.p[1];

        vec4 normal(mesh->normals_x[i], mesh->normals_y[i], mesh->normals_z[i], 0.f);
        out.normal = mesh->world * normal;
        out.normal.normalise();

        out.rgb.set(mesh->colors_r[i], mesh->colors_g[i], mesh->colors_b[i]);
    }

    // =====================================================
    // 2️⃣ Tile binning
    // =====================================================
    constexpr int TILE_H = 128;
    int H = renderer.canvas.getHeight();
    int tileCount = (H + TILE_H - 1) / TILE_H;

    std::vector<TileBin> bins(tileCount);

    for (int t = 0; t < triCount; ++t)
    {
        triIndices& ind = mesh->triangles[t];

        const Vertex& v0 = vsCache[ind.v[0]];
        const Vertex& v1 = vsCache[ind.v[1]];
        const Vertex& v2 = vsCache[ind.v[2]];

        float minY = std::min({ v0.p.y, v1.p.y, v2.p.y });
        float maxY = std::max({ v0.p.y, v1.p.y, v2.p.y });

        int tileMin = std::max(0, (int)minY / TILE_H);
        int tileMax = std::min(tileCount - 1, (int)maxY / TILE_H);

        for (int tile = tileMin; tile <= tileMax; ++tile)
            bins[tile].triIndices.push_back(t);
    }

    // =====================================================
    // 3️⃣ Tile-based MT draw
    // =====================================================
    for (int tile = 0; tile < tileCount; ++tile)
    {
        int yMin = tile * TILE_H;
        int yMax = std::min(yMin + TILE_H - 1, H - 1);

        pool.enqueue([&, tile, yMin, yMax]() {
            for (int triIdx : bins[tile].triIndices)
            {
                triIndices& ind = mesh->triangles[triIdx];
                if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.f ||
                    fabs(vsCache[ind.v[1]].p[2]) > 1.f ||
                    fabs(vsCache[ind.v[2]].p[2]) > 1.f)) {
                    triangle tri(
                        vsCache[ind.v[0]],
                        vsCache[ind.v[1]],
                        vsCache[ind.v[2]]
                    );

                    tri.draw_AVX2_Optimized3_MT(
                        renderer,
                        L,
                        mesh->ka,
                        mesh->kd,
                        yMin,
                        yMax
                    );
                }
            }
            });
    }

    pool.waitAll();
}


void scene1_SoA_t() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    ThreadPool2 pool;
    bool running = true;

    std::vector<Mesh_SoA*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh_SoA* m = new Mesh_SoA();
        *m = Mesh_SoA::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh_SoA();
        *m = Mesh_SoA::makeCube(1.f);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }
        // Opt:
        L.omega_i.normalise();

        for (auto& m : scene)
            render_transfirst_SoA_AVX2_Optimized_MT(renderer, m, camera, L, pool);
        //render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L);
        //render_transfirst_SoA_Scalar(renderer, m, camera, L);

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

void draw_AVX2(Renderer& renderer, Light& L, float ka, float kd)
{
    vec2D minV, maxV;
    getBoundsWindow(renderer.canvas, minV, maxV);
    if (area < 1.f) return;

    int minX = (int)minV.x;
    int minY = (int)minV.y;
    int maxX = (int)ceil(maxV.x);
    int maxY = (int)ceil(maxV.y);

    //Edge e1 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
    //Edge e2 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
    //Edge e0 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));
    Edge e0 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
    Edge e1 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
    Edge e2 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));


    float invArea = 1.0f / area;

    float px = float(minX);
    float py = float(minY);

    float w0_row = e0.A * px + e0.B * py + e0.C;
    float w1_row = e1.A * px + e1.B * py + e1.C;
    float w2_row = e2.A * px + e2.B * py + e2.C;

    float dz_dx =
        (v[0].p[2] * e0.A +
            v[1].p[2] * e1.A +
            v[2].p[2] * e2.A) * invArea;

    float dz_dy =
        (v[0].p[2] * e0.B +
            v[1].p[2] * e1.B +
            v[2].p[2] * e2.B) * invArea;

    vec4 dn_dx =
        (v[0].normal * e0.A +
            v[1].normal * e1.A +
            v[2].normal * e2.A) * invArea;

    vec4 dn_dy =
        (v[0].normal * e0.B +
            v[1].normal * e1.B +
            v[2].normal * e2.B) * invArea;

    colour dc_dx =
        (v[0].rgb * e0.A +
            v[1].rgb * e1.A +
            v[2].rgb * e2.A) * invArea;

    colour dc_dy =
        (v[0].rgb * e0.B +
            v[1].rgb * e1.B +
            v[2].rgb * e2.B) * invArea;

    float z_row =
        (v[0].p[2] * w0_row +
            v[1].p[2] * w1_row +
            v[2].p[2] * w2_row) * invArea;

    vec4 n_row =
        (v[0].normal * w0_row +
            v[1].normal * w1_row +
            v[2].normal * w2_row) * invArea;

    colour c_row =
        (v[0].rgb * w0_row +
            v[1].rgb * w1_row +
            v[2].rgb * w2_row) * invArea;

    // AVX constants
    const __m256 zero = _mm256_setzero_ps();
    const __m256 idx = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
    //const __m256 idx = _mm256_setr_ps(7, 6, 5, 4, 3, 2, 1, 0);

    __m256 xOffset = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
    __m256 w0 = _mm256_add_ps(_mm256_set1_ps(w0_row),
        _mm256_mul_ps(xOffset, _mm256_set1_ps(e0.A)));
    __m256 w1 = _mm256_add_ps(_mm256_set1_ps(w1_row),
        _mm256_mul_ps(xOffset, _mm256_set1_ps(e1.A)));
    __m256 w2 = _mm256_add_ps(_mm256_set1_ps(w2_row),
        _mm256_mul_ps(xOffset, _mm256_set1_ps(e2.A)));



    const __m256 A0v = _mm256_set1_ps(e0.A);
    const __m256 A1v = _mm256_set1_ps(e1.A);
    const __m256 A2v = _mm256_set1_ps(e2.A);
    const __m256 dzdxv = _mm256_set1_ps(dz_dx);

    for (int y = minY; y < maxY; ++y)
    {
        float w0 = w0_row;
        float w1 = w1_row;
        float w2 = w2_row;

        float z = z_row;
        vec4  n = n_row;
        colour c = c_row;

        int x = minX;

        for (; x <= maxX - 8; x += 8)
        {
            __m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0),
                _mm256_mul_ps(A0v, idx));
            __m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1),
                _mm256_mul_ps(A1v, idx));
            __m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2),
                _mm256_mul_ps(A2v, idx));

            __m256 mask =
                _mm256_and_ps(
                    _mm256_and_ps(
                        _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
                        _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)),
                    _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ));

            int bits = _mm256_movemask_ps(mask);
            if (bits)
            {
                __m256 zv = _mm256_add_ps(_mm256_set1_ps(z),
                    _mm256_mul_ps(dzdxv, idx));

                alignas(32) float zbuf[8];
                _mm256_store_ps(zbuf, zv);

                for (int i = 0; i < 8; ++i)
                {
                    if (bits & (1 << i))
                    {
                        int xi = x + i;
                        float zi = zbuf[i];

                        if (zi > 0.001f && renderer.zbuffer(xi, y) > zi)
                        {
                            vec4  ni = n + dn_dx * (float)i;
                            colour ci = c + dc_dx * (float)i;

                            //ni.normalise();
                            float dot = std::max(vec4::dot(L.omega_i, ni), 0.0f);

                            colour shaded = (ci * kd) * (L.L * dot) + (L.ambient * ka);

                            shaded.clampColour();

                            unsigned char r, g, b;
                            shaded.toRGB(r, g, b);

                            renderer.canvas.draw(xi, y, r, g, b);
                            renderer.zbuffer(xi, y) = zi;
                        }
                    }
                }
            }

            w0 += e0.A * 8.f;
            w1 += e1.A * 8.f;
            w2 += e2.A * 8.f;
            z += dz_dx * 8.f;
            n += dn_dx * 8.f;
            c += dc_dx * 8.f;
        }

        // scalar tail
        for (; x < maxX; ++x)
        {
            if (w0 >= 0 && w1 >= 0 && w2 >= 0)
            {
                if (z > 0.001f && renderer.zbuffer(x, y) > z)
                {
                    vec4 nn = n;
                    nn.normalise();

                    float dot = std::max(vec4::dot(L.omega_i, nn), 0.0f);
                    colour shaded =
                        (c * kd) * (L.L * dot) +
                        (L.ambient * ka);

                    shaded.clampColour();

                    unsigned char r, g, b;
                    shaded.toRGB(r, g, b);

                    renderer.canvas.draw(x, y, r, g, b);
                    renderer.zbuffer(x, y) = z;
                }
            }

            w0 += e0.A;
            w1 += e1.A;
            w2 += e2.A;
            z += dz_dx;
            n += dn_dx;
            c += dc_dx;
        }

        w0_row += e0.B;
        w1_row += e1.B;
        w2_row += e2.B;
        z_row += dz_dy;
        n_row += dn_dy;
        c_row += dc_dy;
    }
}

void draw_AVX2_Optimized(Renderer& renderer, Light& L, float ka, float kd)
{
    vec2D minV, maxV;
    getBoundsWindow(renderer.canvas, minV, maxV);
    if (area < 1.f) return;

    const int minX = (int)minV.x;
    const int minY = (int)minV.y;
    const int maxX = (int)ceil(maxV.x);
    const int maxY = (int)ceil(maxV.y);

    Edge e0 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
    Edge e1 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
    Edge e2 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));

    const float invArea = 1.0f / area;

    // --- derivatives ---
    const float dz_dx =
        (v[0].p[2] * e0.A + v[1].p[2] * e1.A + v[2].p[2] * e2.A) * invArea;
    const float dz_dy =
        (v[0].p[2] * e0.B + v[1].p[2] * e1.B + v[2].p[2] * e2.B) * invArea;

    const vec4 dn_dx =
        (v[0].normal * e0.A + v[1].normal * e1.A + v[2].normal * e2.A) * invArea;
    const vec4 dn_dy =
        (v[0].normal * e0.B + v[1].normal * e1.B + v[2].normal * e2.B) * invArea;

    const colour dc_dx =
        (v[0].rgb * e0.A + v[1].rgb * e1.A + v[2].rgb * e2.A) * invArea;
    const colour dc_dy =
        (v[0].rgb * e0.B + v[1].rgb * e1.B + v[2].rgb * e2.B) * invArea;

    const vec4   dn_dx_8 = dn_dx * 8.f;
    const colour dc_dx_8 = dc_dx * 8.f;

    // --- AVX constants ---
    const __m256 zero = _mm256_setzero_ps();
    const __m256 step8_0 = _mm256_set1_ps(e0.A * 8.f);
    const __m256 step8_1 = _mm256_set1_ps(e1.A * 8.f);
    const __m256 step8_2 = _mm256_set1_ps(e2.A * 8.f);
    const __m256 dzdx8 = _mm256_set1_ps(dz_dx * 8.f);

    alignas(32) float lane[8] = { 0,1,2,3,4,5,6,7 };
    const __m256 lanev = _mm256_load_ps(lane);

    float w0_row = e0.A * minX + e0.B * minY + e0.C;
    float w1_row = e1.A * minX + e1.B * minY + e1.C;
    float w2_row = e2.A * minX + e2.B * minY + e2.C;

    float z_row =
        (v[0].p[2] * w0_row + v[1].p[2] * w1_row + v[2].p[2] * w2_row) * invArea;

    vec4   n_row =
        (v[0].normal * w0_row + v[1].normal * w1_row + v[2].normal * w2_row) * invArea;

    colour c_row =
        (v[0].rgb * w0_row + v[1].rgb * w1_row + v[2].rgb * w2_row) * invArea;

    const __m256 w0_step_lane = _mm256_mul_ps(_mm256_set1_ps(e0.A), lanev);
    const __m256 w1_step_lane = _mm256_mul_ps(_mm256_set1_ps(e1.A), lanev);
    const __m256 w2_step_lane = _mm256_mul_ps(_mm256_set1_ps(e2.A), lanev);
    const __m256 z_step_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lanev);


    for (int y = minY; y < maxY; ++y)
    {
        float w0 = w0_row;
        float w1 = w1_row;
        float w2 = w2_row;

        float z = z_row;
        vec4   n = n_row;
        colour c = c_row;

        int x = minX;

        __m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0), w0_step_lane);
        __m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1), w1_step_lane);
        __m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2), w2_step_lane);
        __m256 zv = _mm256_add_ps(_mm256_set1_ps(z), z_step_lane);

        for (; x <= maxX - 8; x += 8)
        {
            __m256 mask =
                _mm256_and_ps(
                    _mm256_and_ps(_mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
                        _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)),
                    _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ));

            int bits = _mm256_movemask_ps(mask);
            if (bits)
            {
                alignas(32) float zbuf[8];
                _mm256_store_ps(zbuf, zv);

                vec4   n_pix = n;
                colour c_pix = c;

                for (int i = 0; i < 8; ++i)
                {
                    if (bits & (1 << i))
                    {
                        int xi = x + i;
                        float zi = zbuf[i];

                        if (zi > 0.001f && renderer.zbuffer(xi, y) > zi)
                        {
                            vec4 ni = n_pix;
                            float dot = std::max(vec4::dot(L.omega_i, ni), 0.0f);

                            colour shaded =
                                (c_pix * kd) * (L.L * dot) + (L.ambient * ka);

                            shaded.clampColour();
                            unsigned char r, g, b;
                            shaded.toRGB(r, g, b);

                            renderer.canvas.draw(xi, y, r, g, b);
                            renderer.zbuffer(xi, y) = zi;
                        }
                    }
                    n_pix += dn_dx;
                    c_pix += dc_dx;
                }
            }

            w0v = _mm256_add_ps(w0v, step8_0);
            w1v = _mm256_add_ps(w1v, step8_1);
            w2v = _mm256_add_ps(w2v, step8_2);
            zv = _mm256_add_ps(zv, dzdx8);

            n += dn_dx_8;
            c += dc_dx_8;
        }

        // scalar tail
        for (; x < maxX; ++x)
        {
            if (w0 >= 0 && w1 >= 0 && w2 >= 0)
            {
                if (z > 0.001f && renderer.zbuffer(x, y) > z)
                {
                    vec4 nn = n;
                    nn.normalise();
                    float dot = std::max(vec4::dot(L.omega_i, nn), 0.0f);

                    colour shaded =
                        (c * kd) * (L.L * dot) + (L.ambient * ka);

                    shaded.clampColour();
                    unsigned char r, g, b;
                    shaded.toRGB(r, g, b);

                    renderer.canvas.draw(x, y, r, g, b);
                    renderer.zbuffer(x, y) = z;
                }
            }
            w0 += e0.A; w1 += e1.A; w2 += e2.A;
            z += dz_dx;
            n += dn_dx;
            c += dc_dx;
        }

        w0_row += e0.B;
        w1_row += e1.B;
        w2_row += e2.B;
        z_row += dz_dy;
        n_row += dn_dy;
        c_row += dc_dy;
    }
}

void draw_AVX2_Optimized2(Renderer& renderer, Light& L, float ka, float kd)
{
    if (area < 1.f) return;
    vec2D minV, maxV;
    getBoundsWindow(renderer.canvas, minV, maxV);

    const int minX = (int)std::floor(minV.x);
    const int minY = (int)std::floor(minV.y);
    const int maxX = (int)std::ceil(maxV.x);
    const int maxY = (int)std::ceil(maxV.y);

    Edge e0 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
    Edge e1 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
    Edge e2 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));

    //auto isTopLeft = [](vec2D p0, vec2D p1) {
    //    return (p0.y == p1.y ? (p1.x < p0.x) : (p1.y > p0.y));
    //    };
    //const bool tl0 = isTopLeft(vec2D(v[1].p), vec2D(v[2].p));
    //const bool tl1 = isTopLeft(vec2D(v[2].p), vec2D(v[0].p));
    //const bool tl2 = isTopLeft(vec2D(v[0].p), vec2D(v[1].p));

    const float invArea = 1.0f / area;

    const float dz_dx =
        (v[0].p[2] * e0.A + v[1].p[2] * e1.A + v[2].p[2] * e2.A) * invArea;
    const float dz_dy =
        (v[0].p[2] * e0.B + v[1].p[2] * e1.B + v[2].p[2] * e2.B) * invArea;

    const vec4 dn_dx =
        (v[0].normal * e0.A + v[1].normal * e1.A + v[2].normal * e2.A) * invArea;
    const vec4 dn_dy =
        (v[0].normal * e0.B + v[1].normal * e1.B + v[2].normal * e2.B) * invArea;

    const colour dc_dx =
        (v[0].rgb * e0.A + v[1].rgb * e1.A + v[2].rgb * e2.A) * invArea;
    const colour dc_dy =
        (v[0].rgb * e0.B + v[1].rgb * e1.B + v[2].rgb * e2.B) * invArea;

    // --- SIMD constants ---
    const __m256 zero = _mm256_setzero_ps();
    //const __m256 lane = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256 lane = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);

    const __m256 w0_lane = _mm256_mul_ps(_mm256_set1_ps(e0.A), lane);
    const __m256 w1_lane = _mm256_mul_ps(_mm256_set1_ps(e1.A), lane);
    const __m256 w2_lane = _mm256_mul_ps(_mm256_set1_ps(e2.A), lane);
    const __m256 z_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lane);

    const __m256 n_x_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx.x), lane);
    const __m256 n_y_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx.y), lane);
    const __m256 n_z_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx.z), lane);

    const __m256 c_r_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx.r), lane);
    const __m256 c_g_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx.g), lane);
    const __m256 c_b_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx.b), lane);

    const __m256 w0_step8 = _mm256_set1_ps(e0.A * 8.f);
    const __m256 w1_step8 = _mm256_set1_ps(e1.A * 8.f);
    const __m256 w2_step8 = _mm256_set1_ps(e2.A * 8.f);
    const __m256 z_step8 = _mm256_set1_ps(dz_dx * 8.f);

    const __m256 n_step8x = _mm256_set1_ps(dn_dx.x * 8.f);
    const __m256 n_step8y = _mm256_set1_ps(dn_dx.y * 8.f);
    const __m256 n_step8z = _mm256_set1_ps(dn_dx.z * 8.f);

    const __m256 c_step8r = _mm256_set1_ps(dc_dx.r * 8.f);
    const __m256 c_step8g = _mm256_set1_ps(dc_dx.g * 8.f);
    const __m256 c_step8b = _mm256_set1_ps(dc_dx.b * 8.f);

    const __m256 Lr = _mm256_set1_ps(L.L.r);
    const __m256 Lg = _mm256_set1_ps(L.L.g);
    const __m256 Lb = _mm256_set1_ps(L.L.b);

    const __m256 L_omega_i_x = _mm256_set1_ps(L.omega_i.x);
    const __m256 L_omega_i_y = _mm256_set1_ps(L.omega_i.y);
    const __m256 L_omega_i_z = _mm256_set1_ps(L.omega_i.z);

    const __m256 ka_step8 = _mm256_set1_ps(ka);
    const __m256 kd_step8 = _mm256_set1_ps(kd);

    const __m256 ambinet_r_ka = _mm256_mul_ps(_mm256_set1_ps(L.ambient.r), _mm256_set1_ps(ka));
    const __m256 ambinet_g_ka = _mm256_mul_ps(_mm256_set1_ps(L.ambient.g), _mm256_set1_ps(ka));
    const __m256 ambinet_b_ka = _mm256_mul_ps(_mm256_set1_ps(L.ambient.b), _mm256_set1_ps(ka));

    float w0_row = e0.A * minX + e0.B * minY + e0.C;
    float w1_row = e1.A * minX + e1.B * minY + e1.C;
    float w2_row = e2.A * minX + e2.B * minY + e2.C;

    __m256 w0_row_v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
    __m256 w1_row_v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
    __m256 w2_row_v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);

    float z_row =
        (v[0].p[2] * w0_row + v[1].p[2] * w1_row + v[2].p[2] * w2_row) * invArea;

    vec4 n_row =
        (v[0].normal * w0_row + v[1].normal * w1_row + v[2].normal * w2_row) * invArea;

    colour c_row =
        (v[0].rgb * w0_row + v[1].rgb * w1_row + v[2].rgb * w2_row) * invArea;

    __m256 _255 = _mm256_set1_ps(255.0f);
    __m256 _001 = _mm256_set1_ps(0.001f);

    //L.omega_i.normalise();

    __m256 w0_row_8 = _mm256_set1_ps(w0_row);

    for (int y = minY; y <= maxY; ++y)
    {
        //__m256 w0v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
        //__m256 w1v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
        //__m256 w2v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);
        //__m256 zv = _mm256_add_ps(_mm256_set1_ps(z_row), z_lane);
        // This method seems faster than the upper one!???
        __m256 w0v = _mm256_set1_ps(w0_row);
        w0v = _mm256_add_ps(w0v, w0_lane);
        __m256 w1v = _mm256_set1_ps(w1_row);
        w1v = _mm256_add_ps(w1v, w1_lane);
        __m256 w2v = _mm256_set1_ps(w2_row);
        w2v = _mm256_add_ps(w2v, w2_lane);
        __m256 zv = _mm256_set1_ps(z_row);
        zv = _mm256_add_ps(zv, z_lane);

        //__m256 nx = _mm256_set1_ps(n_row.x);
        //__m256 ny = _mm256_set1_ps(n_row.y);
        //__m256 nz = _mm256_set1_ps(n_row.z);

        //__m256 cr = _mm256_set1_ps(c_row.r);
        //__m256 cg = _mm256_set1_ps(c_row.g);
        //__m256 cb = _mm256_set1_ps(c_row.b);

        //__m256 nx = _mm256_add_ps(_mm256_set1_ps(n_row.x), n_x_lane);
        //__m256 ny = _mm256_add_ps(_mm256_set1_ps(n_row.y), n_y_lane);
        //__m256 nz = _mm256_add_ps(_mm256_set1_ps(n_row.z), n_z_lane);
        //__m256 cr = _mm256_add_ps(_mm256_set1_ps(c_row.r), c_r_lane);
        //__m256 cg = _mm256_add_ps(_mm256_set1_ps(c_row.g), c_g_lane);
        //__m256 cb = _mm256_add_ps(_mm256_set1_ps(c_row.b), c_b_lane);

        __m256 nx = _mm256_set1_ps(n_row.x);
        nx = _mm256_add_ps(nx, n_x_lane);
        __m256 ny = _mm256_set1_ps(n_row.y);
        ny = _mm256_add_ps(ny, n_y_lane);
        __m256 nz = _mm256_set1_ps(n_row.z);
        nz = _mm256_add_ps(nz, n_z_lane);
        __m256 cr = _mm256_set1_ps(c_row.r);
        cr = _mm256_add_ps(cr, c_r_lane);
        __m256 cg = _mm256_set1_ps(c_row.g);
        cg = _mm256_add_ps(cg, c_g_lane);
        __m256 cb = _mm256_set1_ps(c_row.b);
        cb = _mm256_add_ps(cb, c_b_lane);

        int x = minX;

        for (; x <= maxX - 8; x += 8)
        {

            //__m256 inside =
            //    _mm256_and_ps(
            //        _mm256_and_ps(_mm256_cmp_ps(w0v, zero, _CMP_GE_OQ),
            //            _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ)),
            //        _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ));

            __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
            __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
            __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
            __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
            inside = _mm256_and_ps(inside, w2v_zero);

            //__m256 inside0 = tl0 ? _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ) : _mm256_cmp_ps(w0v, zero, _CMP_GT_OQ);
            //__m256 inside1 = tl1 ? _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ) : _mm256_cmp_ps(w1v, zero, _CMP_GT_OQ);
            //__m256 inside2 = tl2 ? _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ) : _mm256_cmp_ps(w2v, zero, _CMP_GT_OQ);
            //__m256 inside = _mm256_and_ps(_mm256_and_ps(inside0, inside1), inside2);

            int mask = _mm256_movemask_ps(inside);
            bool active = (mask != 0);
            int final_mask = 0;

            if (mask != 0)
            {
                __m256 zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));
                //__m256 depth_ok =
                //    _mm256_and_ps(
                //        _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
                //        _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ));

                __m256 zv_001 = _mm256_cmp_ps(zv, _001, _CMP_GE_OQ);
                __m256 zbuf_zv = _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ);
                __m256 depth_ok = _mm256_and_ps(zv_001, zbuf_zv);

                final_mask = _mm256_movemask_ps(_mm256_and_ps(inside, depth_ok));
                //if (final_mask == 0x00)
                //    active = false;
            }

            if (final_mask)
            {
                // normalize normal
                //__m256 len = _mm256_sqrt_ps(
                //    _mm256_add_ps(
                //        _mm256_add_ps(_mm256_mul_ps(nx, nx), _mm256_mul_ps(ny, ny)),
                //        _mm256_mul_ps(nz, nz)));

                __m256 xmul = _mm256_mul_ps(nx, nx);
                __m256 ymul = _mm256_mul_ps(ny, ny);
                __m256 zmul = _mm256_mul_ps(nz, nz);
                __m256 len = _mm256_add_ps(xmul, ymul);
                len = _mm256_add_ps(len, zmul);
                len = _mm256_sqrt_ps(len);

                //__m256 valid = _mm256_castsi256_ps(_mm256_set1_epi32(final_mask));
                //len = _mm256_blendv_ps(_mm256_set1_ps(1.f), len, valid);

                nx = _mm256_div_ps(nx, len);
                ny = _mm256_div_ps(ny, len);
                nz = _mm256_div_ps(nz, len);

                //__m256 dot =
                //    _mm256_max_ps(
                //        _mm256_add_ps(
                //            _mm256_add_ps(_mm256_mul_ps(nx, L_omega_i_x),
                //                _mm256_mul_ps(ny, L_omega_i_y)),
                //            _mm256_mul_ps(nz, L_omega_i_z)),
                //        zero);

                __m256 ndotl_x = _mm256_mul_ps(nx, L_omega_i_x);
                __m256 ndotl_y = _mm256_mul_ps(ny, L_omega_i_y);
                __m256 ndotl_z = _mm256_mul_ps(nz, L_omega_i_z);
                __m256 dot = _mm256_add_ps(ndotl_x, ndotl_y);
                dot = _mm256_add_ps(dot, ndotl_z);
                dot = _mm256_max_ps(dot, zero);


                //__m256 r =
                //    _mm256_add_ps(
                //        _mm256_mul_ps(_mm256_mul_ps(cr, kd_step8),
                //            _mm256_mul_ps(Lr, dot)),
                //        ambinet_r_ka);

                __m256 r = _mm256_mul_ps(cr, kd_step8);
                r = _mm256_mul_ps(r, Lr);
                r = _mm256_mul_ps(r, dot);
                r = _mm256_add_ps(r, ambinet_r_ka);
                r = _mm256_mul_ps(r, _255);

                //__m256 g =
                //    _mm256_add_ps(
                //        _mm256_mul_ps(_mm256_mul_ps(cg, kd_step8),
                //            _mm256_mul_ps(Lg, dot)),
                //        ambinet_g_ka);

                __m256 g = _mm256_mul_ps(cg, kd_step8);
                g = _mm256_mul_ps(g, Lg);
                g = _mm256_mul_ps(g, dot);
                g = _mm256_add_ps(g, ambinet_g_ka);
                g = _mm256_mul_ps(g, _255);

                //__m256 b =
                //    _mm256_add_ps(
                //        _mm256_mul_ps(_mm256_mul_ps(cb, kd_step8),
                //            _mm256_mul_ps(Lb, dot)),
                //        ambinet_b_ka);

                __m256 b = _mm256_mul_ps(cb, kd_step8);
                b = _mm256_mul_ps(b, Lb);
                b = _mm256_mul_ps(b, dot);
                b = _mm256_add_ps(b, ambinet_b_ka);
                b = _mm256_mul_ps(b, _255);

                //r = _mm256_mul_ps(r, _255);
                //g = _mm256_mul_ps(g, _255);
                //b = _mm256_mul_ps(b, _255);

                if (final_mask == 0xFF)
                {
                    alignas(32) float rr[8], gg[8], bb[8], zz[8];
                    _mm256_store_ps(rr, r);
                    _mm256_store_ps(gg, g);
                    _mm256_store_ps(bb, b);
                    //_mm256_store_ps(zz, zv);
                    _mm256_store_ps(&renderer.zbuffer(x, y), zv);

                    //for (int i = 0; i < 8; i+= 4)
                    //{
                    //    //if (final_mask & (1 << i))
                    //    //{
                    //        //renderer.canvas.draw(
                    //        //    x + i, y,
                    //        //    (unsigned char)(rr[i] * 255.f),
                    //        //    (unsigned char)(gg[i] * 255.f),
                    //        //    (unsigned char)(bb[i] * 255.f)
                    //        //);
                    //        renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]), (unsigned char)(gg[i]), (unsigned char)(bb[i]));
                    //        renderer.canvas.draw(x+ 1 + i, y, (unsigned char)(rr[i+1]), (unsigned char)(gg[i+1]), (unsigned char)(bb[i+1]));
                    //        renderer.canvas.draw(x + 2+ i, y, (unsigned char)(rr[i + 2]), (unsigned char)(gg[i+2]), (unsigned char)(bb[i+2]));
                    //        renderer.canvas.draw(x + 3 +i, y, (unsigned char)(rr[i+3]), (unsigned char)(gg[i+3]), (unsigned char)(bb[i+3]));
                    //        //renderer.zbuffer(x + i, y) = zz[i];
                    //    //}
                    //}
                    //for (int i = 0; i < 8; ++i)
                    //{
                    //    //if (final_mask & (1 << i))
                    //    //{
                    //        //renderer.canvas.draw(
                    //        //    x + i, y,
                    //        //    (unsigned char)(rr[i] * 255.f),
                    //        //    (unsigned char)(gg[i] * 255.f),
                    //        //    (unsigned char)(bb[i] * 255.f)
                    //        //);
                    //        renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]), (unsigned char)(gg[i]), (unsigned char)(bb[i]));
                    //        //renderer.zbuffer(x + i, y) = zz[i];
                    //    //}
                    //}
                    //renderer.canvas.draw(x    , y, (unsigned char)(rr[0]), (unsigned char)(gg[0]), (unsigned char)(bb[0]));
                    //renderer.canvas.draw(x + 1, y, (unsigned char)(rr[1]), (unsigned char)(gg[1]), (unsigned char)(bb[1]));
                    //renderer.canvas.draw(x + 2, y, (unsigned char)(rr[2]), (unsigned char)(gg[2]), (unsigned char)(bb[2]));
                    //renderer.canvas.draw(x + 3, y, (unsigned char)(rr[3]), (unsigned char)(gg[3]), (unsigned char)(bb[3]));
                    //renderer.canvas.draw(x + 4, y, (unsigned char)(rr[4]), (unsigned char)(gg[4]), (unsigned char)(bb[4]));
                    //renderer.canvas.draw(x + 5, y, (unsigned char)(rr[5]), (unsigned char)(gg[5]), (unsigned char)(bb[5]));
                    //renderer.canvas.draw(x + 6, y, (unsigned char)(rr[6]), (unsigned char)(gg[6]), (unsigned char)(bb[6]));
                    //renderer.canvas.draw(x + 7, y, (unsigned char)(rr[7]), (unsigned char)(gg[7]), (unsigned char)(bb[7]));
                    renderer.canvas.draw(x, y, rr, gg, bb);
                }
                else
                {
                    alignas(32) float rr[8], gg[8], bb[8], zz[8];
                    _mm256_store_ps(rr, r);
                    _mm256_store_ps(gg, g);
                    _mm256_store_ps(bb, b);
                    _mm256_store_ps(zz, zv);

                    int m = final_mask;
                    while (m)
                    {
                        // count trailing zeros
                        int i = _tzcnt_u32(m);
                        m &= m - 1;
                        //renderer.canvas.draw(x + i, y,
                        //    (unsigned char)(rr[i] * 255),
                        //    (unsigned char)(gg[i] * 255),
                        //    (unsigned char)(bb[i] * 255));
                        renderer.canvas.draw(
                            x + i, y,
                            (unsigned char)(rr[i]),
                            (unsigned char)(gg[i]),
                            (unsigned char)(bb[i])
                        );
                        renderer.zbuffer(x + i, y) = zz[i];
                    }
                }
            }

            w0v = _mm256_add_ps(w0v, w0_step8);
            w1v = _mm256_add_ps(w1v, w1_step8);
            w2v = _mm256_add_ps(w2v, w2_step8);
            zv = _mm256_add_ps(zv, z_step8);

            nx = _mm256_add_ps(nx, n_step8x);
            ny = _mm256_add_ps(ny, n_step8y);
            nz = _mm256_add_ps(nz, n_step8z);

            cr = _mm256_add_ps(cr, c_step8r);
            cg = _mm256_add_ps(cg, c_step8g);
            cb = _mm256_add_ps(cb, c_step8b);
        }

        for (; x <= maxX; ++x)
        {
            // === 1. Edge test ===
            //float w0 = e0.A * (x + 0.5f) + e0.B * (y + 0.5f) + e0.C;
            //float w1 = e1.A * (x + 0.5f) + e1.B * (y + 0.5f) + e1.C;
            //float w2 = e2.A * (x + 0.5f) + e2.B * (y + 0.5f) + e2.C;
            float w0 = e0.A * x + e0.B * y + e0.C;
            float w1 = e1.A * x + e1.B * y + e1.C;
            float w2 = e2.A * x + e2.B * y + e2.C;

            if (w0 < 0.f || w1 < 0.f || w2 < 0.f)
                continue;

            // === 2. Barycentric interpolation ===
            float alpha = w0 * invArea;
            float beta = w1 * invArea;
            float gamma = w2 * invArea;

            float z = interpolate(alpha, beta, gamma, v[0].p[2], v[1].p[2], v[2].p[2]);

            if (z < 0.001f || z >= renderer.zbuffer(x, y))
                continue;

            // === 3. Normal ===
            vec4 n = interpolate(alpha, beta, gamma, v[0].normal, v[1].normal, v[2].normal);

            n.normalise();

            colour c = interpolate(alpha, beta, gamma, v[0].rgb, v[1].rgb, v[2].rgb);
            c.clampColour();

            // === 5. Lighting ===
            float dot = std::max(vec4::dot(n, L.omega_i), 0.f);

            colour out = (c * kd) * (L.L * dot) + (L.ambient * ka);

            out.clampColour();

            unsigned char r, g, b;
            out.toRGB(r, g, b);

            // === 6. Write ===
            renderer.canvas.draw(x, y, r, g, b);
            renderer.zbuffer(x, y) = z;
        }

        w0_row += e0.B;
        w1_row += e1.B;
        w2_row += e2.B;
        z_row += dz_dy;
        n_row += dn_dy;
        c_row += dc_dy;
    }
}
