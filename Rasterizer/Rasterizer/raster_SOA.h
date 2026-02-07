#pragma once
#include <future>


struct Mesh_SoA_Transformed {
    // transformed vertex data (screen space)
    std::vector<float> transformed_positions_x;
    std::vector<float> transformed_positions_y;
    std::vector<float> transformed_positions_z;
    std::vector<float> transformed_positions_w;

    // transformed normal data (view space)
    std::vector<float> transformed_normals_x;
    std::vector<float> transformed_normals_y;
    std::vector<float> transformed_normals_z;

    // color data
    std::vector<float> colors_r;
    std::vector<float> colors_g;
    std::vector<float> colors_b;

    // view position
    std::vector<float> view_positions_x;
    std::vector<float> view_positions_y;
    std::vector<float> view_positions_z;
    std::vector<float> view_positions_w;

    // triangles indices
    std::vector<triIndices> triangles;

    float ka, kd;

    void ensureCapacity(int size) {
        if (static_cast<int>(transformed_positions_x.size()) < size) {
            transformed_positions_x.resize(size);
            transformed_positions_y.resize(size);
            transformed_positions_z.resize(size);
            transformed_positions_w.resize(size);
            transformed_normals_x.resize(size);
            transformed_normals_y.resize(size);
            transformed_normals_z.resize(size);

            view_positions_x.resize(size);
            view_positions_y.resize(size);
            view_positions_z.resize(size);
            view_positions_w.resize(size);
        }
    }

    void transformBatchSIMD(const matrix& mvp, const matrix& view, const matrix& world, const Mesh_SoA& source,
         int start_idx, int count, float canvas_width, float canvas_height) {

        // constant data
        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 half = _mm256_set1_ps(0.5f);

        const __m256 half_width = _mm256_set1_ps(canvas_width * 0.5f);
        const __m256 half_height = _mm256_set1_ps(canvas_height * 0.5f);
        const __m256 center_x = _mm256_set1_ps(canvas_width * 0.5f);
        const __m256 center_y = _mm256_set1_ps(canvas_height * 0.5f);

        // mvp matrix
        const float* m = mvp.data.a;
        __m256 m00 = _mm256_set1_ps(m[0]);
        __m256 m01 = _mm256_set1_ps(m[1]);
        __m256 m02 = _mm256_set1_ps(m[2]);
        __m256 m03 = _mm256_set1_ps(m[3]);
        __m256 m10 = _mm256_set1_ps(m[4]);
        __m256 m11 = _mm256_set1_ps(m[5]);
        __m256 m12 = _mm256_set1_ps(m[6]);
        __m256 m13 = _mm256_set1_ps(m[7]);
        __m256 m20 = _mm256_set1_ps(m[8]);
        __m256 m21 = _mm256_set1_ps(m[9]);
        __m256 m22 = _mm256_set1_ps(m[10]);
        __m256 m23 = _mm256_set1_ps(m[11]);
        __m256 m30 = _mm256_set1_ps(m[12]);
        __m256 m31 = _mm256_set1_ps(m[13]);
        __m256 m32 = _mm256_set1_ps(m[14]);
        __m256 m33 = _mm256_set1_ps(m[15]);

        // view matrix
        const float* v = view.data.a;
        __m256 v00 = _mm256_set1_ps(v[0]);
        __m256 v01 = _mm256_set1_ps(v[1]);
        __m256 v02 = _mm256_set1_ps(v[2]);
        __m256 v03 = _mm256_set1_ps(v[3]);
        __m256 v10 = _mm256_set1_ps(v[4]);
        __m256 v11 = _mm256_set1_ps(v[5]);
        __m256 v12 = _mm256_set1_ps(v[6]);
        __m256 v13 = _mm256_set1_ps(v[7]);
        __m256 v20 = _mm256_set1_ps(v[8]);
        __m256 v21 = _mm256_set1_ps(v[9]);
        __m256 v22 = _mm256_set1_ps(v[10]);
        __m256 v23 = _mm256_set1_ps(v[11]);
        __m256 v30 = _mm256_set1_ps(v[12]);
        __m256 v31 = _mm256_set1_ps(v[13]);
        __m256 v32 = _mm256_set1_ps(v[14]);
        __m256 v33 = _mm256_set1_ps(v[15]);

        // world matrix
        const float* w = world.data.a;
        __m256 w00 = _mm256_set1_ps(w[0]);
        __m256 w01 = _mm256_set1_ps(w[1]);
        __m256 w02 = _mm256_set1_ps(w[2]);
        __m256 w10 = _mm256_set1_ps(w[4]);
        __m256 w11 = _mm256_set1_ps(w[5]);
        __m256 w12 = _mm256_set1_ps(w[6]);
        __m256 w20 = _mm256_set1_ps(w[8]);
        __m256 w21 = _mm256_set1_ps(w[9]);
        __m256 w22 = _mm256_set1_ps(w[10]);

        // clip data
        const __m256 near_plane = _mm256_set1_ps(0.1f);
        const __m256 far_plane = _mm256_set1_ps(100.0f);
        const __m256 epsilon = _mm256_set1_ps(1e-4f);
        const __m256 minus_one = _mm256_set1_ps(-1.0f);
        const __m256 out_of_screen = _mm256_set1_ps(-10000.0f);

        int i = 0;
        for (; i + 7 < count; i += 8) {
            int actual_count = std::min(8, count - i);
            int store_idx = start_idx + i;

            __m256 vx = loadPartial(&source.positions_x[start_idx + i], actual_count);
            __m256 vy = loadPartial(&source.positions_y[start_idx + i], actual_count);
            __m256 vz = loadPartial(&source.positions_z[start_idx + i], actual_count);
            __m256 vw = loadPartial(&source.positions_w[start_idx + i], actual_count);

            // view transorm view * world * position
            __m256 view_x = _mm256_fmadd_ps(v00, vx,
                _mm256_fmadd_ps(v01, vy,
                    _mm256_fmadd_ps(v02, vz,
                        _mm256_mul_ps(v03, vw))));

            __m256 view_y = _mm256_fmadd_ps(v10, vx,
                _mm256_fmadd_ps(v11, vy,
                    _mm256_fmadd_ps(v12, vz,
                        _mm256_mul_ps(v13, vw))));

            __m256 view_z = _mm256_fmadd_ps(v20, vx,
                _mm256_fmadd_ps(v21, vy,
                    _mm256_fmadd_ps(v22, vz,
                        _mm256_mul_ps(v23, vw))));

            __m256 view_w = _mm256_fmadd_ps(v30, vx,
                _mm256_fmadd_ps(v31, vy,
                    _mm256_fmadd_ps(v32, vz,
                        _mm256_mul_ps(v33, vw))));

            // store
            storePartial(&view_positions_x[store_idx], view_x, actual_count);
            storePartial(&view_positions_y[store_idx], view_y, actual_count);
            storePartial(&view_positions_z[store_idx], view_z, actual_count);
            storePartial(&view_positions_w[store_idx], view_w, actual_count);

            // mvp proj * view_x, proj * view_y, proj * view_z, proj * view_w
            __m256 clip_x = _mm256_fmadd_ps(m00, vx,
                _mm256_fmadd_ps(m01, vy,
                    _mm256_fmadd_ps(m02, vz,
                        _mm256_mul_ps(m03, vw))));

            __m256 clip_y = _mm256_fmadd_ps(m10, vx,
                _mm256_fmadd_ps(m11, vy,
                    _mm256_fmadd_ps(m12, vz,
                        _mm256_mul_ps(m13, vw))));

            __m256 clip_z = _mm256_fmadd_ps(m20, vx,
                _mm256_fmadd_ps(m21, vy,
                    _mm256_fmadd_ps(m22, vz,
                        _mm256_mul_ps(m23, vw))));

            __m256 clip_w = _mm256_fmadd_ps(m30, vx,
                _mm256_fmadd_ps(m31, vy,
                    _mm256_fmadd_ps(m32, vz,
                        _mm256_mul_ps(m33, vw))));

            __m256 w_valid = _mm256_cmp_ps(clip_w, epsilon, _CMP_GT_OQ);

            __m256 inv_w = _mm256_rcp_ps(clip_w);
            inv_w = _mm256_mul_ps(inv_w, _mm256_fnmadd_ps(clip_w, inv_w, _mm256_set1_ps(2.0f)));

            __m256 ndc_x = _mm256_mul_ps(clip_x, inv_w);
            __m256 ndc_y = _mm256_mul_ps(clip_y, inv_w);
            __m256 ndc_z = _mm256_mul_ps(clip_z, inv_w);

            // viewport transform
            // screen_x = (ndc_x + 1.0) * 0.5 * width
            __m256 screen_x = _mm256_mul_ps(
                _mm256_add_ps(ndc_x, one),
                half_width);

            // screen_y = canvas_height - (ndc_y + 1.0) * 0.5 * height
            __m256 screen_y = _mm256_mul_ps(_mm256_sub_ps(one, ndc_y), half_height);

            __m256 invalid_mask = _mm256_cmp_ps(w_valid, zero, _CMP_EQ_OQ);

            screen_x = _mm256_blendv_ps(screen_x, out_of_screen, invalid_mask);
            screen_y = _mm256_blendv_ps(screen_y, out_of_screen, invalid_mask);
            storePartial(&transformed_positions_x[store_idx], screen_x, actual_count);
            storePartial(&transformed_positions_y[store_idx], screen_y, actual_count);

            // Depth: Mapping from [-1,1] in NDC to [0,1]
            __m256 depth = _mm256_fmadd_ps(ndc_z, half, half);
            depth = _mm256_blendv_ps(depth, out_of_screen, invalid_mask);
            storePartial(&transformed_positions_z[store_idx], depth, actual_count);

            // for vertices where w is ineffective, set w = 1.0
            __m256 stored_w = _mm256_blendv_ps(clip_w, _mm256_set1_ps(1.0f), invalid_mask);
            storePartial(&transformed_positions_w[store_idx], stored_w, actual_count);

            // normal matrix transformation
            __m256 nx = loadPartial(&source.normals_x[start_idx + i], actual_count);
            __m256 ny = loadPartial(&source.normals_y[start_idx + i], actual_count);
            __m256 nz = loadPartial(&source.normals_z[start_idx + i], actual_count);

            __m256 world_nx = _mm256_fmadd_ps(w00, nx,
                _mm256_fmadd_ps(w01, ny,
                    _mm256_mul_ps(w02, nz)));

            __m256 world_ny = _mm256_fmadd_ps(w10, nx,
                _mm256_fmadd_ps(w11, ny,
                    _mm256_mul_ps(w12, nz)));

            __m256 world_nz = _mm256_fmadd_ps(w20, nx,
                _mm256_fmadd_ps(w21, ny,
                    _mm256_mul_ps(w22, nz)));

            // normalize
            __m256 len_sq = _mm256_fmadd_ps(world_nx, world_nx,
                _mm256_fmadd_ps(world_ny, world_ny,
                    _mm256_mul_ps(world_nz, world_nz)));

            __m256 inv_len = _mm256_rsqrt_ps(len_sq);
            inv_len = _mm256_mul_ps(inv_len,
                _mm256_fnmadd_ps(_mm256_mul_ps(len_sq, _mm256_mul_ps(inv_len, inv_len)),
                    _mm256_set1_ps(0.5f),
                    _mm256_set1_ps(1.5f)));

            world_nx = _mm256_mul_ps(world_nx, inv_len);
            world_ny = _mm256_mul_ps(world_ny, inv_len);
            world_nz = _mm256_mul_ps(world_nz, inv_len);

            storePartial(&transformed_normals_x[store_idx], world_nx, actual_count);
            storePartial(&transformed_normals_y[store_idx], world_ny, actual_count);
            storePartial(&transformed_normals_z[store_idx], world_nz, actual_count);

            // copy color
            int end = std::min(i + 8, count);
            for (int j = i; j < end; ++j) {
                colors_r[start_idx + j] = source.colors_r[start_idx + j];
                colors_g[start_idx + j] = source.colors_g[start_idx + j];
                colors_b[start_idx + j] = source.colors_b[start_idx + j];
            }
        }
        // last vertexs
        for (; i < count; ++i) {
            int idx = start_idx + i;
            int src_idx = idx;

            vec4 pos(source.positions_x[src_idx], source.positions_y[src_idx],  source.positions_z[src_idx], source.positions_w[src_idx]);

            vec4 clip_pos = mvp * pos;
            clip_pos.divideW();

            float screen_x = (clip_pos[0] + 1.0f) * 0.5f * canvas_width;
            float screen_y = (clip_pos[1] + 1.0f) * 0.5f * canvas_height;
            screen_y = canvas_height - screen_y; 

            transformed_positions_x[idx] = screen_x;
            transformed_positions_y[idx] = screen_y;
            transformed_positions_z[idx] = clip_pos[2];
            transformed_positions_w[idx] = clip_pos[3];

            vec4 normal(source.normals_x[src_idx], source.normals_y[src_idx], source.normals_z[src_idx], 0.0f);

            normal = world * normal;
            normal.normalise();

            transformed_normals_x[idx] = normal[0];
            transformed_normals_y[idx] = normal[1];
            transformed_normals_z[idx] = normal[2];

            colors_r[idx] = source.colors_r[src_idx];
            colors_g[idx] = source.colors_g[src_idx];
            colors_b[idx] = source.colors_b[src_idx];
        }
    }


private:
    static inline __m256 loadPartial(const float* p, int count) {
        static const int maskTable[9][8] = {
            {0}, // unused
            {-1,0,0,0,0,0,0,0},
            {-1,-1,0,0,0,0,0,0},
            {-1,-1,-1,0,0,0,0,0},
            {-1,-1,-1,-1,0,0,0,0},
            {-1,-1,-1,-1,-1,0,0,0},
            {-1,-1,-1,-1,-1,-1,0,0},
            {-1,-1,-1,-1,-1,-1,-1,0},
            {-1,-1,-1,-1,-1,-1,-1,-1}
        };
        return _mm256_maskload_ps(p, _mm256_load_si256((__m256i*)maskTable[count]));
    }

    static inline void storePartial(float* dest, __m256 val, int count)
    {
        static const int maskTable[9][8] = {
            {0}, // unused
            {-1,0,0,0,0,0,0,0},
            {-1,-1,0,0,0,0,0,0},
            {-1,-1,-1,0,0,0,0,0},
            {-1,-1,-1,-1,0,0,0,0},
            {-1,-1,-1,-1,-1,0,0,0},
            {-1,-1,-1,-1,-1,-1,0,0},
            {-1,-1,-1,-1,-1,-1,-1,0},
            {-1,-1,-1,-1,-1,-1,-1,-1}
        };

        __m256i mask = _mm256_load_si256((const __m256i*)maskTable[count]);
        _mm256_maskstore_ps(dest, mask, val);
    }
};

class Tile {
public:
    int x, y;           // top left coordinates
    int width, height;  // size
    //std::atomic<bool> completed;
    Zbuffer<float> zbuffer;
    std::vector<unsigned char> colors;

    Tile() : x(0), y(0), width(0), height(0) {}

    Tile(int _x, int _y, int _w, int _h) : x(_x), y(_y), width(_w), height(_h)
    {
        //zbuffer.resize(width * height, std::numeric_limits<float>::max());
        zbuffer.create(width, height);
        colors.resize(width * height * 3, 0);
    }
};

class RendererSOA_MT {
private:
    int TILE_WIDTH;
    int TILE_HEIGHT; 
    std::vector<Tile> tiles;
    int tiles_x, tiles_y;
    //bool if_multhread;

    int tri_chunk = 1024;
    
public:

    RendererSOA_MT(int canvas_width, int canvas_height, ThreadPool& pool) 
        //: TILE_WIDTH(256), TILE_HEIGHT(384)
        : TILE_WIDTH(1024), TILE_HEIGHT(96)
    {
        TILE_HEIGHT = canvas_height / pool.getThreadCount();
        
        // caculate tiles count
        tiles_x = (canvas_width + TILE_WIDTH - 1) / TILE_WIDTH;
        tiles_y = (canvas_height + TILE_HEIGHT - 1) / TILE_HEIGHT;

        tiles.reserve(tiles_x * tiles_y);

        // create tiles
        for (int ty = 0; ty < tiles_y; ++ty) {
            for (int tx = 0; tx < tiles_x; ++tx) {
                int tile_x = tx * TILE_WIDTH;
                int tile_y = ty * TILE_HEIGHT;
                int tile_w = std::min(TILE_WIDTH, canvas_width - tile_x);
                int tile_h = std::min(TILE_HEIGHT, canvas_height - tile_y);

                tiles.emplace_back(tile_x, tile_y, tile_w, tile_h);
            }
        }
    }

    void renderScene_SoA_Optimized(Renderer& renderer, ThreadPool& pool,
        Mesh_SoA& mesh, matrix& camera, Light& light, float ka, float kd) {

        const int vCount = (int)mesh.positions_x.size();
        const int tileCount = (int)tiles.size();

        int thread_count = pool.getThreadCount();
        int chunkSize = (vCount + thread_count - 1) / thread_count;
        chunkSize = (chunkSize + 7) & ~7; 

        Mesh_SoA_Transformed transformed_mesh;
        transformed_mesh.triangles = mesh.triangles;
        transformed_mesh.ka = ka;
        transformed_mesh.kd = kd;
        transformed_mesh.transformed_positions_x.resize(vCount);
        transformed_mesh.transformed_positions_y.resize(vCount);
        transformed_mesh.transformed_positions_z.resize(vCount);
        transformed_mesh.transformed_positions_w.resize(vCount);
        transformed_mesh.transformed_normals_x.resize(vCount);
        transformed_mesh.transformed_normals_y.resize(vCount);
        transformed_mesh.transformed_normals_z.resize(vCount);
        transformed_mesh.colors_r.resize(vCount);
        transformed_mesh.colors_g.resize(vCount);
        transformed_mesh.colors_b.resize(vCount);
        transformed_mesh.view_positions_x.resize(vCount);
        transformed_mesh.view_positions_y.resize(vCount);
        transformed_mesh.view_positions_z.resize(vCount);
        transformed_mesh.view_positions_w.resize(vCount);

        matrix view = camera * mesh.world;
        matrix p = renderer.perspective * view;

        transformed_mesh.transformBatchSIMD(p, view, mesh.world, mesh,
            0, vCount,
            renderer.canvas.getWidth(), renderer.canvas.getHeight());

        const int triCount = (int)transformed_mesh.triangles.size();
        std::vector<std::vector<int>> triangle_buckets(tileCount);

        // binning
        for (int t = 0; t < triCount; ++t) {
            triIndices& idx = transformed_mesh.triangles[t];

            const vec4 v0_view(
                transformed_mesh.view_positions_x[idx.v[0]],
                transformed_mesh.view_positions_y[idx.v[0]],
                transformed_mesh.view_positions_z[idx.v[0]],
                transformed_mesh.view_positions_w[idx.v[0]]
            );
            const vec4 v1_view(
                transformed_mesh.view_positions_x[idx.v[1]],
                transformed_mesh.view_positions_y[idx.v[1]],
                transformed_mesh.view_positions_z[idx.v[1]],
                transformed_mesh.view_positions_w[idx.v[1]]
            );
            const vec4 v2_view(
                transformed_mesh.view_positions_x[idx.v[2]],
                transformed_mesh.view_positions_y[idx.v[2]],
                transformed_mesh.view_positions_z[idx.v[2]],
                transformed_mesh.view_positions_w[idx.v[2]]
            );
            vec4 e1 = v1_view - v0_view;
            vec4 e2 = v2_view - v0_view;

            vec4 view_normal = vec4::cross(e1, e2);
            float dot_product = vec4::dot(view_normal, -v0_view);
            if (dot_product >= 0.0f)
                continue;

            const float* pos_x = transformed_mesh.transformed_positions_x.data();
            const float* pos_y = transformed_mesh.transformed_positions_y.data();
            float min_x = std::min({ pos_x[idx.v[0]], pos_x[idx.v[1]], pos_x[idx.v[2]] });
            float max_x = std::max({ pos_x[idx.v[0]], pos_x[idx.v[1]], pos_x[idx.v[2]] });
            float min_y = std::min({ pos_y[idx.v[0]], pos_y[idx.v[1]], pos_y[idx.v[2]] });
            float max_y = std::max({ pos_y[idx.v[0]], pos_y[idx.v[1]], pos_y[idx.v[2]] });

            min_x = std::max(0.0f, min_x);
            max_x = std::min((float)renderer.canvas.getWidth() - 1, max_x);
            min_y = std::max(0.0f, min_y);
            max_y = std::min((float)renderer.canvas.getHeight() - 1, max_y);
            if (min_x > max_x || min_y > max_y) continue;

            int min_tx = int(std::floor(min_x)) / TILE_WIDTH;
            int max_tx = int(std::ceil(max_x)) / TILE_WIDTH;
            int min_ty = int(std::floor(min_y)) / TILE_HEIGHT;
            int max_ty = int(std::ceil(max_y)) / TILE_HEIGHT;

            min_tx = std::max(0, min_tx);
            max_tx = std::min(tiles_x - 1, max_tx);
            min_ty = std::max(0, min_ty);
            max_ty = std::min(tiles_y - 1, max_ty);

            for (int ty = min_ty; ty <= max_ty; ++ty) {
                for (int tx = min_tx; tx <= max_tx; ++tx) {
                    int tile_idx = ty * tiles_x + tx;
                    triangle_buckets[tile_idx].push_back(t);
                }
            }
        }
        
        //for (Tile& tile : tiles) {
        //    tile.zbuffer.clear();
        //}


        // 3. 并行瓦片渲染 - 每个tile独立处理
        //pool.beginFrame();
        //for (size_t tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
        //    if (!triangle_buckets[tile_idx].empty()) {
        //        pool.enqueue([&, tile_idx]() {
        //            Tile& tile = tiles[tile_idx];
        //            // 重置瓦片缓冲区
        //            //std::fill(tile.zbuffer.begin(), tile.zbuffer.end(),
        //            //    std::numeric_limits<float>::max());
        //            tile.zbuffer.clear();
        //            std::fill(tile.colors.begin(), tile.colors.end(), 0);
        //            // 渲染该tile中的所有三角形
        //            for (int tri_idx : triangle_buckets[tile_idx]) {
        //                // 这里直接使用你现有的renderTriangleInTile逻辑
        //                // 但是传入transformed_mesh而不是原始mesh
        //                renderTriangleFromSOA_InTile(transformed_mesh, light,
        //                    ka, kd, tri_idx, tile);
        //            }
        //            });
        //    }
        //}
        //pool.waitFrame();
        //pool.waitAll();
        //for (size_t tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
        //    Tile& tile = tiles[tile_idx];
        //    // 重置瓦片缓冲区
        //    //std::fill(tile.zbuffer.begin(), tile.zbuffer.end(),
        //    //    std::numeric_limits<float>::max());
        //    tile.zbuffer.clear();
        //    //std::fill(tile.colors.begin(), tile.colors.end(), 0);
        //    // 渲染该tile中的所有三角形
        //    for (int tri_idx : triangle_buckets[tile_idx]) {
        //        // 这里直接使用你现有的renderTriangleInTile逻辑
        //        // 但是传入transformed_mesh而不是原始mesh
        //        renderTriangleFromSOA_InTile(transformed_mesh, light,
        //            ka, kd, tri_idx, tile);
        //    }
        //}

        for (int tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
            if (!triangle_buckets[tile_idx].empty()) {
                pool.enqueue([&, tile_idx]() {
                    Tile& tile = tiles[tile_idx];
                    // tile.zbuffer.clear();
                    // std::fill(tile.colors.begin(), tile.colors.end(), 0);
                    for (int tri_idx : triangle_buckets[tile_idx]) {
                        const triIndices& ind = transformed_mesh.triangles[tri_idx];
                        // clip out of the screen
                        float z0 = transformed_mesh.transformed_positions_z[ind.v[0]];
                        float z1 = transformed_mesh.transformed_positions_z[ind.v[1]];
                        float z2 = transformed_mesh.transformed_positions_z[ind.v[2]];
                        if (!(fabs(z0) > 1.0f || fabs(z1) > 1.0f || fabs(z2) > 1.0f)) {
                            renderTriangleFromSOA_InTile3(renderer, transformed_mesh, light, ka, kd, tri_idx, tile);
                        }
                    }
                    });
            }
        }
        pool.waitAll();

    }
    void renderScene_SoA_Optimized_MT(Renderer& renderer, ThreadPool& pool,
        Mesh_SoA& mesh, matrix& camera, Light& light, float ka, float kd) {

        const int vCount = (int)mesh.positions_x.size();
        const int tileCount = (int)tiles.size();

        // transformed mesh buffer
        Mesh_SoA_Transformed transformed_mesh;
        transformed_mesh.triangles = mesh.triangles;
        transformed_mesh.ka = ka;
        transformed_mesh.kd = kd;

        transformed_mesh.transformed_positions_x.resize(vCount);
        transformed_mesh.transformed_positions_y.resize(vCount);
        transformed_mesh.transformed_positions_z.resize(vCount);
        transformed_mesh.transformed_positions_w.resize(vCount);
        transformed_mesh.transformed_normals_x.resize(vCount);
        transformed_mesh.transformed_normals_y.resize(vCount);
        transformed_mesh.transformed_normals_z.resize(vCount);
        transformed_mesh.colors_r.resize(vCount);
        transformed_mesh.colors_g.resize(vCount);
        transformed_mesh.colors_b.resize(vCount);
        transformed_mesh.view_positions_x.resize(vCount);
        transformed_mesh.view_positions_y.resize(vCount);
        transformed_mesh.view_positions_z.resize(vCount);
        transformed_mesh.view_positions_w.resize(vCount);


        // mvp matirx
        matrix view = camera * mesh.world;
        matrix p = renderer.perspective * view;

        int thread_count = pool.getThreadCount();
        int chunkSize = (vCount + thread_count - 1) / thread_count;
        chunkSize = (chunkSize + 7) & ~7;

        // use mul thread to transform the vertexs
        for (int chunk_start = 0; chunk_start < vCount; chunk_start += chunkSize) {
            int chunk_end = std::min(chunk_start + chunkSize, vCount);
            int chunk_count = chunk_end - chunk_start;

            if (chunk_count > 0) {
                pool.enqueue([&, chunk_start, chunk_count]() {
                    transformed_mesh.transformBatchSIMD( p, view, mesh.world, mesh,
                        chunk_start, chunk_count, renderer.canvas.getWidth(), renderer.canvas.getHeight()
                    );
                    });
            }
        }
        pool.waitAll();


        // triangle binning to tiles
        const int triCount = (int)transformed_mesh.triangles.size();
        std::vector<std::vector<int>> triangle_buckets(tileCount);

        // create local buckets for each thread to avoid use lock
        std::vector<std::vector<std::vector<int>>> thread_local_buckets( 
            thread_count,
            std::vector<std::vector<int>>(tileCount) );

        // set the per thread triangle manully, avoid passing too few triangles 
        //const int tri_chunk = 1024;
        tri_chunk = triCount / thread_count;
        const int binning_threads = std::min(thread_count, (triCount + tri_chunk - 1) / tri_chunk);
       
        // binning
        for (int thread_id = 0; thread_id < binning_threads; ++thread_id) {
            pool.enqueue([&, thread_id]() {

                int start_tri = thread_id * tri_chunk;
                int end_tri = std::min(start_tri + tri_chunk, triCount);
                auto& local_buckets = thread_local_buckets[thread_id];

                for (int t = start_tri; t < end_tri; ++t) {
                    triIndices& idx = transformed_mesh.triangles[t];

                    // back culling
                    const vec4 v0_view(
                        transformed_mesh.view_positions_x[idx.v[0]],
                        transformed_mesh.view_positions_y[idx.v[0]],
                        transformed_mesh.view_positions_z[idx.v[0]],
                        transformed_mesh.view_positions_w[idx.v[0]]
                    );
                    const vec4 v1_view(
                        transformed_mesh.view_positions_x[idx.v[1]],
                        transformed_mesh.view_positions_y[idx.v[1]],
                        transformed_mesh.view_positions_z[idx.v[1]],
                        transformed_mesh.view_positions_w[idx.v[1]]
                    );
                    const vec4 v2_view(
                        transformed_mesh.view_positions_x[idx.v[2]],
                        transformed_mesh.view_positions_y[idx.v[2]],
                        transformed_mesh.view_positions_z[idx.v[2]],
                        transformed_mesh.view_positions_w[idx.v[2]]
                    );

                    vec4 e1 = v1_view - v0_view;
                    vec4 e2 = v2_view - v0_view;
                    vec4 view_normal = vec4::cross(e1, e2);
                    float dot_product = vec4::dot(view_normal, -v0_view);

                    if (dot_product >= 0.0f)
                        continue;

                    // comput bounding box
                    const float* pos_x = transformed_mesh.transformed_positions_x.data();
                    const float* pos_y = transformed_mesh.transformed_positions_y.data();
                    float min_x = std::min({ pos_x[idx.v[0]], pos_x[idx.v[1]], pos_x[idx.v[2]] });
                    float max_x = std::max({ pos_x[idx.v[0]], pos_x[idx.v[1]], pos_x[idx.v[2]] });
                    float min_y = std::min({ pos_y[idx.v[0]], pos_y[idx.v[1]], pos_y[idx.v[2]] });
                    float max_y = std::max({ pos_y[idx.v[0]], pos_y[idx.v[1]], pos_y[idx.v[2]] });

                    // clip to cancas
                    min_x = std::max(0.0f, min_x);
                    max_x = std::min((float)renderer.canvas.getWidth() - 1, max_x);
                    min_y = std::max(0.0f, min_y);
                    max_y = std::min((float)renderer.canvas.getHeight() - 1, max_y);

                    if (min_x > max_x || min_y > max_y) continue;

                    // get the tiles
                    int min_tx = int(std::floor(min_x)) / TILE_WIDTH;
                    int max_tx = int(std::ceil(max_x)) / TILE_WIDTH;
                    int min_ty = int(std::floor(min_y)) / TILE_HEIGHT;
                    int max_ty = int(std::ceil(max_y)) / TILE_HEIGHT;

                    min_tx = std::max(0, min_tx);
                    max_tx = std::min(tiles_x - 1, max_tx);
                    min_ty = std::max(0, min_ty);
                    max_ty = std::min(tiles_y - 1, max_ty);

                    // add to thread local bucket buffer
                    for (int ty = min_ty; ty <= max_ty; ++ty) {
                        for (int tx = min_tx; tx <= max_tx; ++tx) {
                            int tile_idx = ty * tiles_x + tx;
                            local_buckets[tile_idx].push_back(t);
                        }
                    }
                }
                });
        }
        pool.waitAll();
        
        // merge the bucket
        for (int thread_id = 0; thread_id < binning_threads; ++thread_id) {
            for (int tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
                if (!thread_local_buckets[thread_id][tile_idx].empty()) {
                    triangle_buckets[tile_idx].insert(
                        triangle_buckets[tile_idx].end(),
                        thread_local_buckets[thread_id][tile_idx].begin(),
                        thread_local_buckets[thread_id][tile_idx].end()
                    );
                }
            }
        }

        // draw
        for (int tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
            if (!triangle_buckets[tile_idx].empty()) {
                pool.enqueue([&, tile_idx]() {
                    Tile& tile = tiles[tile_idx];
                    // tile.zbuffer.clear();
                    // std::fill(tile.colors.begin(), tile.colors.end(), 0);
                    for (int tri_idx : triangle_buckets[tile_idx]) {
                        const triIndices& ind = transformed_mesh.triangles[tri_idx];
                        // clip out of the screen
                        float z0 = transformed_mesh.transformed_positions_z[ind.v[0]];
                        float z1 = transformed_mesh.transformed_positions_z[ind.v[1]];
                        float z2 = transformed_mesh.transformed_positions_z[ind.v[2]];
                        if (!(fabs(z0) > 1.0f || fabs(z1) > 1.0f || fabs(z2) > 1.0f)) {
                            renderTriangleFromSOA_InTile3( renderer, transformed_mesh, light, ka, kd, tri_idx, tile );
                        }
                    }
                    });
            }
        }
        pool.waitAll();
    }
    void renderScene_SoA_Optimized_ST(Renderer& renderer,
        Mesh_SoA& mesh, matrix& camera, Light& light, float ka, float kd) {

        const int vCount = (int)mesh.positions_x.size();
        const int tileCount = (int)tiles.size();

        Mesh_SoA_Transformed transformed_mesh;
        transformed_mesh.triangles = mesh.triangles;
        transformed_mesh.ka = ka;
        transformed_mesh.kd = kd;
        transformed_mesh.transformed_positions_x.resize(vCount);
        transformed_mesh.transformed_positions_y.resize(vCount);
        transformed_mesh.transformed_positions_z.resize(vCount);
        transformed_mesh.transformed_positions_w.resize(vCount);
        transformed_mesh.transformed_normals_x.resize(vCount);
        transformed_mesh.transformed_normals_y.resize(vCount);
        transformed_mesh.transformed_normals_z.resize(vCount);
        transformed_mesh.colors_r.resize(vCount);
        transformed_mesh.colors_g.resize(vCount);
        transformed_mesh.colors_b.resize(vCount);
        transformed_mesh.view_positions_x.resize(vCount);
        transformed_mesh.view_positions_y.resize(vCount);
        transformed_mesh.view_positions_z.resize(vCount);
        transformed_mesh.view_positions_w.resize(vCount);

        matrix view = camera * mesh.world;
        matrix p = renderer.perspective * view;

        transformed_mesh.transformBatchSIMD(p, view, mesh.world, mesh,
            0, vCount,
            renderer.canvas.getWidth(), renderer.canvas.getHeight());

        const int triCount = (int)transformed_mesh.triangles.size();
        std::vector<std::vector<int>> triangle_buckets(tileCount);

        // binning
        for (int t = 0; t < triCount; ++t) {
            triIndices& idx = transformed_mesh.triangles[t];

            const vec4 v0_view(
                transformed_mesh.view_positions_x[idx.v[0]],
                transformed_mesh.view_positions_y[idx.v[0]],
                transformed_mesh.view_positions_z[idx.v[0]],
                transformed_mesh.view_positions_w[idx.v[0]]
            );
            const vec4 v1_view(
                transformed_mesh.view_positions_x[idx.v[1]],
                transformed_mesh.view_positions_y[idx.v[1]],
                transformed_mesh.view_positions_z[idx.v[1]],
                transformed_mesh.view_positions_w[idx.v[1]]
            );
            const vec4 v2_view(
                transformed_mesh.view_positions_x[idx.v[2]],
                transformed_mesh.view_positions_y[idx.v[2]],
                transformed_mesh.view_positions_z[idx.v[2]],
                transformed_mesh.view_positions_w[idx.v[2]]
            );
            vec4 e1 = v1_view - v0_view;
            vec4 e2 = v2_view - v0_view;

            vec4 view_normal = vec4::cross(e1, e2);
            float dot_product = vec4::dot(view_normal, -v0_view);
            if (dot_product >= 0.0f)
                continue;

            const float* pos_x = transformed_mesh.transformed_positions_x.data();
            const float* pos_y = transformed_mesh.transformed_positions_y.data();
            float min_x = std::min({ pos_x[idx.v[0]], pos_x[idx.v[1]], pos_x[idx.v[2]] });
            float max_x = std::max({ pos_x[idx.v[0]], pos_x[idx.v[1]], pos_x[idx.v[2]] });
            float min_y = std::min({ pos_y[idx.v[0]], pos_y[idx.v[1]], pos_y[idx.v[2]] });
            float max_y = std::max({ pos_y[idx.v[0]], pos_y[idx.v[1]], pos_y[idx.v[2]] });

            min_x = std::max(0.0f, min_x);
            max_x = std::min((float)renderer.canvas.getWidth() - 1, max_x);
            min_y = std::max(0.0f, min_y);
            max_y = std::min((float)renderer.canvas.getHeight() - 1, max_y);
            if (min_x > max_x || min_y > max_y) continue;

            int min_tx = int(std::floor(min_x)) / TILE_WIDTH;
            int max_tx = int(std::ceil(max_x)) / TILE_WIDTH;
            int min_ty = int(std::floor(min_y)) / TILE_HEIGHT;
            int max_ty = int(std::ceil(max_y)) / TILE_HEIGHT;

            min_tx = std::max(0, min_tx);
            max_tx = std::min(tiles_x - 1, max_tx);
            min_ty = std::max(0, min_ty);
            max_ty = std::min(tiles_y - 1, max_ty);

            for (int ty = min_ty; ty <= max_ty; ++ty) {
                for (int tx = min_tx; tx <= max_tx; ++tx) {
                    int tile_idx = ty * tiles_x + tx;
                    triangle_buckets[tile_idx].push_back(t);
                }
            }
        }

        for (int tile_idx = 0; tile_idx < tileCount; ++tile_idx) {
            if (triangle_buckets[tile_idx].empty()) continue;
            Tile& tile = tiles[tile_idx];
            // tile.zbuffer.clear();
            // std::fill(tile.colors.begin(), tile.colors.end(), 0);
            for (int tri_idx : triangle_buckets[tile_idx]) {
                const triIndices& ind = transformed_mesh.triangles[tri_idx];
                float z0 = transformed_mesh.transformed_positions_z[ind.v[0]];
                float z1 = transformed_mesh.transformed_positions_z[ind.v[1]];
                float z2 = transformed_mesh.transformed_positions_z[ind.v[2]];
                if (!(fabs(z0) > 1.0f || fabs(z1) > 1.0f || fabs(z2) > 1.0f)) {
                    renderTriangleFromSOA_InTile3(
                        renderer, transformed_mesh, light, ka, kd,
                        tri_idx, tile
                    );
                }
            }
        }
    }

    private:
        static vec4 makeEdge(const float& v0x, const float& v0y, const float& v1x, const float& v1y) {
            vec4 e;
            e.x = v0y - v1y;
            e.y = v1x - v0x;
            e.z = v0x * v1y - v1x * v0y;
            return e;
        }

        void renderTriangleFromSOA_InTile3(Renderer& renderer, const Mesh_SoA_Transformed& mesh, Light& light, float ka, float kd,
            int triangle_idx, Tile& tile) {
            // load vertex data 
            const triIndices& idx = mesh.triangles[triangle_idx];

            // load vertex
            float v0_x = mesh.transformed_positions_x[idx.v[0]];
            float v1_x = mesh.transformed_positions_x[idx.v[1]];
            float v2_x = mesh.transformed_positions_x[idx.v[2]];

            float v0_y = mesh.transformed_positions_y[idx.v[0]];
            float v1_y = mesh.transformed_positions_y[idx.v[1]];
            float v2_y = mesh.transformed_positions_y[idx.v[2]];

            float v0_z = mesh.transformed_positions_z[idx.v[0]];
            float v1_z = mesh.transformed_positions_z[idx.v[1]];
            float v2_z = mesh.transformed_positions_z[idx.v[2]];

            // validity check
            if (!isfinite(v0_x) || !isfinite(v0_y) || !isfinite(v0_z) ||
                !isfinite(v1_x) || !isfinite(v1_y) || !isfinite(v1_z) ||
                !isfinite(v2_x) || !isfinite(v2_y) || !isfinite(v2_z)) {
                return;
            }

            // Calculate the area of ​​a triangle (for barycenter interpolation).
            vec4 e0 = makeEdge(v1_x, v1_y, v2_x, v2_y);
            vec4 e1 = makeEdge(v2_x, v2_y, v0_x, v0_y);
            vec4 e2 = makeEdge(v0_x, v0_y, v1_x, v1_y);

            float area = v0_x * e0.x + v0_y * e0.y + e0.z;
            if (area < 1e-6f) return;

            float invArea = 1.0f / area;

            // load the transformed normal data
            float n0_x = mesh.transformed_normals_x[idx.v[0]];
            float n0_y = mesh.transformed_normals_y[idx.v[0]];
            float n0_z = mesh.transformed_normals_z[idx.v[0]];

            float n1_x = mesh.transformed_normals_x[idx.v[1]];
            float n1_y = mesh.transformed_normals_y[idx.v[1]];
            float n1_z = mesh.transformed_normals_z[idx.v[1]];

            float n2_x = mesh.transformed_normals_x[idx.v[2]];
            float n2_y = mesh.transformed_normals_y[idx.v[2]];
            float n2_z = mesh.transformed_normals_z[idx.v[2]];

            // load color
            float c0_r = mesh.colors_r[idx.v[0]];
            float c0_g = mesh.colors_g[idx.v[0]];
            float c0_b = mesh.colors_b[idx.v[0]];

            float c1_r = mesh.colors_r[idx.v[1]];
            float c1_g = mesh.colors_g[idx.v[1]];
            float c1_b = mesh.colors_b[idx.v[1]];

            float c2_r = mesh.colors_r[idx.v[2]];
            float c2_g = mesh.colors_g[idx.v[2]];
            float c2_b = mesh.colors_b[idx.v[2]];

            // compute increment
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

            float min_x = std::max((float)tile.x, std::min({ v0_x, v1_x, v2_x }));
            float max_x = std::min((float)(tile.x + tile.width), std::max({ v0_x, v1_x, v2_x }));
            float min_y = std::max((float)tile.y, std::min({ v0_y, v1_y, v2_y }));
            float max_y = std::min((float)(tile.y + tile.height), std::max({ v0_y, v1_y, v2_y }));

            if (min_x > max_x || min_y > max_y) return;

            int tile_min_x = (int)std::floor(min_x) - tile.x;
            int tile_max_x = (int)std::ceil(max_x) - tile.x;
            int tile_min_y = (int)std::floor(min_y) - tile.y;
            int tile_max_y = (int)std::ceil(max_y) - tile.y;

            tile_min_x = std::max(0, tile_min_x);
            tile_max_x = std::min(tile.width - 1, tile_max_x);
            tile_min_y = std::max(0, tile_min_y);
            tile_max_y = std::min(tile.height - 1, tile_max_y);

            if (tile_min_x > tile_max_x || tile_min_y > tile_max_y) return;

            int global_min_x = tile.x + tile_min_x;
            int global_max_x = tile.x + tile_max_x;
            int global_min_y = tile.y + tile_min_y;
            int global_max_y = tile.y + tile_max_y;

            // --- AVX2常量 ---
            const __m256 zero = _mm256_setzero_ps();
            const __m256 lane = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
            const __m256 _255 = _mm256_set1_ps(255.0f);
            const __m256 _0001 = _mm256_set1_ps(0.001f);
            const __m256 _01 = _mm256_set1_ps(0.1f);
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
            const float start_x = (float)global_min_x;
            float global_y0 = (float)global_min_y;

            float w0_row = e0.x * start_x + e0.y * global_y0 + e0.z;
            float w1_row = e1.x * start_x + e1.y * global_y0 + e1.z;
            float w2_row = e2.x * start_x + e2.y * global_y0 + e2.z;

            float z_row = (v0_z * w0_row + v1_z * w1_row + v2_z * w2_row) * invArea;
            float nx_row = (n0_x * w0_row + n1_x * w1_row + n2_x * w2_row) * invArea;
            float ny_row = (n0_y * w0_row + n1_y * w1_row + n2_y * w2_row) * invArea;
            float nz_row = (n0_z * w0_row + n1_z * w1_row + n2_z * w2_row) * invArea;
            float cr_row = (c0_r * w0_row + c1_r * w1_row + c2_r * w2_row) * invArea;
            float cg_row = (c0_g * w0_row + c1_g * w1_row + c2_g * w2_row) * invArea;
            float cb_row = (c0_b * w0_row + c1_b * w1_row + c2_b * w2_row) * invArea;

            int canvas_width = renderer.canvas.getWidth();
            int canvas_height = renderer.canvas.getHeight();

            for (int y = global_min_y; y <= global_max_y; ++y)
            {

                if (y < 0 || y >= canvas_height) {
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
                    continue;
                }

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

                int x = global_min_x;

                for (; x <= global_max_x - 7; x += 8)
                {
                    // check x range
                    if (x < 0 || x + 7 >= canvas_width) {
                        break; 
                    }

                    __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
                    __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
                    __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
                    __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
                    inside = _mm256_and_ps(inside, w2v_zero);

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

                    __m256 zbuf;
                    if (x >= 0 && x + 8 <= canvas_width) {
                        zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));
                    }
                    else {
                        alignas(32) float zb[8];
                        for (int i = 0; i < 8; ++i) {
                            int px = x + i;
                            zb[i] = (px >= 0 && px < canvas_width)
                                ? renderer.zbuffer(px, y)
                                : std::numeric_limits<float>::infinity();
                        }
                        zbuf = _mm256_load_ps(zb);
                    }

                    __m256 zv_001 = _mm256_cmp_ps(zv, _0001, _CMP_GE_OQ);
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

                    if (final_mask == 0xFF)
                    {
                        alignas(32) float rr[8], gg[8], bb[8];
                        _mm256_store_ps(rr, r);
                        _mm256_store_ps(gg, g);
                        _mm256_store_ps(bb, b);

                        if (x >= 0 && x + 8 <= canvas_width) {
                            _mm256_storeu_ps(&renderer.zbuffer(x, y), zv);
                        }
                        else {
                            alignas(32) float zz[8];
                            _mm256_store_ps(zz, zv);
                            for (int i = 0; i < 8; ++i) {
                                int px = x + i;
                                if (px >= 0 && px < canvas_width) {
                                    renderer.zbuffer(px, y) = zz[i];
                                }
                            }
                        }
                        // draw
                        //for (int i = 0; i < 8; i += 4)
                        //{
                        //    renderer.canvas.draw(x + i, y, (unsigned char)(rr[i]), (unsigned char)(gg[i]), (unsigned char)(bb[i]));
                        //    renderer.canvas.draw(x + 1 + i, y, (unsigned char)(rr[i + 1]), (unsigned char)(gg[i + 1]), (unsigned char)(bb[i + 1]));
                        //    renderer.canvas.draw(x + 2 + i, y, (unsigned char)(rr[i + 2]), (unsigned char)(gg[i + 2]), (unsigned char)(bb[i + 2]));
                        //    renderer.canvas.draw(x + 3 + i, y, (unsigned char)(rr[i + 3]), (unsigned char)(gg[i + 3]), (unsigned char)(bb[i + 3]));
                        //}
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
                            int i = _tzcnt_u32(m);
                            m &= m - 1;

                            int px = x + i;
                            if (px >= 0 && px < canvas_width) {
                                renderer.canvas.draw(
                                    px, y,
                                    (unsigned char)(rr[i]),
                                    (unsigned char)(gg[i]),
                                    (unsigned char)(bb[i])
                                );
                                renderer.zbuffer(px, y) = zz[i];
                            }
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

                // tail
                int tail_count = global_max_x - x + 1;
                if (tail_count > 0 && x >= 0 && x < canvas_width) {
                    __m256i lane_i = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256 tailMask =
                        _mm256_castsi256_ps(
                            _mm256_cmpgt_epi32(
                                _mm256_set1_epi32(tail_count),
                                lane_i));

                    __m256 w0v_zero = _mm256_cmp_ps(w0v, zero, _CMP_GE_OQ);
                    __m256 w1v_zero = _mm256_cmp_ps(w1v, zero, _CMP_GE_OQ);
                    __m256 w2v_zero = _mm256_cmp_ps(w2v, zero, _CMP_GE_OQ);
                    __m256 inside = _mm256_and_ps(w0v_zero, w1v_zero);
                    inside = _mm256_and_ps(inside, w2v_zero);

                    inside = _mm256_and_ps(inside, tailMask);

                    int mask = _mm256_movemask_ps(inside);
                    if (mask != 0)
                    {
                        alignas(32) float zb[8];
                        for (int i = 0; i < 8; ++i) {
                            int px = x + i;
                            zb[i] = (px >= 0 && px < canvas_width && px <= global_max_x)
                                ? renderer.zbuffer(px, y)
                                : std::numeric_limits<float>::infinity();
                        }
                        __m256 zbuf = _mm256_load_ps(zb);

                        __m256 zv_001 = _mm256_cmp_ps(zv, _0001, _CMP_GE_OQ);
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

                            int m = final_mask;
                            while (m)
                            {
                                int i = _tzcnt_u32(m);
                                m &= m - 1;
                                int px = x + i;
                                if (px >= 0 && px < canvas_width && px <= global_max_x)
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

};