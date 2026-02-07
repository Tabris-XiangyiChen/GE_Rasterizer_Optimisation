#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>

#include <cmath>
//#include "matrix.h"
#include "Types.h"
#include "colour.h"
#include "mesh.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"
//#include "tile.h"
#include "raster_SOA.h"

#include <thread>
#include <atomic>


// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, bool if_AVX = false) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;
    //// Opt:
    //L.omega_i.normalise();

    //// Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        if (vec4::dot(mesh->world * mesh->vertices[ind.v[0]].normal, mesh->world * mesh->vertices[ind.v[0]].p - vec4(0.0f, 0.0f, -camera.data.a[11], 1.0f)) >= 0.0f) continue;
        
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

        if(if_AVX)
            tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
        else
            tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}

// SoA struture
void render_SoA(Renderer& renderer, Mesh_SoA* mesh, matrix& camera, Light& L, bool if_AVX = false) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;

    L.omega_i.normalise();

    const int screenW = renderer.canvas.getWidth();
    const int screenH = renderer.canvas.getHeight();

    // Iterate through all triangles
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3];

        for (int i = 0; i < 3; i++) {
            int idx = ind.v[i];

            // Load vertex position from SoA
            vec4 pos(mesh->positions_x[idx], mesh->positions_y[idx], mesh->positions_z[idx], mesh->positions_w[idx]);

            // Apply combined transformation
            t[i].p = p * pos;
            t[i].p.divideW();

            // Transform normal into world space
            vec4 normal(mesh->normals_x[idx], mesh->normals_y[idx], mesh->normals_z[idx], 0.f);
            t[i].normal = mesh->world * normal;
            t[i].normal.normalise();

            // Map to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * float(screenW);
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * float(screenH);
            t[i].p[1] = float(screenH) - t[i].p[1]; // invert Y

            // Copy vertex color from SoA
            t[i].rgb.set(mesh->colors_r[idx], mesh->colors_g[idx], mesh->colors_b[idx]);
        }

        // Clip triangle
        if (fabs(t[0].p[2]) > 1.f || fabs(t[1].p[2]) > 1.f || fabs(t[2].p[2]) > 1.f)
            continue;

        // Draw triangle
        triangle tri(t[0], t[1], t[2]);
        if (if_AVX)
            tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
        else
            tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}

// transform trangles first
void render_transfirst(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, bool if_AVX = false) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix view = camera * mesh->world;
    matrix p = renderer.perspective * view;

    L.omega_i.normalise();
    
    const int vCount = (int)mesh->vertices.size();
    const int triCount = (int)mesh->triangles.size();

    // Vertex transform cache
    std::vector<Vertex> vsCache(vCount);
    // view space postion for back culling
    std::vector<vec4> view_positions(vCount);
    // Transform vertexs first
    for (int i = 0; i < vCount; ++i)
    {
        Vertex& out = vsCache[i];
        const Vertex& in = mesh->vertices[i];

        // view pos
        view_positions[i] = view * in.p;

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

    // Iterate through all triangles in the mesh
    for (unsigned int triIdx = 0; triIdx < triCount; ++triIdx) {
        triIndices& ind = mesh->triangles[triIdx];

        vec4& v0_view = view_positions[ind.v[0]];
        vec4& v1_view = view_positions[ind.v[1]];
        vec4& v2_view = view_positions[ind.v[2]];
        vec4 e1 = v1_view - v0_view;
        vec4 e2 = v2_view - v0_view;
        vec4 view_normal(
            e1[1] * e2[2] - e1[2] * e2[1],  // x
            e1[2] * e2[0] - e1[0] * e2[2],  // y  
            e1[0] * e2[1] - e1[1] * e2[0],  // z
            0.0f
        );
        if (vec4::dot(view_normal, -v0_view) >= 0.0f)
            continue;
        // clip
        if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.0f || fabs(vsCache[ind.v[1]].p[2]) > 1.0f || fabs(vsCache[ind.v[2]].p[2]) > 1.0f))
        {
            triangle tri(vsCache[ind.v[0]], vsCache[ind.v[1]], vsCache[ind.v[2]]);
            if (if_AVX)
                tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
            else
                tri.draw(renderer, L, mesh->ka, mesh->kd);
        }
    }
}

void render_transfirst_SoA_Scalar(Renderer& renderer, Mesh_SoA* mesh, matrix& camera, Light& L, bool if_AVX = false) {
    matrix view = camera * mesh->world;
    matrix p = renderer.perspective * view;

    L.omega_i.normalise();

    const int vCount = (int)mesh->positions_x.size();
    const int triCount = (int)mesh->triangles.size();

    // Vertex cache (SoA -> AOS for temp triangle)
    std::vector<Vertex> vsCache(vCount);
    std::vector<vec4> view_positions(vCount);

    // Transform all vertices first
    for (int i = 0; i < vCount; ++i) {
        Vertex& out = vsCache[i];

        // Load position from SoA
        vec4 pos(mesh->positions_x[i], mesh->positions_y[i], mesh->positions_z[i], mesh->positions_w[i]);

        view_positions[i] = view * pos;

        out.p = p * pos;
        out.p.divideW();

        // Map to screen space
        out.p[0] = (out.p[0] + 1.f) * 0.5f * renderer.canvas.getWidth();
        out.p[1] = (out.p[1] + 1.f) * 0.5f * renderer.canvas.getHeight();
        out.p[1] = float(renderer.canvas.getHeight()) - out.p[1];

        // Transform normal
        vec4 normal(mesh->normals_x[i], mesh->normals_y[i], mesh->normals_z[i], 0.f);
        out.normal = mesh->world * normal;
        out.normal.normalise();

        // Color
        out.rgb.set(mesh->colors_r[i], mesh->colors_g[i], mesh->colors_b[i]);
    }

    // Draw triangles
    for (int triIdx = 0; triIdx < triCount; ++triIdx) {
        triIndices& ind = mesh->triangles[triIdx];
        // back culling
        vec4& v0_view = view_positions[ind.v[0]];
        vec4& v1_view = view_positions[ind.v[1]];
        vec4& v2_view = view_positions[ind.v[2]];
        vec4 e1 = v1_view - v0_view;
        vec4 e2 = v2_view - v0_view;
        vec4 view_normal(
            e1[1] * e2[2] - e1[2] * e2[1],  // x
            e1[2] * e2[0] - e1[0] * e2[2],  // y  
            e1[0] * e2[1] - e1[1] * e2[0],  // z
            0.0f
        );
        if (vec4::dot(view_normal, -v0_view) >= 0.0f)
            continue;

        if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.0f ||
            fabs(vsCache[ind.v[1]].p[2]) > 1.0f ||
            fabs(vsCache[ind.v[2]].p[2]) > 1.0f)) {

            triangle tri(vsCache[ind.v[0]], vsCache[ind.v[1]], vsCache[ind.v[2]]);
            if (if_AVX)
                tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
            else
                tri.draw(renderer, L, mesh->ka, mesh->kd);
        }
    }
}

void render_transfirst_SoA_AVX2_Optimized(Renderer& renderer, Mesh_SoA* mesh, matrix& camera, Light& L, bool if_AVX = false) {

    matrix view = camera * mesh->world;
    matrix p = renderer.perspective * view;

    L.omega_i.normalise();

    const int vCount = (int)mesh->positions_x.size();
    const int triCount = (int)mesh->triangles.size();

    std::vector<Vertex> vsCache(vCount);
    // world space postion for back culling
    std::vector<vec4> view_positions(vCount);

    const float width = float(renderer.canvas.getWidth());
    const float height = float(renderer.canvas.getHeight());
    const __m256 halfWidth = _mm256_set1_ps(0.5f * width);
    const __m256 halfHeight = _mm256_set1_ps(0.5f * height);
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 canvasHeight = _mm256_set1_ps(height);

    // matrix value for SIMD
    const __m256 view_m0 = _mm256_set1_ps(view.data.a[0]);
    const __m256 view_m1 = _mm256_set1_ps(view.data.a[1]);
    const __m256 view_m2 = _mm256_set1_ps(view.data.a[2]);
    const __m256 view_m3 = _mm256_set1_ps(view.data.a[3]);
    const __m256 view_m4 = _mm256_set1_ps(view.data.a[4]);
    const __m256 view_m5 = _mm256_set1_ps(view.data.a[5]);
    const __m256 view_m6 = _mm256_set1_ps(view.data.a[6]);
    const __m256 view_m7 = _mm256_set1_ps(view.data.a[7]);
    const __m256 view_m8 = _mm256_set1_ps(view.data.a[8]);
    const __m256 view_m9 = _mm256_set1_ps(view.data.a[9]);
    const __m256 view_m10 = _mm256_set1_ps(view.data.a[10]);
    const __m256 view_m11 = _mm256_set1_ps(view.data.a[11]);
    const __m256 view_m12 = _mm256_set1_ps(view.data.a[12]);
    const __m256 view_m13 = _mm256_set1_ps(view.data.a[13]);
    const __m256 view_m14 = _mm256_set1_ps(view.data.a[14]);
    const __m256 view_m15 = _mm256_set1_ps(view.data.a[15]);

    const float* m = p.data.a;
    const __m256 p00 = _mm256_set1_ps(m[0]);
    const __m256 p01 = _mm256_set1_ps(m[1]);
    const __m256 p02 = _mm256_set1_ps(m[2]);
    const __m256 p03 = _mm256_set1_ps(m[3]);
    const __m256 p10 = _mm256_set1_ps(m[4]);
    const __m256 p11 = _mm256_set1_ps(m[5]);
    const __m256 p12 = _mm256_set1_ps(m[6]);
    const __m256 p13 = _mm256_set1_ps(m[7]);
    const __m256 p20 = _mm256_set1_ps(m[8]);
    const __m256 p21 = _mm256_set1_ps(m[9]);
    const __m256 p22 = _mm256_set1_ps(m[10]);
    const __m256 p23 = _mm256_set1_ps(m[11]);
    const __m256 p30 = _mm256_set1_ps(m[12]);
    const __m256 p31 = _mm256_set1_ps(m[13]);
    const __m256 p32 = _mm256_set1_ps(m[14]);
    const __m256 p33 = _mm256_set1_ps(m[15]);

    const float* w = mesh->world.data.a;
    __m256 w00 = _mm256_set1_ps(w[0]);
    __m256 w01 = _mm256_set1_ps(w[1]);
    __m256 w02 = _mm256_set1_ps(w[2]);
    __m256 w10 = _mm256_set1_ps(w[4]);
    __m256 w11 = _mm256_set1_ps(w[5]);
    __m256 w12 = _mm256_set1_ps(w[6]);
    __m256 w20 = _mm256_set1_ps(w[8]);
    __m256 w21 = _mm256_set1_ps(w[9]);
    __m256 w22 = _mm256_set1_ps(w[10]);

    int i = 0;
    for (; i + 7 < vCount; i += 8) {
        __m256 vx = _mm256_load_ps(&mesh->positions_x[i]);
        __m256 vy = _mm256_load_ps(&mesh->positions_y[i]);
        __m256 vz = _mm256_load_ps(&mesh->positions_z[i]);
        __m256 vw = _mm256_load_ps(&mesh->positions_w[i]);

        // view position
        __m256 viewX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(view_m0, vx),
                _mm256_mul_ps(view_m1, vy)),
            _mm256_add_ps(_mm256_mul_ps(view_m2, vz),
                _mm256_mul_ps(view_m3, vw))
        );
        __m256 viewY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(view_m4, vx),
                _mm256_mul_ps(view_m5, vy)),
            _mm256_add_ps(_mm256_mul_ps(view_m6, vz),
                _mm256_mul_ps(view_m7, vw))
        );
        __m256 viewZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(view_m8, vx),
                _mm256_mul_ps(view_m9, vy)),
            _mm256_add_ps(_mm256_mul_ps(view_m10, vz),
                _mm256_mul_ps(view_m11, vw))
        );
        __m256 viewW = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(view_m12, vx),
                _mm256_mul_ps(view_m13, vy)),
            _mm256_add_ps(_mm256_mul_ps(view_m14, vz),
                _mm256_mul_ps(view_m15, vw))
        );

        float viewX8[8], viewY8[8], viewZ8[8], viewW8[8];
        _mm256_store_ps(viewX8, viewX);
        _mm256_store_ps(viewY8, viewY);
        _mm256_store_ps(viewZ8, viewZ);
        _mm256_store_ps(viewW8, viewW);
        for (int j = 0; j < 8; ++j) {
            view_positions[i + j] = vec4(viewX8[j], viewY8[j], viewZ8[j], viewW8[j]);
        }


        __m256 outX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p00, vx),
                _mm256_mul_ps(p01, vy)),
            _mm256_add_ps(_mm256_mul_ps(p02, vz),
                _mm256_mul_ps(p03, vw))
        );
        __m256 outY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p10, vx),
                _mm256_mul_ps(p11, vy)),
            _mm256_add_ps(_mm256_mul_ps(p12, vz),
                _mm256_mul_ps(p13, vw))
        );
        __m256 outZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p20, vx),
                _mm256_mul_ps(p21, vy)),
            _mm256_add_ps(_mm256_mul_ps(p22, vz),
                _mm256_mul_ps(p23, vw))
        );
        __m256 outW = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p30, vx),
                _mm256_mul_ps(p31, vy)),
            _mm256_add_ps(_mm256_mul_ps(p32, vz),
                _mm256_mul_ps(p33, vw))
        );

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

        // normal matrix transformation
        __m256 nx = _mm256_load_ps(&mesh->normals_x[i]);
        __m256 ny = _mm256_load_ps(&mesh->normals_y[i]);
        __m256 nz = _mm256_load_ps(&mesh->normals_z[i]);

        __m256 outNX = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(w00, nx),
                _mm256_mul_ps(w01, ny)),
            _mm256_mul_ps(w02, nz)
        );
        __m256 outNY = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(w10, nx),
                _mm256_mul_ps(w11, ny)),
            _mm256_mul_ps(w12, nz)
        );
        __m256 outNZ = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(w20, nx),
                _mm256_mul_ps(w21, ny)),
            _mm256_mul_ps(w22, nz)
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

        // store in vertex cache
        for (int j = 0; j < 8; ++j) {
            Vertex& out = vsCache[i + j];
            out.p = vec4(((float*)&outX)[j], ((float*)&outY)[j], ((float*)&outZ)[j], 1.f);
            out.normal = vec4(((float*)&outNX)[j], ((float*)&outNY)[j], ((float*)&outNZ)[j], 0.f);
            out.rgb.set(mesh->colors_r[i + j], mesh->colors_g[i + j], mesh->colors_b[i + j]);
        }
    }

    // last vertexs
    for (; i < vCount; ++i) {
        Vertex& out = vsCache[i];
        vec4 pos(mesh->positions_x[i], mesh->positions_y[i], mesh->positions_z[i], mesh->positions_w[i]);

        view_positions[i] = view * pos;
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

    for (int triIdx = 0; triIdx < triCount; ++triIdx) {
        triIndices& ind = mesh->triangles[triIdx];
        // back culling
        vec4& v0_view = view_positions[ind.v[0]];
        vec4& v1_view = view_positions[ind.v[1]];
        vec4& v2_view = view_positions[ind.v[2]];
        vec4 e1 = v1_view - v0_view;
        vec4 e2 = v2_view - v0_view;
        vec4 view_normal = vec4::cross(e1, e2);
        if (vec4::dot(view_normal, -v0_view) >= 0.0f)
            continue;

        if (!(fabs(vsCache[ind.v[0]].p[2]) > 1.f ||
            fabs(vsCache[ind.v[1]].p[2]) > 1.f ||
            fabs(vsCache[ind.v[2]].p[2]) > 1.f)) {

            triangle tri(vsCache[ind.v[0]], vsCache[ind.v[1]], vsCache[ind.v[2]]);
            if (if_AVX)
                tri.draw_AVX2_Optimized3(renderer, L, mesh->ka, mesh->kd);
            else
                tri.draw(renderer, L, mesh->ka, mesh->kd);
        }
    }
}


// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // camera is just a matrix
    matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
   // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        // Render each object in the scene
        for (auto& m : scene)
            render(renderer, m, camera, L);

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1(bool if_trans_first = false, bool if_AVX = false) {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // Opt:
    L.omega_i.normalise();
    bool running = true;

    std::vector<Mesh*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh();
        *m = Mesh::makeCube(1.f);
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

        for (auto& m : scene)
        {
            if (if_trans_first)
            {
                if(if_AVX)
                    render_transfirst(renderer, m, camera, L, true);
                else
                    render_transfirst(renderer, m, camera, L, false);
            }
            else
            {
                if (if_AVX)
                    render(renderer, m, camera, L, true);
                else
                    render(renderer, m, camera, L, false);
            }
        }

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

void scene1_SoA(bool if_trans_first = false, bool if_AVX = false, bool if_AVX_at_transf = false) {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // Opt:
    L.omega_i.normalise();
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

        for (auto& m : scene)
        {
            if (if_trans_first)
            {
                if (if_AVX)
                    if (if_AVX_at_transf)
                        render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L, true);
                    else
                        render_transfirst_SoA_Scalar(renderer, m, camera, L, true);
                else
                    if (if_AVX_at_transf)
                        render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L, false);
                    else
                        render_transfirst_SoA_Scalar(renderer, m, camera, L, false);
            }
            else
            {
                if (if_AVX)
                    render_SoA(renderer, m, camera, L, true);
                else
                    render_SoA(renderer, m, camera, L, false);
            }
        }

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

void scene1_SoA_MT() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // Opt:
    L.omega_i.normalise();

    ThreadPool threads;
    //RendererSOA_MT renderer_mt(renderer.canvas.getWidth(), renderer.canvas.getHeight(), std::thread::hardware_concurrency());
    RendererSOA_MT renderer_mt(renderer.canvas.getWidth(), renderer.canvas.getHeight());


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

        for (auto& m : scene) {

            renderer_mt.renderScene_SoA_Optimized(renderer, threads, *m, camera, L, m->ka, m->kd);
            //renderer_mt.renderScene_SoA_Optimized_MT(renderer, threads, *m, camera, L, m->ka, m->kd);
            //renderer_mt.renderScene_SoA_Optimized_ST(renderer, *m, camera, L, m->ka, m->kd);

        }

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2(bool if_trans_first = false, bool if_AVX = false) {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // Opt:
    L.omega_i.normalise();

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
        {
            if (if_trans_first)
            {
                if (if_AVX)
                    render_transfirst(renderer, m, camera, L, true);
                else
                    render_transfirst(renderer, m, camera, L, false);
            }
            else
            {
                if (if_AVX)
                    render(renderer, m, camera, L, true);
                else
                    render(renderer, m, camera, L, false);
            }
        }

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

void scene2_SoA(bool if_trans_first = false, bool if_AVX = false, bool if_AVX_at_transf = false) {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // Opt:
    L.omega_i.normalise();

    std::vector<Mesh_SoA*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh_SoA* m = new Mesh_SoA();
            *m = Mesh_SoA::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh_SoA* sphere = new Mesh_SoA();
    *sphere = Mesh_SoA::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
        {
            if (if_trans_first)
            {
                if (if_AVX)
                    if (if_AVX_at_transf)
                        render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L, true);
                    else
                        render_transfirst_SoA_Scalar(renderer, m, camera, L, true);
                else
                    if (if_AVX_at_transf)
                        render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L, false);
                    else
                        render_transfirst_SoA_Scalar(renderer, m, camera, L, false);
            }
            else
            {
                if (if_AVX)
                    render_SoA(renderer, m, camera, L, true);
                else
                    render_SoA(renderer, m, camera, L, false);
            }
        }
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}
void scene2_SoA_MT(bool if_trans_first = false, bool if_AVX = false, bool if_AVX_at_transf = false) {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    // Opt:
    L.omega_i.normalise();

    std::vector<Mesh_SoA*> scene;

    ThreadPool threads;
    RendererSOA_MT renderer_mt(renderer.canvas.getWidth(), renderer.canvas.getHeight());

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh_SoA* m = new Mesh_SoA();
            *m = Mesh_SoA::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh_SoA* sphere = new Mesh_SoA();
    *sphere = Mesh_SoA::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
        {
            renderer_mt.renderScene_SoA_Optimized(renderer, threads, *m, camera, L, m->ka, m->kd);
            //renderer_mt.renderScene_SoA_Optimized_MT(renderer, threads, *m, camera, L, m->ka, m->kd);
            //renderer_mt.renderScene_SoA_Optimized_ST(renderer, *m, camera, L, m->ka, m->kd);
        }
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}


void scene3_AoS()
{
    Renderer renderer;
    matrix camera;
    Light L{
        vec4(0.f, 1.f, 1.f, 0.f),
        colour(1.0f, 1.0f, 1.0f),
        colour(0.2f, 0.2f, 0.2f)
    };

    constexpr int NX = 20;
    constexpr int NY = 10;
    constexpr int NZ = 20;
    constexpr float SPACING = 2.5f;

    std::vector<Mesh*> scene;
    scene.reserve(NX * NY * NZ);

    // ----------------------------
    // Create massive cube grid
    // ----------------------------
    for (int z = 0; z < NZ; ++z)
    {
        for (int y = 0; y < NY; ++y)
        {
            for (int x = 0; x < NX; ++x)
            {
                Mesh* m = new Mesh();
                *m = Mesh::makeCube(1.f);

                float px = (x - NX * 0.5f) * SPACING;
                float py = (y - NY * 0.5f) * SPACING;
                float pz = -z * SPACING;

                m->world =
                    matrix::makeTranslation(px, py, pz) *
                    makeRandomRotation();

                scene.push_back(m);
            }
        }
    }

    float zoffset = 5.f;
    float speed = 0.25f;

    auto start = std::chrono::high_resolution_clock::now();
    int frame = 0;

    L.omega_i.normalise();

    while (!renderer.canvas.keyPressed(VK_ESCAPE))
    {
        renderer.canvas.checkInput();
        renderer.clear();

        // Camera flies through geometry
        camera = matrix::makeTranslation(0, 0, -zoffset);
        zoffset += speed;

        // Continuous animation (avoid static transforms)
        for (auto& m : scene)
        {
            m->world = m->world * matrix::makeRotateXYZ(0.01f, 0.02f, 0.015f);
            //render_transfirst(renderer, m, camera, L);
            render(renderer, m, camera, L);
        }

        renderer.present();

        if (++frame % 120 == 0)
        {
            auto now = std::chrono::high_resolution_clock::now();
            std::cout << "AoS 120 frames: "
                << std::chrono::duration<double, std::milli>(now - start).count()
                << " ms\n";
            start = now;
        }

        if (zoffset > NZ * SPACING)
            zoffset = 0.f;
    }

    for (auto& m : scene)
        delete m;
}

void scene3_SoA()
{
    Renderer renderer;
    matrix camera;
    Light L{
        vec4(0.f, 1.f, 1.f, 0.f),
        colour(1.0f, 1.0f, 1.0f),
        colour(0.2f, 0.2f, 0.2f)
    };

    constexpr int NX = 20;
    constexpr int NY = 10;
    constexpr int NZ = 20;
    constexpr float SPACING = 2.5f;

    std::vector<Mesh_SoA*> scene;
    scene.reserve(NX * NY * NZ);

    // ----------------------------
    // Create massive cube grid
    // ----------------------------
    for (int z = 0; z < NZ; ++z)
    {
        for (int y = 0; y < NY; ++y)
        {
            for (int x = 0; x < NX; ++x)
            {
                Mesh_SoA* m = new Mesh_SoA();
                *m = Mesh_SoA::makeCube(1.f);

                float px = (x - NX * 0.5f) * SPACING;
                float py = (y - NY * 0.5f) * SPACING;
                float pz = -z * SPACING;

                m->world =
                    matrix::makeTranslation(px, py, pz) *
                    makeRandomRotation();

                scene.push_back(m);
            }
        }
    }

    float zoffset = 5.f;
    float speed = 0.25f;

    auto start = std::chrono::high_resolution_clock::now();
    int frame = 0;

    L.omega_i.normalise();

    while (!renderer.canvas.keyPressed(VK_ESCAPE))
    {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset);
        zoffset += speed;

        for (auto& m : scene)
        {
            m->world = m->world * matrix::makeRotateXYZ(0.01f, 0.02f, 0.015f);

            // 核心测试点
            render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L);
        }

        renderer.present();

        if (++frame % 120 == 0)
        {
            auto now = std::chrono::high_resolution_clock::now();
            std::cout << "SoA AVX2 120 frames: "
                << std::chrono::duration<double, std::milli>(now - start).count()
                << " ms\n";
            start = now;
        }

        if (zoffset > NZ * SPACING)
            zoffset = 0.f;
    }

    for (auto& m : scene)
        delete m;
}

void scene4(bool if_trans_first = false, bool if_AVX = false)
{
    Renderer renderer;
    matrix camera = matrix::makeIdentity();

    Light L{
        vec4(0.f, 1.f, 1.f, 0.f),
        colour(1.0f, 1.0f, 1.0f),
        colour(0.2f, 0.2f, 0.2f)
    };
    L.omega_i.normalise();

    std::vector<Mesh*> scene;

    // 1. Create high-poly spheres
    int LAT = 40;
    int LON = 80;

    for (int i = 0; i < 6; ++i)
    {
        Mesh* s = new Mesh();
        *s = Mesh::makeSphere(2.5f, LAT, LON);

        float x = -6.0f + (i % 3) * 6.0f;
        float y = (i < 3) ? 3.5f : -3.5f;

        s->world = matrix::makeTranslation(x, y, -10.f);
        scene.push_back(s);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    float angle = 0.f;

    while (running)
    {
        renderer.canvas.checkInput();
        renderer.clear();

        if (renderer.canvas.keyPressed(VK_ESCAPE))
            break;

        angle += 0.01f;

        // Animate only top row
        for (int i = 0; i < 3; ++i)
        {
            scene[i]->world =
                scene[i]->world *
                matrix::makeRotateXYZ(0.01f, 0.02f, 0.015f);
        }

        for (auto& m : scene)
        {
            if (if_trans_first)
            {
                if (if_AVX)
                    render_transfirst(renderer, m, camera, L, true);
                else
                    render_transfirst(renderer, m, camera, L, false);
            }
            else
            {
                if (if_AVX)
                    render(renderer, m, camera, L, true);
                else
                    render(renderer, m, camera, L, false);
            }
        }

        renderer.present();
        // 5. Measure

        if (++cycle % 120 == 0)
        {
            end = std::chrono::high_resolution_clock::now();
            std::cout << cycle / 120 << " :" << "[AoS] "
                << std::chrono::duration<double, std::milli>(end - start).count()
                << " ms\n";
            start = std::chrono::high_resolution_clock::now();
        }
    }

    for (auto& m : scene)
        delete m;
}

void scene4_SoA(bool if_trans_first = false, bool if_AVX = false, bool if_AVX_at_transf = false)
{
    Renderer renderer;
    matrix camera = matrix::makeIdentity();

    Light L{  vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    L.omega_i.normalise();

    std::vector<Mesh_SoA*> scene;

    constexpr int LAT = 40;
    constexpr int LON = 80;

    for (int i = 0; i < 6; ++i)
    {
        Mesh_SoA* s = new Mesh_SoA();
        *s = Mesh_SoA::makeSphere(2.5f, LAT, LON);

        float x = -6.0f + (i % 3) * 6.0f;
        float y = (i < 3) ? 3.5f : -3.5f;

        s->world = matrix::makeTranslation(x, y, -10.f);
        scene.push_back(s);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;

    while (running)
    {
        renderer.canvas.checkInput();
        renderer.clear();

        if (renderer.canvas.keyPressed(VK_ESCAPE))
            break;


        for (int i = 0; i < 3; ++i)
        {
            scene[i]->world =
                scene[i]->world *
                matrix::makeRotateXYZ(0.01f, 0.02f, 0.015f);
        }

        for (auto& m : scene)
        {
            if (if_trans_first)
            {
                if (if_AVX)
                    if (if_AVX_at_transf)
                        render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L, true);
                    else
                        render_transfirst_SoA_Scalar(renderer, m, camera, L, true);
                else
                    if (if_AVX_at_transf)
                        render_transfirst_SoA_AVX2_Optimized(renderer, m, camera, L, false);
                    else
                        render_transfirst_SoA_Scalar(renderer, m, camera, L, false);
            }
            else
            {
                if (if_AVX)
                    render_SoA(renderer, m, camera, L, true);
                else
                    render_SoA(renderer, m, camera, L, false);
            }
        }

        renderer.present();

        if (++cycle % 120 == 0)
        {
            end = std::chrono::high_resolution_clock::now();
            std::cout << cycle / 120 << " :" << "[SoA SIMD] "
                << std::chrono::duration<double, std::milli>(end - start).count()
                << " ms\n";
            start = std::chrono::high_resolution_clock::now();
        }
    }

    for (auto& m : scene)
        delete m;
}
void scene4_SoA_MT(bool if_trans_first = false, bool if_AVX = false, bool if_AVX_at_transf = false)
{
    Renderer renderer;
    matrix camera = matrix::makeIdentity();

    Light L{  vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };
    L.omega_i.normalise();

    std::vector<Mesh_SoA*> scene;
    ThreadPool threads;
    RendererSOA_MT renderer_mt(renderer.canvas.getWidth(), renderer.canvas.getHeight());

    constexpr int LAT = 40;
    constexpr int LON = 80;

    for (int i = 0; i < 6; ++i)
    {
        Mesh_SoA* s = new Mesh_SoA();
        *s = Mesh_SoA::makeSphere(2.5f, LAT, LON);

        float x = -6.0f + (i % 3) * 6.0f;
        float y = (i < 3) ? 3.5f : -3.5f;

        s->world = matrix::makeTranslation(x, y, -10.f);
        scene.push_back(s);
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;

    while (running)
    {
        renderer.canvas.checkInput();
        renderer.clear();

        if (renderer.canvas.keyPressed(VK_ESCAPE))
            break;

        for (int i = 0; i < 3; ++i)
        {
            scene[i]->world =
                scene[i]->world *
                matrix::makeRotateXYZ(0.01f, 0.02f, 0.015f);
        }


        // SIMD render path
        for (auto& m : scene)
        {
            //renderer_mt.renderScene_SoA_Optimized_MT(renderer, threads, *m, camera, L, m->ka, m->kd);
            //renderer_mt.renderScene_SoA_Optimized(renderer, threads, *m, camera, L, m->ka, m->kd);
            renderer_mt.renderScene_SoA_Optimized_ST(renderer, *m, camera, L, m->ka, m->kd);
        }

        renderer.present();

        if (++cycle % 120 == 0)
        {
            end = std::chrono::high_resolution_clock::now();
            std::cout << cycle / 120 << " :" << "[SoA SIMD MT] " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
            start = std::chrono::high_resolution_clock::now();
        }
    }

    for (auto& m : scene)
        delete m;
}

// Entry point of the application
// No input variables
int main() {
    // Uncomment the desired scene function to run

    // if using transform all vertexs before draw
    bool if_trans_first = true;
    // if using SIMD AVX2 when rendering pixels
    bool if_AVX = true;
    // if using SIMD AVX2 when transform all vertexs (only for SoA structure)
    bool if_AVX_at_transf = true;

    //sceneTest(); 
    
    //
    //scene1(if_trans_first, if_AVX);

    //scene1_SoA(if_trans_first, if_AVX, if_AVX_at_transf);
    // always use transform vertexs first and SIMD
    //scene1_SoA_MT();

    //scene2();
    //scene2(if_trans_first, if_AVX);

    //scene2_SoA(if_trans_first, if_AVX, if_AVX_at_transf);
    // always use transform vertexs first and SIMD
    //scene2_SoA_MT();

    //scene3_AoS();
    //scene3_SoA();
    

    //scene4();
    //scene4(if_trans_first, if_AVX);
    
    //scene4_SoA(if_trans_first, if_AVX, if_AVX_at_transf);
    scene4_SoA_MT();

    return 0;
}