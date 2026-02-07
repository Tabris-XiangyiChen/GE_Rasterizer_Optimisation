#pragma once

#include "mesh.h"
//#include "colour.h"
#include "Types.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// Simple support class for a 2D vector
class vec2D {
public:
    float x, y;

    // Default constructor initializes both components to 0
    vec2D() { x = y = 0.f; };

    // Constructor initializes components with given values
    vec2D(float _x, float _y) : x(_x), y(_y) {}

    // Constructor initializes components from a vec4
    vec2D(vec4 v) {
        x = v[0];
        y = v[1];
    }

    // Display the vector components
    void display() { std::cout << x << '\t' << y << std::endl; }

    // Overloaded subtraction operator for vector subtraction
    vec2D operator- (vec2D& v) {
        vec2D q;
        q.x = x - v.x;
        q.y = y - v.y;
        return q;
    }
};

struct Edge {
    float A, B, C;
};
static Edge makeEdge(const vec2D& v0, const vec2D& v1) {
    Edge e;
    e.A = v0.y - v1.y;
    e.B = v1.x - v0.x;
    e.C = v0.x * v1.y - v1.x * v0.y;
    return e;
}
static vec4 makeEdge0(const vec2D& v0, const vec2D& v1) {
    vec4 e;
    e.x = v0.y - v1.y;
    e.y = v1.x - v0.x;
    e.z = v0.x * v1.y - v1.x * v0.y;
    return e;
}

// Class representing a triangle for rendering purposes
class triangle {
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    colour col[3];     // Colors for each vertex of the triangle

public:
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        // Calculate the 2D area of the triangle
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);
        area = std::fabs(e1.x * e2.y - e1.y * e2.x);
    }
    //Opt: 
    //triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
    //    v[0] = v1;
    //    v[1] = v2;
    //    v[2] = v3;
    //    // Calculate the 2D area of the triangle
    //    area = std::fabs((v[1].p[0] - v[0].p[0]) * (v[2].p[1] - v[0].p[1])
    //        - (v[1].p[1] - v[0].p[1]) * (v[2].p[0] - v[0].p[0]));
    //}

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    float getC(vec2D v1, vec2D v2, vec2D p) {
        vec2D e = v2 - v1;
        vec2D q = p - v1;
        return q.y * e.x - q.x * e.y;
    }
    inline float getC(float x0, float y0, float x1, float y1, float px, float py)
    {
        return (py - y0) * (x1 - x0) - (px - x0) * (y1 - y0);
    }


    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    //bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
    //    alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
    //    beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
    //    gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;
    //    if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
    //    return true;
    //}
    // Opt: Use mul, avoid create new vec2D
    bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
        float invArea = 1.0f / area;
        //alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) * invArea;
        //beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) * invArea;
        //gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) * invArea;
        alpha = getC(v[0].p[0], v[0].p[1], v[1].p[0], v[1].p[1], p.x, p.y) * invArea;
        beta = getC(v[1].p[0], v[1].p[1], v[2].p[0], v[2].p[1], p.x, p.y) * invArea;
        gamma = getC(v[2].p[0], v[2].p[1], v[0].p[0], v[0].p[1], p.x, p.y) * invArea;
        if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
        return true;
    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }

    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;
        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);
        // Skip very small triangles
        if (area < 1.f) return;
        // Opt: One time per frame
        //L.omega_i.normalise();
        // Iterate over the bounding box and check each pixel
        for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
            for (int x = (int)(minV.x); x < (int)ceil(maxV.x); x++) {
                float alpha, beta, gamma;
                // Check if the pixel lies inside the triangle
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {
                    // Interpolate color, depth, and normals
                    colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                    vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                    // Opt:
                    normal.normalise();
                    // Perform Z-buffer test and apply shading
                    if (renderer.zbuffer(x, y) > depth && depth > 0.001f) {
                        // typical shader begin
                        L.omega_i.normalise();
                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka); // using kd instead of ka for ambient
                        // typical shader end
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }
    // Opt: LEE（Linear Expression Evaluation）
    void draw_LEE1(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;
        // Light direction normalize ONCE per triangle
        //L.omega_i.normalise();
        // Triangles
        vec2D p0(v[0].p);
        vec2D p1(v[1].p);
        vec2D p2(v[2].p);
        // Build edge equations
        // E(x,y) = A*x + B*y + C
        // E0
        float A0 = p0.y - p1.y;
        float B0 = p1.x - p0.x;
        float C0 = p0.x * p1.y - p1.x * p0.y;
        // E1
        float A1 = p1.y - p2.y;
        float B1 = p2.x - p1.x;
        float C1 = p1.x * p2.y - p2.x * p1.y;
        // E2
        float A2 = p2.y - p0.y;
        float B2 = p0.x - p2.x;
        float C2 = p2.x * p0.y - p0.x * p2.y;
        float invArea = 1.0f / area;
        // Boundray
        int minX = (int)minV.x;
        int maxX = (int)ceil(maxV.x);
        int minY = (int)minV.y;
        int maxY = (int)ceil(maxV.y);
        // Compute barycentric edge values at top-left pixel
        float px = minX + 0.5f;
        float py = minY + 0.5f;
        float w0_row = A0 * px + B0 * py + C0;
        float w1_row = A1 * px + B1 * py + C1;
        float w2_row = A2 * px + B2 * py + C2;
        // Scan all pixels(Scanline)
        for (int y = minY; y < maxY; ++y) {
            float w0 = w0_row;
            float w1 = w1_row;
            float w2 = w2_row;
            for (int x = minX; x < maxX; ++x) {
                // Inside test
                float alpha = w0 * invArea;
                float beta = w1 * invArea;
                float gamma = w2 * invArea;
                if (!(alpha < 0.f || beta < 0.f || gamma < 0.f)) {
                    // Interplate
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                    if (depth > 0.001f && renderer.zbuffer(x, y) > depth) 
                    {
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();
                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                        //
                        normal.normalise();
                        float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot) + (L.ambient * ka); // using kd instead of ka for ambient
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
                // Step right (LEE)
                w0 += A0;
                w1 += A1;
                w2 += A2;
            }
            // Step down one scanline
            w0_row += B0;
            w1_row += B1;
            w2_row += B2;
        }
    }

    void draw_LEE2(Renderer& renderer, Light& L, float ka, float kd)
    {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;
        // Normalize light direction ONCE per triangle
        //L.omega_i.normalise();
        int minX = (int)minV.x;
        int minY = (int)minV.y;
        int maxX = (int)ceil(maxV.x);
        int maxY = (int)ceil(maxV.y);
        // Build edge equations
        Edge e0 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
        Edge e1 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
        Edge e2 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));
        float invArea = 1.0f / area;
        // Compute barycentric edge values at top-left pixel
        float px = minX + 0.5f;
        float py = minY + 0.5f;
        float w0_row = e0.A * px + e0.B * py + e0.C;
        float w1_row = e1.A * px + e1.B * py + e1.C;
        float w2_row = e2.A * px + e2.B * py + e2.C;
        // Precompute attribute gradients (LEE)
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
        // Initial interpolated values at top-left
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
        // Scanline rasterization
        for (int y = minY; y < maxY; ++y)
        {
            float w0 = w0_row;
            float w1 = w1_row;
            float w2 = w2_row;
            float z = z_row;
            vec4  n = n_row;
            colour c = c_row;
            for (int x = minX; x < maxX; ++x)
            {
                if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                {
                    if (renderer.zbuffer(x, y) > z && z > 0.001f)
                    {
                        // No per-pixel normalisation
                        n.normalise();
                        float dot = std::max(vec4::dot(L.omega_i, n), 0.0f);
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
                // Step right (LEE)
                w0 += e0.A;
                w1 += e1.A;
                w2 += e2.A;
                z += dz_dx;
                n += dn_dx;
                c += dc_dx;
            }
            // Step down one scanline
            w0_row += e0.B;
            w1_row += e1.B;
            w2_row += e2.B;
            z_row += dz_dy;
            n_row += dn_dy;
            c_row += dc_dy;
        }
    }

    void draw_AVX2_Optimized3(Renderer& renderer, Light& L, float ka, float kd)
    {
        if (area < 1.f) return;
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);


        int W = renderer.canvas.getWidth();
        int H = renderer.canvas.getHeight();

        //const int minX = (int)std::floor(minV.x);
        //const int minY = (int)std::floor(minV.y);
        //const int maxX = (int)std::ceil(maxV.x);
        //const int maxY = (int)std::ceil(maxV.y);
        int minX = std::clamp((int)std::floor(minV.x), 0, W - 1);
        int minY = std::clamp((int)std::floor(minV.y), 0, H - 1);
        int maxX = std::clamp((int)std::ceil(maxV.x), 0, W - 1);
        int maxY = std::clamp((int)std::ceil(maxV.y), 0, H - 1);

        vec4 e0 = makeEdge0(vec2D(v[1].p), vec2D(v[2].p));
        vec4 e1 = makeEdge0(vec2D(v[2].p), vec2D(v[0].p));
        vec4 e2 = makeEdge0(vec2D(v[0].p), vec2D(v[1].p));

        //auto isTopLeft = [](vec2D p0, vec2D p1) {
        //    return (p0.y == p1.y ? (p1.x < p0.x) : (p1.y > p0.y));
        //    };
        //const bool tl0 = isTopLeft(vec2D(v[1].p), vec2D(v[2].p));
        //const bool tl1 = isTopLeft(vec2D(v[2].p), vec2D(v[0].p));
        //const bool tl2 = isTopLeft(vec2D(v[0].p), vec2D(v[1].p));

        const float invArea = 1.0f / area;

        const float dz_dx = (v[0].p[2] * e0.x + v[1].p[2] * e1.x + v[2].p[2] * e2.x) * invArea;
        const float dz_dy = (v[0].p[2] * e0.y + v[1].p[2] * e1.y + v[2].p[2] * e2.y) * invArea;

        const vec4 dn_dx = (v[0].normal * e0.x + v[1].normal * e1.x + v[2].normal * e2.x) * invArea;
        const vec4 dn_dy = (v[0].normal * e0.y + v[1].normal * e1.y + v[2].normal * e2.y) * invArea;

        const colour dc_dx = (v[0].rgb * e0.x + v[1].rgb * e1.x + v[2].rgb * e2.x) * invArea;
        const colour dc_dy = (v[0].rgb * e0.y + v[1].rgb * e1.y + v[2].rgb * e2.y) * invArea;

        // --- SIMD constants ---
        const __m256 zero = _mm256_setzero_ps();
        //const __m256 lane = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256 lane = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

        const __m256 w0_lane = _mm256_mul_ps(_mm256_set1_ps(e0.x), lane);
        const __m256 w1_lane = _mm256_mul_ps(_mm256_set1_ps(e1.x), lane);
        const __m256 w2_lane = _mm256_mul_ps(_mm256_set1_ps(e2.x), lane);
        const __m256 z_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dx), lane);

        const __m256 n_x_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx.x), lane);
        const __m256 n_y_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx.y), lane);
        const __m256 n_z_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dx.z), lane);

        const __m256 c_r_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx.r), lane);
        const __m256 c_g_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx.g), lane);
        const __m256 c_b_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dx.b), lane);

        const __m256 w0_dy_lane = _mm256_mul_ps(_mm256_set1_ps(e0.y), lane);
        const __m256 w1_dy_lane = _mm256_mul_ps(_mm256_set1_ps(e1.y), lane);
        const __m256 w2_dy_lane = _mm256_mul_ps(_mm256_set1_ps(e2.y), lane);
        const __m256 z_dy_lane = _mm256_mul_ps(_mm256_set1_ps(dz_dy), lane);

        const __m256 n_dy_x_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dy.x), lane);
        const __m256 n_dy_y_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dy.y), lane);
        const __m256 n_dy_z_lane = _mm256_mul_ps(_mm256_set1_ps(dn_dy.z), lane);

        const __m256 c_dy_r_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dy.r), lane);
        const __m256 c_dy_g_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dy.g), lane);
        const __m256 c_dy_b_lane = _mm256_mul_ps(_mm256_set1_ps(dc_dy.b), lane);

        const __m256 w0_step8 = _mm256_set1_ps(e0.x * 8.f);
        const __m256 w1_step8 = _mm256_set1_ps(e1.x * 8.f);
        const __m256 w2_step8 = _mm256_set1_ps(e2.x * 8.f);
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

        float w0_row = e0.x * minX + e0.y * minY + e0.z;
        float w1_row = e1.x * minX + e1.y * minY + e1.z;
        float w2_row = e2.x * minX + e2.y * minY + e2.z;

        float w0_dy = e0.y;
        float w1_dy = e1.y;
        float w2_dy = e2.y;

        __m256 w0_row_v = _mm256_add_ps(_mm256_set1_ps(w0_row), w0_lane);
        __m256 w1_row_v = _mm256_add_ps(_mm256_set1_ps(w1_row), w1_lane);
        __m256 w2_row_v = _mm256_add_ps(_mm256_set1_ps(w2_row), w2_lane);

        float z_row =
            (v[0].p[2] * w0_row + v[1].p[2] * w1_row + v[2].p[2] * w2_row) * invArea;
        __m256 z_row_v = _mm256_add_ps(_mm256_set1_ps(z_row), z_lane);

        vec4 n_row =
            (v[0].normal * w0_row + v[1].normal * w1_row + v[2].normal * w2_row) * invArea;
        __m256 nx_row_v = _mm256_add_ps(_mm256_set1_ps(n_row.x), n_x_lane);
        __m256 ny_row_v = _mm256_add_ps(_mm256_set1_ps(n_row.y), n_y_lane);
        __m256 nz_row_v = _mm256_add_ps(_mm256_set1_ps(n_row.z), n_z_lane);

        colour c_row =
            (v[0].rgb * w0_row + v[1].rgb * w1_row + v[2].rgb * w2_row) * invArea;

        __m256 cr_row_v = _mm256_add_ps(_mm256_set1_ps(c_row.r), c_r_lane);
        __m256 cg_row_v = _mm256_add_ps(_mm256_set1_ps(c_row.g), c_g_lane);
        __m256 cb_row_v = _mm256_add_ps(_mm256_set1_ps(c_row.b), c_b_lane);

        __m256 _255 = _mm256_set1_ps(255.0f);
        __m256 _001 = _mm256_set1_ps(0.001f);
        __m256 _1 = _mm256_set1_ps(1.0f);

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
            //__m256 w0v = w0_row_v;
            //__m256 w1v = w1_row_v;
            //__m256 w2v = w2_row_v;
            //__m256 zv = z_row_v;

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

            //__m256 nx = nx_row_v;
            //__m256 ny = ny_row_v;
            //__m256 nz = nz_row_v;
            //__m256 cr = cr_row_v;
            //__m256 cg = cg_row_v;
            //__m256 cb = cb_row_v;

            int x = minX;

            for (; x <= maxX - 7; x += 8)
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
                if(mask == 0)
                {
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
                    continue;
                }

                __m256 zbuf = _mm256_loadu_ps(&renderer.zbuffer(x, y));
                //__m256 depth_ok =
                //    _mm256_and_ps(
                //        _mm256_cmp_ps(zv, _001, _CMP_GE_OQ),
                //        _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ));

                __m256 zv_001 = _mm256_cmp_ps(zv, _001, _CMP_GE_OQ);
                __m256 zbuf_zv = _mm256_cmp_ps(zbuf, zv, _CMP_GE_OQ);
                __m256 depth_ok = _mm256_and_ps( zv_001, zbuf_zv);

                int final_mask = _mm256_movemask_ps(_mm256_and_ps(inside, depth_ok));
                

                if (final_mask == 0)
                {
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

                nx = _mm256_add_ps(nx, n_step8x);
                ny = _mm256_add_ps(ny, n_step8y);
                nz = _mm256_add_ps(nz, n_step8z);

                cr = _mm256_add_ps(cr, c_step8r);
                cg = _mm256_add_ps(cg, c_step8g);
                cb = _mm256_add_ps(cb, c_step8b);
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

            //int tail_start = maxX - maxX % 8 - 1;
            int tail_count = maxX - x + 1;
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
                        zb[i] = (px <= maxX)
                            ? renderer.zbuffer(px, y)
                            : 1.0f;
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
                            if (px <= maxX)
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


            w0_row += w0_dy;
            w1_row += w1_dy;
            w2_row += w2_dy;
            z_row += dz_dy;
            n_row += dn_dy;
            c_row += dc_dy;
            //w0_row_v = _mm256_add_ps(w0_row_v, w0_dy_lane);
            //w1_row_v = _mm256_add_ps(w1_row_v, w1_dy_lane);
            //w2_row_v = _mm256_add_ps(w2_row_v, w2_dy_lane);
            //z_row_v = _mm256_add_ps(z_row_v, z_dy_lane);
            //nx_row_v = _mm256_add_ps(nx_row_v, n_dy_x_lane);
            //ny_row_v = _mm256_add_ps(ny_row_v, n_dy_y_lane);
            //nz_row_v = _mm256_add_ps(nz_row_v, n_dy_z_lane);
            //cr_row_v = _mm256_add_ps(cr_row_v, c_dy_r_lane);
            //cg_row_v = _mm256_add_ps(cg_row_v, c_dy_g_lane);
            //cb_row_v = _mm256_add_ps(cb_row_v, c_dy_b_lane);
        }
    }

    void draw_SSE(Renderer& renderer, Light& L, float ka, float kd)
    {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;

        //L.omega_i.normalise();

        int minX = (int)minV.x;
        int minY = (int)minV.y;
        int maxX = (int)ceil(maxV.x);
        int maxY = (int)ceil(maxV.y);

        // Build edge equations
        Edge e0 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
        Edge e1 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
        Edge e2 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));

        float invArea = 1.0f / area;

        float px = minX + 0.5f;
        float py = minY + 0.5f;

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

        // SSE constants
        const __m128 zero = _mm_setzero_ps();
        const __m128 idx = _mm_setr_ps(0, 1, 2, 3);

        const __m128 A0v = _mm_set1_ps(e0.A);
        const __m128 A1v = _mm_set1_ps(e1.A);
        const __m128 A2v = _mm_set1_ps(e2.A);
        const __m128 dzdxv = _mm_set1_ps(dz_dx);

        for (int y = minY; y < maxY; ++y)
        {
            float w0 = w0_row;
            float w1 = w1_row;
            float w2 = w2_row;

            float z = z_row;
            vec4 n = n_row;
            colour c = c_row;

            int x = minX;

            // SSE vectorized loop (4 pixels at a time)
            for (; x <= maxX - 4; x += 4)
            {
                __m128 w0v = _mm_add_ps(_mm_set1_ps(w0), _mm_mul_ps(A0v, idx));
                __m128 w1v = _mm_add_ps(_mm_set1_ps(w1), _mm_mul_ps(A1v, idx));
                __m128 w2v = _mm_add_ps(_mm_set1_ps(w2), _mm_mul_ps(A2v, idx));

                __m128 mask = _mm_and_ps(
                    _mm_and_ps(
                        _mm_cmpge_ps(w0v, zero),
                        _mm_cmpge_ps(w1v, zero)),
                    _mm_cmpge_ps(w2v, zero)
                );

                int bits = _mm_movemask_ps(mask);

                if (bits)
                {
                    __m128 zv = _mm_add_ps(_mm_set1_ps(z), _mm_mul_ps(dzdxv, idx));
                    alignas(16) float zbuf[4];
                    _mm_store_ps(zbuf, zv);

                    for (int i = 0; i < 4; ++i)
                    {
                        if (bits & (1 << i))
                        {
                            int xi = x + i;
                            float zi = zbuf[i];

                            if (zi > 0.001f && renderer.zbuffer(xi, y) > zi)
                            {
                                vec4 ni = n + dn_dx * (float)i;
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

                w0 += e0.A * 4.f;
                w1 += e1.A * 4.f;
                w2 += e2.A * 4.f;
                z += dz_dx * 4.f;
                n += dn_dx * 4.f;
                c += dc_dx * 4.f;
            }

            // Scalar tail
            for (; x < maxX; ++x)
            {
                if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                {
                    if (z > 0.001f && renderer.zbuffer(x, y) > z)
                    {
                        vec4 nn = n;
                        nn.normalise();
                        float dot = std::max(vec4::dot(L.omega_i, nn), 0.0f);
                        colour shaded = (c * kd) * (L.L * dot) + (L.ambient * ka);
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

    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    void getBounds(vec2D& minV, vec2D& maxV) {
        minV = vec2D(v[0].p);
        maxV = vec2D(v[0].p);
        for (unsigned int i = 1; i < 3; i++) {
            minV.x = std::min(minV.x, v[i].p[0]);
            minV.y = std::min(minV.y, v[i].p[1]);
            maxV.x = std::max(maxV.x, v[i].p[0]);
            maxV.y = std::max(maxV.y, v[i].p[1]);
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) {
        getBounds(minV, maxV);
        minV.x = std::max(minV.x, static_cast<float>(0));
        minV.y = std::max(minV.y, static_cast<float>(0));
        maxV.x = std::min(maxV.x, static_cast<float>(canvas.getWidth()));
        maxV.y = std::min(maxV.y, static_cast<float>(canvas.getHeight()));
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas) {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        for (int y = (int)minV.y; y < (int)maxV.y; y++) {
            for (int x = (int)minV.x; x < (int)maxV.x; x++) {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() {
        for (unsigned int i = 0; i < 3; i++) {
            v[i].p.display();
        }
        std::cout << std::endl;
    }

    void draw_AVX2_clipped(Renderer& renderer,
        Light& L,
        float ka,
        float kd,
        int yMin,
        int yMax)
    {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);

        int minY = std::max((int)minV.y, yMin);
        int maxY = std::min((int)ceil(maxV.y), yMax);

        if (minY >= maxY) return;

        L.omega_i.normalise();

        int minX = (int)minV.x;
        //int minY = (int)minV.y;
        int maxX = (int)ceil(maxV.x);
        //int maxY = (int)ceil(maxV.y);

        //Edge e1 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
        //Edge e2 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
        //Edge e0 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));
        Edge e0 = makeEdge(vec2D(v[1].p), vec2D(v[2].p));
        Edge e1 = makeEdge(vec2D(v[2].p), vec2D(v[0].p));
        Edge e2 = makeEdge(vec2D(v[0].p), vec2D(v[1].p));


        float invArea = 1.0f / area;

        float px = minX + 0.5f;
        float py = minY + 0.5f;

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

                                ni.normalise();
                                float dot = std::max(vec4::dot(L.omega_i, ni), 0.0f);

                                colour shaded =
                                    (ci * kd) * (L.L * dot) +
                                    (L.ambient * ka);

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


};
