#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>

// The `colour` class represents an RGB colour with floating-point precision.
// It provides various utilities for manipulating and converting colours.
class colour {
public:
    union {
        struct {
            float r, g, b; // Red, Green, and Blue components of the colour
            float padding;
        };
        float rgb[4];     // Array representation of the RGB components
        //float rgb[3];     // Array representation of the RGB components
    };

public:
    // Enum for indexing the RGB components
    enum Colour { RED = 0, GREEN = 1, BLUE = 2 };

    // Constructor to initialize the colour with specified RGB values.
    // Default values are 0 (black).
    // Input Variables:
    // - _r: Red component (default 0.0f)
    // - _g: Green component (default 0.0f)
    // - _b: Blue component (default 0.0f)
    colour(float _r = 0, float _g = 0, float _b = 0) : r(_r), g(_g), b(_b), padding(0) {}
    //colour(float _r = 0, float _g = 0, float _b = 0) : r(_r), g(_g), b(_b) {}

    // Sets the RGB components of the colour.
    // Input Variables:
    // - _r: Red component
    // - _g: Green component
    // - _b: Blue component
    void set(float _r, float _g, float _b) { r = _r, g = _g, b = _b; }

    // Accesses the specified component of the colour by index.
    // Input Variables:
    // - c: Index of the component (RED, GREEN, or BLUE)
    // Returns a reference to the specified component.
    float& operator[] (Colour c) { return rgb[c]; }

    // Assigns the values of another colour to this one.
    // Input Variables:
    // - c: The source color
    void operator = (colour c) {
        r = c.r;
        g = c.g;
        b = c.b;
    }
    //// Opt:
    //colour& operator = (const colour& c)
    //{
    //    r = c.r;
    //    g = c.g;
    //    b = c.b;
    //    return *this;
    //}

    colour operator*(float scalar) const {
        return { r * scalar, g * scalar, b * scalar};
    }

    // Clamps the RGB components of the colour to the range [0, 1].
    void clampColour() {
        r = std::min(r, 1.0f);
        g = std::min(g, 1.0f);
        b = std::min(b, 1.0f);
    }

    // Converts the floating-point RGB values to integer values (0-255).
    // Output Variables:
    // - cr: Red component as an unsigned char
    // - cg: Green component as an unsigned char
    // - cb: Blue component as an unsigned char
    void toRGB(unsigned char& cr, unsigned char& cg, unsigned char& cb) {
        cr = static_cast<unsigned char>(std::floor(r * 255));
        cg = static_cast<unsigned char>(std::floor(g * 255));
        cb = static_cast<unsigned char>(std::floor(b * 255));
    }

    // Scales the RGB components of the colour by a scalar value.
    // Input Variables:
    // - scalar: The scaling factor
    // Returns a new `colour` object with scaled components.
    colour operator * (const float scalar) {
        colour c;
        c.r = r * scalar;
        c.g = g * scalar;
        c.b = b * scalar;
        return c;
    }
    // Opt: SIMD SSE
    //colour operator*(const float scalar) {
    //    colour out;
    //    __m128 v = _mm_load_ps(this->rgb);          // r g b _
    //    __m128 s = _mm_set1_ps(scalar);       // s s s s
    //    v = _mm_mul_ps(v, s);
    //    _mm_store_ps(out.rgb, v);
    //    out.rgb[3] = 0.0f;
    //    return out;
    //}

    // Multiplies the RGB components of this colour with another colour.
    // Input Variables:
    // - col: The other color to multiply with
    // Returns a new `colour` object with multiplied components.
    colour operator * (const colour& col) {
        colour c;
        c.r = r * col.r;
        c.g = g * col.g;
        c.b = b * col.b;
        return c;
    }
    //colour operator* (const colour& col)
    //{
    //    colour result;
    //    __m128 a = _mm_load_ps(this->rgb);
    //    __m128 b = _mm_load_ps(col.rgb);
    //    a = _mm_mul_ps(a, b);
    //    _mm_store_ps(result.rgb, a);
    //    return result;
    //}

    // Adds the RGB components of another colour to this one.
    // Input Variables:
    // - _c: The other colour to add
    // Returns a new `colour` object with added components.
    colour operator + (const colour& _c) {
        colour c;
        c.r = r + _c.r;
        c.g = g + _c.g;
        c.b = b + _c.b;
        return c;
    }
    //colour operator+(const colour& c) const {
    //    colour out;
    //    __m128 a = _mm_load_ps(this->rgb);
    //    __m128 b = _mm_load_ps(c.rgb);
    //    __m128 r = _mm_add_ps(a, b);
    //    _mm_store_ps(out.rgb, r);
    //    out.rgb[3] = 0.0f;
    //    return out;
    //}


    colour& operator += (const colour& _c) {
        r = r + _c.r;
        g = g + _c.g;
        b = b + _c.b;
        return *this;
    }
};