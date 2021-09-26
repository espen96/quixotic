#version 150

#define NUMCONTROLS 100
//#define vertexinfo
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
uniform float GameTime;

/*
Control Map:
[0] sunDir.x
[1] sunDir.y
[2] sunDir.z
[3] arctan(ProjMat[0][0])
[4] arctan(ProjMat[1][1])
[5] ProjMat[1][0]
[6] ProjMat[0][1]
[7] ProjMat[1][2]
[8] ProjMat[1][3]
[9] ProjMat[2][0]
[10] ProjMat[2][1]
[11] ProjMat[2][2]
[12] ProjMat[2][3]
[13] ProjMat[3][0]
[14] ProjMat[3][1]
[15] ProjMat[3][2]
[16] ModelViewMat[0][0]
[17] ModelViewMat[0][1]
[18] ModelViewMat[0][2]
[19] ModelViewMat[1][0]
[20] ModelViewMat[1][1]
[21] ModelViewMat[1][2]
[22] ModelViewMat[2][0]
[23] ModelViewMat[2][1]
[24] ModelViewMat[2][2]
[25] FogColor
[26] SkyColor
[27] SkyColor
[28] SkyColor
[29] SkyColor
*/

// returns control pixel index or -1 if not control
int inControl(vec2 screenCoord, float screenWidth) {
    if (screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}
int inControl2(vec2 screenCoord, vec4 glpos) {
    if (screenCoord.y < 1.0) {
        float screenWidth = round(screenCoord.x * 2.0 / (glpos.x / glpos.w + 1.0));
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}

// discards the current pixel if it is control
void discardControl(vec2 screenCoord, float screenWidth) {
    if (screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            discard;
        }
    }
}

// discard but for when ScreenSize is not given
void discardControlGLPos(vec2 screenCoord, vec4 glpos) {
    if (screenCoord.y < 1.0) {
        float screenWidth = round(screenCoord.x * 2.0 / (glpos.x / glpos.w + 1.0));
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            discard;
        }
    }
}

// get screen coordinates of a particular control index
vec2 getControl(int index, vec2 screenSize) {
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}

int intmod(int i, int base) {
    return i - (i / base * base);
}

vec3 encodeInt(int i) {
    int s = int(i < 0) * 128;
    i = abs(i);
    int r = intmod(i, 256);
    i = i / 256;
    int g = intmod(i, 256);
    i = i / 256;
    int b = intmod(i, 128);
    return vec3(float(r) / 255.0, float(g) / 255.0, float(b + s) / 255.0);
}

int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

vec3 encodeFloat(float i) {
    return encodeInt(int(i * FPRECISION));
}

float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}


/*
 * Created by Onnowhere (https://github.com/onnowhere)
 * Utility functions for Minecraft core vertex shaders
 */

#define LIGHT0_DIRECTION vec3(0.2, 1.0, -0.7) // Default light 0 direction everywhere except in inventory
#define LIGHT1_DIRECTION vec3(-0.2, 1.0, 0.7) // Default light 1 direction everywhere except in nether and inventory

/*
 * Returns the FOV in degrees
 * Calculates using the fact that top/near = tan(theta / 2)
 */
float getFOV(mat4 ProjMat) {
    return atan(1.0, ProjMat[1][1]) * 114.591559;
}

/*
 * Returns if rendering in a GUI
 * In the GUI, near is 1000 and far is 3000, so -(far+near)/(far-near) = -2.0
 */
bool isGUI(mat4 ProjMat) {
    return ProjMat[3][2] == -2.0;
}

/*
 * Returns if rendering in the main menu background panorama
 * Checks the far clipping plane value so this should only be used with position_tex_color
 */
bool isPanorama(mat4 ProjMat) {
    float far = ProjMat[3][2] / (ProjMat[2][2] + 1);
    return far < 9.99996 && far > 9.99995;
}

/*
 * Returns if rendering in the nether given light directions
 * In the nether, the light directions are parallel but in opposite directions
 */
bool isNether(vec3 light0, vec3 light1) {
    return abs(light0) == abs(light1);
}

/*
 * Returns camera to world space matrix given light directions
 * Creates matrix by comparing world space light directions to camera space light directions
 */
mat3 getWorldMat(vec3 light0, vec3 light1) {
    if (isNether(light0, light1)) {
        // Cannot determine matrix in the nether due to parallel light directions
        return mat3(0.0);
    }
    mat3 V = mat3(normalize(LIGHT0_DIRECTION), normalize(LIGHT1_DIRECTION), normalize(cross(LIGHT0_DIRECTION, LIGHT1_DIRECTION)));
    mat3 W = mat3(normalize(light0), normalize(light1), normalize(cross(light0, light1)));
    return W * inverse(V);
}

/*
 * Returns far clipping plane distance
 * Evaluates far clipping plane by extracting it from the projection matrix
 */
float getFarClippingPlane(mat4 ProjMat) {
    vec4 distProbe = inverse(ProjMat) * vec4(0.0, 0.0, 1.0, 1.0);
    return length(distProbe.xyz / distProbe.w);
}

/*
 * Returns render distance based on far clipping plane
 * Uses far clipping plane distance to get render distance in chunks
 */
float getRenderDistance(mat4 ProjMat) {
    return round(getFarClippingPlane(ProjMat) / 64.0);
}

/*
 * Returns orthographic transformation matrix
 * Creates matrix by extracting values from projection matrix
 */
mat4 getOrthoMat(mat4 ProjMat, float Zoom) {
    float far = getFarClippingPlane(ProjMat);
    float near = 0.05; // Fixed distance that should never change
    float left = -(0.5 / (ProjMat[0][0] / (2.0 * near))) / Zoom;
    float right = -left;
    float top = (0.5 / (ProjMat[1][1] / (2.0 * near))) / Zoom;
    float bottom = -top;

    return mat4(2.0 / (right - left),               0.0,                                0.0,                            0.0,
                0.0,                                2.0 / (top - bottom),               0.0,                            0.0,
                0.0,                                0.0,                                -2.0 / (far - near),            0.0,
                -(right + left) / (right - left),   -(top + bottom) / (top - bottom),   -(far + near) / (far - near),   1.0);
}



vec3 ScreenSpaceDither( vec2 vScreenPos )
{
	// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR
	//vec3 vDither = vec3( dot( vec2( 171.0, 231.0 ), vScreenPos.xy + iTime ) );
    vec3 vDither = vec3( dot( vec2( 171.0, 231.0 ), vScreenPos.xy +(GameTime*16000)) );
    vDither.rgb = fract( vDither.rgb / vec3( 103.0, 71.0, 97.0 ) );
    
    //note: apply triangular pdf
    //vDither.r = remap_noise_tri_erp(vDither.r)*2.0-0.5;
    //vDither.g = remap_noise_tri_erp(vDither.g)*2.0-0.5;
    //vDither.b = remap_noise_tri_erp(vDither.b)*2.0-0.5;
    
    return vDither.rgb ; //note: looks better without 0.375...

    //note: not sure why the 0.5-offset is there...
    //vDither.rgb = fract( vDither.rgb / vec3( 103.0, 71.0, 97.0 ) ) - vec3( 0.5, 0.5, 0.5 );
	//return (vDither.rgb / 255.0) * 0.375;
}
vec3 encodeFloat24(float val) {
    uint sign = val > 0.0 ? 0u : 1u;
    uint exponent = uint(log2(abs(val)));
    uint mantissa = uint((abs(val) / exp2(float(exponent)) - 1.0) * 131072.0);
    return vec3(
        (sign << 7u) | ((exponent + 31u) << 1u) | (mantissa >> 16u),
        (mantissa >> 8u) & 255u,
        mantissa & 255u
    ) / 255.0;
}

float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}

