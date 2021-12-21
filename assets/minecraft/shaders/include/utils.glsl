#version 150

#define NUMCONTROLS 200
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
[5] ProjMat[0][1]
[6] ProjMat[1][0]
[7] ProjMat[2][1]
[8] ProjMat[3][1]
[9] ProjMat[0][2]
[10] ProjMat[1][2]
[11] ProjMat[2][2]
[12] ProjMat[3][2]
[13] ProjMat[0][3]
[14] ProjMat[1][3]
[15] ProjMat[2][3]
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
    if(screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if(fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}
int inControl2(vec2 screenCoord, vec4 glpos) {
    if(screenCoord.y < 1.0) {
        float screenWidth = round(screenCoord.x * 2.0 / (glpos.x / glpos.w + 1.0));
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if(fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}

// discards the current pixel if it is control
void discardControl(vec2 screenCoord, float screenWidth) {
    if(screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if(fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            discard;
        }
    }
}

// discard but for when ScreenSize is not given
void discardControlGLPos(vec2 screenCoord, vec4 glpos) {
    if(screenCoord.y < 1.0) {
        float screenWidth = round(screenCoord.x * 2.0 / (glpos.x / glpos.w + 1.0));
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if(fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
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
    if(isNether(light0, light1)) {
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

    return mat4(2.0 / (right - left), 0.0, 0.0, 0.0, 0.0, 2.0 / (top - bottom), 0.0, 0.0, 0.0, 0.0, -2.0 / (far - near), 0.0, -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1.0);
}

#define steps 15.0
vec3 ScreenSpaceDither(vec2 vScreenPos) {
    vec3 vDither = vec3(dot(vec2(131.0, 312.0), vScreenPos.xy + fract(GameTime * 100)));
    vDither.rgb = fract(vDither.rgb / vec3(103.0, 71.0, 97.0)) * vec3(2.0, 2.0, 2.0) - vec3(0.5, 0.5, 0.5);
    return (vDither.rgb / steps);
}
vec3 encodeFloat24(float val) {
    uint sign = val > 0.0 ? 0u : 1u;
    uint exponent = uint(log2(abs(val)));
    uint mantissa = uint((abs(val) / exp2(float(exponent)) - 1.0) * 131072.0);
    return vec3((sign << 7u) | ((exponent + 31u) << 1u) | (mantissa >> 16u), (mantissa >> 8u) & 255u, mantissa & 255u) / 255.0;
}
float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * (GameTime*1000));
}

float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}

float packUnorm2x4(vec2 xy) {
    return dot(floor(15.0 * xy + 0.5), vec2(1.0 / 255.0, 16.0 / 255.0));
}
float packUnorm2x2(vec2 xy) {
    return dot(floor(4.0 * xy + 0.5), vec2(1.0 / 16.0, 16.0 / 16.0));
}
float packUnorm2x4(float x, float y) {
    return packUnorm2x4(vec2(x, y));
}
vec2 unpackUnorm2x4(float pack) {
    vec2 xy;
    xy.x = modf(pack * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}
vec2 unpackUnorm2x2(float pack) {
    vec2 xy;
    xy.x = modf(pack * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

//Dithering from Jodie
float Bayer2(vec2 a) {
    a = floor(a + (GameTime*100));
    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}

#define Bayer4(a)   (Bayer2(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer8(a)   (Bayer4(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer16(a)  (Bayer8(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer32(a)  (Bayer16( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer64(a)  (Bayer32( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer128(a) (Bayer64( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer256(a) (Bayer128(0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer512(a) (Bayer256(0.5 * (a)) * 0.25 + Bayer2(a))
float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

// simple x-y decorrelated noise seems enough
#define stepnoise0(p, size) rnd( floor(p/size)*size ) 
#define rnd(U) fract(sin( 1e3*(U)*mat2(1,-7.131, 12.9898, 1.233) )* 43758.5453)

//   joeedh's original noise (cleaned-up)
vec2 stepnoise(vec2 p, float size) {
    p = floor((p + 10.) / size) * size;          // is p+10. useful ?   
    p = fract(p * .1) + 1. + p * vec2(2, 3) / 1e4;
    p = fract(1e5 / (.1 * p.x * (p.y + vec2(0, 1)) + 1.));
    p = fract(1e5 / (p * vec2(.1234, 2.35) + 1.));
    return p;
}

// --- stippling mask  : regular stippling + per-tile random offset + tone-mapping

#define SEED1 1.705
#define DMUL  8.12235325       // are exact DMUL and -.5 important ?

float mask(vec2 p) {

    p += (stepnoise0(p, 5.5) - .5) * DMUL;   // bias [-2,2] per tile otherwise too regular
    float f = fract(p.x * SEED1 + p.y / (SEED1 + .15555)); //  weights: 1.705 , 0.5375

    //return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the image, see tests )
    // returned value in [0,37.2] , but < 0.57 with P=50% 

    return (pow(f, 150.) + 1.3 * f) * 0.43478260869; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%    
}


const float ITER = 36.;
const float FREQ = 0.7*3.14159;

vec2 hash21(float p)
{
	vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

//a wave with its gradient
vec4 sinC(vec2 uv, float freq, float i)
{
    vec2 ang = 2.*3.14159*hash21(i+ITER*float(GameTime/24000));
 	vec2 k = vec2(sin(ang.x), cos(ang.x));
    float x = freq*dot(uv,k)  - ang.y;
    float amp = 1.;//pow(freq, -3.5);
    float H = amp*sin(x);
    float Hdx = amp*cos(x);
    return vec4(H, Hdx*k*freq, 0.3*amp);
}

vec3 map(vec2 uv) {
    float freq = FREQ;
    float amp = 1.;

    vec4 h = vec4(0.0);
    for(float i = 0.; i < ITER; i++) 
    {        
        h += sinC(uv, freq, i);
    	freq *= 1.0 + 0.3/ITER;
    }
    return 0.5*h.xyz/h.w + 0.5;
}
float hash21(vec2 p)
{
	return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
// I verified value distribution is flat by taking a sample and
// importing into GIMP, taking a histogram, which looks flat.

// I had no tool for frequency analysis handy!
// well Fabrice pointed me at his stuff so 
// I made https://shadertoy.com/view/3tlczs from it

//float diff(float x)
//{
//    return .5 + .5 * (dFdx(x) + dFdy(x)); //fwidth(x);
//}
// 1st derivative in direction, central difference
// f'(x) = (f(x+y) - f(x-y)) / (2|y|)
#define diff2(f, x, y) (((f((x) + (y))) - (f((x) - (y)))) / (2.*length(y)))

// 2nd derivative
// f''(x) = ((f'(x+y) - f'(x)) / (|y|) 
//         - (f'(x) - f'(x-y)) / (|y|)) / (|y|)
#define diff3(f, x, y) (((f((x) + (y))) + (f((x) - (y))) - 2.*(f((x)))) / dot(y,y))
    // N.B. in 2D, this winds up being a plus-pattern high-pass filter
    // with center coeff -.5 and all the rest .125
    // after being normalized and all

float bluenoise(vec2 p)
{
    float s = .125;
    float h3 = .5 + s * (
          diff3(hash21, p, vec2(1,0)) 
        + diff3(hash21, p, vec2(0,1))
        );
    return h3;
}
    // with coordinate scale of 2., looks more like blue noise should.
    // value samples looks gaussian distributed to me;  some averaging happened somehow.
    // perhaps due to combination of two dimensional axes?
    // h2 histogram in GIMP looks gaussian bell curve shaped if the gamma is correct
    // but that apparently signifies nothing about the frequency distribution?!
	// since the original white noise range is 0 to 1,
    // the deltas can be -1 to 1, and there's 2 of them, so they need averaged,
    // which apparently gives us gaussian value distribution, bunched near mid-gray.
// h2 is violet noise, from what I understand
float violetnoise(vec2 p)
{
    p *= 2.; // without, the resolution seems halved, pixels doubled  -- or used to! fine, now, but comes out more distorted, smeared, without
    float s = .5; //sqrt(.5); //
    float h2 = .5 + s * (
          diff2(hash21, p, vec2(1,0))
        + diff2(hash21, p, vec2(0,1))
        );
    return h2;
}

float luma4(vec3 color) {
    return dot(color, vec3(0.21, 0.72, 0.07));
}

vec3 toLinear(vec3 sRGB) {
    return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}
