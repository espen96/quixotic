#version 150
#extension GL_ARB_gpu_shader5 : enable
uniform sampler2D DiffuseSampler;
uniform sampler2D TranslucentDepthSampler;

uniform float Time;
uniform vec2 ScreenSize;

in vec2 oneTexel;
in vec2 texCoord;
in mat4 gbufferProjection;
in float near;
in float far;
in float overworld;
in vec3 sunVec;
in mat4 gbufferProjectionInverse;

out vec4 fragColor;

float sqr(float x) {
    return x * x;
}
float pow3(float x) {
    return sqr(x) * x;
}
float pow4(float x) {
    return sqr(x) * sqr(x);
}
float pow5(float x) {
    return pow4(x) * x;
}
float pow6(float x) {
    return pow5(x) * x;
}
float pow8(float x) {
    return pow4(x) * pow4(x);
}
float pow16(float x) {
    return pow8(x) * pow8(x);
}
float pow32(float x) {
    return pow16(x) * pow16(x);
}
////////////////////////////////
    #define sssMin 22
    #define sssMax 47
    #define lightMin 48
    #define lightMax 72
    #define roughMin 73
    #define roughMax 157
    #define metalMin 158
    #define metalMax 251
//////////////////////////////////////////////////////////////////////////////////////////
vec2 unpackUnorm2x4(float pack) {
    vec2 xy;
    xy.x = modf(pack * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

/////////////////////////////////////////////////////////////////////////

float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

#define  projMAD2(m, v) (diagonal3(m) * (v) + vec3(0,0,m[3].b))

#define viewMAD(m, v) (mat3(m) * (v) + (m)[3].xyz)
#define diag3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define diag4(mat) vec4(diag3(mat), (mat)[2].w)

#define isaturate(x) clamp(x, 0.0, 1.0)
#define ircp(x) (1.0 / x)

#define sstep(x, low, high) smoothstep(low, high, x)
#define stex(x) texture(x, coord)
#define stexLod(x, lod) textureLod(x, coord, lod)
#define landMask(x) (x < 1.0)
#define icube_smooth(x) (x * x) * (3.0 - 2.0 * x)

#define expf(x) exp2((x) * rlog2)
#define log10(x) (log(x) * rcp(log(10.0)))

#define isnan3(a) (isnan(a.x) || isnan(a.y) || isnan(a.z))
#define isinf3(a) (isinf(a.x) || isinf(a.y) || isinf(a.z))

#define isnan4(a) (isnan(a.x) || isnan(a.y) || isnan(a.z) || isnan(a.w))
#define isinf4(a) (isinf(a.x) || isinf(a.y) || isinf(a.z) || isinf(a.w))

float max2(vec2 x) {
    return max(x.x, x.y);
}

float max3(float x, float y, float z) {
    return max(x, max(y, z));
}
float max3(vec3 x) {
    return max(x.x, max(x.y, x.z));
}
float min3(float x, float y, float z) {
    return min(x, min(y, z));
}
float min3(vec3 x) {
    return min(x.x, min(x.y, x.z));
}

vec3 screenSpaceToViewSpace(vec3 screenPosition, mat4 projectionInverse) {
    screenPosition = screenPosition * 2.0 - 1.0;

    vec3 viewPosition = vec3(vec2(projectionInverse[0].x, projectionInverse[1].y) * screenPosition.xy + projectionInverse[3].xy, projectionInverse[3].z);

    viewPosition /= projectionInverse[2].w * screenPosition.z + projectionInverse[3].w;

    return viewPosition;
}

float screenSpaceToViewSpace(float depth, mat4 projectionInverse) {
    depth = depth * 2.0 - 1.0;
    return projectionInverse[3].z / (projectionInverse[2].w * depth + projectionInverse[3].w);
}

vec3 viewSpaceToScreenSpace(vec3 viewPosition, mat4 projection) {
    vec3 screenPosition = vec3(projection[0].x, projection[1].y, projection[2].z) * viewPosition + projection[3].xyz;
    screenPosition /= -viewPosition.z;

    return screenPosition * 0.5 + 0.5;
}

float viewSpaceToScreenSpace(float depth, mat4 projection) {
    return ((projection[2].z * depth + projection[3].z) / -depth) * 0.5 + 0.5;
}

vec3 viewSpaceToSceneSpace(in vec3 viewPosition, in mat4 modelViewInverse) {
    return mat3(modelViewInverse) * viewPosition + modelViewInverse[3].xyz;
}

vec3 sceneSpaceToShadowView(in vec3 scenePosition, in mat4 shadowMV) {
    return mat3(shadowMV) * scenePosition + shadowMV[3].xyz;
}

vec3 shadowViewToShadowClip(in vec3 shadowView, in mat4 shadowProj) {
    return mat3(shadowProj) * shadowView + shadowProj[3].xyz;
}
float bayer2(vec2 c) {
    c = 0.5 * floor(c);
    return fract(1.5 * fract(c.y) + c.x);
}
float bayer4(vec2 c) {
    return 0.25 * bayer2(0.5 * c) + bayer2(c);
}
float bayer8(vec2 c) {
    return 0.25 * bayer4(0.5 * c) + bayer2(c);
}
float bayer16(vec2 c) {
    return 0.25 * bayer8(0.5 * c) + bayer2(c);
}
float bayer32(vec2 c) {
    return 0.25 * bayer16(0.5 * c) + bayer2(c);
}
float bayer64(vec2 c) {
    return 0.25 * bayer32(0.5 * c) + bayer2(c);
}
float bayer128(vec2 c) {
    return 0.25 * bayer64(0.5 * c) + bayer2(c);
}

float maxof(vec2 x) {
    return max(x.x, x.y);
}
float maxof(vec3 x) {
    return max(max(x.x, x.y), x.z);
}
float maxof(vec4 x) {
    return max(max(x.x, x.y), max(x.z, x.w));
}
float minof(vec2 x) {
    return min(x.x, x.y);
}
float minof(vec3 x) {
    return min(min(x.x, x.y), x.z);
}
float minof(vec4 x) {
    return min(min(x.x, x.y), min(x.z, x.w));
}

vec2 sincos(float x) {
    return vec2(sin(x), cos(x));
}

vec2 circleMap(in float point) {
    return vec2(cos(point), sin(point));
}

vec2 Rotate(vec2 vector, float angle) {
    vec2 sc = sincos(angle);
    return vec2(sc.y * vector.x + sc.x * vector.y, sc.y * vector.y - sc.x * vector.x);
}

vec3 Rotate(vec3 vector, vec3 axis, float angle) {
	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    vec2 sc = sincos(angle);
    return sc.y * vector + sc.x * cross(axis, vector) + (1.0 - sc.y) * dot(axis, vector) * axis;
}

//#define max3f(x, y, z) max(x, max(y, z))

//#define max3(x) max(x.x, max(x.y, x.z))

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}
float R2_dither() {
    vec2 alpha = vec2(0.75487765, 0.56984026);
    return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0 / 1.6180339887 * Time);
}

float rayTraceShadow(vec3 dir, vec3 position, float dither) {
    float stepSize = map(linZ( texture(TranslucentDepthSampler, texCoord).r) * dither, 0, 100, 25, 50);
    int maxSteps = 50;

    vec3 clipPosition = nvec3(gbufferProjection * nvec4(position)) * 0.5 + 0.5;
    float rayLength = ((position.z + dir.z * sqrt(3.0) * far) > -sqrt(3.0) * near) ? (-sqrt(3.0) * near - position.z) / dir.z : sqrt(3.0) * far;

    vec3 end = toClipSpace3(position + dir * rayLength);
    vec3 direction = end - clipPosition;

    float len = max(abs(direction.x) / oneTexel.x, abs(direction.y) / oneTexel.y) / stepSize;

    vec3 maxLengths = (step(0., direction) - clipPosition) / direction;
    float mult = min(min(maxLengths.x, maxLengths.y), maxLengths.z);
    vec3 stepv = direction / len;

    int iterations = min(int(min(len, mult * len) - 2), maxSteps);

    vec3 spos = clipPosition + stepv / stepSize;

    for(int i = 0; i < int(iterations); i++) {
        spos += stepv * dither;
        float sp = texture(TranslucentDepthSampler, spos.xy).x;

        if(sp < spos.z + 0.000001) {

            float dist = abs(linZ(sp) - linZ(spos.z)) / linZ(spos.z);

            if(dist < 0.05)
                return exp2(position.z / 8.);

        }

    }
    return 1.0;
}
#define _saturate(x) clamp(x, 0.0, 1.0)
#define _saturateInt(x) clamp(x, 0, 1)
vec2 viewSize = ScreenSize;
float saturate(in float x) {
    return _saturate(x);
}
int saturate(in int x) {
    return _saturateInt(x);
}
vec2 saturate(in vec2 x) {
    return _saturate(x);
}
vec3 saturate(in vec3 x) {
    return _saturate(x);
}
vec4 saturate(in vec4 x) {
    return _saturate(x);
}
const float phi = sqrt(5.0) * 0.5 + 0.5;
float ascribeDepth(in float depth, in float amount) {
    depth = screenSpaceToViewSpace(depth, gbufferProjectionInverse);
    depth *= 1.0 + amount;
    return viewSpaceToScreenSpace(depth, gbufferProjection);
}

bool raytraceIntersection(in sampler2D depthtexture, in vec3 startPosition, in vec3 rayDirection, out vec3 hitPosition, in uint stride, in float depthLeniency, in float maxSteps) {
    hitPosition = startPosition;
    startPosition = screenSpaceToViewSpace(startPosition, gbufferProjectionInverse);

    vec3 rayStep = startPosition + abs(startPosition.z) * rayDirection;
    rayStep = viewSpaceToScreenSpace(rayStep, gbufferProjection) - hitPosition;
    rayStep *= minof((step(0.0, rayStep) - hitPosition) / rayStep);

    hitPosition.xy *= viewSize;
    rayStep.xy *= viewSize;

    rayStep /= maxof(abs(rayStep.xy));

    float dither = floor(stride * fract(fract(Time * (1.0 / phi)) + bayer128(gl_FragCoord.st)) + 1.0);

    vec3 steps = (step(0.0, rayStep) * vec3(viewSize - 1.0, 1.0) - hitPosition) / rayStep;
    steps.z += float(stride);
    steps = clamp(steps, 0.0, maxSteps);
    float tMax = min(minof(steps), maxof(viewSize));

    vec3 rayOrigin = hitPosition;

    float ascribeAmount = depthLeniency * float(stride) * oneTexel.y * gbufferProjectionInverse[1].y;

    bool intersected = false;
    float t = dither;
    while(t < tMax && !intersected) {
        float stepStride = t == dither ? dither : float(stride);

        hitPosition = rayOrigin + t * rayStep;

        float maxZ = hitPosition.z;
        float minZ = hitPosition.z - stepStride * abs(rayStep.z);

        float depth = texelFetch(depthtexture, ivec2(hitPosition.xy), 0).r;
        float ascribedDepth = ascribeDepth(depth, ascribeAmount);

        intersected = maxZ >= depth && minZ <= ascribedDepth;
        intersected = intersected && depth < 1.0;
        if(saturate(hitPosition.xy) == hitPosition.xy) {
            intersected = false;
            break;
        }
        if(!intersected) {
            t += float(stride);
        }
    }

    if(intersected) {
        bool refinementIntersection = true;
        float refinementStride = stride;
        for(int i = 0; i < findMSB(stride); ++i) {
            t += (refinementIntersection && t > 0.0 ? -1.0 : 1.0) * (refinementStride *= 0.5);
            hitPosition = rayOrigin + t * rayStep;

            float maxZ = hitPosition.z;
            float minZ = hitPosition.z - stride * abs(rayStep.z);

            float depth = texelFetch(depthtexture, ivec2(hitPosition.xy), 0).r;
            float ascribedDepth = ascribeDepth(depth, ascribeAmount);

            refinementIntersection = maxZ >= depth && minZ <= ascribedDepth;
            intersected = intersected && depth < 1.0;
        }
    }

    hitPosition.xy *= oneTexel;

    return intersected;
}

// simplified version of joeedh's https://www.shadertoy.com/view/Md3GWf
// see also https://www.shadertoy.com/view/MdtGD7

// --- checkerboard noise : to decorelate the pattern between size x size tiles 

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

void main() {
    vec4 outcol = vec4(0.0, 0.0, 0.0, 1.0);
    float depth = texture(TranslucentDepthSampler, texCoord).r;

    vec3 screenPos = vec3(texCoord, depth);
    vec3 clipPos = screenPos * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;

    bool sky = depth >= 1.0;

    if(!sky && overworld == 1.0) {
        float noise = R2_dither();

    
            //vec3 hitPosition;
            //vec3 rayDirection = reflect(rayDirection, mat3(gbufferModelView) * sd.flatNormal);
            //bool hit = raytraceIntersection(depthtex1, pd.screenPosition[0], rayDirection, hitPosition, REFLECTION_RAY_STRIDE * 2u, 16.0, pixelCount * rcp(2.0));
            // bool raytraceIntersection(in sampler2D depthtexture, in vec3 startPosition, in vec3 rayDirection, out vec3 hitPosition, in uint stride, in float depthLeniency, in float maxSteps) {
            //bool shadow =  raytraceIntersection(TranslucentDepthSampler, screenPos, sunVec, vec3(1.0) ,uint(10.0), 0.1, 50);

        float screenShadow = rayTraceShadow(sunVec, viewPos, noise);

        outcol.rgb = clamp(vec3(screenShadow), 0.01, 1);

    }

    fragColor = outcol;

}
