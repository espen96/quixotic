#version 150
#extension GL_ARB_gpu_shader5 : enable

vec4 textureGatherOffsets(sampler2D sampler, vec2 texCoord, ivec2[4] offsets, int channel)
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    return vec4(
        texelFetch(sampler, coord + offsets[0], 0)[channel], texelFetch(sampler, coord + offsets[1], 0)[channel],
        texelFetch(sampler, coord + offsets[2], 0)[channel], texelFetch(sampler, coord + offsets[3], 0)[channel]);
}

uniform sampler2D cloudsample;
uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TranslucentSampler;

uniform vec2 ScreenSize;
uniform vec2 OutSize;
uniform float Time;
in vec4 exposure;
in vec2 rodExposureDepth;
in mat4 gbufferModelView;
in mat4 gbufferProjectionInverse;
in mat4 gbufferProjection;
mat4 gbufferModelViewInverse = inverse(gbufferModelView);

vec2 getControl(int index, vec2 screenSize)
{
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}
vec2 start = getControl(0, OutSize);
vec2 inc = vec2(2.0 / OutSize.x, 0.0);
vec2 pbr1 = vec4((texture(DiffuseSampler, start + 100.0 * inc))).xy;
vec2 pbr2 = vec4((texture(DiffuseSampler, start + 101.0 * inc))).xy;
vec2 pbr3 = vec4((texture(DiffuseSampler, start + 102.0 * inc))).xy;
vec2 pbr4 = vec4((texture(DiffuseSampler, start + 103.0 * inc))).xy;

in vec3 ambientUp;
in vec3 ambientLeft;
in vec3 ambientRight;
in vec3 ambientB;
in vec3 ambientF;
in vec3 ambientDown;
in vec3 suncol;
in vec3 nsunColor;

in vec2 oneTexel;
in vec4 fogcol;

in vec2 texCoord;

in float near;
in float far;
in float end;
in float overworld;

in float rainStrength;
in vec3 sunVec;

in vec3 sunPosition;
in vec3 sunPosition2;
in vec3 sunPosition3;
in float skyIntensityNight;
in float skyIntensity;
in float sunElevation;

out vec4 fragColor;

#define TORCH_R 1.0
#define TORCH_G 0.7
#define TORCH_B 0.5

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FUDGE 32.0
#define MIN_LIGHT_AMOUNT 1.0 //[0.0 0.5 1.0 1.5 2.0 3.0 4.0 5.0]

#define Dirt_Amount 0.01

#define Dirt_Mie_Phase 0.4

#define Dirt_Absorb_R 0.65
#define Dirt_Absorb_G 0.85
#define Dirt_Absorb_B 1.05

#define Water_Absorb_R 0.25422
#define Water_Absorb_G 0.03751
#define Water_Absorb_B 0.01150

const float pi = 3.141592653589793238462643383279502884197169;

#define CLOUDS_QUALITY 0.1
float sqr(float x)
{
    return x * x;
}
float pow3(float x)
{
    return sqr(x) * x;
}
float pow4(float x)
{
    return sqr(x) * sqr(x);
}
float pow5(float x)
{
    return pow4(x) * x;
}
float pow6(float x)
{
    return pow5(x) * x;
}
float pow8(float x)
{
    return pow4(x) * pow4(x);
}
float pow16(float x)
{
    return pow8(x) * pow8(x);
}
float pow32(float x)
{
    return pow16(x) * pow16(x);
}
////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
vec2 unpackUnorm2x4(float pack)
{
    vec2 xy;
    xy.x = modf(pack * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

float map(float value, float min1, float max1, float min2, float max2)
{
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
float luma(vec3 color)
{
    return dot(color, vec3(0.299, 0.587, 0.114));
}

/////////////////////////////////////////////////////////////////////////

vec3 toLinear(vec3 sRGB)
{
    return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

float invLinZ(float lindepth)
{
    return -((2.0 * near / lindepth) - far - near) / (far - near);
}
float linZ(float depth)
{
    if (overworld != 1.0)
        return (2.0 * near * far) / (far + near - depth * (far - near));

    if (overworld == 1.0)
        return (2.0 * near) / (far + near - depth * (far - near));
}

vec4 backProject(vec4 vec)
{
    vec4 tmp = inverse(gbufferProjection * gbufferModelView) * vec;
    return tmp / tmp.w;
}

vec3 normVec(vec3 vec)
{
    return vec * inversesqrt(dot(vec, vec));
}

vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    color = pow(color, vec3(0.45454545454));
    return color;
}
vec3 lumaBasedReinhardToneMapping2(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    // color = pow(color, vec3(0.45454545454));
    return color;
}

float facos(float inX)
{
    const float C0 = 1.56467;
    const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}

vec3 drawSun(float cosY, float sunInt, vec3 nsunlight, vec3 inColor)
{
    return inColor + nsunlight * 1133 * pow3(smoothstep(0.99955640208, 0.99985963575, cosY)) * 0.62;
}

// Return random noise in the range [0.0, 1.0], as a function of x.
float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float StableStarField(in vec2 vSamplePos, float fThreshhold)
{
    vec2 floorSample = floor(vSamplePos);
    float StarVal = hash12(floorSample);

    float v1 = clamp(StarVal / (1.0 - fThreshhold) - fThreshhold / (1.0 - fThreshhold), 0.0, 1.0);

    StarVal = v1 * 30.0 * skyIntensityNight;
    return StarVal;
}

float stars(vec3 fragpos)
{
    float elevation = clamp(fragpos.y, 0., 1.);
    vec2 uv = fragpos.xz / (1. + elevation);

    return StableStarField(uv * 700., 0.999) * 0.5 * 0.3;
}
const float pidiv = 0.31830988618; // 1/pi

// Mie phase function
float phaseg(float x, float g)
{
    float gg = sqr(g);
    return ((-0.25 * gg + 0.25) * pidiv) * pow(-2.0 * g * x + gg + 1.0, -1.5);
}

vec2 Nnoise(vec2 coord)
{
    float x = sin(coord.x * 100.0) * 0.1 + sin((coord.x * 200.0) + 3.0) * 0.05 +
              fract(cos((coord.x * 19.0) + 1.0) * 33.33) * 0.15;
    float y = sin(coord.y * 100.0) * 0.1 + sin((coord.y * 200.0) + 3.0) * 0.05 +
              fract(cos((coord.y * 19.0) + 1.0) * 33.33) * 0.25;
    return vec2(x, y);
}

vec3 reinhard(vec3 x)
{
    x *= 1.66;
    return x / (1.0 + x);
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

#define projMAD2(m, v) (diagonal3(m) * (v) + vec3(0, 0, m[3].b))

vec3 toClipSpace3(vec3 viewSpacePosition)
{
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

vec3 worldToView(vec3 worldPos)
{
    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos + gbufferModelView[3];

    return pos.xyz;
}

vec3 nvec3(vec4 pos)
{
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos)
{
    return vec4(pos.xyz, 1.0);
}

float cdist(vec2 coord)
{
    return max(abs(coord.x - 0.5), abs(coord.y - 0.5)) * 1.85;
}
float cdist2(vec2 coord)
{
    vec2 vec = abs(coord * 2.0 - 1.0);
    float d = max(vec.x, vec.y);
    return 1.0 - d * d;
}

vec3 cosineHemisphereSample2(vec2 Xi)
{
    float r = sqrt(Xi.x);
    float theta = 2.0 * 3.14159265359 * Xi.y;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec3(x, y, sqrt(max(0.0f, 1 - Xi.x)));
}

vec3 TangentToWorld2(vec3 N, vec3 H)
{
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(UpVector, N));
    vec3 B = cross(N, T);

    return vec3((T * H.x) + (B * H.y) + (N * H.z));
}

vec2 R2_samples2(int n)
{
    vec2 alpha = vec2(0.75487765, 0.56984026);
    return fract(alpha * n);
}

vec3 reinhard_jodie(vec3 v)
{
    float l = luma(v);
    vec3 tv = v / (1.0f + v);
    tv = mix(v / (1.0f + l), tv, tv);
    return tv;
}

#define tau 6.2831853071795864769252867665590

#define AOQuality 0 //[0 1 2]
#define AORadius 2.0

float dither5x3()
{
    const int ditherPattern[15] = int[15](9, 3, 7, 12, 0, 11, 5, 1, 14, 8, 2, 13, 10, 4, 6);

    vec2 position = floor(mod(vec2(texCoord.s * ScreenSize.x, texCoord.t * ScreenSize.y), vec2(5.0, 3.0)));

    int dither = ditherPattern[int(position.x) + int(position.y) * 5];

    return float(dither) * 0.0666666666666667f;
}
#define g(a) (-4. * a.x * a.y + 3. * a.x + a.y * 2.)

float bayer16x16(vec2 p)
{
    vec2 m0 = vec2(mod(floor(p * 0.125), 2.));
    vec2 m1 = vec2(mod(floor(p * 0.25), 2.));
    vec2 m2 = vec2(mod(floor(p * 0.5), 2.));
    vec2 m3 = vec2(mod(floor(p), 2.));

    return (g(m0) + g(m1) * 4.0 + g(m2) * 16.0 + g(m3) * 64.0) * 0.003921568627451;
}
#undef g

float dither = bayer16x16(gl_FragCoord.xy);

// Dithering from Jodie
float bayer2(vec2 a)
{
    a = floor(a);
    return fract(dot(a, vec2(.5, a.y * .75)));
}

#define bayer4(a) (bayer2(.5 * (a)) * .25 + bayer2(a))
#define bayer8(a) (bayer4(.5 * (a)) * .25 + bayer2(a))
#define bayer16(a) (bayer8(.5 * (a)) * .25 + bayer2(a))
#define bayer32(a) (bayer16(.5 * (a)) * .25 + bayer2(a))
#define bayer64(a) (bayer32(.5 * (a)) * .25 + bayer2(a))
#define bayer128(a) (bayer64(.5 * (a)) * .25 + bayer2(a))

float dither64 = bayer64(gl_FragCoord.xy);
vec2 OffsetDist(float x)
{
    float n = fract(x * 8.0) * 3.1415;
    return vec2(cos(n), sin(n)) * x;
}
float linZt(float depth)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}
float dbao2(sampler2D depth)
{
    float ao = 0.0;

    float depthsamp = texture2D(depth, texCoord).r;
    if (depthsamp >= 1.0)
        return 1.0;

    float hand = float(depthsamp < 0.56);
    depthsamp = linZt(depthsamp);
    float aspectRatio = ScreenSize.x / ScreenSize.y;

    dither = dither64;

    float currentStep = 0.2 * dither + 0.2;

    float radius = 0.35;
    float fovScale = 0.729;
    float distScale = max(1 * depthsamp + near, 5.0);
    vec2 scale = radius * vec2(1.0 / aspectRatio, 1.0) * fovScale / distScale;
    float mult = (0.7 / radius) * (far - near);

    for (int i = 0; i < 4; i++)
    {
        vec2 offset = OffsetDist(currentStep) * scale;
        float angle = 0.0, dist = 0.0;

        for (int i = 0; i < 2; i++)
        {
            float sampleDepth = linZt(texture2D(depth, texCoord + offset).r);
            float sampl = (depthsamp - sampleDepth) * mult;
            angle += clamp(0.5 - sampl, 0.0, 1.0);
            dist += clamp(0.25 * sampl - 1.0, 0.0, 1.0);
            offset = -offset;
        }

        ao += clamp(angle + dist, 0.0, 1.0);
        currentStep += 0.2;
    }
    ao *= 0.25;

    return ao;
}

float dbao(sampler2D depth)
{
    float ao = 0.0;
    float aspectRatio = ScreenSize.x / ScreenSize.y;

    const int aoloop = 2; // 3
    const int aoside = AOQuality + 2;

    float radius = AORadius * 0.5 / pow(2.0, AOQuality * 0.5);
    float dither2 = fract(dither5x3() - dither64);
    float d = linZ(texture(depth, texCoord.xy).r);
    const float piangle = 0.0174603175;

    float rot = 360 / aoside * (dither2 + fract(Time * 0.125));

    float size = radius * dither64;
    float sd = 0.0;
    float angle = 0.0;
    float dist = 0.0;
    vec2 scale = vec2(1.0 / aspectRatio, 1.0) * gbufferProjection[1][1] / (2.74747742 * max(far * d, 6.0));

    for (int i = 0; i < aoloop; i++)
    {
        for (int j = 0; j < aoside; j++)
        {
            sd = linZ(texture(depth, texCoord.xy + vec2(cos(rot * piangle), sin(rot * piangle)) * size * scale).r);
            float samples = far * (d - sd) / size;
            angle = clamp(0.5 - samples, 0.0, 1.0);
            dist = clamp(0.0625 * samples, 0.0, 1.0);
            sd = linZ(texture(depth, texCoord.xy - vec2(cos(rot * piangle), sin(rot * piangle)) * size * scale).r);
            samples = far * (d - sd) / size;
            angle += clamp(0.5 - samples, 0.0, 1.0);
            dist += clamp(0.0625 * samples, 0.0, 1.0);
            ao += clamp(angle + dist, 0.0, 1.0);
            rot += 180.0 / aoside;
        }
        rot += 180.0 / aoside;
        size += radius * ((AOQuality * 0.5) * dither64 + 1.0);
        radius += (AOQuality * 0.5) * radius;
        angle = 0.0;
        dist = 0.0;
    }

    ao /= aoloop * aoside;

    return pow(ao, AOQuality * 0.25 + 1.5);
}

// simplified version of joeedh's https://www.shadertoy.com/view/Md3GWf
// see also https://www.shadertoy.com/view/MdtGD7

// --- checkerboard noise : to decorelate the pattern between size x size tiles

// simple x-y decorrelated noise seems enough
#define stepnoise0(p, size) rnd(floor(p / size) * size)
#define rnd(U) fract(sin(1e3 * (U)*mat2(1, -7.131, 12.9898, 1.233)) * 43758.5453)

//   joeedh's original noise (cleaned-up)
vec2 stepnoise(vec2 p, float size)
{
    p = floor((p + 10.) / size) * size; // is p+10. useful ?
    p = fract(p * .1) + 1. + p * vec2(2, 3) / 1e4;
    p = fract(1e5 / (.1 * p.x * (p.y + vec2(0, 1)) + 1.));
    p = fract(1e5 / (p * vec2(.1234, 2.35) + 1.));
    return p;
}

// --- stippling mask  : regular stippling + per-tile random offset +
// tone-mapping

#define SEED1 1.705
#define DMUL 8.12235325 // are exact DMUL and -.5 important ?

float mask(vec2 p)
{
    p += (stepnoise0(p, 5.5) - .5) * DMUL;                 // bias [-2,2] per tile otherwise too regular
    float f = fract(p.x * SEED1 + p.y / (SEED1 + .15555)); //  weights: 1.705 , 0.5375

    // return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the
    // image, see tests ) returned value in [0,37.2] , but < 0.57 with P=50%

    return (pow(f, 150.) + 1.3 * f) * 0.43478260869; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%
}

vec3 viewToWorld(vec3 viewPos)
{
    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = gbufferModelViewInverse * pos;

    return pos.xyz;
}
#define fsign(a) (clamp((a)*1e35, 0., 1.) * 2. - 1.)

float interleaved_gradientNoise()
{
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y) + Time / 1.6128);
}
float triangularize(float dither)
{
    float center = dither * 2.0 - 1.0;
    dither = center * inversesqrt(abs(center));
    return clamp(dither - fsign(center), 0.0, 1.0);
}
vec3 fp10Dither(vec3 color, float dither)
{
    const vec3 mantissaBits = vec3(6., 6., 5.);
    vec3 exponent = floor(log2(color));
    return color + dither * exp2(-mantissaBits) * exp2(exponent);
}

vec3 getDepthPoint(vec2 coord, float depth)
{
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; // convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;

    return pos.xyz;
}
vec3 constructNormal(float depthA, vec2 texCoords, sampler2D depthtex, float water)
{
    vec2 offsetB = vec2(0.0, oneTexel.y);
    vec2 offsetC = vec2(oneTexel.x, 0.0);
    float depthB = texture(depthtex, texCoords + offsetB).r;
    float depthC = texture(depthtex, texCoords + offsetC).r;
    vec3 A = getDepthPoint(texCoords, depthA);
    // A += pow4(texture(DiffuseSampler, texCoord).g) * 0.01 * 1 - water;

    vec3 B = getDepthPoint(texCoords + offsetB, depthB);
    // B += pow4(texture(DiffuseSampler, texCoord + offsetB * 3.0).g) * 0.01 * 1 - water;

    vec3 C = getDepthPoint(texCoords + offsetC, depthC);
    // C += pow4(texture(DiffuseSampler, texCoord + offsetC * 3.0).g) * 0.01 * 1 - water;

    vec3 AB = normalize(B - A);
    vec3 AC = normalize(C - A);

    vec3 normal = -cross(AB, AC);
    // normal.z = -normal.z;

    return normalize(normal);
}
float brightnessContrast(float value, float brightness, float contrast)
{
    return (value - 0.5) * contrast + 0.5 + brightness;
}
float getRawDepth(vec2 uv)
{
    return texture(TranslucentDepthSampler, uv).x;
}

// inspired by keijiro's depth inverse projection
// https://github.com/keijiro/DepthInverseProjection
// constructs view space ray at the far clip plane from the screen uv
// then multiplies that ray by the linear 01 depth
vec3 viewSpacePosAtScreenUV(vec2 uv)
{
    vec3 viewSpaceRay = (gbufferProjectionInverse * vec4(uv * 2.0 - 1.0, 1.0, 1.0) * near).xyz;
    float rawDepth = getRawDepth(uv);
    return viewSpaceRay * (linZ(rawDepth) + pow8(luma(texture(DiffuseSampler, uv).xyz)) * 0.00);
}
vec3 viewSpacePosAtPixelPosition(vec2 vpos)
{
    vec2 uv = vpos * oneTexel.xy;
    return viewSpacePosAtScreenUV(uv);
}

vec2 unpackUnorm2x4v2(vec4 pack)
{
    vec2 xy;
    float pack2 = (pack.x + pack.y + pack.z + pack.z) / 4;
    xy.x = modf(pack2 * 255.0 / 16.0, xy.y);
    return xy * vec2(1.06666666667, 1.0 / 15.0);
}

void main()
{
    vec4 outcol = vec4(0.0, 0.0, 0.0, 1.0);
    float depth = texture(TranslucentDepthSampler, texCoord).r;
    float depth2 = texture(DiffuseDepthSampler, texCoord).r;
    float noise = clamp(mask(gl_FragCoord.xy + (Time * 100)), 0, 1);
    bool isWater = (texture(TranslucentSampler, texCoord).a * 255 == 200);

    vec3 screenPos = vec3(texCoord, depth);
    vec3 clipPos = screenPos * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 p3 = mat3(gbufferModelViewInverse) * viewPos;
    vec3 view = normVec(p3);
    ivec2 texoffsets[4] = ivec2[](ivec2(0, 1), ivec2(1, 0), -ivec2(0, 1), -ivec2(1, 0));
    ivec2 texoffsets2[4] = ivec2[](ivec2(0, -1), ivec2(-1, 0), -ivec2(0, -1), -ivec2(-1, 0));

    vec4 depthgather = textureGatherOffsets(DiffuseDepthSampler, texCoord, texoffsets2, 0);

    bool sky = depth2 >= 1.0;
    bool edgex = depthgather.x >= 1.0;
    bool edgey = depthgather.y >= 1.0;
    bool edgez = depthgather.z >= 1.0;
    bool edgew = depthgather.w >= 1.0;
    float vdots = dot(view, sunPosition2);
    bool skycheck = abs(float(edgex) + float(edgey) + float(edgez) + float(edgew)) > 0.0;

    if (sky && overworld == 1.0)
    {

        vec4 cloud = vec4(0, 0, 0, 1);
        cloud = texture(cloudsample, texCoord * CLOUDS_QUALITY);
        vec3 atmosphere = toLinear(cloud.rgb);
        if (view.y > 0.)
        {

            atmosphere += ((stars(view) * 2.0) * clamp(1 - (rainStrength * 1), 0, 1)) * 0.05;
            atmosphere += drawSun(vdots, 0, suncol.rgb * 0.006, vec3(0.0)) * clamp(1 - (rainStrength * 1), 0, 1);
        }

        outcol.rgb = lumaBasedReinhardToneMapping(atmosphere);
    }
    else
    {

        vec2 texCoord = texCoord;
        vec3 wnormal = vec3(0.0);
        vec3 normal = normalize(constructNormal(depth, texCoord, TranslucentDepthSampler, float(isWater)));

        float mod2 = gl_FragCoord.x + gl_FragCoord.y;
        float res = mod(mod2, 2.0f);

        vec4 OutTexel3 = (texture(DiffuseSampler, texCoord).rgba);
        vec4 cbgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 2);
        vec4 crgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 0);
        float lmx = clamp(mix(OutTexel3.b, dot(cbgather, vec4(1.0)) / 4, res), 0.0, 1);
        float lmy = clamp(mix(OutTexel3.r, dot(crgather, vec4(1.0)) / 4, res), 0.0, 1);

        vec4 depthgather = textureGatherOffsets(TranslucentDepthSampler, texCoord, texoffsets, 0);
        vec4 lmgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 3);
        float depthtest = dot(depthgather, vec4(1.0)) * 0.25;
        depthtest = round(clamp(float(depthtest - depth) * 10000 - 1, 0, 1));

        vec2 lmtrans = unpackUnorm2x4(OutTexel3.a);
        vec2 lmtrans10 = unpackUnorm2x4v2(lmgather);
        lmtrans10 = mix(lmtrans10, lmtrans, depthtest);

        OutTexel3.r = clamp(mix(dot(crgather, vec4(1.0)) / 4, OutTexel3.r, res), 0.0, 1);
        OutTexel3.b = clamp(mix(dot(cbgather, vec4(1.0)) / 4, OutTexel3.b, res), 0.0, 1);

        if (skycheck && res != 1)
        {
            OutTexel3.rb *= 0.5;
        }

        vec3 OutTexel = toLinear(OutTexel3.rgb);

        if (overworld == 1.0)
        {
            float ao = 1.0;
            int isEyeInWater = 0;
            int isEyeInLava = 0;
            if (fogcol.a > 0.078 && fogcol.a < 0.079)
                isEyeInWater = 1;
            if (fogcol.r == 0.6 && fogcol.b == 0.0)
                isEyeInLava = 1;

            float screenShadow = clamp((pow32(lmx)) * 100, 0.0, 1.0) * lmx;
            ao = dbao2(TranslucentDepthSampler);

            vec3 normal3 = (normal);
            normal = viewToWorld(normal3);
            vec3 ambientCoefs = normal / dot(abs(normal), vec3(1.0));
            float minLight = MIN_LIGHT_AMOUNT * 0.007 /
                             (exposure.x + rodExposureDepth.x / (rodExposureDepth.x + 1.0) * exposure.x * 1.);

            vec3 ambientLight = ambientUp * clamp(ambientCoefs.y, 0., 1.);
            ambientLight += ambientDown * clamp(-ambientCoefs.y, 0., 1.);
            ambientLight += ambientRight * clamp(ambientCoefs.x, 0., 1.);
            ambientLight += ambientLeft * clamp(-ambientCoefs.x, 0., 1.);
            ambientLight += ambientB * clamp(ambientCoefs.z, 0., 1.);
            ambientLight += ambientF * clamp(-ambientCoefs.z, 0., 1.);
            ambientLight *= 1.0;
            ambientLight *= (1.0 + rainStrength * 0.2);

            float lumAC = luma(ambientLight);
            vec3 diff = ambientLight - lumAC;
            ambientLight = ambientLight + diff * (-lumAC * 1.0 + 0.5);
            ambientLight =
                clamp(ambientLight * (pow8(lmx) * 1.5) +
                          (pow3(lmy) * 3.0) * (vec3(TORCH_R, TORCH_G, TORCH_B) * vec3(TORCH_R, TORCH_G, TORCH_B)),
                      0.0025, 10.0);

            float shadeDir = max(0.0, dot(normal, sunPosition2));
            shadeDir *= screenShadow;

            shadeDir = clamp(shadeDir * pow3(lmx) * ao, 0, 1);
            vec3 suncol = (suncol)*clamp(skyIntensity * 3.0, 0.15, 1);
            vec3 shading = ((suncol * 3.75) * shadeDir) + ambientLight * ao;

            shading = mix(ambientLight, shading, 1 - (rainStrength * lmx));

            if (isWater)
                shading = ambientLight;
            vec3 dlight = (OutTexel * shading);
            outcol.rgb = lumaBasedReinhardToneMapping(dlight);

            // outcol.a = clamp(grCol, 0, 1);
            // outcol.rgb = vec3(lumaBasedReinhardToneMapping(shading));
            ///---------------------------------------------
            // outcol.rgb = lumaBasedReinhardToneMapping(clamp(vec3(pbr.rgb), 0.01, 1));
            // if(luma(ambientLight )>1.0) outcol.rgb = vec3(1.0,0,0);
            ///---------------------------------------------
        }
        else
        {

            float ao = 1.0;
            ao = dbao2(TranslucentDepthSampler);
            float lumC = luma(fogcol.rgb);
            vec3 diff = fogcol.rgb - lumC;

            vec3 ambientLight = clamp((diff + 0.1) * (0.25) + (pow3(lmy) * 2.0) * (vec3(TORCH_R, TORCH_G, TORCH_B) *
                                                                                   vec3(TORCH_R, TORCH_G, TORCH_B)),
                                      0.0005, 10.0);
            outcol.rgb = lumaBasedReinhardToneMapping(OutTexel.rgb * ambientLight * ao);
        }
    }

    if (isWater)
    {

        vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B) * fogcol.rgb;
        vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
        vec3 totEpsilon = dirtEpsilon * Dirt_Amount + waterEpsilon;
        outcol.rgb *= exp(-length(viewPos) * totEpsilon);
    }
    fragColor = outcol;
    /*
    vec4 numToPrint = vec4(gbufferProjection[2].xyzw);

    // Define text to draw
    clearTextBuffer();
    c('R'); c(':'); c(' '); floatToDigits(numToPrint.r);
    printTextAt(1.0, 1.0);

    clearTextBuffer();
    c('G'); c(':'); c(' '); floatToDigits(numToPrint.g);
    printTextAt(1.0, 2.0);

    clearTextBuffer();
    c('B'); c(':'); c(' '); floatToDigits(numToPrint.b);
    printTextAt(1.0, 3.0);

    clearTextBuffer();
    c('A'); c(':'); c(' '); floatToDigits(numToPrint.a);
    printTextAt(1.0, 4.0);

    fragColor += colour;
    */
}
