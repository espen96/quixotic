#version 150
#extension GL_ARB_gpu_shader5 : enable
uniform sampler2D DiffuseSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D cloudsample;
uniform sampler2D shadow;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D PreviousFrameSampler;

uniform vec2 ScreenSize;
uniform float Time;

in vec3 ambientUp;
in vec3 ambientLeft;
in vec3 ambientRight;
in vec3 ambientB;
in vec3 ambientF;
in vec3 ambientDown;
in vec3 suncol;
in vec3 zMults;

in vec2 oneTexel;
in vec4 fogcol;

in vec2 texCoord;

in mat3 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 wgbufferModelViewInverse;
in mat4 gbufferProjection;

in float near;
in float far;
in float end;
in float overworld;

in float rainStrength;
in vec3 sunVec;

in vec3 sunPosition3;
in float skyIntensityNight;

mat4 gbufferProjectionInverse = inverse(gbufferProjection);

out vec4 fragColor;

#define AOStrength 1.0
#define steps 6

#define TORCH_R 1.0 
#define TORCH_G 0.7 
#define TORCH_B 0.5 

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FUDGE 32.0

#define Dirt_Amount 0.01 

#define Dirt_Mie_Phase 0.4  //Values close to 1 will create a strong peak around the sun and weak elsewhere, values close to 0 means uniform fog. 

#define Dirt_Absorb_R 0.65 
#define Dirt_Absorb_G 0.85 
#define Dirt_Absorb_B 1.05

#define Water_Absorb_R 0.25422
#define Water_Absorb_G 0.03751
#define Water_Absorb_B 0.01150

#define BASE_FOG_AMOUNT 0.2 //Base fog amount amount (does not change the "cloudy" fog)
#define CLOUDY_FOG_AMOUNT 1.0 
#define FOG_TOD_MULTIPLIER 1.0 //Influence of time of day on fog amount
#define FOG_RAIN_MULTIPLIER 1.0 //Influence of rain on fog amount

#define SSAO_SAMPLES 4

#define NORMDEPTHTOLERANCE 1.0

#define CLOUDS_QUALITY 0.5
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

vec4 pbr(vec2 in1, vec2 in2, vec3 test) {

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);

    vec4 alphatest = vec4(0.0);
    vec4 pbr = vec4(0.0);

    float maps1 = mix(in1.x, in2.x, res);
    float maps2 = mix(in2.x, in1.x, res);

    maps1 = map(maps1, 0, 1, 128, 255);
    if(maps1 == 128)
        maps1 = 0.0;
    maps2 = map(maps2, 0, 1, 0, 128);

    float maps = (maps1 + maps2) / 255;
    float expanded = int(maps * 255);

    if(expanded >= sssMin && expanded <= sssMax)
        alphatest.g = maps; // SSS
    float sss = map(alphatest.g * 255, sssMin, sssMax, 0, 1);

    if(expanded >= lightMin && expanded <= lightMax)
        alphatest.r = maps; // Emissives
    float emiss = map(alphatest.r * 255, lightMin, lightMax, 0, 1);

    if(expanded >= roughMin && expanded <= roughMax)
        alphatest.b = maps; // Roughness
    float rough = map(alphatest.b * 255, roughMin, roughMax, 0, 1);

    if(expanded >= metalMin && expanded <= metalMax)
        alphatest.a = maps; // Metals
    float metal = map(alphatest.a * 255, metalMin, metalMax, 0, 1);

    pbr = vec4(emiss, sss, rough, metal);

    if(pbr.b * 255 < 17) {
        float lum = luma(test);
        vec3 diff = test - lum;
        test = clamp(vec3(length(diff)), 0.01, 1);

        if(test.r > 0.3)
            test *= 0.3;

        if(test.r < 0.05)
            test *= 5.0;
        if(test.r < 0.05)
            test *= 2.0;
        test = clamp(test * 1.5 - 0.1, 0, 1);
        pbr.b = clamp(test.r, 0, 1);
    }

    return pbr;
}

/////////////////////////////////////////////////////////////////////////

vec3 toLinear(vec3 sRGB) {
    return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

float invLinZ(float lindepth) {
    return -((2.0 * near / lindepth) - far - near) / (far - near);
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}
float linZ2(float depth) {
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

vec4 backProject(vec4 vec) {
    vec4 tmp = wgbufferModelViewInverse * vec;
    return tmp / tmp.w;
}

vec3 normVec(vec3 vec) {
    return vec * inversesqrt(dot(vec, vec));
}

vec3 lumaBasedReinhardToneMapping(vec3 color) {
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    color = pow(color, vec3(0.45454545454));
    return color;
}

vec4 textureGood(sampler2D sam, vec2 uv) {
    vec2 res = textureSize(sam, 0);

    vec2 st = uv * res - 0.5;

    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = textureLod(sam, (iuv + vec2(0.5, 0.5)) / res, 0);
    vec4 b = textureLod(sam, (iuv + vec2(1.5, 0.5)) / res, 0);
    vec4 c = textureLod(sam, (iuv + vec2(0.5, 1.5)) / res, 0);
    vec4 d = textureLod(sam, (iuv + vec2(1.5, 1.5)) / res, 0);

    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
vec3 skyLut(vec3 sVector, vec3 sunVec, float cosT, sampler2D lut) {
    float mCosT = clamp(cosT, 0.0, 1.);
    float cosY = dot(sunVec, sVector);
    float x = ((cosY * cosY) * (cosY * 0.5 * 256.) + 0.5 * 256. + 18. + 0.5) * oneTexel.x;
    float y = (mCosT * 256. + 1.0 + 0.5) * oneTexel.y;

    return textureGood(lut, vec2(x, y)).rgb;
}

vec3 drawSun(float cosY, float sunInt, vec3 nsunlight, vec3 inColor) {
    return inColor + nsunlight / 0.0008821203 * pow3(smoothstep(cos(0.0093084168595 * 3.2), cos(0.0093084168595 * 1.8), cosY)) * 0.62;
}

// Return random noise in the range [0.0, 1.0], as a function of x.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
// Convert Noise2d() into a "star field" by stomping everthing below fThreshhold to zero.
float NoisyStarField(in vec2 vSamplePos, float fThreshhold) {
    float StarVal = hash12(vSamplePos);
    StarVal = clamp(StarVal / (1.0 - fThreshhold) - fThreshhold / (1.0 - fThreshhold), 0.0, 1.0);

    return StarVal;
}

float StableStarField(in vec2 vSamplePos, float fThreshhold) {

    vec2 floorSample = floor(vSamplePos);
    float v1 = NoisyStarField(floorSample, fThreshhold);

    float StarVal = v1 * 30.0 * skyIntensityNight;
    return StarVal;
}

float stars(vec3 fragpos) {

    float elevation = clamp(fragpos.y, 0., 1.);
    vec2 uv = fragpos.xz / (1. + elevation);

    return StableStarField(uv * 700., 0.999) * 0.5 * (0.3 - 0.3 * 0);
}

const float pi = 3.141592653589793238462643383279502884197169;

//Mie phase function
float phaseg(float x, float g) {
    float gg = g * g;
    return (gg * -0.25 / 3.14 + 0.25 / 3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}

vec2 Nnoise(vec2 coord) {
    float x = sin(coord.x * 100.0) * 0.1 + sin((coord.x * 200.0) + 3.0) * 0.05 + fract(cos((coord.x * 19.0) + 1.0) * 33.33) * 0.15;
    float y = sin(coord.y * 100.0) * 0.1 + sin((coord.y * 200.0) + 3.0) * 0.05 + fract(cos((coord.y * 19.0) + 1.0) * 33.33) * 0.25;
    return vec2(x, y);
}

vec3 reinhard(vec3 x) {
    x *= 1.66;
    return x / (1.0 + x);
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

#define  projMAD2(m, v) (diagonal3(m) * (v) + vec3(0,0,m[3].b))

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

vec2 GGX_FV(float dotLH, float roughness) {
    float alpha = roughness * roughness;

	// F
    float F_a, F_b;
    float dotLH5 = pow5(1.0f - dotLH);
    F_a = 1.0f;
    F_b = dotLH5;

	// V
    float vis;
    float k = alpha * 0.5;
    float k2 = k * k;
    float invK2 = 1.0f - k2;
    vis = 1 / (dotLH * dotLH * invK2 + k2);

    return vec2(F_a * vis, F_b * vis);
}

float GGX_D(float dotNH, float roughness) {
    float alpha = roughness * roughness;
    float alphaSqr = alpha * alpha;
    float pi = 3.14159f;
    float denom = dotNH * dotNH * (alphaSqr - 1.0) + 1.0f;

    float D = alphaSqr / (pi * denom * denom);
    return D;
}

float GGX(vec3 N, vec3 V, vec3 L, float roughness, float F0) {
    vec3 H = normalize(V + L);

    float dotNL = clamp(dot(N, L), 0, 1);
    float dotLH = clamp(dot(L, H), 0, 1);
    float dotNH = clamp(dot(N, H), 0, 1);

    float D = GGX_D(dotNH, roughness);
    vec2 FV_helper = GGX_FV(dotLH, roughness);
    float FV = F0 * FV_helper.x + (1.0f - F0) * FV_helper.y;
    float specular = dotNL * D * FV;

    return specular;
}

vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos + gbufferModelView[3];

    return pos.xyz;
}

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

float cdist(vec2 coord) {
    return max(abs(coord.x - 0.5), abs(coord.y - 0.5)) * 1.85;
}

vec3 rayTrace(vec3 dir, vec3 position, float dither) {

    float stepSize = 15;
    int maxSteps = 30;
    int maxLength = 30;

    vec3 clipPosition = nvec3(gbufferProjection * nvec4(position)) * 0.5 + 0.5;

    float rayLength = ((position.z + dir.z * sqrt(3.0) * maxLength) > -sqrt(3.0) * near) ? (-sqrt(3.0) * near - position.z) / dir.z : sqrt(3.0) * maxLength;

    vec3 end = toClipSpace3(position + dir * rayLength);
    vec3 direction = end - clipPosition;  //convert to clip space

    float len = max(abs(direction.x) / oneTexel.x, abs(direction.y) / oneTexel.y) / stepSize;

	//get at which length the ray intersects with the edge of the screen
    vec3 maxLengths = (step(0., direction) - clipPosition) / direction;
    float mult = min(min(maxLengths.x, maxLengths.y), maxLengths.z);

    vec3 stepv = direction / len;

    int iterations = min(int(min(len, mult * len) - 2), maxSteps);	

	//Do one iteration for closest texel (good contact shadows)
    vec3 spos = clipPosition + stepv / stepSize * 4.0;

    spos += stepv * dither;

    for(int i = 0; i < iterations; i++) {
        float sp = linZ(texture(TranslucentDepthSampler, spos.xy).x);
        float currZ = linZ(spos.z);
        if(sp < currZ) {
            float dist = abs(sp - currZ) / currZ;
            if(dist <= 0.036)
                return vec3(spos.xy, invLinZ(sp));
        }
        spos += stepv;
    }

    return vec3(1.1);
}

vec4 SSR(vec3 fragpos, float fragdepth, vec3 normal, float noise) {

    vec3 pos = vec3(0.0);

    vec4 color = vec4(0.0);

    vec3 reflectedVector = reflect(normalize(fragpos), normalize(normal));

    pos = rayTrace(reflectedVector, fragpos, noise);

    if(pos.z < 1.0 - 1e-5) {

        color = texture(PreviousFrameSampler, pos.st);
        color.rgb *= 5.0;
    }

    return color;
}

vec3 reinhard_jodie(vec3 v) {
    float l = luma(v);
    vec3 tv = v / (1.0f + v);
    tv = mix(v / (1.0f + l), tv, tv);
    return pow(tv, vec3(0.45454545454));
}

vec3 toScreenSpace(vec2 p, float depth) {
    vec4 fragposition = gbufferProjectionInverse * vec4(vec3(p, texture(TranslucentDepthSampler, p).x) * 2.0 - 1.0, 1.0);
    return fragposition.xyz /= fragposition.w;
}
int bitfieldReverse(int a) {
    a = ((a & 0x55555555) << 1) | ((a & 0xAAAAAAAA) >> 1);
    a = ((a & 0x33333333) << 2) | ((a & 0xCCCCCCCC) >> 2);
    a = ((a & 0x0F0F0F0F) << 4) | ((a & 0xF0F0F0F0) >> 4);
    a = ((a & 0x00FF00FF) << 8) | ((a & 0xFF00FF00) >> 8);
    a = ((a & 0x0000FFFF) << 16) | ((a & 0xFFFF0000) >> 16);
    return a;
}
#define tau 6.2831853071795864769252867665590

vec2 hammersley(int i, int N) {
    return vec2(float(i) / float(N), float(bitfieldReverse(i)) * 2.3283064365386963e-10);
}
vec2 circlemap(vec2 p) {
    return (vec2(cos((p).y * tau), sin((p).y * tau)) * p.x);
}
float jaao(vec2 p, vec3 normal, float noise, float depth, float radius) {

		// By Jodie. With some modifications

    float ao = 1.0;
    vec3 p3 = toScreenSpace(p, depth);
    vec2 clipRadius = radius * vec2(ScreenSize.x / ScreenSize.y, 1.0) / length(p3);

    vec3 v = normalize(-p3);

    float nvisibility = 0.0;
    float vvisibility = 0.0;

    for(int i = 0; i < steps; i++) {
        vec2 circlePoint = circlemap(hammersley(i * 15 + 1, 16 * steps)) * clipRadius;

        circlePoint *= noise + 0.1;

        vec3 o = toScreenSpace(circlePoint + p, depth) - p3;
        vec3 o2 = toScreenSpace(circlePoint * .25 + p, depth) - p3;
        float l = length(o);
        float l2 = length(o2);
        o /= l;
        o2 /= l2;

        nvisibility += clamp(1. - max(dot(o, normal) - clamp((l - radius) / radius, 0., 1.), dot(o2, normal) - clamp((l2 - radius) / radius, 0., 1.)), 0., 1.);

        vvisibility += clamp(1. - max(dot(o, v) - clamp((l - radius) / radius, 0., 1.), dot(o2, v) - clamp((l2 - radius) / radius, 0., 1.)), 0., 1.);

    }

    ao = min(vvisibility * 2.0, nvisibility) / float(steps);

    return ao;

}

float rayTraceShadow(vec3 dir, vec3 position, float dither) {
    float stepSize = map(100 * dither, 0, 100, 25, 75);
    int maxSteps = 11;

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
    float sp = linZ(texture2D(TranslucentDepthSampler, spos.xy).x);

    if(sp < spos.z + 0.000001) {
        float dist = abs(linZ(sp) - linZ(spos.z)) / linZ(spos.z);

        if(dist <= 0.05)
            return 0.0;
    }

    for(int i = 0; i < int(iterations); i++) {
        spos += stepv * dither;
        if(spos.x < 0.0 || spos.y < 0.0 || spos.z < 0.0 || spos.x > 1.0 || spos.y > 1.0 || spos.z > 1.0 || clamp(clipPosition.xy, 0, 1) != clipPosition.xy)
            break;

        float sp = texture(TranslucentDepthSampler, spos.xy).x;

        if(sp >= 1.0)
            break;
        if(sp < spos.z + 0.000001) {

            float dist = abs(linZ(sp) - linZ(spos.z)) / linZ(spos.z);

            if(dist < 0.05)
                return 0.0;

        }

    }
    return 1.0;
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
float ld(float depth) {
    return 1.0 / (zMults.y - depth * zMults.z);		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}
vec4 textureGood2(sampler2D sam, vec2 uv) {
    vec2 res = textureSize(sam, 0);

    vec2 st = uv * res - 0.5;

    vec2 iuv = floor(st);
    vec2 fuv = fract(st);

    vec4 a = textureLod(sam, (iuv + vec2(0.5, 0.5)) / res, 0);
    vec4 b = textureLod(sam, (iuv + vec2(1.5, 0.5)) / res, 0);
    vec4 c = textureLod(sam, (iuv + vec2(0.5, 1.5)) / res, 0);
    vec4 d = textureLod(sam, (iuv + vec2(1.5, 1.5)) / res, 0);

    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
vec4 BilateralUpscale(sampler2D tex, sampler2D depth, vec2 coord, float frDepth, float rendres) {
    coord = coord;
    vec4 vl = vec4(0.0);
    float sum = 0.0;

    mat3x3 weights;
    ivec2 scaling = ivec2(1.0 / rendres);
    ivec2 posD = ivec2(coord * rendres) * scaling;
    ivec2 posVl = ivec2(coord * rendres);
    float dz = zMults.x;
    ivec2 pos = (ivec2(gl_FragCoord.xy + Time) % 2) * 2;
	//pos = ivec2(1,-1);

    ivec2 tcDepth = posD + ivec2(-2, -2) * scaling + pos * scaling;
    float dsample = ld(texelFetch(depth, tcDepth, 0).r);
    float w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch(tex, posVl + ivec2(-2) + pos, 0) * w;
    sum += w;

    tcDepth = posD + ivec2(-2, 0) * scaling + pos * scaling;
    dsample = ld(texelFetch(depth, tcDepth, 0).r);
    w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch(tex, posVl + ivec2(-2, 0) + pos, 0) * w;
    sum += w;

    tcDepth = posD + ivec2(0) + pos * scaling;
    dsample = ld(texelFetch(depth, tcDepth, 0).r);
    w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch(tex, posVl + ivec2(0) + pos, 0) * w;
    sum += w;

    tcDepth = posD + ivec2(0, -2) * scaling + pos * scaling;
    dsample = ld(texelFetch(depth, tcDepth, 0).r);
    w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch(tex, posVl + ivec2(0, -2) + pos, 0) * w;
    sum += w;

    return vl / sum;
}
// avoid hardware interpolation
vec4 sample_biquadratic_exact(sampler2D channel, vec2 uv) {
    vec2 res = (textureSize(channel, 0).xy);
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    const ivec3 e = ivec3(-1, 0, 1);
    vec4 s00 = texelFetch(channel, t + e.xx, 0);
    vec4 s01 = texelFetch(channel, t + e.xy, 0);
    vec4 s02 = texelFetch(channel, t + e.xz, 0);
    vec4 s12 = texelFetch(channel, t + e.yz, 0);
    vec4 s11 = texelFetch(channel, t + e.yy, 0);
    vec4 s10 = texelFetch(channel, t + e.yx, 0);
    vec4 s20 = texelFetch(channel, t + e.zx, 0);
    vec4 s21 = texelFetch(channel, t + e.zy, 0);
    vec4 s22 = texelFetch(channel, t + e.zz, 0);
    vec2 q0 = (q + 1.0) / 2.0;
    vec2 q1 = q / 2.0;
    vec4 x0 = mix(mix(s00, s01, q0.y), mix(s01, s02, q1.y), q.y);
    vec4 x1 = mix(mix(s10, s11, q0.y), mix(s11, s12, q1.y), q.y);
    vec4 x2 = mix(mix(s20, s21, q0.y), mix(s21, s22, q1.y), q.y);
    return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}

void main() {
    vec4 outcol = vec4(0.0, 0.0, 0.0, 1.0);
    float depth = texture(TranslucentDepthSampler, texCoord).r;

    vec4 OutTexel3 = (texture(DiffuseSampler, texCoord).rgba);
    vec3 OutTexel = toLinear(OutTexel3.rgb);

    vec3 screenPos = vec3(texCoord, depth);
    vec3 clipPos = screenPos * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 p3 = mat3(gbufferModelViewInverse) * viewPos;
    vec3 view = normVec(p3);
    bool isWater = (texture(TranslucentSampler, texCoord).a * 255 == 200);
    float frDepth = ld(depth);
    //float depthtest = (depth+depthb+depthc+depthd+depthe)/5;

    bool sky = depth >= 1.0;

    if(sky && overworld == 1.0) {

        vec3 atmosphere = skyLut(view, sunPosition3.xyz, view.y, temporals3Sampler);

        if(view.y > 0.) {
            atmosphere += (stars(view) * 2.0) * clamp(1 - (rainStrength * 1), 0, 1);
            atmosphere += drawSun(dot(sunPosition3, view), 0, suncol.rgb / 150., vec3(0.0)) * clamp(1 - (rainStrength * 1), 0, 1) * 20;
            atmosphere += drawSun(dot(-sunPosition3, view), 0, suncol.rgb, vec3(0.0)) * clamp(1 - (rainStrength * 1), 0, 1);
            vec4 cloud = texture(cloudsample, texCoord * CLOUDS_QUALITY);
            atmosphere = atmosphere * cloud.a + (cloud.rgb);
        }

        atmosphere = (clamp(atmosphere * 1.1, 0, 2));
        outcol.rgb = reinhard(atmosphere);
    } else {

        float mod2 = gl_FragCoord.x + gl_FragCoord.y;
        float res = mod(mod2, 2.0f);
        ivec2 texoffsets[4] = ivec2[] (ivec2(0, 1), ivec2(1, 0), -ivec2(0, 1), -ivec2(1, 0));
        vec4 depthgather = textureGatherOffsets(TranslucentDepthSampler, texCoord, texoffsets, 0);
        vec4 lmgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 3);

        vec2 lmtrans = unpackUnorm2x4(OutTexel3.a);

        vec2 lmtrans2 = unpackUnorm2x4(lmgather.z);
        float depthb = depthgather.z;
        lmtrans2 *= 1.0 - (depthb - depth);

        vec2 lmtrans3 = unpackUnorm2x4(lmgather.x);
        float depthc = depthgather.x;
        lmtrans3 *= 1.0 - (depthc - depth);

        vec2 lmtrans4 = unpackUnorm2x4(lmgather.y);
        float depthd = depthgather.y;
        lmtrans4 *= 1.0 - (depthd - depth);

        vec2 lmtrans5 = unpackUnorm2x4(lmgather.w);
        float depthe = depthgather.w;
        lmtrans5 *= 1.0 - (depthe - depth);

        float lmy = mix(lmtrans.y, (lmtrans2.y + lmtrans3.y + lmtrans4.y + lmtrans5.y) / 4, res);
        float lmx = mix((lmtrans2.y + lmtrans3.y + lmtrans4.y + lmtrans5.y) / 4, lmtrans.y, res);

        if(overworld == 1.0) {

            float noise = clamp(mask(gl_FragCoord.xy + (Time * 100)), 0, 1);

            vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));

            float postlight = 1;

            if(lmx > 0.95) {
                lmx *= 0.75;
                lmy = 0.1;
                postlight = 0.0;
            }

            vec3 lightmap = texture(temporals3Sampler, vec2(lmy, lmx) * (oneTexel * 17)).xyz;

            float normalstrength = (1 - luma(OutTexel3.rgb)) * 0.2;
            ivec2 texoffsets[4] = ivec2[] (ivec2(0, 3), ivec2(3, 0), -ivec2(0, 3), -ivec2(3, 0));
            vec4 normoffset = pow(textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 2) * 1.5, vec4(8.0)) * normalstrength;

            vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
            fragpos.rgb += pow8(OutTexel3.b * 1.5) * normalstrength;

            vec3 p2 = backProject(vec4(scaledCoord + 2.0 * vec2(0.0, oneTexel.y), depthc, 1.0)).xyz;
            p2.rgb += (normoffset.x);
            p2 = p2 - fragpos;

            vec3 p3 = backProject(vec4(scaledCoord + 2.0 * vec2(oneTexel.x, 0.0), depthd, 1.0)).xyz;
            p3.rgb += (normoffset.y);
            p3 = p3 - fragpos;

            vec3 p4 = backProject(vec4(scaledCoord - 2.0 * vec2(0.0, oneTexel.y), depthb, 1.0)).xyz;
            p4.rgb += (normoffset.z);
            p4 = p4 - fragpos;

            vec3 p5 = backProject(vec4(scaledCoord - 2.0 * vec2(oneTexel.x, 0.0), depthe, 1.0)).xyz;
            p5.rgb += (normoffset.w);
            p5 = p5 - fragpos;

            vec3 normal = normalize(cross(p2, p3)) + normalize(cross(-p4, p3)) + normalize(cross(p2, -p5)) + normalize(cross(-p4, -p5));
            normal = normal == vec3(0.0) ? vec3(0.0, 1.0, 0.0) : normalize(-normal);

            vec3 normal3 = worldToView(normal);

            vec3 ambientCoefs = normal / dot(abs(normal), vec3(1.0));

            vec3 ambientLight = ambientUp * clamp(ambientCoefs.y, 0., 1.);
            ambientLight += ambientDown * clamp(-ambientCoefs.y, 0., 1.);
            ambientLight += ambientRight * clamp(ambientCoefs.x, 0., 1.);
            ambientLight += ambientLeft * clamp(-ambientCoefs.x, 0., 1.);
            ambientLight += ambientB * clamp(ambientCoefs.z, 0., 1.);
            ambientLight += ambientF * clamp(-ambientCoefs.z, 0., 1.);
            ambientLight *= (1.0 + rainStrength * 0.2);
            ambientLight = clamp(ambientLight * (pow(lmx, 8.0) * 1.5) + lmy * vec3(TORCH_R, TORCH_G, TORCH_B), 0, 2.0);

            vec4 pbr = pbr(lmtrans, unpackUnorm2x4((texture(DiffuseSampler, texCoord + vec2(oneTexel.y)).a)), OutTexel3.rgb);

            float sssa = pbr.g;
            float smoothness = pbr.a * 255 > 1.0 ? pbr.a : pbr.b;

            float light = pbr.r;
            vec3 f0 = pbr.a * 255 > 1.0 ? vec3(0.8) : vec3(0.04);
            vec3 reflections = vec3(0.0);

            if(pbr.a * 255 > 1.0) {
                float ldepth = linZ2(depth);
                vec3 normal2 = normal3 + (noise * (1 - smoothness));
                vec3 fragpos3 = (vec4(texCoord, ldepth, 1.0)).xyz;
                fragpos3 *= ldepth;
                vec3 avgSky = mix(lightmap * 0.5, ambientLight, lmx);
                vec4 reflection = vec4(SSR(viewPos.xyz, depth, normal2, noise));

                float normalDotEye = dot(normal, normalize(fragpos3));
                float fresnel = pow5(clamp(1.0 + normalDotEye, 0.0, 1.0));
                fresnel = fresnel * 0.98 + 0.02;
                fresnel *= max(1.0 - 0 * 0.5 * 1, 0.5);
                fresnel *= 1.0 - 1 * 0.3;

                reflection = mix(vec4(avgSky, 1), reflection, reflection.a);
                reflections += ((reflection.rgb) * (fresnel * OutTexel));
                OutTexel *= 0.075;
                reflections = max(vec3(0.0), reflections);
            }
            OutTexel *= mix(vec3(1.0), lightmap, postlight);
            float ao = 1.0 * ((1.0 - AOStrength) + jaao(texCoord, normal3, noise, depth, 1.2) * AOStrength);

            vec3 sunPosition2 = mix(sunPosition3, -sunPosition3, clamp(skyIntensityNight * 3, 0, 1));
            float shadeDir = max(0.0, dot(normal, sunPosition2));

            float screenShadow = BilateralUpscale(shadow, TranslucentDepthSampler, gl_FragCoord.xy, frDepth, 0.5).x;
            screenShadow = clamp(((screenShadow + lmy) * clamp((pow32(lmx)) * 100, 0.1, 1.0)), 0.1, 1.0) * lmx;
            shadeDir *= screenShadow;
            shadeDir += max(0.0, (max(phaseg(dot(view, sunPosition2), 0.5) * 2.0 + (max(0.0, screenShadow * ao * 2 - 1) * 0.5), phaseg(dot(view, sunPosition2), 0.1)) * pi * 1.6) * float(sssa) * lmx);
            shadeDir = clamp(shadeDir, 0, 1);

            float sunSpec = ((GGX(normal, -normalize(view), sunPosition2, 1 - smoothness, f0.x)));

            vec3 shading = vec3(0.0);

            shading = (suncol * shadeDir) + ambientLight;
            shading += (sunSpec * suncol) * shadeDir;

            shading += lightmap * 0.1;

            shading = mix(ambientLight, shading, 1 - (rainStrength * lmx));
            if(light > 0.001)
                shading.rgb = vec3(light * 2.0);
            shading = max(vec3(0.1), shading);

            vec3 dlight = (OutTexel * shading) + reflections;

            outcol.rgb = lumaBasedReinhardToneMapping(dlight * ao);

            outcol.rgb *= 1.0 + max(0.0, light);
        ///---------------------------------------------
            //outcol.rgb = clamp(vec3(shadeDir), 0.01, 1);
        ///---------------------------------------------
        } else {
            if(end != 1.0) {
                vec2 p_m = texCoord;
                vec2 p_d = p_m;
                p_d.xy -= Time * 0.1;
                vec2 dst_map_val = vec2(Nnoise(p_d.xy));
                vec2 dst_offset = dst_map_val.xy;

                dst_offset *= 2.0;
                dst_offset *= 0.0025;
                dst_offset *= (1. - p_m.t);

                vec2 texCoord = p_m.st + dst_offset;
                OutTexel = toLinear(texture(DiffuseSampler, texCoord).rgb);
            }

            outcol.rgb = mix(reinhard_jodie(OutTexel.rgb * ((((lmx + 0.15) * vec3(1.0)) + (pow3(lmy) * vec3(TORCH_R, TORCH_G, TORCH_G))))), fogcol.rgb * 0.5, pow(depth, 2048));
            //if(light > 0.001) outcol.rgb *= clamp(vec3(2.0 - 1 * 2) * light * 2, 1.0, 10.0);
        }

        if(isWater) {

            vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B) * fogcol.rgb;
            vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
            vec3 totEpsilon = dirtEpsilon * Dirt_Amount + waterEpsilon;
            outcol.rgb *= clamp(exp(-length(viewPos) * totEpsilon), 0.2, 1.0);

        }

    }

    fragColor = outcol;

}
