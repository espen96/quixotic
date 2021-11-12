#version 150
#extension GL_ARB_gpu_shader5 : enable
uniform sampler2D DiffuseSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D noisetex;

uniform vec2 ScreenSize;
uniform vec2 OutSize;
uniform float Time;

in mat4 gbufferModelView;
in mat4 gbufferProjectionInverse;
in mat4 gbufferProjection;
mat4 gbufferModelViewInverse = inverse(gbufferModelView);
in float sShadows;
// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05

vec2 getControl(int index, vec2 screenSize) {
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
in float skys;
in float cloudy;

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

#define SSAO_SAMPLES 4

#define NORMDEPTHTOLERANCE 1.0
const float pi = 3.141592653589793238462643383279502884197169;

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
    #define sssMin pbr1.x*255
    #define sssMax pbr1.y*255
    #define lightMin pbr2.x*255
    #define lightMax pbr2.y*255
    #define roughMin pbr3.x*255
    #define roughMax pbr3.y*255
    #define metalMin pbr4.x*255
    #define metalMax pbr4.y*255
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
    if(overworld != 1.0)
        return (2.0 * near * far) / (far + near - depth * (far - near));

    if(overworld == 1.0)
        return (2.0 * near) / (far + near - depth * (far - near));

}

vec4 backProject(vec4 vec) {
    vec4 tmp = inverse(gbufferProjection * gbufferModelView) * vec;
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
vec3 lumaBasedReinhardToneMapping2(vec3 color) {
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    //color = pow(color, vec3(0.45454545454));
    return color;
}
vec4 textureGood(sampler2D sam, vec2 uv) {
    vec2 res = textureSize(sam, 0);

    vec2 st = uv * res - 0.5;

    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture(sam, (iuv + vec2(0.5, 0.5)) / res, 0);
    vec4 b = texture(sam, (iuv + vec2(1.5, 0.5)) / res, 0);
    vec4 c = texture(sam, (iuv + vec2(0.5, 1.5)) / res, 0);
    vec4 d = texture(sam, (iuv + vec2(1.5, 1.5)) / res, 0);

    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
vec3 skyLut(vec3 sVector, vec3 sunVec, float cosT, sampler2D lut) {
    float mCosT = clamp(cosT, 0.0, 1.);
    float cosY = dot(sunVec, sVector);
    float x = ((cosY * cosY) * (cosY * 0.5 * 256.) + 0.5 * 256. + 18. + 0.5) * oneTexel.x;
    float y = (mCosT * 256. + 1.0 + 0.5) * oneTexel.y;

    return textureGood(lut, vec2(x, y)).rgb;
}
float facos(float inX) {

    const float C0 = 1.56467;
    const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}
vec3 skyLut2(vec3 sVector, vec3 sunVec, float cosT, float rainStrength) {
	#define SKY_BRIGHTNESS_DAY 1.0
	#define SKY_BRIGHTNESS_NIGHT 1.0;	
    float mCosT = clamp(cosT, 0.0, 1.0);
    float cosY = dot(sunVec, sVector);
    float Y = facos(cosY);
    const float a = -0.8;
    const float b = -0.1;
    const float c = 3.0;
    const float d = -7.;
    const float e = 0.35;

  //luminance (cie model)
    vec3 daySky = vec3(0.0);
    vec3 moonSky = vec3(0.0);
	// Day
    if(skyIntensity > 0.00001) {
        float L0 = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * Y) - exp(d * pi / 2.)) + e * cosY * cosY);
        vec3 skyColor0 = mix(vec3(0.05, 0.5, 1.) / 1.5, vec3(0.4, 0.5, 0.6) / 1.5, rainStrength);
        vec3 normalizedSunColor = nsunColor;
        vec3 skyColor = mix(skyColor0, normalizedSunColor, 1.0 - pow(1.0 + L0, -1.2)) * (1.0 - rainStrength);
        daySky = pow(L0, 1.0 - rainStrength) * skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15. * SKY_BRIGHTNESS_DAY;
    }
	// Night
    else if(skyIntensityNight > 0.00001) {
        float L0Moon = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * (pi - Y)) - exp(d * pi / 2.)) + e * cosY * cosY);
        moonSky = pow(L0Moon, 1.0 - rainStrength) * skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) * SKY_BRIGHTNESS_NIGHT;
    }
    return (daySky + moonSky);
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

float StableStarField(in vec2 vSamplePos, float fThreshhold) {

    vec2 floorSample = floor(vSamplePos);
    float StarVal = hash12(floorSample);

    float v1 = clamp(StarVal / (1.0 - fThreshhold) - fThreshhold / (1.0 - fThreshhold), 0.0, 1.0);

    StarVal = v1 * 30.0 * skyIntensityNight;
    return StarVal;
}

float stars(vec3 fragpos) {

    float elevation = clamp(fragpos.y, 0., 1.);
    vec2 uv = fragpos.xz / (1. + elevation);

    return StableStarField(uv * 700., 0.999) * 0.5 * (0.3 - 0.3 * 0);
}

//Mie phase function
float phaseg(float x, float g) {
    float gg = g * g;
    return (gg * -0.25 / pi + 0.25 / pi) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
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
float cdist2(vec2 coord) {
    vec2 vec = abs(coord * 2.0 - 1.0);
    float d = max(vec.x, vec.y);
    return 1.0 - d * d;
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
        color.rgb *= 1.0;
    }

    return color;
}

vec3 reinhard_jodie(vec3 v) {
    float l = luma(v);
    vec3 tv = v / (1.0f + v);
    tv = mix(v / (1.0f + l), tv, tv);
    return tv;
}

#define tau 6.2831853071795864769252867665590

#define AOQuality 0   //[0 1 2] Increases the quality of Ambient Occlusion from 0 to 2, 0 is default
#define AORadius 2.0 //[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0] //Changes the radius of Ambient Occlusion to be larger or smaller, 1.0 is default

float dither5x3() {
    const int ditherPattern[15] = int[15] (9, 3, 7, 12, 0, 11, 5, 1, 14, 8, 2, 13, 10, 4, 6);

    vec2 position = floor(mod(vec2(texCoord.s * ScreenSize.x, texCoord.t * ScreenSize.y), vec2(5.0, 3.0)));

    int dither = ditherPattern[int(position.x) + int(position.y) * 5];

    return float(dither) * 0.0666666666666667f;
}
#define g(a) (-4.*a.x*a.y+3.*a.x+a.y*2.)

float bayer16x16(vec2 p) {

    vec2 m0 = vec2(mod(floor(p * 0.125), 2.));
    vec2 m1 = vec2(mod(floor(p * 0.25), 2.));
    vec2 m2 = vec2(mod(floor(p * 0.5), 2.));
    vec2 m3 = vec2(mod(floor(p), 2.));

    return (g(m0) + g(m1) * 4.0 + g(m2) * 16.0 + g(m3) * 64.0) * 0.003921568627451;
}
#undef g

float dither = bayer16x16(gl_FragCoord.xy);

//Dithering from Jodie
float bayer2(vec2 a) {
    a = floor(a);
    return fract(dot(a, vec2(.5, a.y * .75)));
}

#define bayer4(a)   (bayer2( .5*(a))*.25+bayer2(a))
#define bayer8(a)   (bayer4( .5*(a))*.25+bayer2(a))
#define bayer16(a)  (bayer8( .5*(a))*.25+bayer2(a))
#define bayer32(a)  (bayer16(.5*(a))*.25+bayer2(a))
#define bayer64(a)  (bayer32(.5*(a))*.25+bayer2(a))
#define bayer128(a) (bayer64(.5*(a))*.25+bayer2(a))

float dither64 = bayer64(gl_FragCoord.xy);
vec2 OffsetDist(float x) {
    float n = fract(x * 8.0) * 3.1415;
    return vec2(cos(n), sin(n)) * x;
}
float linZt(float depth) {

    return (2.0 * near) / (far + near - depth * (far - near));

}
float dbao2(sampler2D depth) {
    float ao = 0.0;

    float depthsamp = texture2D(depth, texCoord).r;
    if(depthsamp >= 1.0)
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

    for(int i = 0; i < 4; i++) {
        vec2 offset = OffsetDist(currentStep) * scale;
        float angle = 0.0, dist = 0.0;

        for(int i = 0; i < 2; i++) {
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
const float angle = radians(360.0 / float(6));
const float angleSin = sin(angle);
const float angleCos = cos(angle);
const mat2 rotationMatrix = mat2(angleCos, angleSin, -angleSin, angleCos);
float dbao3(sampler2D depth) {
    float deptho = texture2D(depth, texCoord).r;

    vec2 tapOffset = vec2(0.0, 1.0 / 512.0); // Fixed step for varying resolutions
    float dist = 1.0 - pow(deptho, 64.0);

    float occlusion = 0.0;
    for(int ii = 0; ii < 6; ++ii) {
        for(int jj = 0; jj < 12; ++jj) {
            float mul = float(jj + 1) * dist;
            float tapValue = texture2D(depth, texCoord + (tapOffset * mul)).r;
            float rangeCheck = clamp(smoothstep(0.0, 1.0, mul / abs(deptho - tapValue)), 0.0, 1.0);
            occlusion += tapValue >= deptho ? rangeCheck : 0.0;
        }
        tapOffset = rotationMatrix * tapOffset;
    }
    return occlusion / float(6 * 12);
}
const float sky_planetRadius = 6731e3;

const float PI = 3.141592;
vec3 cameraPosition = vec3(0, abs((cloudy)), 0);
const float cloud_height = 1500.;
const float maxHeight = 1650.;
int maxIT_clouds = 15;
const float cdensity = 0.2;

///////////////////////////

//Cloud without 3D noise, is used to exit early lighting calculations if there is no cloud
float cloudCov(in vec3 pos, vec3 samplePos) {
    float mult = max(pos.y - 2000.0, 0.0) / 2000.0;
    float mult2 = max(-pos.y + 2000.0, 0.0) / 500.0;
    float coverage = clamp(texture(noisetex, fract(samplePos.xz / 12500.)).x * 1. + 0.5 * rainStrength, 0.0, 1.0);
    float cloud = coverage * coverage * 1.0 - mult * mult * mult * 3.0 - mult2 * mult2;
    return max(cloud, 0.0);
}
//Erode cloud with 3d Perlin-worley noise, actual cloud value

float cloudVol(in vec3 pos, in vec3 samplePos, in float cov) {
    float mult2 = (pos.y - 1500) / 2500 + rainStrength * 0.4;

    float cloud = clamp(cov - 0.11 * (0.2 + mult2), 0.0, 1.0);
    return cloud;

}

vec4 renderClouds(vec3 fragpositi, vec3 color, float dither, vec3 sunColor, vec3 moonColor, vec3 avgAmbient) {

    vec4 fragposition = gbufferModelViewInverse * vec4(fragpositi, 1.0);

    vec3 worldV = normalize(fragposition.rgb);
    float VdotU = worldV.y;
    maxIT_clouds = int(clamp(maxIT_clouds / sqrt(VdotU), 0.0, maxIT_clouds));

    vec3 dV_view = worldV;

    vec3 progress_view = dV_view * dither + cameraPosition;

    float total_extinction = 1.0;

    worldV = normalize(worldV) * 300000. + cameraPosition; //makes max cloud distance not dependant of render distance
    if(worldV.y < cloud_height)
        return vec4(0.0, 0.0, 0.0, 1.0);	//don't trace if no intersection is possible

    dV_view = normalize(dV_view);

	//setup ray to start at the start of the cloud plane and end at the end of the cloud plane
    dV_view *= max(maxHeight - cloud_height, 0.0) / dV_view.y / maxIT_clouds;

    vec3 startOffset = dV_view * clamp(dither, 0.0, 1.0);
    progress_view = startOffset + cameraPosition + dV_view * (cloud_height - cameraPosition.y) / (dV_view.y);

    float mult = length(dV_view);

    color = vec3(0.0);
    float SdotV = dot(sunVec, normalize(fragpositi));
	//fake multiple scattering approx 1 (from horizon zero down clouds)
    float mieDay = max(phaseg(SdotV, 0.22), phaseg(SdotV, 0.2));
    float mieNight = max(phaseg(-SdotV, 0.22), phaseg(-SdotV, 0.2));

    vec3 sunContribution = mieDay * sunColor * 3.14;
    vec3 moonContribution = mieNight * moonColor * 3.14;
    float ambientMult = exp(-(1.25 + 0.8 * clamp(rainStrength, 0.75, 1)) * cdensity * 50.0);
    vec3 skyCol0 = avgAmbient * ambientMult;

    for(int i = 0; i < maxIT_clouds; i++) {
        vec3 curvedPos = progress_view;
        vec2 xz = progress_view.xz - cameraPosition.xz;
        curvedPos.y -= sqrt(pow(sky_planetRadius, 2.0) - dot(xz, xz)) - sky_planetRadius;
        vec3 samplePos = curvedPos * vec3(1.0, 1.0 / 32.0, 1.0) / 4 + (sunElevation * 1000) * vec3(0.5, 0.0, 0.5);

        float coverageSP = cloudCov(curvedPos, samplePos);
        if(coverageSP > 0.05) {
            float cloud = cloudVol(curvedPos, samplePos, coverageSP);
            if(cloud > 0.05) {
                float mu = cloud * cdensity;

				//fake multiple scattering approx 2  (from horizon zero down clouds)
                float h = 0.35 - 0.35 * clamp(progress_view.y / 4000. - 1500. / 4000., 0.0, 1.0);
                float powder = 1.0 - exp(-mu * mult);
                float Shadow = mix(1.0, powder, h);
                float ambientPowder = mix(1.0, powder, h * ambientMult);
                vec3 S = vec3(sunContribution * Shadow + Shadow * moonContribution + skyCol0 * ambientPowder);

                vec3 Sint = (S - S * exp(-mult * mu)) / (mu);
                color += mu * Sint * total_extinction;
                total_extinction *= exp(-mu * mult);
                if(total_extinction < 1e-5)
                    break;
            }

        }

        progress_view += dV_view;

    }

    float cosY = normalize(dV_view).y;

    color.rgb = mix(color.rgb * vec3(0.2, 0.21, 0.21), color.rgb, 1 - rainStrength);
    return mix(vec4(color, clamp(total_extinction, 0.0, 1.0)), vec4(0.0, 0.0, 0.0, 1.0), 1 - smoothstep(0.02, 0.20, cosY));

}
float dbao(sampler2D depth) {
    float ao = 0.0;
    float aspectRatio = ScreenSize.x / ScreenSize.y;

    const int aoloop = 2; //3
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

    for(int i = 0; i < aoloop; i++) {
        for(int j = 0; j < aoside; j++) {
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

float rayTraceShadow(vec3 dir, vec3 position, float dither, float depth) {

    float stepSize = clamp(linZ(depth) * 10.0, 15, 200);
    int maxSteps = int(clamp(invLinZ(depth) * 10.0, 15, 50));
    dither *= 1.0 - pow(depth, 256);
    vec3 clipPosition = nvec3(gbufferProjection * nvec4(position)) * 0.5 + 0.5;
    float rayLength = ((position.z + dir.z * sqrt(3.0) * far) > -sqrt(3.0) * near) ? (-sqrt(3.0) * near - position.z) / dir.z : sqrt(3.0) * far;

    vec3 end = toClipSpace3(position + dir * rayLength);
    //vec3 end = nvec3(gbufferProjection * nvec4(position + dir * rayLength)) * 0.5 + 0.5;
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

        if(sp < spos.z + 0.00000001) {

            float dist = abs(linZ(sp) - linZ(spos.z)) / linZ(spos.z);

            if(dist < 0.05)
                return exp2(position.z / 8.);

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

vec3 viewToWorld(vec3 viewPos) {

    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = gbufferModelViewInverse * pos;

    return pos.xyz;
}
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)

float interleaved_gradientNoise() {
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y) + Time / 1.6128);
}
float triangularize(float dither) {
    float center = dither * 2.0 - 1.0;
    dither = center * inversesqrt(abs(center));
    return clamp(dither - fsign(center), 0.0, 1.0);
}
vec3 fp10Dither(vec3 color, float dither) {
    const vec3 mantissaBits = vec3(6., 6., 5.);
    vec3 exponent = floor(log2(color));
    return color + dither * exp2(-mantissaBits) * exp2(exponent);
}

vec3 getDepthPoint(vec2 coord, float depth) {
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; //convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;

    return pos.xyz;
}
vec3 constructNormal(float depthA, vec2 texCoords, sampler2D depthtex, float water) {
    vec2 offsetB = vec2(0.0, oneTexel.y * 0.6);
    vec2 offsetC = vec2(oneTexel.x * 0.6, 0.0);
    float depthB = texture(depthtex, texCoords + offsetB).r;
    float depthC = texture(depthtex, texCoords + offsetC).r;
    vec3 A = getDepthPoint(texCoords, depthA);
    A += pow4(texture(DiffuseSampler, texCoord).g) * 0.01 * 1 - water;

    vec3 B = getDepthPoint(texCoords + offsetB, depthB);
    B += pow4(texture(DiffuseSampler, texCoord + offsetB * 3.0).g) * 0.01 * 1 - water;

    vec3 C = getDepthPoint(texCoords + offsetC, depthC);
    C += pow4(texture(DiffuseSampler, texCoord + offsetC * 3.0).g) * 0.01 * 1 - water;

    vec3 AB = normalize(B - A);
    vec3 AC = normalize(C - A);

    vec3 normal = -cross(AB, AC);
	// normal.z = -normal.z;

    return normalize(normal);
}

float getRawDepth(vec2 uv) {
    return texture(TranslucentDepthSampler, uv).x;
}

            // inspired by keijiro's depth inverse projection
            // https://github.com/keijiro/DepthInverseProjection
            // constructs view space ray at the far clip plane from the screen uv
            // then multiplies that ray by the linear 01 depth
vec3 viewSpacePosAtScreenUV(vec2 uv) {
    vec3 viewSpaceRay = (gbufferProjectionInverse * vec4(uv * 2.0 - 1.0, 1.0, 1.0) * near).xyz;
    float rawDepth = getRawDepth(uv);
    return viewSpaceRay * (linZ(rawDepth) + pow8(luma(texture(DiffuseSampler, uv).xyz)) * 0.00);
}
vec3 viewSpacePosAtPixelPosition(vec2 vpos) {
    vec2 uv = vpos * oneTexel.xy;
    return viewSpacePosAtScreenUV(uv);
}

vec3 viewNormalAtPixelPosition(vec2 vpos) {
                // get current pixel's view space position
    vec3 viewSpacePos_c = viewSpacePosAtPixelPosition(vpos + vec2(0.0, 0.0));

                // get view space position at 1 pixel offsets in each major direction
    vec3 viewSpacePos_l = viewSpacePosAtPixelPosition(vpos + vec2(-1.0, 0.0));
    vec3 viewSpacePos_r = viewSpacePosAtPixelPosition(vpos + vec2(1.0, 0.0));
    vec3 viewSpacePos_d = viewSpacePosAtPixelPosition(vpos + vec2(0.0, -1.0));
    vec3 viewSpacePos_u = viewSpacePosAtPixelPosition(vpos + vec2(0.0, 1.0));

                // get the difference between the current and each offset position
    vec3 l = viewSpacePos_c - viewSpacePos_l;
    vec3 r = viewSpacePos_r - viewSpacePos_c;
    vec3 d = viewSpacePos_c - viewSpacePos_d;
    vec3 u = viewSpacePos_u - viewSpacePos_c;

                // pick horizontal and vertical diff with the smallest z difference
    vec3 hDeriv = abs(l.z) < abs(r.z) ? l : r;
    vec3 vDeriv = abs(d.z) < abs(u.z) ? d : u;

                // get view space normal from the cross product of the two smallest offsets
    vec3 viewNormal = normalize(cross(hDeriv, vDeriv));

    return viewNormal;
}

vec3 viewNormalAtPixelPosition2(vec2 vpos) {
                // screen uv from vpos
    vec2 uv = vpos * oneTexel.xy;

                // current pixel's depth
    float c = getRawDepth(uv);

                // get current pixel's view space position
    vec3 viewSpacePos_c = viewSpacePosAtScreenUV(uv);

                // get view space position at 1 pixel offsets in each major direction
    vec3 viewSpacePos_l = viewSpacePosAtScreenUV(uv + vec2(-1.0, 0.0) * oneTexel.xy);
    vec3 viewSpacePos_r = viewSpacePosAtScreenUV(uv + vec2(1.0, 0.0) * oneTexel.xy);
    vec3 viewSpacePos_d = viewSpacePosAtScreenUV(uv + vec2(0.0, -1.0) * oneTexel.xy);
    vec3 viewSpacePos_u = viewSpacePosAtScreenUV(uv + vec2(0.0, 1.0) * oneTexel.xy);

                // get the difference between the current and each offset position
    vec3 l = viewSpacePos_c - viewSpacePos_l;
    vec3 r = viewSpacePos_r - viewSpacePos_c;
    vec3 d = viewSpacePos_c - viewSpacePos_d;
    vec3 u = viewSpacePos_u - viewSpacePos_c;

                // get depth values at 1 & 2 pixels offsets from current along the horizontal axis
    vec4 H = vec4(getRawDepth(uv + vec2(-1.0, 0.0) * oneTexel.xy), getRawDepth(uv + vec2(1.0, 0.0) * oneTexel.xy), getRawDepth(uv + vec2(-2.0, 0.0) * oneTexel.xy), getRawDepth(uv + vec2(2.0, 0.0) * oneTexel.xy));

                // get depth values at 1 & 2 pixels offsets from current along the vertical axis
    vec4 V = vec4(getRawDepth(uv + vec2(0.0, -1.0) * oneTexel.xy), getRawDepth(uv + vec2(0.0, 1.0) * oneTexel.xy), getRawDepth(uv + vec2(0.0, -2.0) * oneTexel.xy), getRawDepth(uv + vec2(0.0, 2.0) * oneTexel.xy));

                // current pixel's depth difference from slope of offset depth samples
                // differs from original article because we're using non-linear depth values
                // see article's comments
    vec2 he = abs((2 * H.xy - H.zw) - c);
    vec2 ve = abs((2 * V.xy - V.zw) - c);

                // pick horizontal and vertical diff with the smallest depth difference from slopes
    vec3 hDeriv = he.x < he.y ? l : r;
    vec3 vDeriv = ve.x < ve.y ? d : u;

                // get view space normal from the cross product of the best derivatives
    vec3 viewNormal = normalize(cross(hDeriv, vDeriv));

    return viewNormal;
}

vec2 unpackUnorm2x4v2(vec4 pack) {
    vec2 xy;
    float pack2 = (pack.x + pack.y + pack.z + pack.z) / 4;
    xy.x = modf(pack2 * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}
////////////////////

#define DEBUG
#define DEBUG_PROGRAM 30
#define DEBUG_BRIGHTNESS 10.0 // [1/65536.0 1/32768.0 1/16384.0 1/8192.0 1/4096.0 1/2048.0 1/1024.0 1/512.0 1/256.0 1/128.0 1/64.0 1/32.0 1/16.0 1/8.0 1/4.0 1/2.0 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024.0 2048.0 4096.0 8192.0 16384.0 32768.0 65536.0]
#define DRAW_DEBUG_VALUE

vec3 Debug = vec3(1.0);

// Write the direct variable onto the screen
void show(bool x) {
    Debug = vec3(float(x));
}
void show(float x) {
    Debug = vec3(x);
}
void show(vec2 x) {
    Debug = vec3(x, 0.0);
}
void show(vec3 x) {
    Debug = x;
}
void show(vec4 x) {
    Debug = x.rgb;
}

void inc2(bool x) {
    Debug += vec3(float(x));
}
void inc2(float x) {
    Debug += vec3(x);
}
void inc2(vec2 x) {
    Debug += vec3(x, 0.0);
}
void inc2(vec3 x) {
    Debug += x;
}
void inc2(vec4 x) {
    Debug += x.rgb;
}

#ifdef DRAW_DEBUG_VALUE
	// Display the value of the variable on the debug value viewer
	#define showval(x) if (all(equal(ivec2(gl_FragCoord.xy), ivec2(ScreenSize/2)))) show(x);
	#define incval(x)  if (all(equal(ivec2(gl_FragCoord.xy), ivec2(ScreenSize/2)))) inc2(x);
#else
	#define showval(x)
	#define incval(x)
#endif

vec4 SampleTextureCatmullRom(sampler2D tex, vec2 uv, vec2 texSize) {
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 *= oneTexel;
    texPos3 *= oneTexel;
    texPos12 *= oneTexel;

    vec4 result = vec4(0.0);
    result += texture(tex, vec2(texPos0.x, texPos0.y)) * w0.x * w0.y;
    result += texture(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture(tex, vec2(texPos3.x, texPos0.y)) * w3.x * w0.y;

    result += texture(tex, vec2(texPos0.x, texPos12.y)) * w0.x * w12.y;
    result += texture(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture(tex, vec2(texPos3.x, texPos12.y)) * w3.x * w12.y;

    result += texture(tex, vec2(texPos0.x, texPos3.y)) * w0.x * w3.y;
    result += texture(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture(tex, vec2(texPos3.x, texPos3.y)) * w3.x * w3.y;

    return result;
}
/***********************************************************************/
/* Text Rendering */
const int _A = 0x64bd29, _B = 0x749d27, _C = 0xe0842e, _D = 0x74a527, _E = 0xf09c2f, _F = 0xf09c21, _G = 0xe0b526, _H = 0x94bd29, _I = 0xf2108f, _J = 0x842526, _K = 0x9284a9, _L = 0x10842f, _M = 0x97a529, _N = 0x95b529, _O = 0x64a526, _P = 0x74a4e1, _Q = 0x64acaa, _R = 0x749ca9, _S = 0xe09907, _T = 0xf21084, _U = 0x94a526, _V = 0x94a544, _W = 0x94a5e9, _X = 0x949929, _Y = 0x94b90e, _Z = 0xf4106f, _0 = 0x65b526, _1 = 0x431084, _2 = 0x64904f, _3 = 0x649126, _4 = 0x94bd08, _5 = 0xf09907, _6 = 0x609d26, _7 = 0xf41041, _8 = 0x649926, _9 = 0x64b904, _APST = 0x631000, _PI = 0x07a949, _UNDS = 0x00000f, _HYPH = 0x001800, _TILD = 0x051400, _PLUS = 0x011c40, _EQUL = 0x0781e0, _SLSH = 0x041041, _EXCL = 0x318c03, _QUES = 0x649004, _COMM = 0x000062, _FSTP = 0x000002, _QUOT = 0x528000, _BLNK = 0x000000, _COLN = 0x000802, _LPAR = 0x410844, _RPAR = 0x221082;

const ivec2 MAP_SIZE = ivec2(5, 5);

int GetBit(int bitMap, int index) {
    return (bitMap >> index) & 1;
}

float DrawChar(int charBitMap, inout vec2 anchor, vec2 charSize, vec2 uv) {
    uv = (uv - anchor) / charSize;

    anchor.x += charSize.x;

    if(!all(lessThan(abs(uv - vec2(0.5)), vec2(0.5))))
        return 0.0;

    uv *= MAP_SIZE;

    int index = int(uv.x) % MAP_SIZE.x + int(uv.y) * MAP_SIZE.x;

    return float(GetBit(charBitMap, index));
}

const int STRING_LENGTH = 15;
int[STRING_LENGTH] drawString;

float DrawString(inout vec2 anchor, vec2 charSize, int stringLength, vec2 uv) {
    uv = (uv - anchor) / charSize;

    anchor.x += charSize.x * stringLength;

    if(!all(lessThan(abs(uv / vec2(stringLength, 1.0) - vec2(0.5)), vec2(0.5))))
        return 0.0;

    int charBitMap = drawString[int(uv.x)];

    uv *= MAP_SIZE;

    int index = int(uv.x) % MAP_SIZE.x + int(uv.y) * MAP_SIZE.x;

    return float(GetBit(charBitMap, index));
}

#define log10(x) (log2(x) / log2(10.0))

float DrawInt(int val, inout vec2 anchor, vec2 charSize, vec2 uv) {
    if(val == 0)
        return DrawChar(_0, anchor, charSize, uv);

    const int _DIGITS[10] = int[10] (_0, _1, _2, _3, _4, _5, _6, _7, _8, _9);

    bool isNegative = val < 0.0;

    if(isNegative)
        drawString[0] = _HYPH;

    val = abs(val);

    int posPlaces = int(ceil(log10(abs(val) + 0.001)));
    int strIndex = posPlaces - int(!isNegative);

    while(val > 0) {
        drawString[strIndex--] = _DIGITS[val % 10];
        val /= 10;
    }

    return DrawString(anchor, charSize, posPlaces + int(isNegative), texCoord);
}

float DrawFloat(float val, inout vec2 anchor, vec2 charSize, int negPlaces, vec2 uv) {
    int whole = int(val);
    int part = int(fract(abs(val)) * pow(10, negPlaces));

    int posPlaces = max(int(ceil(log10(abs(val)))), 1);

    anchor.x -= charSize.x * (posPlaces + int(val < 0) + 0.25);
    float ret = 0.0;
    ret += DrawInt(whole, anchor, charSize, uv);
    ret += DrawChar(_FSTP, anchor, charSize, texCoord);
    anchor.x -= charSize.x * 0.3;
    ret += DrawInt(part, anchor, charSize, uv);

    return ret;
}

void DrawDebugText() {
	#if (defined DEBUG) && (defined DRAW_DEBUG_VALUE) && (DEBUG_PROGRAM != 50)
    vec2 charSize = vec2(0.009) * ScreenSize.yy / ScreenSize;
    vec2 texPos = vec2(charSize.x / 1.5, 1.0 - charSize.y * 2.0);

    if(texCoord.x > charSize.x * 12.0 || texCoord.y < 1 - charSize.y * 12) {
        return;
    }

    vec3 color = fragColor.rgb;
    float text = 0.0;

    vec3 val = vec3(1.0);

    drawString = int[STRING_LENGTH] (_D, _E, _B, _U, _G, 0, _S, _T, _A, _T, _S, 0, 0, 0, 0);
    text += DrawString(texPos, charSize, 11, texCoord);
    color += text * vec3(1.0, 1.0, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

    texPos.x = charSize.x / 1.0, 1.0;
    texPos.y -= charSize.y * 2;

    text = 0.0;
    drawString = int[STRING_LENGTH] (_R, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    text += DrawString(texPos, charSize, 2, texCoord);
    texPos.x += charSize.x * 5.0;
    text += DrawFloat(val.r, texPos, charSize, 3, texCoord);
    color += text * vec3(1.0, 0.0, 0.0) * sqrt(clamp(abs(val.r), 0.2, 1.0));

    texPos.x = charSize.x / 1.0, 1.0;
    texPos.y -= charSize.y * 1.4;

    text = 0.0;
    drawString = int[STRING_LENGTH] (_G, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    text += DrawString(texPos, charSize, 2, texCoord);
    texPos.x += charSize.x * 5.0;
    text += DrawFloat(val.g, texPos, charSize, 3, texCoord);
    color += text * vec3(0.0, 1.0, 0.0) * sqrt(clamp(abs(val.g), 0.2, 1.0));

    texPos.x = charSize.x / 1.0, 1.0;
    texPos.y -= charSize.y * 1.4;

    text = 0.0;
    drawString = int[STRING_LENGTH] (_B, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    text += DrawString(texPos, charSize, 2, texCoord);
    texPos.x += charSize.x * 5.0;
    text += DrawFloat(val.b, texPos, charSize, 3, texCoord);
    color += text * vec3(0.0, 0.8, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

    texPos.x = charSize.x / 1.0, 1.0;
    texPos.y -= charSize.y * 1.4;

    fragColor.rgb = color;
	#endif
}
void main() {
    vec4 outcol = vec4(0.0, 0.0, 0.0, 1.0);
    float depth = texture(TranslucentDepthSampler, texCoord).r;
    float noise = clamp(mask(gl_FragCoord.xy + (Time * 100)), 0, 1);

    vec3 screenPos = vec3(texCoord, depth);
    vec3 clipPos = screenPos * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;
    vec3 p3 = mat3(gbufferModelViewInverse) * viewPos;
    vec3 view = normVec(p3);

    bool sky = depth >= 1.0;

    float comp = 1.0 - near / far / far;			//distances above that are considered as sky

    vec4 tpos = gbufferProjection * vec4((-sunPosition), 1.0);
    tpos = vec4(tpos.xyz / tpos.w, 1.0);
    vec2 pos1 = tpos.xy / tpos.z;
	vec2 lightPos = pos1*0.5+0.5;
    vec2 ntc2 = texCoord;
    vec2 deltatexcoord = vec2(lightPos - ntc2);

    vec2 noisetc = lightPos - deltatexcoord * clamp(noise,0,1);
    float gr = 0.0;

    vec4 Samplee = textureGather(TranslucentDepthSampler, noisetc);
    gr += dot(step(vec4(comp), Samplee), vec4(0.25));

    float grCol = clamp(gr*2-0.1,0,1);
    grCol = mix(1.0,grCol,clamp(cdist2(noisetc),0,1));
    grCol = mix(1.0,grCol,clamp(dot((sunPosition2), normalize(view))*2-1.2,0,1));
    grCol = mix(1.0,grCol,float(!sky));

    grCol = clamp(grCol,0,1);

    if(sky && overworld == 1.0) {

        //float frDepth = ld(depth2);
        //vec3 atmosphere = skyLut(view, sunPosition3.xyz, view.y, temporals3Sampler) + (noise / 12);
        vec3 atmosphere = skyLut2(view.xyz, sunPosition3, view.y, rainStrength * 0.5) * skys;
        atmosphere = (atmosphere);

        if(view.y > 0.) {
            vec3 avgamb = vec3(0.0);
            avgamb += ambientUp;
            avgamb += ambientDown;
            avgamb += ambientRight;
            avgamb += ambientLeft;
            avgamb += ambientB;
            avgamb += ambientF;
            avgamb *= (1.0 + rainStrength * 0.2);
            //vec4 cloud = texture(cloudsample, texCoord * CLOUDS_QUALITY);
            //vec4 cloud = BilateralUpscale(cloudsample, TranslucentDepthSampler, gl_FragCoord.xy, frDepth, 0.5);
            vec4 cloud = renderClouds(viewPos, avgamb, noise, suncol, suncol, avgamb).rgba;

            atmosphere += (stars(view) * 2.0) * clamp(1 - (rainStrength * 1), 0, 1);
            atmosphere += drawSun(dot(sunPosition2, view), 0, suncol.rgb / 150., vec3(0.0)) * clamp(1 - (rainStrength * 1), 0, 1) * 20;
            atmosphere = atmosphere * cloud.a + (cloud.rgb);
        }

        atmosphere = (clamp(atmosphere * 1.1, 0, 2));
        outcol.rgb = lumaBasedReinhardToneMapping(atmosphere);
        outcol.a = clamp(grCol, 0, 1);
        fragColor = outcol;
  	//DrawDebugText();
        return;
    } else {

        bool isWater = (texture(TranslucentSampler, texCoord).a * 255 == 200);

        vec2 texCoord = texCoord;
        vec3 wnormal = vec3(0.0);
        vec3 normal = normalize(constructNormal(depth, texCoord, TranslucentDepthSampler, float(isWater)));
        if(!isWater)normal = viewNormalAtPixelPosition2(gl_FragCoord.xy);

        vec2 texCoord2 = texCoord;
        if(isWater) {
            wnormal = normalize(viewToWorld(normal));

            float displ = (wnormal.z / (length(viewPos) / far) / 2000.);
            vec2 refractedCoord = texCoord + (displ * 0.5);
            if(texture(TranslucentSampler, refractedCoord).r <= 0.0)
                refractedCoord = texCoord;
            texCoord2 = refractedCoord;
        }

        vec4 OutTexel3 = (texture(DiffuseSampler, texCoord2).rgba);
        vec3 OutTexel = toLinear(OutTexel3.rgb);
        float mod2 = gl_FragCoord.x + gl_FragCoord.y;
        float res = mod(mod2, 2.0f);
        ivec2 texoffsets[4] = ivec2[] (ivec2(0, 1), ivec2(1, 0), -ivec2(0, 1), -ivec2(1, 0));
        vec4 depthgather = textureGatherOffsets(TranslucentDepthSampler, texCoord, texoffsets, 0);
        vec4 lmgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 3);
        float depthtest = (depthgather.x + depthgather.y + depthgather.z + depthgather.z) * 0.25;
        depthtest = round(clamp(float(depthtest - depth) * 10000 - 1, 0, 1));

        vec2 lmtrans = unpackUnorm2x4(OutTexel3.a);
        vec2 lmtrans10 = unpackUnorm2x4v2(lmgather);
        lmtrans10 = mix(lmtrans10, lmtrans, depthtest);

        float lmy = clamp(mix(lmtrans.y, lmtrans10.y / 4, res), 0.0, 1);
        float lmx = clamp(mix(lmtrans10.y, lmtrans.y, res), 0.0, 1);
        vec4 pbr = pbr(lmtrans, unpackUnorm2x4(lmgather.x), OutTexel3.rgb);
        float light = pbr.r;

        if(overworld == 1.0) {
            float ao = 1.0;
            float postlight = 1;
            int isEyeInWater = 0;
            int isEyeInLava = 0;
            if(fogcol.a > 0.078 && fogcol.a < 0.079)
                isEyeInWater = 1;
            if(fogcol.r == 0.6 && fogcol.b == 0.0)
                isEyeInLava = 1;

            if(lmx > 0.95) {
                lmx *= 0.75;
                lmy = 0.1;
                postlight = 0.0;
            }
            vec3 origin = backProject(vec4(0.0, 0.0, 0.0, 1.0)).xyz;

            float screenShadow = clamp((pow32(lmx)) * 100, 0.0, 1.0) * lmx;

            if(screenShadow > 0.0 && lmy < 0.9 && !isWater && isEyeInWater == 0) {
                ao = dbao(TranslucentDepthSampler);

              
                    screenShadow *= rayTraceShadow(sunVec + (origin * 0.1), viewPos, noise, depth) + lmy;
            }

            screenShadow = clamp(screenShadow, 0.1, 1.0);
            vec3 normal3 = (normal);
            normal = viewToWorld(normal3);
            vec3 ambientCoefs = normal / dot(abs(normal), vec3(1.0));

            vec3 ambientLight = ambientUp * clamp(ambientCoefs.y, 0., 1.);
            ambientLight += ambientDown * clamp(-ambientCoefs.y, 0., 1.);
            ambientLight += ambientRight * clamp(ambientCoefs.x, 0., 1.);
            ambientLight += ambientLeft * clamp(-ambientCoefs.x, 0., 1.);
            ambientLight += ambientB * clamp(ambientCoefs.z, 0., 1.);
            ambientLight += ambientF * clamp(-ambientCoefs.z, 0., 1.);
            ambientLight *= (1.0 + rainStrength * 0.2);

            ambientLight = clamp(ambientLight * (pow8(lmx) * 1.5) + (pow3(lmy) * 3.0) * (vec3(TORCH_R, TORCH_G, TORCH_B) * vec3(TORCH_R, TORCH_G, TORCH_B)), 0.01, 10.0);

            float sssa = pbr.g;
            float smoothness = pbr.a * 255 > 1.0 ? pbr.a : pbr.b;

            vec3 f0 = pbr.a * 255 > 1.0 ? vec3(0.8) : vec3(0.04);
            vec3 reflections = vec3(0.0);

            if(pbr.a * 255 > 1.0 && !isWater) {
                vec3 normal2 = normal3 + (noise * (1 - smoothness));

                vec3 avgSky = mix(vec3(0.0), ambientLight, lmx);
                vec4 reflection = vec4(SSR(viewPos.xyz, depth, normal2, noise));

                float normalDotEye = dot(normal, normalize(viewPos));
                float fresnel = pow5(clamp(1.0 + normalDotEye, 0.0, 1.0));
                fresnel = fresnel * 0.98 + 0.02;
                fresnel *= max(1.0 - 0 * 0.5 * 1, 0.5);
                fresnel *= 1.0 - 1 * 0.3;

                reflection = mix(vec4(avgSky, 1), reflection, reflection.a);
                reflections += ((reflection.rgb) * (fresnel * OutTexel));
                OutTexel *= 0.075;
                reflections = max(vec3(0.0), reflections);
            }

            float shadeDir = max(0.0, dot(normal, sunPosition2));

            shadeDir *= screenShadow;
            shadeDir += max(0.0, (max(phaseg(dot(view, sunPosition2), 0.5) * 2.0, phaseg(dot(view, sunPosition2), 0.1)) * pi * 1.6) * float(sssa) * lmx) * (max(0.1, (screenShadow * ao) * 2 - 1));
            shadeDir = clamp(shadeDir * pow3(lmx) * ao, 0, 1);

            float sunSpec = ((GGX(normal, -normalize(view), sunPosition2, 1 - smoothness, f0.x)));
            vec3 suncol = suncol * clamp(skyIntensity * 3.0, 0.15, 1);
            vec3 shading = (suncol * shadeDir) + ambientLight * ao;
            shading += (sunSpec * suncol) * shadeDir;

            shading = mix(ambientLight, shading, 1 - (rainStrength * lmx));
            if(light > 0.001)
                shading.rgb = vec3(light * 2.0);
            //shading = max(vec3(0.0005), shading);
            if(postlight != 1.0)
                shading = mix(vec3(1.0), shading, 0.5);
            if(isWater)
                shading = ambientLight;
            vec3 dlight = (OutTexel * shading) + reflections;
            outcol.rgb = lumaBasedReinhardToneMapping(dlight);

            outcol.rgb *= 1.0 + max(0.0, light);
            outcol.a = clamp(grCol, 0, 1);

        ///---------------------------------------------
            //outcol.rgb = clamp(vec3(sunPosition), 0.01, 1);
            //if(luma(ambientLight )>1.0) outcol.rgb = vec3(1.0,0,0);
        ///---------------------------------------------
        } else {
            vec2 dst_map_val = vec2(0);
            if(end != 1.0) {
                vec2 p_m = texCoord;

                vec2 p_d = texCoord - Time * 0.1;

                dst_map_val = fract(vec2(sin(length(fract(p_d) + Time * 0.2) * 100.0)));
                vec2 dst_offset = dst_map_val.xy;

                dst_offset *= 0.001;
                dst_offset *= (1. - texCoord.t);

                vec2 texCoord = texCoord + dst_offset;
                OutTexel = toLinear(texture(DiffuseSampler, texCoord).rgb);
            }

            float ao = 1.0;
            ao = dbao2(TranslucentDepthSampler);
            float lumC = luma(fogcol.rgb);
            vec3 diff = fogcol.rgb - lumC;

            vec3 ambientLight = clamp(vec3(diff) * ((max(0.15, lmx)) * 1.5) + (pow3(lmy) * 3.0) * (vec3(TORCH_R, TORCH_G, TORCH_B) * vec3(TORCH_R, TORCH_G, TORCH_B)), 0, 10.0);

            outcol.rgb = lumaBasedReinhardToneMapping(OutTexel.rgb * ambientLight * ao);
            //outcol.rgb = vec3(ao);
            if(light > 0.001)
                outcol.rgb *= clamp(vec3(2.0 - 1 * 2) * light * 2, 1.0, 10.0);
        //outcol.rgb = vec3(texCoord,0);

        }
        if(isWater) {

            vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B) * fogcol.rgb;
            vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
            vec3 totEpsilon = dirtEpsilon * Dirt_Amount + waterEpsilon;
            outcol.rgb *= clamp(exp(-length(viewPos) * totEpsilon), 0.2, 1.0);

        }

    }

    fragColor = outcol + (noise / 128);
      	//DrawDebugText();
}
