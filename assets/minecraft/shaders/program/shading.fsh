#version 150
#extension GL_ARB_gpu_shader5 : enable
uniform sampler2D DiffuseSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D cloudsample;
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
in vec3 nsunColor;
flat in vec3 zMults;

in vec2 oneTexel;
in vec4 fogcol;

in vec2 texCoord;

in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 wgbufferModelViewInverse;
in mat4 gbufferProjectionInverse;
in mat4 gbufferProjection;

in float near;
in float far;
in float end;
in float overworld;

in float rainStrength;
in vec3 sunVec;

in vec3 sunPosition2;
in vec3 sunPosition3;
in float skyIntensityNight;
in float skyIntensity;


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
	#define SKY_BRIGHTNESS_DAY 0.4
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
    return pow(tv, vec3(0.45454545454));
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

float dbao(sampler2D depth) {
    float ao = 0.0;
    float aspectRatio = ScreenSize.x / ScreenSize.y;

    const int aoloop = 3;
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
float ld(float depth) {
    return 1.0 / (zMults.y - depth * zMults.z);		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}

vec4 BilateralUpscale(sampler2D tex, sampler2D depth, vec2 coord, float frDepth, float rendres) {

    vec4 vl = vec4(0.0);
    float sum = 0.0;
    ivec2 posVl = ivec2(coord / 2.0);

    ivec2 pos = (ivec2(gl_FragCoord.xy + (Time * 1000)) % 2) * 1;

    float w = 1e-5;
    vl += texelFetch(tex, posVl + ivec2(-0.5) + pos, 0) * w;
    sum += w;

    vl += texelFetch(tex, posVl + ivec2(-0.5, 0) + pos, 0) * w;
    sum += w;

    vl += texelFetch2D(tex, posVl + ivec2(0) + pos, 0) * w;
    sum += w;

    vl += texelFetch(tex, posVl + ivec2(0, -0.5) + pos, 0) * w;
    sum += w;

    return vl / sum;
}
vec4 BilateralUpscale2(sampler2D tex, sampler2D depth, vec2 coord, float frDepth, float rendres) {

    vec4 vl = vec4(0.0);
    float sum = 0.0;
    mat3x3 weights;
    const ivec2 scaling = ivec2(1.0 / 1);

    ivec2 posD = ivec2(coord / 1.0) * scaling;
    ivec2 posVl = ivec2(coord / 1.0);

    float dz = zMults.x;
    ivec2 pos = (ivec2(gl_FragCoord.xy + (Time * 1000)) % 2) * 2;
	//pos = ivec2(1,-1);

    ivec2 tcDepth = posD + ivec2(-4, -4) * scaling + pos * scaling;
    float dsample = ld(texelFetch2D(depth, tcDepth, 0).r);
    float w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch2D(tex, posVl + ivec2(-2) + pos, 0) * w;
    sum += w;

    tcDepth = posD + ivec2(-10, 0) * scaling + pos * scaling;
    dsample = ld(texelFetch2D(depth, tcDepth, 0).r);
    w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch2D(tex, posVl + ivec2(-2, 0) + pos, 0) * w;
    sum += w;

    tcDepth = posD + ivec2(0) + pos * scaling;
    dsample = ld(texelFetch2D(depth, tcDepth, 0).r);
    w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch2D(tex, posVl + ivec2(0) + pos, 0) * w;
    sum += w;

    tcDepth = posD + ivec2(0, -4) * scaling + pos * scaling;
    dsample = ld(texelFetch2D(depth, tcDepth, 0).r);
    w = abs(dsample - frDepth) < dz ? 1.0 : 1e-5;
    vl += texelFetch2D(tex, posVl + ivec2(0, -2) + pos, 0) * w;
    sum += w;

    return vl / sum;
}

vec3 viewToWorld(vec3 viewPos) {

    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = gbufferModelViewInverse * pos;

    return pos.xyz;
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
vec3 constructNormal(float depthA, vec2 texcoords, sampler2D depthtex, float water) {
    vec2 offsetB = vec2(0.0, oneTexel.y);
    vec2 offsetC = vec2(oneTexel.x, 0.0);
    float depthB = texture(depthtex, texcoords + offsetB).r;
    float depthC = texture(depthtex, texcoords + offsetC).r;
    vec3 A = getDepthPoint(texcoords, depthA);
    A += pow4(texture(DiffuseSampler, texCoord).g) * 0.01 * 1 - water;

    vec3 B = getDepthPoint(texcoords + offsetB, depthB);
    B += pow4(texture(DiffuseSampler, texCoord + offsetB * 3.0).g) * 0.01 * 1 - water;

    vec3 C = getDepthPoint(texcoords + offsetC, depthC);
    C += pow4(texture(DiffuseSampler, texCoord + offsetC * 3.0).g) * 0.01 * 1 - water;

    vec3 AB = normalize(B - A);
    vec3 AC = normalize(C - A);

    vec3 normal = -cross(AB, AC);
	// normal.z = -normal.z;

    return normalize(normal);
}
vec2 unpackUnorm2x4v2(vec4 pack) {
    vec2 xy;
    float pack2 = (pack.x + pack.y + pack.z + pack.z) / 4;
    xy.x = modf(pack2 * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
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

    if(sky && overworld == 1.0) {

        //float frDepth = ld(depth2);
        //vec3 atmosphere = skyLut(view, sunPosition3.xyz, view.y, temporals3Sampler) + (noise / 12);
        vec3 atmosphere = skyLut2(view.xyz, sunPosition3, view.y, rainStrength);

        if(view.y > 0.) {
            vec4 cloud = texture(cloudsample, texCoord * CLOUDS_QUALITY);
            //vec4 cloud = BilateralUpscale(cloudsample, TranslucentDepthSampler, gl_FragCoord.xy, frDepth, 0.5);

            atmosphere += (stars(view) * 2.0) * clamp(1 - (rainStrength * 1), 0, 1);
            atmosphere += drawSun(dot(sunPosition2, view), 0, suncol.rgb / 150., vec3(0.0)) * clamp(1 - (rainStrength * 1), 0, 1) * 20;
            atmosphere = atmosphere * cloud.a + (cloud.rgb);
        }

        atmosphere = (clamp(atmosphere * 1.1, 0, 2));
        outcol.rgb = reinhard(atmosphere);
        fragColor = outcol;

        return;
    } else {

        bool isWater = (texture(TranslucentSampler, texCoord).a * 255 == 200);

        vec2 texCoord = texCoord;
        vec3 wnormal = vec3(0.0);
        vec3 normal = normalize(constructNormal(depth, texCoord, TranslucentDepthSampler, float(isWater)));

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
                screenShadow *= rayTraceShadow(sunVec + (origin * 0.1), viewPos, noise, depth) + lmy;
                ao = dbao(TranslucentDepthSampler);
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

            ambientLight = clamp(ambientLight * (pow8(lmx) * 1.5) + (pow3(lmy)*3.0) * (vec3(TORCH_R, TORCH_G, TORCH_B)*vec3(TORCH_R, TORCH_G, TORCH_B)), 0, 10.0);

            vec4 pbr = pbr(lmtrans, unpackUnorm2x4(lmgather.x), OutTexel3.rgb);

            float sssa = pbr.g;
            float smoothness = pbr.a * 255 > 1.0 ? pbr.a : pbr.b;

            float light = pbr.r;
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
            shadeDir = clamp(shadeDir*pow3(lmx) * ao, 0, 1);

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

        ///---------------------------------------------
            //outcol.rgb = clamp(vec3(shading), 0.01, 1);
            //if(luma(ambientLight )>1.0) outcol.rgb = vec3(1.0,0,0);
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

    fragColor = outcol + (noise / 255);
}
