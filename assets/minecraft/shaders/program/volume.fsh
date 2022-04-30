#version 150
#extension GL_EXT_gpu_shader4_1 : enable
uniform sampler2D MainSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;
uniform vec2 ScreenSize;
uniform float Time;


in vec3 avgSky;
in mat4 gbufferProjection;

in vec2 texCoord;
in vec2 oneTexel;
flat in vec4 fogcol;
in mat4 gbufferModelViewInverse;
in mat4 gbufferProjectionInverse;
flat in float near;
flat in float far;
flat in float overworld;
flat in float end;

flat in float sunElevation;
flat in vec3 sunPosition;
flat in float fogAmount;
flat in vec2 eyeBrightnessSmooth;
in vec3 suncol;
#define VL_SAMPLES 2
#define Ambient_Mult 1.0
#define SEA_LEVEL 70
#define ATMOSPHERIC_DENSITY 0.5
#define fog_mieg1 0.40
#define fog_mieg2 0.10
#define fog_coefficientRayleighR 5.8
#define fog_coefficientRayleighG 1.35
#define fog_coefficientRayleighB 3.31

#define fog_coefficientMieR 3.0
#define fog_coefficientMieG 3.0
#define fog_coefficientMieB 3.0
#define SUNBRIGHTNESS 20
#define Dirt_Amount 0.005 // How much dirt there is in water

#define Dirt_Scatter_R 0.6 // How much dirt diffuses red
#define Dirt_Scatter_G 0.6 // How much dirt diffuses green
#define Dirt_Scatter_B 0.6 // How much dirt diffuses blue

#define Dirt_Absorb_R 0.65 // How much dirt absorbs red
#define Dirt_Absorb_G 0.85 // How much dirt absorbs green
#define Dirt_Absorb_B 1.05 // How much dirt absorbs blue

#define Water_Absorb_R 0.25422 // How much water absorbs red
#define Water_Absorb_G 0.03751 // How much water absorbs green
#define Water_Absorb_B 0.01150 // How much water absorbs blue

#define Dirt_Mie_Phase 0.4

vec3 cameraPosition = vec3(0, 72, 0);

out vec4 fragColor;
const float pi = 3.141592653589793238462643383279502884197169;
float cdist(vec2 coord)
{
    vec2 vec = abs(coord * 2.0 - 1.0);
    float d = max(vec.x, vec.y);
    return 1.0 - d * d;
}
vec4 lightCol = vec4(suncol, float(sunElevation > 1e-5) * 2 - 1.);
vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    return color;
}
float LinearizeDepth(float depth)
{
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

float luma(vec3 color)
{
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec4 backProject(vec4 vec)
{
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
}


///////////////////////////////////

float facos(float inX)
{
    const float C0 = 1.56467;
    const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}

float phaseRayleigh(float cosTheta)
{
    vec2 mul_add = vec2(0.1, 0.28) / facos(-1.0);
    return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for
                                             // energy conservation
}
float phaseg(float x, float g)
{
    float gg = g * g;
    return (gg * -0.25 + 0.25) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5) / 3.1415;
}
float cloudVol(in vec3 pos)
{
    float unifCov = exp2(-max(pos.y - SEA_LEVEL, 0.0) / 50.);
    float cloud = unifCov * 60. * fogAmount;
    return cloud;
}

mat2x3 getVolumetricRays(float dither, vec3 fragpos, vec3 ambientUp)
{
    vec3 wpos = fragpos;
    vec3 dVWorld = (wpos - gbufferModelViewInverse[3].xyz);

    float maxLength = min(length(dVWorld), far) / length(dVWorld);
    dVWorld *= maxLength;


    float SdotV = dot(sunPosition, normalize(fragpos)) * lightCol.a;
    float dL = length(dVWorld);

    float mie = max(phaseg(SdotV, fog_mieg1), 0.07692307692);
    float rayL = phaseRayleigh(SdotV);

    vec3 ambientCoefs = dVWorld / dot(abs(dVWorld), vec3(1.));

    vec3 ambientLight = ambientUp;

    vec3 skyCol0 = (8.0 * ambientLight * Ambient_Mult) / 16.0;
    vec3 sunColor = (8.0 * lightCol.rgb) / 1.5;

    vec3 rC = vec3(fog_coefficientRayleighR * 1e-6, fog_coefficientRayleighG * 1e-5, fog_coefficientRayleighB * 1e-5);
    vec3 mC = vec3(fog_coefficientMieR * 1e-6, fog_coefficientMieG * 1e-6, fog_coefficientMieB * 1e-6);

    float expFactor = 11.0;

    float d = (pow(expFactor, float(1 + 0)) / expFactor - 1.0 / expFactor) / (1 - 1.0 / expFactor);
    float dd = pow(expFactor, float(1 + 0)) * log(expFactor) / (expFactor - 1.0);
    vec3 progressW = gbufferModelViewInverse[3].xyz + cameraPosition + d * dVWorld;
    float densityVol = cloudVol(progressW);

    float density = densityVol * ATMOSPHERIC_DENSITY * 300.;
    vec2 airCoef = exp2(-max(1.0, 0.0) / vec2(8.0e3, 1.2e3) * vec2(6.0, 7.0)) * 6.0;

    vec3 rL = rC * airCoef.x;
    vec3 m = (airCoef.y + density) * mC;
    vec3 vL0 = sunColor * (rayL * rL + m * mie) + skyCol0 * (rL + m);
    
    vec3 vL = (vL0 - vL0 * exp(-(rL + m) * dd * dL)) / ((rL + m) + 0.00000001);
    vec3 absorbance = clamp(exp(-(rL + m) * dd * dL), 0.0, 1.0);

    return mat2x3(vL, absorbance);
}

void waterVolumetrics(inout vec3 inColor, vec3 rayEnd, float estEyeDepth, float rayLength, float dither,
                      vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL,
                      float sunElevation, float depth)
{
    float spCount = 0.5;


    float maxZ = min(rayLength, 32.0) / (1e-8 + rayLength);

    rayLength *= maxZ;
    float dY = normalize(rayEnd).y * rayLength;
    vec3 absorbance = vec3(1.0);
    vec3 vL = vec3(0.0);
    float phase = phaseg(VdotL, Dirt_Mie_Phase);
    float expFactor = 11.0;

    float d = (pow(expFactor, float(dither) * float(spCount)) / expFactor - 1.0 * expFactor) / (1 - 1.0 / expFactor);
    float dd = pow(expFactor, float(dither) * float(spCount)) * log(expFactor) * float(spCount) / (expFactor - 1.0);

    vec3 ambientMul = exp(-max(estEyeDepth - dY * d, 0.0) * waterCoefs * 1.1);
    vec3 sunMul = exp(-max((estEyeDepth - dY * d), 0.0) / abs(sunElevation) * waterCoefs);
    vec3 light = (lightSource * phase * sunMul + ambientMul * ambient) * scatterCoef;
    absorbance *= exp(-dd * rayLength * waterCoefs);
    vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs * absorbance;

    inColor += vL;
}


void main()
{
    float depth = texture(TranslucentDepthSampler, texCoord).r;
    float depth2 = texture(DiffuseDepthSampler, texCoord).r;
    vec2 texCoord = texCoord;
    vec2 dst_map_val = vec2(0);
    if (end != 1.0 && overworld != 1.0)
    {
        vec2 p_d = texCoord - Time * 0.1;

        dst_map_val = fract(vec2(sin(length(fract(p_d) + Time * 0.2) * 100.0)));
        vec2 dst_offset = dst_map_val.xy;

        dst_offset *= 0.001;
        dst_offset *= (1. - texCoord.t);

        texCoord = texCoord + dst_offset;
    }

    vec2 texCoord2 = texCoord;
    float lum = luma(fogcol.rgb);
    vec3 diff = fogcol.rgb - lum;
    vec3 test = clamp(vec3(0.0) + diff * (-lum * 1.0 + 2), 0, 1);
    bool isWater = (texture(TranslucentSampler, texCoord).a * 255 == 200);
    int isEyeInWater = 0;
    int isEyeInLava = 0;
    if (fogcol.a > 0.078 && fogcol.a < 0.079)
        isEyeInWater = 1;
    if (fogcol.r == 0.6 && fogcol.b == 0.0)
        isEyeInLava = 1;

    vec3 vl = vec3(0.);
    float estEyeDepth = max(62.90 - cameraPosition.y, 0.0);

    vec3 OutTexel = texture(MainSampler, texCoord).rgb;
    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));
    vec3 screenPos = vec3(texCoord, depth);
    vec3 clipPos = screenPos * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;

    vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
    fragColor.rgb = OutTexel;

    if (overworld == 1.0)
    {
        vec3 direct;
        direct = suncol;

        float df = length(fragpos);

        if (isEyeInWater == 1 && overworld == 1)
        {
            float dirtAmount = Dirt_Amount;
            vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B) * fogcol.rgb;
            vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
            vec3 totEpsilon = dirtEpsilon * dirtAmount + waterEpsilon;
            vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B);
            fragColor.rgb *= clamp(exp(-df * totEpsilon), 0.2, 1.0);

            waterVolumetrics(vl, fragpos, estEyeDepth, length(fragpos), 1, totEpsilon, scatterCoef, avgSky * 10,
                             direct.rgb, dot(normalize(fragpos), normalize(sunPosition)), sunElevation, depth);
            vec3 abso = exp(-length(fragpos) * totEpsilon);

            fragColor.rgb *= abso;

            fragColor.rgb += lumaBasedReinhardToneMapping(vl);
        }

        else if (isEyeInWater == 0)
        {
            mat2x3 vl = getVolumetricRays(1.0, fragpos, avgSky);
            float lumC = luma(vl[0]);
            vec3 diff = vl[0] - lumC;
            vl[0] = vl[0] + diff * (-lumC * 1.25 + 0.5);

            fragColor.rgb *= vl[1];
            fragColor.rgb += lumaBasedReinhardToneMapping(vl[0]);
            float absorbance = dot(vl[1], vec3(0.22, 0.71, 0.07));
            if (isEyeInWater == 1)
                absorbance = 1;
            fragColor.rgb *= vl[1];
            fragColor.rgb += lumaBasedReinhardToneMapping(vl[0]);
           // if (depth2 >= 1 && isWater)
           //     fragColor.rgb = OutTexel.rgb;
        }
    }
    else
    {
        fragColor.rgb = mix(fragColor.rgb * 2.0, fogcol.rgb * 0.5, pow(depth, 256));
        fragColor.a = pow(depth, 256);
    }

    if (isEyeInLava == 1)
    {
        fragColor.rgb *= exp(-length(fragpos) * vec3(0.2, 0.7, 4.0) * 4.);
        fragColor.rgb += vec3(4.0, 0.5, 0.1) * 0.5;
    }
    if (fogcol.r == 0 && fogcol.g == 0 && fogcol.b == 0)
    {
        fragColor.rgb *= exp(-length(fragpos) * vec3(1.0) * 0.25);
    }
}
