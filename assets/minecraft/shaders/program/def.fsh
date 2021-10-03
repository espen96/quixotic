#version 150

uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec4 skycol;

uniform sampler2D DiffuseSampler;
uniform sampler2D prevsky;
in float skyIntensity;
in vec3 nsunColor;
in vec3 sunDir;
in float skyIntensityNight;
in float rainStrength;
in float sunIntensity;
in float moonIntensity;


 in vec3 ambientUp;
 in vec3 ambientLeft;
 in vec3 ambientRight;
 in vec3 ambientB;
 in vec3 ambientF;
 in vec3 ambientDown;
 in vec3 avgSky;
out vec4 fragColor;


#define PI 3.141592


#define SKY_BRIGHTNESS_DAY 0.4//[0.0 0.5 0.75 1.0 1.2 1.4 1.6 1.8 2.0]
#define SKY_BRIGHTNESS_NIGHT 1.0 //[0.0 0.5 0.75 1.0 1.2 1.4 1.6 1.8 2.0]
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)


float facos(float inX) {

	const float C0 = 1.56467;
	const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : PI - res;
}

vec3 ScreenSpaceDither( vec2 vScreenPos )
{
    vec3 vDither = vec3(dot(vec2(131.0, 312.0), vScreenPos.xy + fract(Time*2048)));
    vDither.rgb = fract(vDither.rgb / vec3(103.0, 71.0, 97.0)) * vec3(2.0,2.0,2.0) - vec3(0.5, 0.5, 0.5);
    return (vDither.rgb / 15);
}


vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.2));
	return color;
}


#define MIN_LIGHT_AMOUNT 0.225 //[0.0 0.5 1.0 1.5 2.0 3.0 4.0 5.0]
#define TORCH_AMOUNT 1.0 //[0.0 0.5 0.75 1. 1.2 1.4 1.6 1.8 2.0]
#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.6 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.3 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]

vec3 reinhard_extended(vec3 v, float max_white)
{
    vec3 numerator = v * (1.0f + (v / vec3(max_white * max_white)));
    return numerator / (1.0f + v);
}
vec3 reinhard(vec3 v)
{
    return v / (1.0f + v);
}
float luminance(vec3 v)
{
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out)
{
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}
vec3 reinhard_extended_luminance(vec3 v, float max_white_l)
{
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return change_luminance(v, l_new);
}
vec3 reinhard_jodie(vec3 v)
{
    float l = luminance(v);
    vec3 tv = v / (1.0f + v);
    return mix(v / (1.0f + l), tv, tv);
}
/////////////////

const float TAU 	= PI * 2.0;
const float hPI 	= PI * 0.5;
const float rPI 	= 1.0 / PI;
const float rTAU 	= 1.0 / TAU;

const float PHI		= sqrt(5.0) * 0.5 + 0.5;
const float rLOG2	= 1.0 / log(2.0);


const float sunAngularSize = 0.533333;
const float moonAngularSize = 0.516667;

//Sky coefficients and heights

#define airNumberDensity 2.5035422e25
#define ozoneConcentrationPeak 8e-6
const float ozoneNumberDensity = airNumberDensity * ozoneConcentrationPeak;
#define ozoneCrossSection vec3(4.51103766177301e-21, 3.2854797958699e-21, 1.96774621921165e-22)

#define sky_planetRadius 6731e3

#define sky_atmosphereHeight 110e3
#define sky_scaleHeights vec2(8.0e3, 1.2e3)

#define sky_mieg 0.80 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
#define sky_coefficientRayleighR 5.8 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.0 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.0 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0 ]
#define sky_coefficientRayleighG 1.35 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.0 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.0 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0 ]
#define sky_coefficientRayleighB 3.31 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.0 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.0 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0 ]

#define sky_coefficientRayleigh vec3(sky_coefficientRayleighR*1e-6, sky_coefficientRayleighG*1e-5, sky_coefficientRayleighB*1e-5)


#define sky_coefficientMieR 3.0 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.0 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.0 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0 ]
#define sky_coefficientMieG 3.0 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.0 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.0 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0 ]
#define sky_coefficientMieB 3.0 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.0 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.0 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9.0 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0 ]

#define sky_coefficientMie vec3(sky_coefficientMieR*1e-6, sky_coefficientMieG*1e-6, sky_coefficientMieB*1e-6) // Should be >= 2e-6
const vec3 sky_coefficientOzone = (ozoneCrossSection * (ozoneNumberDensity * 1.2e-6)); // ozone cross section * (ozone number density * (cm ^ 3))

const vec2 sky_inverseScaleHeights = 1.0 / sky_scaleHeights;
const vec2 sky_scaledPlanetRadius = sky_planetRadius * sky_inverseScaleHeights;
const float sky_atmosphereRadius = sky_planetRadius + sky_atmosphereHeight;
const float sky_atmosphereRadiusSquared = sky_atmosphereRadius * sky_atmosphereRadius;

#define sky_coefficientsScattering mat2x3(sky_coefficientRayleigh, sky_coefficientMie)
const mat3   sky_coefficientsAttenuation = mat3(sky_coefficientRayleigh, sky_coefficientMie * 1.11, sky_coefficientOzone); // commonly called the extinction coefficient

#define sun_illuminance 128000.0 //[10000.0 20000.0 30000.0 40000.0 50000.0 60000.0 70000.0 80000.0 90000.0 100000.0 110000.0 120000.0 130000.0 140000.0 160000.0]
#define moon_illuminance 125.0 //[0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 1000.0 10000.0 100000.0]

#define sunColorR 1.0 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
#define sunColorG 0.898 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
#define sunColorB 0.826 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]

#define sunColorBase (vec3(sunColorR,sunColorG,sunColorB) * sun_illuminance)
//#define sunColorBase blackbody(5778) * sun_illuminance
#define moonColorR 0.9080 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
#define moonColorG 0.9121 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
#define moonColorB 0.8948 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]

#define moonColorBase (vec3(moonColorR, moonColorG, moonColorB) * moon_illuminance )    //Fake Purkinje effect

float sky_rayleighPhase(float cosTheta) {
	const vec2 mul_add = vec2(0.1, 0.28) * rPI;
	return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for energy conservation
}

float sky_miePhase(float cosTheta, const float g) {
	float gg = g * g;
	return (gg * -0.25 + 0.25) * rPI * pow(-(2.0 * g) * cosTheta + (gg + 1.0), -1.5);
}

vec2 sky_phase(float cosTheta, const float g) {
	return vec2(sky_rayleighPhase(cosTheta), sky_miePhase(cosTheta, g));
}

vec3 sky_density(float centerDistance) {
	vec2 rayleighMie = exp(centerDistance * -sky_inverseScaleHeights + sky_scaledPlanetRadius);

	// Ozone distribution curve by Sergeant Sarcasm - https://www.desmos.com/calculator/j0wozszdwa
	float ozone = exp(-max(0.0, (35000.0 - centerDistance) - sky_planetRadius) * (1.0 / 5000.0))
	            * exp(-max(0.0, (centerDistance - 35000.0) - sky_planetRadius) * (1.0 / 15000.0));
	return vec3(rayleighMie, ozone);
}

// Seb Lagarde Approx
float ATanPos(float x)
{
    float t0 = (x < 1.0f) ? x : 1.0f / x;
    float t1 = (-0.218891 * t0 + 1.01991) * t0; // p(x)
    return (x < 1.0f) ? t1: PI/2.0 - t1; // undo range reduction
}

float ATan(float x)
{
    float t0 = ATanPos(abs(x));
    return (x < 0.0f) ? -t0: t0; // undo range reduction
}


vec3 sky_airmass(vec3 position, vec3 direction, const float steps) {
	float l = length(position);
	vec2 rayLeighMie = exp(sky_scaledPlanetRadius - l * sky_inverseScaleHeights)  / sky_inverseScaleHeights;
	// Integral of the lorentzian approximation of ozone distribution : 1/(1+((x-40000)/5000)^2)
	float l2 = (l - 40000 - sky_planetRadius)/7000.;
	float ozone =  0.5 * PI *7000. - ATan(l2)*7000.;
	// The integral path is longer near the horizon, it is "streching" the vertical path
	return vec3(rayLeighMie, ozone) / clamp(direction.y*0.82+0.18*(position.y-sky_planetRadius)/110e3,1e-10,1.0);
}

vec3 sky_opticalDepth(vec3 position, vec3 direction, const float steps) {
	return sky_coefficientsAttenuation * sky_airmass(position, direction, steps);
}

vec3 sky_transmittance(vec3 position, vec3 direction, const float steps) {
	return exp2(-sky_opticalDepth(position, direction, steps) * rLOG2);
}

// No intersection if returned y component is < 0.0
vec2 rsi(vec3 position, vec3 direction, float radius) {
	float PoD = dot(position, direction);
	float radiusSquared = radius * radius;

	float delta = PoD * PoD + radiusSquared - dot(position, position);
	if (delta < 0.0) return vec2(-1.0);
	      delta = sqrt(delta);

	return -PoD + vec2(-delta, delta);
}

vec3 calculateAtmosphere(vec3 background, vec3 viewVector, vec3 upVector, vec3 sunVector, vec3 moonVector, out vec2 pid, out vec3 transmittance, const int iSteps, float noise) {
	const int jSteps = 4;

	vec3 viewPosition = vec3(0.0,sky_planetRadius + 0,0.0);

	vec2 aid = rsi(viewPosition, viewVector, sky_atmosphereRadius);
	if (aid.y < 0.0) {transmittance = vec3(1.0); return vec3(0.0);}

	pid = rsi(viewPosition, viewVector, sky_planetRadius * 0.998);
	bool planetIntersected = pid.y >= 0.0;

	vec2 sd = vec2((planetIntersected && pid.x < 0.0) ? pid.y : max(aid.x, 0.0), (planetIntersected && pid.x > 0.0) ? pid.x : aid.y);

	float stepSize  = (sd.y - sd.x) * (1.0 / iSteps);
	vec3  increment = viewVector * stepSize;
	vec3  position  = viewPosition + increment * (0.34*noise);
	float SdotV = dot(viewVector, sunVector );
	vec2 phaseSun  = sky_phase(SdotV, sky_mieg);
	vec2 phaseMoon = sky_phase(-SdotV, sky_mieg);

	vec3 scatteringSun     = vec3(0.0);
	vec3 scatteringMoon    = vec3(0.0);
	vec3 scatteringAmbient = vec3(0.0);
	transmittance = vec3(1.0);

	for (int i = 0; i < iSteps; ++i, position += increment) {
		vec3 density          = sky_density(length(position));
		if (density.y > 1e35) break;
		vec3 stepAirmass      = density * stepSize;
		vec3 stepOpticalDepth = sky_coefficientsAttenuation * stepAirmass;

		vec3 stepTransmittance       = exp2(-stepOpticalDepth * rLOG2);
		vec3 stepTransmittedFraction = (stepTransmittance - 1.0) / -stepOpticalDepth;
		vec3 stepScatteringVisible   = transmittance * stepTransmittedFraction;

		scatteringSun  += sky_coefficientsScattering * (stepAirmass.xy * phaseSun ) * stepScatteringVisible * sky_transmittance(position, sunVector,  jSteps);
		scatteringMoon += sky_coefficientsScattering * (stepAirmass.xy * phaseMoon) * stepScatteringVisible * sky_transmittance(position, moonVector, jSteps);
		// Nice way to fake multiple scattering.
		scatteringAmbient += sky_coefficientsScattering * stepAirmass.xy * stepScatteringVisible;

		transmittance *= stepTransmittance;
	}

	vec3 scattering = scatteringSun * sunColorBase + scatteringAmbient * background + scatteringMoon*moonColorBase;

	return scattering;
}

//////////////////
vec2 sincos(float x){
    return vec2(sin(x), cos(x));
}

vec3 cartToSphere(vec2 coord) {
	coord *= vec2(TAU, PI);
	vec2 lon = sincos(coord.x) * sin(coord.y);
	return vec3(lon.x, 2.0/PI*coord.y-1.0, lon.y);
}
void main() {
//    vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );

if (gl_FragCoord.x < 17. && gl_FragCoord.y < 17.){


vec3 avgAmbient = (ambientUp + ambientLeft + ambientRight + ambientB + ambientF + ambientDown)/6.*(1.0+rainStrength*0.2);

  float skyLut = floor(gl_FragCoord.y)/15.;
  float sky_lightmap = pow(skyLut,2.23);
  float torchLut = floor(gl_FragCoord.x)/15.;
  torchLut *= torchLut;
  float torch_lightmap = ((torchLut*torchLut)*(torchLut*torchLut))*(torchLut*10.)+torchLut;
	float avgEyeIntensity = ((sunIntensity*120.+moonIntensity*4.)+skyIntensity*230.+skyIntensityNight*4.)*sky_lightmap;
	float exposure =  0.18/log2(max(avgEyeIntensity*0.16+1.0,1.13));
  vec3 ambient = (((avgAmbient)*10.0)*sky_lightmap*log2(1.13+sky_lightmap*1.5)+torch_lightmap*0.05*vec3(TORCH_R,TORCH_G,TORCH_B)*TORCH_AMOUNT)*exposure * vec3(1.0,0.96,0.96)+MIN_LIGHT_AMOUNT*0.001*vec3(0.75,1.0,1.25);
  fragColor = vec4(reinhard_jodie(ambient*10.),1.0);
}

if (gl_FragCoord.x > 8. && gl_FragCoord.x < 9.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 ){




  //luminance (cie model)
	vec3 daySky = vec3(0.0);
	vec3 moonSky = vec3(0.0);
	// Day
	if (skyIntensity > 0.00001)
	{
		vec3 skyColor = nsunColor;
		daySky = skyIntensity*skyColor*vec3(0.8,0.9,1.)*15.*1.5;
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		moonSky = (skyIntensityNight*vec3(0.08,0.12,0.18)*vec3(0.4))*3.0;
	}
  fragColor.rgb =  ((daySky + moonSky) )*(1.0-rainStrength);


}




if (gl_FragCoord.x > 18. && gl_FragCoord.y > 1.){
  float cosY = clamp(floor(gl_FragCoord.x - 18.0)/256.*2.0-1.0,-1.0+1e-5,1.0-1e-5);
  cosY = pow(abs(cosY),1/3.0)*fsign(cosY);
  float mCosT = clamp(floor(gl_FragCoord.y-1.0)/256.,0.0,1.0);
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
	if (skyIntensity > 0.00001)
	{
	  float L0 = (1.0+a*exp(b/mCosT))*(1.0+c*(exp(d*Y)-exp(d*3.1415/2.))+e*cosY*cosY);
		vec3 skyColor0 = mix(vec3(0.05,0.5,1.)/1.5,vec3(0.4,0.5,0.6)/1.5,rainStrength*2);
		vec3 normalizedSunColor = nsunColor;

		vec3 skyColor = mix(skyColor0,normalizedSunColor,1.0-pow(1.0+L0,-1.2))*(1.0-rainStrength);
		daySky = pow(L0,1.0-rainStrength)*skyIntensity*skyColor*vec3(0.8,0.9,1.)*15.*SKY_BRIGHTNESS_DAY;
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		float L0Moon = (1.0+a*exp(b/mCosT))*(1.0+c*(exp(d*(PI-Y))-exp(d*3.1415/2.))+e*cosY*cosY);
		moonSky = pow(L0Moon,1.0-rainStrength)*skyIntensityNight*vec3(0.08,0.12,0.18)*vec3(0.4)*SKY_BRIGHTNESS_NIGHT;
	}
/*
  vec2 p = clamp(floor(gl_FragCoord.xy-vec2(18.,1.))/256.+0/256.,0.0,1.0);
  vec3 viewVector = cartToSphere(p);

	vec2 planetSphere = vec2(0.0);
	vec3 sky = vec3(0.0);
	vec3 skyAbsorb = vec3(0.0);
  vec3 WsunVec = sunDir;
	sky = calculateAtmosphere(vec3(0.0), viewVector, vec3(0.0,1.0,0.0), WsunVec, -WsunVec, planetSphere, skyAbsorb, 10, luminance(ScreenSpaceDither(gl_FragCoord.xy)))*0.00015;

  fragColor.rgb = sky;
  */
  fragColor.rgb = ( ((daySky + moonSky) ) )+(ScreenSpaceDither(gl_FragCoord.xy)/32);
}


}
