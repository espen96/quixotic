#version 150

uniform sampler2D noisetex;
uniform sampler2D DiffuseDepthSampler;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;
in vec3 avgSky;
in vec3 sc;

in mat4 gbufferModelViewInverse;
in float sunElevation;
in float rainStrength;
in float cloudy;
in vec3 sunVec;

out vec4 fragColor;
#define CLOUDS_QUALITY 0.85 

////////////////////////////////////////////////



const float sky_planetRadius = 6731e3;


float frameTimeCounter = sunElevation * 1000;
#define cloud_LevelOfDetail 1		// Number of fbm noise iterations for on-screen clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define cloud_ShadowLevelOfDetail -1	// Number of fbm noise iterations for the shadowing of on-screen clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define cloud_LevelOfDetailLQ -1 // Number of fbm noise iterations for reflected clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define cloud_ShadowLevelOfDetailLQ -1	// Number of fbm noise iterations for the shadowing of reflected clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define minRayMarchSteps 25		// Number of ray march steps towards zenith for on-screen clouds	[20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200]
#define maxRayMarchSteps 50		// Number of ray march steps towards horizon for on-screen clouds	[20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200]
#define minRayMarchStepsLQ 10	// Number of ray march steps towards zenith for reflected clouds	[5  10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90 95 100]
#define maxRayMarchStepsLQ 25		// Number of ray march steps towards horizon for reflected clouds	[  5  10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90 95 100]
#define cloudMieG 0.55 // Values close to 1 will create a strong peak of luminance around the sun and weak elsewhere, values close to 0 means uniform fog. [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]
#define cloudMieG2 0.2 // Multiple scattering approximation. Values close to 1 will create a strong peak of luminance around the sun and weak elsewhere, values close to 0 means uniform fog. [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]
#define cloudMie2Multiplier 0.7 // Multiplier for multiple scattering approximation  [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]




#define cloudDensity 0.0199		// Cloud Density, 0.02-0.04 is around irl values	[0.0010 0.0011 0.0013 0.0015 0.0017 0.0020 0.0023 0.0026 0.0030 0.0034 0.0039 0.0045 0.0051 0.0058 0.0067 0.0077 0.0088 0.0101 0.0115 0.0132 0.0151 0.0173 0.0199 0.0228 0.0261 0.0299 0.0342 0.0392 0.0449 0.0514 0.0589 0.0675 0.0773 0.0885 0.1014 0.1162 0.1331 0.1524 0.1746 0.2000]
#define cloudCoverage -0.24			// Cloud coverage	[-1.00 -0.98 -0.96 -0.94 -0.92 -0.90 -0.88 -0.86 -0.84 -0.82 -0.80 -0.78 -0.76 -0.74 -0.72 -0.70 -0.68 -0.66 -0.64 -0.62 -0.60 -0.58 -0.56 -0.54 -0.52 -0.50 -0.48 -0.46 -0.44 -0.42 -0.40 -0.38 -0.36 -0.34 -0.32 -0.30 -0.28 -0.26 -0.24 -0.22 -0.20 -0.18 -0.16 -0.14 -0.12 -0.10 -0.08 -0.06 -0.04 -0.02 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50 0.52 0.54 0.56 0.58 0.60 0.62 0.64 0.66 0.68 0.70 0.72 0.74 0.76 0.78 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98 1.00]
#define fbmAmount 1.00 		// Amount of noise added to the cloud shape	[0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50 0.52 0.54 0.56 0.58 0.60 0.62 0.64 0.66 0.68 0.70 0.72 0.74 0.76 0.78 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.18 1.20 1.22 1.24 1.26 1.28 1.30 1.32 1.34 1.36 1.38 1.40 1.42 1.44 1.46 1.48 1.50 1.52 1.54 1.56 1.58 1.60 1.62 1.64 1.66 1.68 1.70 1.72 1.74 1.76 1.78 1.80 1.82 1.84 1.86 1.88 1.90 1.92 1.94 1.96 1.98 2.00 2.02 2.04 2.06 2.08 2.10 2.12 2.14 2.16 2.18 2.20 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.38 2.40 2.42 2.44 2.46 2.48 2.50 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 2.72 2.74 2.76 2.78 2.80 2.82 2.84 2.86 2.88 2.90 2.92 2.94 2.96 2.98 3.00]
#define fbmPower1 4.00	// Higher values increases high frequency details of the cloud shape	[1.50 1.52 1.54 1.56 1.58 1.60 1.62 1.64 1.66 1.68 1.70 1.72 1.74 1.76 1.78 1.80 1.82 1.84 1.86 1.88 1.90 1.92 1.94 1.96 1.98 2.00 2.02 2.04 2.06 2.08 2.10 2.12 2.14 2.16 2.18 2.20 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.38 2.40 2.42 2.44 2.46 2.48 2.50 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 2.72 2.74 2.76 2.78 2.80 2.82 2.84 2.86 2.88 2.90 2.92 2.94 2.96 2.98 3.00 3.02 3.04 3.06 3.08 3.10 3.12 3.14 3.16 3.18 3.20 3.22 3.24 3.26 3.28 3.30 3.32 3.34 3.36 3.38 3.40 3.42 3.44 3.46 3.48 3.50 3.52 3.54 3.56 3.58 3.60 3.62 3.64 3.66 3.68 3.70 3.72 3.74 3.76 3.78 3.80 3.82 3.84 3.86 3.88 3.90 3.92 3.94 3.96 3.98 4.00]
#define fbmPower2 2.00	// Lower values increases high frequency details of the cloud shape	[1.50 1.52 1.54 1.56 1.58 1.60 1.62 1.64 1.66 1.68 1.70 1.72 1.74 1.76 1.78 1.80 1.82 1.84 1.86 1.88 1.90 1.92 1.94 1.96 1.98 2.00 2.02 2.04 2.06 2.08 2.10 2.12 2.14 2.16 2.18 2.20 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.38 2.40 2.42 2.44 2.46 2.48 2.50 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 2.72 2.74 2.76 2.78 2.80 2.82 2.84 2.86 2.88 2.90 2.92 2.94 2.96 2.98 3.00 3.02 3.04 3.06 3.08 3.10 3.12 3.14 3.16 3.18 3.20 3.22 3.24 3.26 3.28 3.30 3.32 3.34 3.36 3.38 3.40 3.42 3.44 3.46 3.48 3.50 3.52 3.54 3.56 3.58 3.60 3.62 3.64 3.66 3.68 3.70 3.72 3.74 3.76 3.78 3.80 3.82 3.84 3.86 3.88 3.90 3.92 3.94 3.96 3.98 4.00]

const float PI = 3.141592;
vec3 cameraPosition = vec3(0,abs((cloudy)),0);
const float cloud_height = 1500.;
const float maxHeight = 1650.;
int maxIT_clouds = 15;
const float steps = 15.0;
const float cdensity = 0.2;

///////////////////////////

//Mie phase function
float phaseg(float x, float g) {
	float gg = g * g;
	return (gg * -0.25 / 3.14 + 0.25 / 3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}

vec4 textureGood(sampler2D sam, vec2 uv) {
	vec2 res = textureSize(sam, 0) * 0.75;

	vec2 st = uv * res - 0.5;

	vec2 iuv = floor(st);
	vec2 fuv = fract(st);

	vec4 a = textureLod(sam, (iuv + vec2(0.5, 0.5)) / res, 0);
	vec4 b = textureLod(sam, (iuv + vec2(1.5, 0.5)) / res, 0);
	vec4 c = textureLod(sam, (iuv + vec2(0.5, 1.5)) / res, 0);
	vec4 d = textureLod(sam, (iuv + vec2(1.5, 1.5)) / res, 0);

	return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}

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

	vec4 fragposition = vec4(fragpositi, 1.0);
	
	vec3 worldV = normalize(fragposition.rgb);
	float VdotU = worldV.y;
	maxIT_clouds = int(clamp(maxIT_clouds / sqrt(VdotU), 0.0, maxIT_clouds));

	vec3 dV_view = worldV;

	vec3 progress_view = dV_view * dither + cameraPosition;

	float total_extinction = 1.0;

	worldV = normalize(worldV) * 300000. + cameraPosition; //makes max cloud distance not dependant of render distance
	if(worldV.y < cloud_height) return vec4(0.0, 0.0, 0.0, 1.0);	//don't trace if no intersection is possible

	dV_view = normalize(dV_view);

	//setup ray to start at the start of the cloud plane and end at the end of the cloud plane
	dV_view *= max(maxHeight - cloud_height, 0.0) / dV_view.y / maxIT_clouds;

	vec3 startOffset = dV_view * clamp(dither, 0.0, 1.0);
	progress_view = startOffset + cameraPosition + dV_view * (cloud_height - cameraPosition.y) / (dV_view.y);

	float mult = length(dV_view);

	color = vec3(0.0);
	float SdotV = dot(sunVec, normalize(fragpositi));
	//fake multiple scattering approx 1 (from horizon zero down clouds)
	float mieDay = max(phaseg(SdotV, 0.2), phaseg(SdotV, 0.2));
	float mieNight = max(phaseg(-SdotV, 0.2), phaseg(-SdotV, 0.2));

	vec3 sunContribution = mieDay * sunColor * 3.14;
	vec3 moonContribution = mieNight * moonColor * 3.14;
	float ambientMult = exp(-(1.25 + 0.8 * clamp(rainStrength, 0.75, 1)) * cdensity * 50.0);
	vec3 skyCol0 = avgAmbient * ambientMult;

	for(int i = 0; i < maxIT_clouds; i++) {
		vec3 curvedPos = progress_view;
		vec2 xz = progress_view.xz - cameraPosition.xz;
		curvedPos.y -= sqrt(pow(sky_planetRadius, 2.0) - dot(xz, xz)) - sky_planetRadius;
		vec3 samplePos = curvedPos * vec3(1.0, 1.0 / 32.0, 1.0) / 4 + frameTimeCounter * vec3(0.5, 0.0, 0.5);

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

vec4 backProject(vec4 vec) {
	vec4 tmp = gbufferModelViewInverse * vec;
	return tmp / tmp.w;
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

void main() {

    //vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
	float noise = mask(gl_FragCoord.xy + (Time * 4000));

	float depth = 1.0;

	vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));
	vec3 sc = sc * (1 - ((rainStrength) * 0.5));
	vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
	vec4 cloud = renderClouds(fragpos, avgSky, noise, sc, sc, avgSky).rgba;

	fragColor = cloud;

}
