#version 150

uniform sampler2D noisetex;
uniform sampler2D TranslucentDepthSampler;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;
in vec3 avgSky;
in vec3 sc;
in mat4 gbufferProjectionInverse;

in mat4 gbufferModelViewInverse;
in float sunElevation;
in float rainStrength;
in float cloudy;
in vec3 sunVec;

in float ambientMult;
in vec3 skyCol0;
in vec3 curvedPos;
in vec3 samplePos;

out vec4 fragColor;
#define CLOUDS_QUALITY 0.5 
#define VOLUMETRIC_CLOUDS

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

////////////////////////////////////////////////

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
	float mult = max(pos.y - 2000.0, 0.0) * 0.0005;
	float mult2 = max(-pos.y + 2000.0, 0.0) * 0.002;
	float coverage = clamp(texture(noisetex, fract(samplePos.xz * 0.00008)).x + 0.5 * rainStrength, 0.0, 1.0);
	float cloud = sqr(coverage) - pow3(mult) * 3.0 - sqr(mult2);
	return max(cloud, 0.0);
}
//Erode cloud with 3d Perlin-worley noise, actual cloud value

float cloudVol(in vec3 pos, in vec3 samplePos, in float cov) {
	float mult2 = (pos.y - 1500) * 0.0004 + rainStrength * 0.4;

	float cloud = clamp(cov - 0.11 * (0.2 + mult2), 0.0, 1.0);
	return cloud;

}
const float pi = 3.141592653589793238462643383279502884197169;

const float pidiv = 0.31830988618; // 1/pi

//Mie phase function
float phaseg(float x, float g) {
	float gg = sqr(g);
	return ((-0.25 * gg + 0.25) * pidiv) * pow(-2.0 * g * x + gg + 1.0, -1.5);
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

	vec3 sunContribution = mieDay * sunColor * pi;
	vec3 moonContribution = mieNight * moonColor * pi;

	for(int i = 0; i < maxIT_clouds; i++) {
		vec3 curvedPos = progress_view;
		vec2 xz = progress_view.xz - cameraPosition.xz;

		curvedPos.y -= sqrt((sky_planetRadius * sky_planetRadius) - dot(xz, xz)) - sky_planetRadius;
		vec3 samplePos = curvedPos * vec3(1.0, 0.03125, 1.0) * 0.25 + (sunElevation * 1000);

		float coverageSP = cloudCov(curvedPos, samplePos);
		if(coverageSP > 0.07) {
			float cloud = cloudVol(curvedPos, samplePos, coverageSP);
			if(cloud > 0.05) {
				float mu = cloud * cdensity;

				//fake multiple scattering approx 2  (from horizon zero down clouds)
				float h = 0.35 - 0.35 * clamp((progress_view.y - 1500.0) * 0.00025, 0.0, 1.0);
				float powder = 1.0 - exp(-mu * mult);
				float Shadow = mix(1.0, powder, h);
				float ambientPowder = mix(1.0, powder, h * ambientMult);
				vec3 S = vec3(sunContribution * Shadow + Shadow * moonContribution + skyCol0 * ambientPowder);

				vec3 Sint = (S - S * exp(-mult * mu)) / (mu);
				color += mu * Sint * total_extinction;
				total_extinction *= exp(-mu * mult);
				if(total_extinction < 0.1)
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

float dither64 = bayer16(gl_FragCoord.xy);
vec3 lumaBasedReinhardToneMapping(vec3 color) {
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= clamp(toneMappedLuma / luma, 0, 10);
    //color = pow(color, vec3(0.45454545454));
	return color;
}
vec3 reinhard(vec3 x) {
	x *= 1.66;
	return x / (1.0 + x);
}
void main() {

    //vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
	float noise = mask(gl_FragCoord.xy + (Time * 4000));

	float depth = 1.0;

	vec2 halfResTC = vec2(gl_FragCoord.xy / CLOUDS_QUALITY);
	#ifdef VOLUMETRIC_CLOUDS
	bool doClouds = false;
	for(int i = 0; i < floor(1.0 / CLOUDS_QUALITY) + 1.0; i++) {
		for(int j = 0; j < floor(1.0 / CLOUDS_QUALITY) + 1.0; j++) {
			if(texelFetch(TranslucentDepthSampler, ivec2(halfResTC) + ivec2(i, j), 0).x >= 1.0)
				doClouds = true;
		}
	}
	if(doClouds) {
		vec3 sc = sc * (1 - ((rainStrength) * 0.5));
		vec3 screenPos = vec3(halfResTC * oneTexel, depth);
		vec3 clipPos = screenPos * 2.0 - 1.0;
		vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
		vec3 viewPos = tmp.xyz / tmp.w;

		vec4 cloud = renderClouds(viewPos, avgSky, dither64, sc, sc, avgSky).rgba;
		cloud.rgb = lumaBasedReinhardToneMapping(cloud.rgb);
		fragColor = cloud;
	} else
		fragColor = vec4(0.0, 0.0, 0.0, 1.0);

	#else
	fragColor = vec4(0.0, 0.0, 0.0, 1.0);
	#endif

}
