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

float frameTimeCounter = sunElevation * 1000;

const float PI = 3.141592;
vec3 cameraPosition = vec3(0,abs((cloudy)),0);
const float cloud_height = 1500.;
const float maxHeight = 1650.;
int maxIT_clouds = 15;
const float steps = 15.0;
const float cdensity = 0.20;

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
	//if(worldV.y < cloud_height) return vec4(0.0, 0.0, 0.0, 1.0);	//don't trace if no intersection is possible

	dV_view = normalize(dV_view);

		//setup ray to start at the start of the cloud plane and end at the end of the cloud plane
	dV_view *= max(maxHeight - cloud_height, 0.0) / dV_view.y / maxIT_clouds;

	vec3 startOffset = dV_view * clamp(dither, 0, 1);
	progress_view = startOffset + cameraPosition + dV_view * (cloud_height - cameraPosition.y) / (dV_view.y);

	float mult = length(dV_view);

	color = vec3(0.0);
	float SdotV = dot(sunVec, normalize(fragpositi));
		//fake multiple scattering approx 1 (from horizon zero down clouds)
	float mieDay = max(phaseg(SdotV, 0.4), phaseg(SdotV, 0.2));
	float mieNight = max(phaseg(-SdotV, 0.4), phaseg(-SdotV, 0.2));

	vec3 sunContribution = mieDay * sunColor * 3.14;
	vec3 moonContribution = mieNight * moonColor * 3.14;
	float ambientMult = exp(-(1 + 0.24 + 0.8 * clamp(rainStrength, 0.75, 1)) * cdensity * 50.0);
	vec3 skyCol0 = avgAmbient * ambientMult;

	for(int i = 0; i < maxIT_clouds; i++) {
		vec3 curvedPos = progress_view;
		vec2 xz = progress_view.xz - cameraPosition.xz;
		curvedPos.y -= sqrt(pow(6731e3, 2.0) - dot(xz, xz)) - 6731e3;
		vec3 samplePos = curvedPos * vec3(1.0, 1. / 32., 1.0) / 4 + frameTimeCounter * vec3(0.5, 0., 0.5);

		float coverageSP = cloudCov(curvedPos, samplePos);
		if(coverageSP > 0.05) {
			float cloud = cloudVol(curvedPos, samplePos, coverageSP);
			if(cloud > 0.05) {
				float mu = cloud * cdensity;

				//fake multiple scattering approx 2  (from horizon zero down clouds)
				float h = 0.35 - 0.35 * clamp(progress_view.y / 4000. - 1500. / 4000., 0.0, 1.0);
				float powder = 1.0 - exp(-mu * mult);
				float Shadow = mix(0.5, powder, h);
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
    float dither2 = fract(dither5x3() - dither64);

	float depth = texture(DiffuseDepthSampler, texCoord).r;

	vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));
	vec3 sc = sc * (1 - ((rainStrength) * 0.5));
	vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
	vec4 cloud = renderClouds(fragpos, avgSky, noise, sc, sc, avgSky).rgba;

	fragColor = cloud;

}
