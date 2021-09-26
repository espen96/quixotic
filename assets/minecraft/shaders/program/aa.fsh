#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;
uniform mat4 ProjMat;

in vec3 flareColor;

in float GameTime;
in vec2 texCoord;
in vec2 texCoord2;
in vec2 oneTexel;
in vec3 sunDir;
in vec4 fogcol;
in vec4 skycol;
in vec4 rain;
in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in float near;
in float far;

in float aspectRatio;
in float cosFOVrad;
in float tanFOVrad;
in mat4 gbPI;
in mat4 gbP;


flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in vec3 zenithColor;
flat in vec3 sunColor;
flat in vec3 sunColorCloud;
flat in vec3 moonColor;
flat in vec3 moonColorCloud;
flat in vec3 lightSourceColor;
flat in vec3 avgSky;



out vec4 fragColor;

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0




vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}


const mat3 ACESInputMat =
mat3(0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ODT_SAT => XYZ => D60_2_D65 => sRGB
const mat3 ACESOutputMat =
mat3( 1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);
vec3 LinearTosRGB(in vec3 color)
{
    vec3 x = color * 12.92f;
    vec3 y = 1.055f * pow(clamp(color,0.0,1.0), vec3(1.0f / 2.4f)) - 0.055f;

    vec3 clr = color;
    clr.r = color.r < 0.0031308f ? x.r : y.r;
    clr.g = color.g < 0.0031308f ? x.g : y.g;
    clr.b = color.b < 0.0031308f ? x.b : y.b;

    return clr;
}

int inControl(vec2 screenCoord, float screenWidth) {
    if (screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}

vec4 getNotControl(sampler2D inSampler, vec2 coords, bool inctrl) {
    if (inctrl) {
        return (texture(inSampler, coords - vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(0.0, oneTexel.y))) / 3.0;
    } else {
        return texture(inSampler, coords);
    }
}



//Color//
  #define LIGHT_MI 1.00 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]
  #define AMBIENT_MI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]

  #define LIGHT_DI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]
  #define AMBIENT_DI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]

  #define LIGHT_EI 1.00 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]
  #define AMBIENT_EI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]

  #define LIGHT_NI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]
  #define AMBIENT_NI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 2.55 2.60 2.65 2.70 2.75 2.80 2.85 2.90 2.95 3.00 3.05 3.10 3.15 3.20 3.25 3.30 3.35 3.40 3.45 3.50 3.55 3.60 3.65 3.70 3.75 3.80 3.85 3.90 3.95 4.00]

  #define AMBIENT_AI 1.0 //[0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00]


float LinearizeDepth(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}

float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

vec4 backProject(vec4 vec) {
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
}



#define PI 3.141592
#define iSteps 16
#define jSteps 8

vec2 rsi(vec3 r0, vec3 rd, float sr) {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    float d = (b*b) - 4.0*a*c;
    if (d < 0.0) return vec2(1e5,-1e5);
    return vec2(
        (-b - sqrt(d))/(2.0*a),
        (-b + sqrt(d))/(2.0*a)
    );
}
vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}
vec3 atmosphere(vec3 r, vec3 r0, vec3 pSun, float iSun, float rPlanet, float rAtmos, vec3 kRlh, float kMie, float shRlh, float shMie, float g) {
    // Normalize the sun and view directions.
    pSun = normalize(pSun);
    r = normalize(r);

    // Calculate the step size of the primary ray.
    vec2 p = rsi(r0, r, rAtmos);
    if (p.x > p.y) return vec3(0,0,0);
    p.y = min(p.y, rsi(r0, r, rPlanet).x);
    float iStepSize = (p.y - p.x) / float(iSteps);

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    vec3 totalRlh = vec3(0,0,0);
    vec3 totalMie = vec3(0,0,0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;
    float pRlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
    float pMie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {

        // Calculate the primary ray sample position.
        vec3 iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        float iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        // Calculate the step size of the secondary ray.
        float jStepSize = rsi(iPos, pSun, rAtmos).y / float(jSteps);

        // Initialize the secondary ray time.
        float jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        // Sample the secondary ray.
        for (int j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            vec3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            float jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh += exp(-jHeight / shRlh) * jStepSize;
            jOdMie += exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime += jStepSize;
        }

        // Calculate attenuation.
        vec3 attn = exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)));

        // Accumulate scattering.
        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;

        // Increment the primary ray time.
        iTime += iStepSize;

    }

    // Calculate and return the final color.
    return iSun * (pRlh * kRlh * totalRlh + pMie * kMie * totalMie);
}

vec2 tapLocation2(int sampleNumber, float spinAngle,int nb, float nbRot,float r0)
{
    float alpha = (float(sampleNumber*1.0f + r0) * (1.0 / (nb)));
    float angle = alpha * (nbRot * 6.28) + spinAngle*6.28;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}

float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y);
}
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}




float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}



float GetLinearDepth(float depth) {
   return (2.0 * near) / (far + near - depth * (far - near));
}

vec2 OffsetDist(float x, int s) {
	float n = fract(x * 1.414) * 3.1415;
    return vec2(cos(n), sin(n)) * x / s;
}

vec2 OffsetDist(float x) {
	float n = fract(x * 8.0) * 3.1415;
    return vec2(cos(n), sin(n)) * x;
}
float AmbientOcclusion(sampler2D depth, vec2 coord, float dither) {
	float ao = 0.0;
	float aspectRatio = ScreenSize.x/ScreenSize.y;


		int samples = 6;


	float d = texture(depth, coord).r;
	if(d >= 1.0) return 1.0;
	float hand = float(d < 0.56);
	d = GetLinearDepth(d);
	
	float sampleDepth = 0.0, angle = 0.0, dist = 0.0;
	float fovScale = gbufferModelViewInverse[1][1] / 1.37;
	float distScale = max((far - near) * d + near, 6.0);
	vec2 scale = 0.35 * vec2(1.0 / aspectRatio, 1.0) * fovScale / distScale;
	scale *= vec2(0.5, 1.0);

	for(int i = 1; i <= samples; i++) {
		vec2 offset = OffsetDist(i + dither, samples) * scale;

		sampleDepth = GetLinearDepth(texture(depth, coord + offset).r);
		float sample = (far - near) * (d - sampleDepth) * 2.0;
		if (hand > 0.5) sample *= 1024.0;
		angle = clamp(0.5 - sample, 0.0, 1.0);
		dist = clamp(0.5 * sample - 1.0, 0.0, 1.0);

		sampleDepth = GetLinearDepth(texture(depth, coord - offset).r);
		sample = (far - near) * (d - sampleDepth) * 2.0;
		if (hand > 0.5) sample *= 1024.0;
		angle += clamp(0.5 - sample, 0.0, 1.0);
		dist += clamp(0.5 * sample - 1.0, 0.0, 1.0);
		
		ao += clamp(angle + dist, 0.0, 1.0);
	}
	ao /= samples;
	
	return ao;
}
  #define SATURATION 1.50 //[0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00]
  #define VIBRANCE 1.50 //[0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00]
vec3 colorSaturation(vec3 color){
	float grayVibrance = (color.r + color.g + color.b) / 3.0;
	float graySaturation = grayVibrance;
	if (SATURATION < 1.00) graySaturation = dot(color, vec3(0.299, 0.587, 0.114));

	float mn = min(color.r, min(color.g, color.b));
	float mx = max(color.r, max(color.g, color.b));
	float sat = (1.0 - (mx - mn)) * (1.0 - mx) * grayVibrance * 5.0;
	vec3 lightness = vec3((mn + mx) * 0.5);

	color = mix(color, mix(color, lightness, 1.0 - VIBRANCE), sat);
	color = mix(color, lightness, (1.0 - lightness) * (2.0 - VIBRANCE) / 2.0 * abs(VIBRANCE - 1.0));
	color = color * SATURATION - graySaturation * (SATURATION - 1.0);

	return color;
}


float GGX (vec3 n, vec3 v, vec3 l, float r, float F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}

float getRoundFragDepth(sampler2D depthTex, vec2 texcoord)
{
	vec3 screenPos = vec3(texcoord, texture(depthTex, texcoord).r);
	vec3 clipPos = screenPos * 2.0 - 1.0;
	vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
	vec3 viewPos = tmp.xyz / tmp.w;
	return length(viewPos);
}
#define FOG_END far/5 // How far away the fog should end. [32 64 128 far]
#define FOG_NEAR 0 // How far away the fog should start. [0 2 4 8 16 32 64]
 vec3 doFog(float depth, vec3 color,vec3 customFogColor)
{
	float viewPos = getRoundFragDepth(DiffuseDepthSampler, texCoord);

	float fogNearValue;
	float fogFarValue;


	{

		fogNearValue = FOG_NEAR;
		fogFarValue = FOG_END;
		color = mix(color, customFogColor, clamp(((length(viewPos)-fogNearValue)/fogFarValue), 0.0, 1.0));

	return color;
} 
} 

uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentSpecSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D ParticlesSampler;
uniform sampler2D ParticlesDepthSampler;
uniform sampler2D PartialCompositeSampler;
uniform sampler2D ItemEntityDepthSampler;
uniform sampler2D WeatherDepthSampler;
uniform sampler2D CloudsDepthSampler;



#define TONEMAP_TOE_STRENGTH    0 // [-1 -0.99 -0.98 -0.97 -0.96 -0.95 -0.94 -0.93 -0.92 -0.91 -0.9 -0.89 -0.88 -0.87 -0.86 -0.85 -0.84 -0.83 -0.82 -0.81 -0.8 -0.79 -0.78 -0.77 -0.76 -0.75 -0.74 -0.73 -0.72 -0.71 -0.7 -0.69 -0.68 -0.67 -0.66 -0.64 -0.63 -0.62 -0.61 -0.6 -0.59 -0.58 -0.57 -0.56 -0.55 -0.54 -0.53 -0.52 -0.51 -0.5 -0.49 -0.48 -0.47 -0.46 -0.45 -0.44 -0.43 -0.42 -0.41 -0.4 -0.39 -0.38 -0.37 -0.36 -0.35 -0.34 -0.33 -0.32 -0.31 -0.3 -0.29 -0.28 -0.27 -0.26 -0.25 -0.24 -0.23 -0.22 -0.21 -0.2 -0.19 -0.18 -0.17 -0.16 -0.15 -0.14 -0.13 -0.12 -0.11 -0.1 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_TOE_LENGTH      0 // [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_LINEAR_SLOPE    1   // Should usually be left at 1
#define TONEMAP_LINEAR_LENGTH   0.5 // [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_SHOULDER_CURVE  0.6 // [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_SHOULDER_LENGTH 1   // Not currently in an actually useful state

#define WHITE_BALANCE 7200 // [2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000]

#define CONTRAST -0.3 // [-1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
#define CONTRAST_MIDPOINT 0.14


#if !defined INCLUDE_UTILITY_COLOR
#define INCLUDE_UTILITY_COLOR

//--// Conversion matrices //-------------------------------------------------//

/*
Rec. 709 / sRGB

Wxy = 0.3127x + 0.329y
Rxy = 0.64x   + 0.33y
Gxy = 0.3x    + 0.6y
Bxy = 0.15x   + 0.06y
*/
const mat3 tmp_r709 = mat3(
	0.64 / 0.33, 0.3 / 0.6, 0.15 / 0.06,
	1.0,         1.0,       1.0,
	0.03 / 0.33, 0.1 / 0.6, 0.79 / 0.06
);
const vec3 lc_r709 = vec3(0.3127 / 0.329, 1.0, 0.3583 / 0.329) * inverse(tmp_r709);

const mat3 R709ToXyz = mat3(lc_r709 * tmp_r709[0], lc_r709, lc_r709 * tmp_r709[2]);
const mat3 XyzToR709 = inverse(R709ToXyz);

/*
Rec. 2020 / Rec. 2100

Wxy = 0.3127x + 0.329y
Rxy = 0.708x  + 0.292y
Gxy = 0.17x   + 0.797y
Bxy = 0.131x  + 0.046y
*/
const mat3 tmp_r2020 = mat3(
	0.708 / 0.292, 0.17  / 0.797, 0.131 / 0.046,
	1.0,           1.0,           1.0,
	0.0   / 0.292, 0.033 / 0.797, 0.823 / 0.046
);
const vec3 lc_r2020 = vec3(0.3127 / 0.3290, 1.0, 0.3583 / 0.3290) * inverse(tmp_r2020);

const mat3 R2020ToXyz = mat3(lc_r2020 * tmp_r2020[0], lc_r2020, lc_r2020 * tmp_r2020[2]);
const mat3 XyzToR2020 = inverse(R2020ToXyz);

//--// Set up working color space

#define USE_R2020
#if defined USE_R2020
const mat3 XyzToRgb = XyzToR2020;
const mat3 RgbToXyz = R2020ToXyz;
#else // R709
const mat3 XyzToRgb = XyzToR709;
const mat3 RgbToXyz = R709ToXyz;
#endif

const mat3 R709ToRgb = R709ToXyz * XyzToRgb;
const mat3 RgbToR709 = RgbToXyz * XyzToR709;

// Variant that divides out the old white point
// Needed to correctly convert fractions of reflected/transmitted light (i.e. the albedo of a surface, transmittance through an atmosphere, and other things where you multiply by the illuminant)
const mat3 R709ToRgb_unlit = mat3(
	R709ToRgb[0] / (R709ToRgb[0].x + R709ToRgb[0].y + R709ToRgb[0].z),
	R709ToRgb[1] / (R709ToRgb[1].x + R709ToRgb[1].y + R709ToRgb[1].z),
	R709ToRgb[2] / (R709ToRgb[2].x + R709ToRgb[2].y + R709ToRgb[2].z)
);

//----------------------------------------------------------------------------//

vec3 RgbToYcocg(vec3 rgb) {
	const mat3 mat = mat3(
		 0.25, 0.5, 0.25,
		 0.5,  0.0,-0.5,
		-0.25, 0.5,-0.25
	);
	return rgb * mat;
}
vec3 YcocgToRgb(vec3 ycocg) {
	float tmp = ycocg.x - ycocg.z;
	return vec3(tmp + ycocg.y, ycocg.x + ycocg.z, tmp - ycocg.y);
}

float LinearToSrgb(float color) {
	return mix(1.055 * pow(color, 1.0 / 2.4) - 0.055, color * 12.92, step(color, 0.0031308));
}
vec3 LinearToSrgb(vec3 color) {
	return mix(1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, color * 12.92, step(color, vec3(0.0031308)));
}
float SrgbToLinear(float color) {
	return mix(pow(color / 1.055 + (0.055 / 1.055), 2.4), color / 12.92, step(color, 0.04045));
}
vec3 SrgbToLinear(vec3 color) {
	return mix(pow(color / 1.055 + (0.055 / 1.055), vec3(2.4)), color / 12.92, step(color, vec3(0.04045)));
}

vec3 Blackbody(float temperature) {
	// https://en.wikipedia.org/wiki/Planckian_locus
	const vec4[2] xc = vec4[2](
		vec4(-0.2661293e9,-0.2343589e6, 0.8776956e3, 0.179910), // 1667k <= t <= 4000k
		vec4(-3.0258469e9, 2.1070479e6, 0.2226347e3, 0.240390)  // 4000k <= t <= 25000k
	);
	const vec4[3] yc = vec4[3](
		vec4(-1.1063814,-1.34811020, 2.18555832,-0.20219683), // 1667k <= t <= 2222k
		vec4(-0.9549476,-1.37418593, 2.09137015,-0.16748867), // 2222k <= t <= 4000k
		vec4( 3.0817580,-5.87338670, 3.75112997,-0.37001483)  // 4000k <= t <= 25000k
	);

	float temperatureSquared = temperature * temperature;
	vec4 t = vec4(temperatureSquared * temperature, temperatureSquared, temperature, 1.0);

	float x = dot(1.0 / t, temperature < 4000.0 ? xc[0] : xc[1]);
	float xSquared = x * x;
	vec4 xVals = vec4(xSquared * x, xSquared, x, 1.0);

	vec3 xyz = vec3(0.0);
	xyz.y = 1.0;
	xyz.z = 1.0 / dot(xVals, temperature < 2222.0 ? yc[0] : temperature < 4000.0 ? yc[1] : yc[2]);
	xyz.x = x * xyz.z;
	xyz.z = xyz.z - xyz.x - 1.0;

	return xyz * XyzToRgb;
}
vec3 rodSample(vec2 Xi)
{
	float r = sqrt(1.0f - Xi.x*Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
#endif
vec3 coneSample(vec2 Xi)
{
	float r = sqrt(1.0f - Xi.x*Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
vec3 cosineHemisphereSample(vec2 Xi)
{
    float r = sqrt(Xi.x);
    float theta = 2.0 * 3.14159265359 * Xi.y;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec3(x, y, sqrt(clamp(1.0 - Xi.x,0.,1.)));
}
#define Clamp01(x) clamp(x, 0, 1)
#define Max0(x) max(x, 0)

float MinOf(vec2 x) { return min(x.x, x.y); }
float MinOf(vec3 x) { return min(min(x.x, x.y), x.z); }
float MaxOf(vec2 x) { return max(x.x, x.y); }
float MaxOf(vec3 x) { return max(max(x.x, x.y), x.z); }

	mat3 ChromaticAdaptationMatrix(vec3 sourceXYZ, vec3 destinationXYZ) {
		const mat3 XyzToLms = mat3(
			 0.7328, 0.4296,-0.1624,
			-0.7036, 1.6975, 0.0061,
			 0.0030, 0.0136, 0.9834
		); // CAT02

		vec3 sourceLMS = sourceXYZ * XyzToLms;
		vec3 destinationLMS = destinationXYZ * XyzToLms;

		vec3 tmp = destinationLMS / sourceLMS;

		mat3 vonKries = mat3(
			tmp.x, 0.0, 0.0,
			0.0, tmp.y, 0.0,
			0.0, 0.0, tmp.z
		);

		return (XyzToLms * vonKries) * inverse(XyzToLms);
	}
	vec3 WhiteBalance(vec3 color) {
		vec3 sourceXYZ = Blackbody(WHITE_BALANCE) * RgbToXyz;
		vec3 destinationXYZ = Blackbody(6500.0) * RgbToXyz;
		mat3 matrix = RgbToXyz * ChromaticAdaptationMatrix(sourceXYZ, destinationXYZ) * XyzToRgb;

		return color * matrix;
	}

	vec3 Contrast(vec3 color) {
		float luminance = dot(color, RgbToXyz[1]);
		float newLuminance = CONTRAST_MIDPOINT * pow(luminance / CONTRAST_MIDPOINT, exp2(CONTRAST));
		return color * Max0(newLuminance / luminance);
	}
	vec3 Saturation(vec3 color) {
		float luminance = dot(color, RgbToXyz[1]);
		float minComp = MinOf(color), maxComp = MaxOf(color);

		// compute the desired output saturation
		//float originalSaturation = maxComp == 0.0 ? 0.0 : Clamp01(1.0 - minComp / maxComp);
		float newSaturation = maxComp == 0.0 ? 0.0 : Clamp01(1.0 - pow(minComp / maxComp, SATURATION));

		// compute fully saturated version of the color (if it exits)
		vec3 saturatedColor = (maxComp - minComp) == 0.0 ? vec3(maxComp) : (color - minComp) / (maxComp - minComp);

		// compute new color from saturated & non-saturated color
		color  = mix(vec3(1.0), saturatedColor, newSaturation);
		color *= luminance / dot(color, RgbToXyz[1]);

		return color;
	}

	vec3 Tonemap(vec3 color) {
		const float toeStrength    = TONEMAP_TOE_STRENGTH;
		const float toeLength      = TONEMAP_TOE_LENGTH * TONEMAP_TOE_LENGTH / 2;
		const float linearSlope    = TONEMAP_LINEAR_SLOPE;
		const float linearLength   = TONEMAP_LINEAR_LENGTH;
		const float shoulderCurve  = TONEMAP_SHOULDER_CURVE;
		const float shoulderLength = TONEMAP_SHOULDER_LENGTH;

		const float toeX     = toeLength;
		const float toeY     = linearSlope * toeLength * (1.0 - toeStrength);
		const float toePower = 1.0 / (1.0 - toeStrength);

		const float tm = toeY * pow(1.0 / toeX, toePower);

		const float lm = linearSlope;
		const float la = toeStrength == 1.0 ? -linearSlope * toeX : toeY - toeY * toePower;

		const float shoulderX = linearLength * (1.0 - toeY) / linearSlope + toeX;
		const float shoulderY = linearLength * (1.0 - toeY) + toeY;

		const float sim = linearSlope * shoulderLength / (1.0 - shoulderY);
		const float sia = -sim * shoulderX;
		const float som = (1.0 - shoulderY) / shoulderLength;
		const float soa = shoulderY;

		for (int i = 0; i < 3; ++i) {
			if (color[i] < toeX) {
				color[i] = tm * pow(color[i], toePower);
			} else if (color[i] < shoulderX) {
				color[i] = lm * color[i] + la;
			} else {
				color[i]  = sim * color[i] + sia;
				color[i] /= pow(pow(color[i], 1.0 / shoulderCurve) + 1.0, shoulderCurve);
				color[i]  = som * color[i] + soa;
			}
		}

		return color;
	}
    float A = 0.2;
float B = 0.25;
float C = 0.10;
float D = 0.35;
float E = 0.02;
float F = 0.3;

vec3 Tonemap_Filmic_UC2(vec3 linearColor, float linearWhite,
	float A, float B, float C, float D, float E, float F) {

	// Uncharted II configurable tonemapper.

	// A = shoulder strength
	// B = linear strength
	// C = linear angle
	// D = toe strength
	// E = toe numerator
	// F = toe denominator
	// Note: E / F = toe angle
	// linearWhite = linear white point value

	vec3 x = linearColor;
	vec3 color = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

	x = vec3(linearWhite);
	vec3 white = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

	return color / white;
}
vec3 Tonemap_Filmic_UC2Default(vec3 linearColor) {

	// Uncharted II fixed tonemapping formula.
	// Gives a warm and gritty image, saturated shadows and bleached highlights.

	return pow(Tonemap_Filmic_UC2(linearColor*4., 11.2, 0.22, 0.3, 0.1, 0.4, 0.025, 0.30),vec3(1.0/2.232));
}
vec3 reinhard(vec3 x){
x *= 1.66;
return pow(x/(1.0+x),vec3(1.0/2.2));
}

vec3 whitePreservingLumaBasedReinhardToneMapping(vec3 color)
{
	float white = 2.;
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma * (1. + luma / (white*white)) / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.0));
	return color;
}
vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.2));
	return color;
}

vec3 ToneMap_Hejl2015(in vec3 hdr)
{
    vec4 vh = vec4(hdr*0.85, 3.0);	//0
    vec4 va = (1.75 * vh) + 0.05;	//0.05
    vec4 vf = ((vh * va + 0.004f) / ((vh * (va + 0.55f) + 0.0491f))) - 0.0821f+0.000633604888;	//((0+0.004)/((0*(0.05+0.55)+0.0491)))-0.0821
    return vf.xyz / vf.www;
}

// with improvments from Bobcao3
vec2 invWidthHeight = vec2(1.0 / ScreenSize.x, 1.0 / ScreenSize.y);



vec3 getSkyColorLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	return texture(lut,vec2(x,y)).rgb;
}


vec3 drawSun(float cosY, float sunInt,vec3 nsunlight,vec3 inColor){
	return inColor+nsunlight/0.0008821203*pow(smoothstep(cos(0.0093084168595*3.2),cos(0.0093084168595*1.8),cosY),3.)*0.62;
}

// Return random noise in the range [0.0, 1.0], as a function of x.
float hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
// Convert Noise2d() into a "star field" by stomping everthing below fThreshhold to zero.
float NoisyStarField( in vec2 vSamplePos, float fThreshhold )
{
    float StarVal = hash12( vSamplePos );
        StarVal = clamp(StarVal/(1.0 - fThreshhold) - fThreshhold/(1.0 - fThreshhold),0.0,1.0);

    return StarVal;
}

// Stabilize NoisyStarField() by only sampling at integer values.
float StableStarField( in vec2 vSamplePos, float fThreshhold )
{
    // Linear interpolation between four samples.
    // Note: This approach has some visual artifacts.
    // There must be a better way to "anti alias" the star field.
    float fractX = fract( vSamplePos.x );
    float fractY = fract( vSamplePos.y );
    vec2 floorSample = floor( vSamplePos );
    float v1 = NoisyStarField( floorSample, fThreshhold );
    float v2 = NoisyStarField( floorSample + vec2( 0.0, 1.0 ), fThreshhold );
    float v3 = NoisyStarField( floorSample + vec2( 1.0, 0.0 ), fThreshhold );
    float v4 = NoisyStarField( floorSample + vec2( 1.0, 1.0 ), fThreshhold );

    float StarVal =   v1 * ( 1.0 - fractX ) * ( 1.0 - fractY )
        			+ v2 * ( 1.0 - fractX ) * fractY
        			+ v3 * fractX * ( 1.0 - fractY )
        			+ v4 * fractX * fractY;
	return StarVal;
}

float stars(vec3 fragpos){

	float elevation = clamp(fragpos.y,0.,1.);
	vec2 uv = fragpos.xz/(1.+elevation);

	return StableStarField(uv*700.,0.999)*0.5*(0.3-0.3*0);
}

#define ffstep(x,y) clamp((y - x) * 1e35,0.0,1.0)

const float pi = 3.141592653589793238462643383279502884197169;

vec2 sphereToCarte(vec3 dir) {
    float lonlat = atan(-dir.x, -dir.z);
    return vec2(lonlat * (0.5/pi) +0.5,0.5*dir.y+0.5);
}

vec3 skyFromTex(vec3 pos,sampler2D sampler){
	vec2 p = sphereToCarte(pos);
	return texture(sampler,p*oneTexel*256.+vec2(18.5,1.5)*oneTexel).rgb;
}
float noise(vec2 uv)
{
    return fract(sin(uv.x * 113. + uv.y * 412.) * 6339.);
}

vec3 noiseSmooth(vec2 uv)
{
    vec2 index = floor(uv);
    
    vec2 pq = fract(uv);
    pq = smoothstep(0., 1., pq);
     
    float topLeft = noise(index);
    float topRight = noise(index + vec2(1, 0.));
    float top = mix(topLeft, topRight, pq.x);
    
    float bottomLeft = noise(index + vec2(0, 1));
    float bottomRight = noise(index + vec2(1, 1));
    float bottom = mix(bottomLeft, bottomRight, pq.x);
    
    return vec3(mix(top, bottom, pq.y));
}



// random1o2i
float noise2D(vec2 p) {
    return fract(sin(dot(p,vec2(127.1,311.7))) * 43758.5453);
}

float interpNoise2D(vec2 uv) {
    float intX = floor(uv.x);
    float fractX = fract(uv.x);
    float intY = floor(uv.y);
    float fractY = fract(uv.y);

    float v1 = noise2D(vec2(intX, intY));
    float v2 = noise2D(vec2(intX + 1.0, intY));
    float v3 = noise2D(vec2(intX, intY + 1.0));
    float v4 = noise2D(vec2(intX + 1.0, intY + 1.0));

    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);
    
    return mix(i1, i2, fractY);
}

float fbm2D(vec2 uv) {
    float total = 0.0;
    float persistence = 0.45;
    int octaves = 8;

    for(int i = 1; i <= octaves; i++) {
        float freq = pow(2.f, float(i));
        float amp = pow(persistence, float(i));

        total += interpNoise2D(uv * freq) * amp;
    }
    return total;
}

vec3 rotateY(vec3 p, float a) {
    return vec3(cos(a) * p.x + sin(a) * p.z, p.y, -sin(a) * p.x + cos(a) * p.z);
}



const vec3 up = vec3(0.0, 1.0, 0.0);

// https://www.color-hex.com/color/87cefa
const vec3 horizonColor = vec3(135.0/255.0, 206.0/255.0, 250.0/255.0);
// https://www.color-hex.com/color/1874cd
const vec3 skyColor = vec3(24.0/255.0, 116.0/255.0, 205.0/255.0);
const vec3 cloudColor = vec3(1.0);
const float cloudPlaneHeight = 10.0;

void Clouds(vec3 dir, out vec3 color) {
    	vec3 sc = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb;

    vec3 cloudPlane = dir*cloudPlaneHeight/dot(dir, up);
    vec2 uv = cloudPlane.xz + (GameTime*10);
	    uv *= 1.0;
    
    float f = fbm2D(uv*0.5) ;
    
    // whipping up the clouds a little so they would not look too much like generic fbm
	vec2 _uv = uv*0.5;
    mat2 rm  = mat2 (vec2(-sin(f+_uv.y), cos(f+_uv.x)), vec2(cos(f+_uv.y), sin(f+_uv.x)));  
    uv += .005*uv*rm;
    float clouds = fbm2D(uv * .05);
    clouds = clamp((clouds - 0.45) * 2.0, 0.0, 1.0);
    color = mix(color, sc.rgb, clouds);
	color = mix(color*0.25,vec3(0.0),clamp(length(abs(cloudPlane)*0.02),0,1));
}



//Mie phase function
float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 /3.14 + 0.25 /3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}
float packUnorm2x4(vec2 xy) {
	return dot(floor(15.0 * xy + 0.5), vec2(1.0 / 255.0, 16.0 / 255.0));
}
float packUnorm2x4(float x, float y) { return packUnorm2x4(vec2(x, y)); }
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

vec3 Uncharted2ToneMapping(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	float exposure = 2.;
	color *= exposure;
	color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
	float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
	color /= white;
	color = pow(color, vec3(1. / 2.2));
	return color;
}



float width = ScreenSize.x; //texture width
float height = ScreenSize.y; //texture height

vec2 inverse_buffer_size = vec2(1.0/width, 1.0/height);

float rgb2lumi(vec4 color) {
    return dot(color.rgb, vec3(0.212, 0.716, 0.072));
    //return color.g;
}

vec2 gradientAt(sampler2D tex, vec2 pos, vec2 sampling)
{   
    vec2 stepx = vec2(sampling.x, 0.0);
    vec2 stepy = vec2(0.0, sampling.y);
    
    /* Explicit Sobel
    float left = rgb2lumi( texture(texture, pos-stepx) ) * 2.0
               + rgb2lumi( texture(texture, pos-stepx-stepy) )
               + rgb2lumi( texture(texture, pos-stepx+stepy) );
    float right = rgb2lumi( texture(texture, pos+stepx) ) * 2.0
                + rgb2lumi( texture(texture, pos+stepx-stepy) )
                + rgb2lumi( texture(texture, pos+stepx+stepy) );
    
    float top = rgb2lumi( texture(texture, pos-stepy) ) * 2.0
              + rgb2lumi( texture(texture, pos-stepy-stepx) )
              + rgb2lumi( texture(texture, pos-stepy+stepx) );
    float bottom = rgb2lumi( texture(texture, pos+stepy) ) * 2.0
                 + rgb2lumi( texture(texture, pos+stepy-stepx) )
                 + rgb2lumi( texture(texture, pos+stepy+stepx) );
    */
    
    // 3x3 derivative kernel operator making use of bilinear interpolation.
    // The sobel operator is not rotationally invariant at all. Scharr is 
    // better. Kroon did some work in finding an optimal kernel as well.
    // The center pixel has an effective coefficient of (1-alpha)*2
    // The outer pixels have an effective coefficient of alpha
    // To calculate alpha from the relation between the two:
    // alpha = 2 / (2+R)
    // Sobel (R=2):         alpha = 0.5
    // Scharr (R=10/3):     alpha = 0.375
    // Kroon (R=61/17):     alpha = 0.358
   /* 
    float alpha = 0.358;
    float left = rgb2lumi( texture(DiffuseSampler, pos-stepx - alpha*stepy) )
               + rgb2lumi( texture(DiffuseSampler, pos-stepx + alpha*stepy) );
    float right = rgb2lumi( texture(DiffuseSampler, pos+stepx - alpha*stepy) )
                + rgb2lumi( texture(DiffuseSampler, pos+stepx + alpha*stepy) );
    
    float top = rgb2lumi( texture(DiffuseSampler, pos-stepy - alpha*stepx) ) 
              + rgb2lumi( texture(DiffuseSampler, pos-stepy + alpha*stepx) );
    float bottom = rgb2lumi( texture(DiffuseSampler, pos+stepy - alpha*stepx) ) 
                 + rgb2lumi( texture(DiffuseSampler, pos+stepy + alpha*stepx) );

   */
    // 5x5 derivative kernel operator making use of bilinear interpolation.
    // Took me a while to calculate these coefficients. I hope I did not make 
    // a mistake. In the end, it turned out not too work better than the 3x3
    // kernel. The diffusion should be in the gradient field, not the original
    // albeido image.
    
    float a1 = 1.12;  // 0.12; 0.04
    float a2 = 1.06;  // 0.06; 0.03
    float a3 = 1.11157;  // 0.11157; 0.125  relation between two center elements
    float a4 = 2.3472;  // 2.3472; 2.3063  relation between center and corner
    float left = rgb2lumi( texture(DiffuseSampler, pos - a3*stepx) ) * a4
               + rgb2lumi( texture(DiffuseSampler, pos - a1*stepx - a2*stepy) )
               + rgb2lumi( texture(DiffuseSampler, pos - a1*stepx + a2*stepy) );
    float right = rgb2lumi( texture(DiffuseSampler, pos + a3*stepx) ) * a4
                + rgb2lumi( texture(DiffuseSampler, pos + a1*stepx - a2*stepy) )
                + rgb2lumi( texture(DiffuseSampler, pos + a1*stepx + a2*stepy) );  
    float top = rgb2lumi( texture(DiffuseSampler, pos - a3*stepy) ) * a4
                 + rgb2lumi( texture(DiffuseSampler, pos - a1*stepy - a2*stepx) )
                 + rgb2lumi( texture(DiffuseSampler, pos - a1*stepy + a2*stepx) );
    float bottom = rgb2lumi( texture(DiffuseSampler, pos + a3*stepy) ) * a4
                 + rgb2lumi( texture(DiffuseSampler, pos + a1*stepy - a2*stepx) )
                 + rgb2lumi( texture(DiffuseSampler, pos + a1*stepy + a2*stepx) );
  
    // Return gradient
    //return vec2(right-left, bottom-top) / (a4+2.0);
    return vec2(right-left, bottom-top) * 0.5;
}


void main() {
float rainStrength = 1-rain.r;
    bool inctrl = inControl(texCoord * OutSize, OutSize.x) > -1;
    float aspectRatio = ScreenSize.x/ScreenSize.y;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);

    float ao = AmbientOcclusion(TranslucentDepthSampler,texCoord,ditherGradNoise()) ;

    float diffuseDepth = texture(DiffuseDepthSampler, texCoord).r;
    vec3 OutTexel = texture(DiffuseSampler, texCoord).rgb;


    // Get centre location
    vec2 pos = texCoord;
    
    // Init value
    vec4 color1 = vec4(0.0, 0.0, 0.0, 0.0); 
    vec4 color2; // to set color later
    
    // Init kernel and number of steps
    //vec4 kernel = vec4(0.399, 0.242, 0.054, 0.004); // Gaussian sigma 1.0
    //vec4 kernel = vec4(0.53, 0.22, 0.015, 0.00018); // Gaussian sigma 0.75
    vec4 kernel = vec4(0.79, 0.11, 0.0026, 0.000001); // Gaussian sigma 0.5
    int sze = 2; 
    
    // Init step size in tex coords
    float dx = 1.0/ScreenSize.x;
    float dy = 1.0/ScreenSize.y;
    
    // Convolve
    for (int y=-sze; y<sze+1; y++)
    {
        for (int x=-sze; x<sze+1; x++)
        {   
            float k = kernel[int(abs(float(x)))] * kernel[int(abs(float(y)))];
            vec2 dpos = vec2(float(x)*dx, float(y)*dy);
            color1 += texture(DiffuseSampler, pos+dpos) * k;
        }
    }


    fragColor = color1;

}
