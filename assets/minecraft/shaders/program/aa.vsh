#version 150

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
out vec2 texCoord;
out vec2 texCoord2;
out vec2 oneTexel;
out vec3 sunDir;
out vec4 fogcol;
out vec4 fogColor;
out vec4 rain;
out float GameTime;
out vec4 skycol;
out mat4 gbufferModelViewInverse;
out mat4 gbufferModelView;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out float near;
out float far;
out float cosFOVrad;
out float tanFOVrad;
uniform vec2 InSize;
uniform float FOV;
out float aspectRatio;
out mat4 gbPI;
out mat4 gbP;
out vec4 fogColor2;
out vec3 flareColor;
out vec3 sunColor;

uniform float Time;

vec3 rodSample(vec2 Xi)
{
	float r = sqrt(1.0f - Xi.x*Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
//Low discrepancy 2D sequence, integration error is as low as sobol but easier to compute : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}
// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05

vec2 getControl(int index, vec2 screenSize) {
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}

int intmod(int i, int base) {
    return i - (i / base * base);
}

vec3 encodeInt(int i) {
    int s = int(i < 0) * 128;
    i = abs(i);
    int r = intmod(i, 256);
    i = i / 256;
    int g = intmod(i, 256);
    i = i / 256;
    int b = intmod(i, 128);
    return vec3(float(r) / 255.0, float(g) / 255.0, float(b + s) / 255.0);
}

int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

vec3 encodeFloat(float i) {
    return encodeInt(int(i * FPRECISION));
}

float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}
vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter)
{
    float alpha = float(sampleNumber+jitter)/nb;
    float angle = (jitter+alpha) * (nbRot * 6.28);

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}


const float pi = 3.141592653589793238462643383279502884197169;

vec2 sphereToCarte(vec3 dir) {
    float lonlat = atan(-dir.x, -dir.z);
    return vec2(lonlat * (0.5/pi) +0.5,0.5*dir.y+0.5);
}

vec3 skyFromTex(vec3 pos,sampler2D sampler){
	vec2 p = sphereToCarte(pos);
	return texture(sampler,p*oneTexel*256.+vec2(18.5,1.5)*oneTexel).rgb;
}
void main() {



    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    texCoord = Position.xy / OutSize;
    oneTexel = 1.0 / OutSize;

    //simply decoding all the control data and constructing the sunDir, ProjMat, ModelViewMat

    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);


    // ProjMat constructed assuming no translation or rotation matrices applied (aka no view bobbing).
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), 0.0, 0.0,
                        decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz),  decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0);

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0,
                            0.0, 0.0, 0.0, 1.0);
    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));
    skycol = vec4((texture(DiffuseSampler, start + 26.0 * inc)));
    GameTime = vec4((texture(DiffuseSampler, start + 27.0 * inc))).r;
    rain = vec4((texture(DiffuseSampler, start + 28.0 * inc)));
    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + inc).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz),
                                                    1.0)).xyz);
    
    gbufferModelViewInverse = inverse(ProjMat * ModeViewMat);
    gbufferModelView = (ProjMat * ModeViewMat);
    gbufferProjection = ProjMat;
    gbufferProjectionInverse = inverse(ProjMat);
    aspectRatio = InSize.x / InSize.y;

    float FOVrad = FOV / 360.0 * 3.1415926535;
    cosFOVrad = cos(FOVrad);
    tanFOVrad = tan(FOVrad);
    gbPI = mat4(2.0 * tanFOVrad * aspectRatio, 0.0,             0.0, 0.0,
                0.0,                           2.0 * tanFOVrad, 0.0, 0.0,
                0.0,                           0.0,             0.0, 0.0,
                -tanFOVrad * aspectRatio,     -tanFOVrad,       1.0, 1.0);

    gbP = mat4(1.0 / (2.0 * tanFOVrad * aspectRatio), 0.0,               0.0, 0.0,
               0.0,                             1.0 / (2.0 * tanFOVrad), 0.0, 0.0,
               0.5,                             0.5,                     1.0, 0.0,
               0.0,                             0.0,                     0.0, 1.0);
  float rainStrength = 0;

  vec3   upPosition =  vec3(0,1,0);
  vec3 sunPosition = sunDir;


float normSunVec = sqrt(sunPosition.x*sunPosition.x+sunPosition.y*sunPosition.y+sunPosition.z*sunPosition.z);
float normUpVec = sqrt(upPosition.x*upPosition.x+upPosition.y*upPosition.y+upPosition.z*upPosition.z);

float sunPosX = sunPosition.x/normSunVec;
float sunPosY = sunPosition.y/normSunVec;
float sunPosZ = sunPosition.z/normSunVec;

vec3 sunVec=vec3(sunPosX,sunPosY,sunPosZ);

float upPosX = upPosition.x/normUpVec;
float upPosY = upPosition.y/normUpVec;
float upPosZ = upPosition.z/normUpVec;

vec3 upVec=vec3(upPosX,upPosY,upPosZ);
float sunElevation = sunPosX*upPosX+sunPosY*upPosY+sunPosZ*upPosZ;

float angSun= -(( pi * 0.5128205128205128 - acos(sunElevation*1.065-0.065))/1.5);
float angMoon= -(( pi * 0.5128205128205128 - acos(-sunElevation*1.065-0.065))/1.5);
float angSky= -(( pi * 0.5128205128205128 - acos(sunElevation*0.95+0.05))/1.5);
float angSkyNight= -(( pi * 0.5128205128205128 -acos(-sunElevation*0.95+0.05))/1.5);

float sunIntensity=max(0.,1.0-exp(angSun));
float fading = clamp(sunElevation+0.095,0.0,0.08)/0.08;
float skyIntensity=max(0.,1.0-exp(angSky))*(1.0-rainStrength*0.4)*pow(fading,5.0);
float moonIntensity=max(0.,1.0-exp(angMoon));
float fading2 = clamp(-sunElevation+0.095,0.0,0.08)/0.08;
float skyIntensityNight=max(0.,1.0-exp(angSkyNight))*(1.0-rainStrength*0.4)*pow(fading2,5.0);

float sunAmount = 27.;
float ambientAmount = 1.2;


float sunElev = pow(clamp(1.0-sunElevation,0.0,1.0),4.0)*1.8;
float sunlightR0=1.0;
float sunlightG0=(0.89*exp(-sunElev*0.57))*(1.0-rainStrength*0.3) + rainStrength*0.3;
float sunlightB0=(0.8*exp(-sunElev*1.4))*(1.0-rainStrength*0.3) + rainStrength*0.3;

float sunlightR=sunlightR0/(sunlightR0+sunlightG0+sunlightB0);
float sunlightG=sunlightG0/(sunlightR0+sunlightG0+sunlightB0);
float sunlightB=sunlightB0/(sunlightR0+sunlightG0+sunlightB0);
vec3 nsunColor=vec3(sunlightR,sunlightG,sunlightB);
vec3 sunColor=vec3(sunlightR*3.*sunAmount*(1.0-rainStrength*0.95),sunlightG*3.*sunAmount*(1.0-rainStrength*0.95),sunlightB*3.*sunAmount*(1.0-rainStrength*0.95));

float fogAmount = 0.005;

vec4 tpos = vec4(sunDir,1.0)*gbufferProjection;
	tpos = vec4(tpos.xyz/tpos.w,1.0);
	vec2 pos1 = tpos.xy/tpos.z;
	vec2 sunPosScreen = pos1*0.5+0.5;
	float sunVis = 0.0;
	const int nVisSamples = 1000;
	vec2 meanCenter = vec2(0.);
	for (int i = 0; i < nVisSamples; i++){
		vec2 spPos = sunPosScreen + tapLocation(i, nVisSamples, 88.0,0.0)*0.035;
		float spSunVis = texture(DiffuseDepthSampler, sunPosScreen + tapLocation(i, nVisSamples, 88.0,0.0)*0.035).r < 1.0 ? 0.0 : 1.0/nVisSamples;
		sunVis += spSunVis;
		meanCenter += spSunVis * spPos;	// Readjust sun position when its partially occluded
	}
	if (sunVis > 0.0)
		meanCenter /= sunVis;
	else
		meanCenter = sunPosScreen;
	vec2 scale = vec2(1.0, aspectRatio)*0.01;
	texCoord2 = (meanCenter - texCoord.xy)/scale;
  float truepos = sign(sunDir.z)*1.0;		//1 -> sun / -1 -> moon
  flareColor = mix(sunColor*skyIntensity+0.00001,3*vec3(0.16, 0.24,0.36)*skyIntensityNight+0.00001,(truepos+1.0)/2.) * (1.0-rainStrength*0.95);

	flareColor = flareColor * sunVis * 3.0 * (0.712-length(meanCenter-0.5));
	fogColor.rgb = texelFetch(temporals3Sampler, ivec2(0,16),0).rgb*5;
	fogColor.a = 0.4/0.6*fogAmount;
//	fogColor.rgb *= (eyeBrightnessSmooth.y/255.+0.006);


}
