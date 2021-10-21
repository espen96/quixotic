#version 150
const float sunPathRotation = -35.0;

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D DiffuseSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D clouds;

uniform vec2 InSize;
uniform float FOV;

 out vec3 ambientUp;
 out vec3 ambientLeft;
 out vec3 ambientRight;
 out vec3 ambientB;
 out vec3 ambientF;
 out vec3 ambientDown;
 out vec3 avgSky;
 out vec3 upPosition;
 out vec3 suncol;


out vec2 texCoord;
out vec2 oneTexel;
out vec3 sunDir;
out vec4 fogcol;

out vec4 rain;




 out mat4 gbufferProjection;
// out mat4 gbufferProjectionInverse;
out float near;
out float far;
out float end;
out float overworld;
out float aspectRatio;

out float rainStrength;
out vec3 sunVec;
out vec3 sunPosition;
out vec3 sunPosition3;
out float skyIntensity;
out float skyIntensityNight;

//out mat4 wgbufferModelViewInverse;
out mat4 wgbufferModelView;

 out mat4 gbufferModelViewInverse;
 out mat4 gbufferModelView;

uniform float Time;
float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
const float pi = 3.141592653589793238462643383279502884197169;

float facos(float inX) {

	const float C0 = 1.56467;
	const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}
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

int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}


float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}
vec3 skyLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	return texture(lut,vec2(x,y)).rgb;


}
float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}

void main() {


     suncol = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb*2.5;

    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);

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

    
    overworld = vec4((texture(DiffuseSampler, start + 28.0 * inc))).r;
    end = vec4((texture(DiffuseSampler, start + 29.0 * inc))).r;

    rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));




    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + inc).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz),
                                                    1.0)).xyz);
    

    gbufferModelViewInverse = inverse(ModeViewMat);
//    wgbufferModelViewInverse = inverse(ProjMat * ModeViewMat);

    gbufferModelView = (ModeViewMat);
    wgbufferModelView = (ProjMat * ModeViewMat);


    gbufferProjection = ProjMat;
//    gbufferProjectionInverse = inverse(ProjMat);
    aspectRatio = InSize.x / InSize.y;




////////////////////////////////////////////////

// 0     = +0.9765 +0.2154
// 6000  = +0.0 +1.0
// 12000 = -0.9765 +0.2154
// 18000 = -0.0 -1.0
// 24000 = +0.9765 +0.2154
float time3 = map(sunDir.y,-1,+1,0,1);
bool time8 = sunDir.y > 0;
float time4 = map(sunDir.x,-1,+1,0,1);
float time5 = mix(12000,0,time4);
float time6 = mix(24000,12000,1-time4);
float time7 = mix(time6,time5,time8);


float worldTime = time7;


const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994)); //radians() is not a const function on some drivers, so multiply by pi/180 manually.

//minecraft's native calculateCelestialAngle() function, ported to GLSL.
float ang = fract(worldTime / 24000.0 - 0.25);
ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) * 6.28318530717959; //0-2pi, rolls over from 2pi to 0 at noon.

sunDir =  vec3(-sin(ang), cos(ang) * sunRotationData);


 rainStrength = 1-rain.r;
vec3 sunDir2 = sunDir;
 sunPosition = mat3(gbufferModelView) * sunDir2;
 sunPosition3 =  sunDir2;
upPosition = vec3(gbufferModelView[1].xyz);
const vec3 cameraPosition = vec3(0.0);


float normSunVec = sqrt(sunPosition.x*sunPosition.x+sunPosition.y*sunPosition.y+sunPosition.z*sunPosition.z);
float normUpVec = sqrt(upPosition.x*upPosition.x+upPosition.y*upPosition.y+upPosition.z*upPosition.z);

float sunPosX = sunPosition.x/normSunVec;
float sunPosY = sunPosition.y/normSunVec;
float sunPosZ = sunPosition.z/normSunVec;
sunVec=vec3(sunPosX,sunPosY,sunPosZ);

float upPosX = upPosition.x/normUpVec;
float upPosY = upPosition.y/normUpVec;
float upPosZ = upPosition.z/normUpVec;

float sunElevation = sunPosX*upPosX+sunPosY*upPosY+sunPosZ*upPosZ;

float angSky= -(( pi * 0.5128205128205128 - facos(sunElevation*0.95+0.05))/1.5);
float angSkyNight= -(( pi * 0.5128205128205128 -facos(-sunElevation*0.95+0.05))/1.5);

float fading = clamp(sunElevation+0.095,0.0,0.08)/0.08;
 skyIntensity=max(0.,1.0-exp(angSky))*(1.0-rainStrength*0.4)*pow(fading,5.0);
float fading2 = clamp(-sunElevation+0.095,0.0,0.08)/0.08;
 skyIntensityNight=max(0.,1.0-exp(angSkyNight))*(1.0-rainStrength*0.4)*pow(fading2,5.0);
///////////////////////////


	ambientUp = texelFetch(temporals3Sampler,ivec2(0,37),0).rgb;
	ambientDown = texelFetch(temporals3Sampler,ivec2(1,37),0).rgb;
	ambientLeft = texelFetch(temporals3Sampler,ivec2(2,37),0).rgb;
	ambientRight = texelFetch(temporals3Sampler,ivec2(3,37),0).rgb;
	ambientB = texelFetch(temporals3Sampler,ivec2(4,37),0).rgb;
	ambientF = texelFetch(temporals3Sampler,ivec2(5,37),0).rgb;
	avgSky = texelFetch(temporals3Sampler,ivec2(11,37),0).rgb;

    gl_Position = vec4(outPos.xy, 0.2, 1.0);



}
