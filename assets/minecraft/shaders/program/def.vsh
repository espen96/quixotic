#version 150

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D DiffuseSampler;
uniform float Time;
out vec2 texCoord;
out vec2 oneTexel;
out vec3 sunDir;

uniform vec2 InSize;

out float aspectRatio;

out vec4 skycol;
out vec4 rain;

out float skyIntensity;
out vec3 nsunColor;
out float skyIntensityNight;
out float rainStrength;

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05

#define PI 3.141592
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

void main() {

    oneTexel = 1.0 / OutSize;
    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    texCoord = Position.xy / OutSize;

    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);



    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0,
                            0.0, 0.0, 0.0, 1.0);
    skycol = vec4((texture(DiffuseSampler, start + 26.0 * inc)));
    rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));



    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + inc).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz),
                                                    1.0)).xyz);
    

    aspectRatio = InSize.x / InSize.y;




////////////////////////////////////////////////
vec3 sunDir2 = normalize(vec3(sunDir.x,sunDir.y,sunDir.z+0.3));

vec3 sunPosition = sunDir2;
const vec3 upPosition = vec3(0,1,0);
rainStrength = (1-(rain.r))*0.75;

float normSunVec = sqrt(sunPosition.x*sunPosition.x+sunPosition.y*sunPosition.y+sunPosition.z*sunPosition.z);
float normUpVec = sqrt(upPosition.x*upPosition.x+upPosition.y*upPosition.y+upPosition.z*upPosition.z);

float sunPosX = sunPosition.x/normSunVec;
float sunPosY = sunPosition.y/normSunVec;
float sunPosZ = sunPosition.z/normSunVec;


float upPosX = upPosition.x/normUpVec;
float upPosY = upPosition.y/normUpVec;
float upPosZ = upPosition.z/normUpVec;

float sunElevation = sunPosX*upPosX+sunPosY*upPosY+sunPosZ*upPosZ;


float angSun= -(( PI * 0.5128205128205128 - acos(sunElevation*1.065-0.065))/1.5);
float angMoon= -(( PI * 0.5128205128205128 - acos(-sunElevation*1.065-0.065))/1.5);
float angSky= -(( PI * 0.5128205128205128 - acos(sunElevation*0.95+0.05))/1.5);
float angSkyNight= -(( PI * 0.5128205128205128 -acos(-sunElevation*0.95+0.05))/1.5);

float sunIntensity=max(0.,1.0-exp(angSun));
float fading = clamp(sunElevation+0.095,0.0,0.08)/0.08;
skyIntensity=max(0.,1.0-exp(angSky))*(1.0-rainStrength*0.4)*pow(fading,5.0);
float moonIntensity=max(0.,1.0-exp(angMoon));
float fading2 = clamp(-sunElevation+0.095,0.0,0.08)/0.08;
skyIntensityNight=max(0.,1.0-exp(angSkyNight))*(1.0-rainStrength*0.4)*pow(fading2,5.0);


float sunElev = pow(clamp(1.0-sunElevation,0.0,1.0),4.0)*1.8;
const float sunlightR0=1.0;
float sunlightG0=(0.89*exp(-sunElev*0.57))*(1.0-rainStrength*0.3) + rainStrength*0.3;
float sunlightB0=(0.8*exp(-sunElev*1.4))*(1.0-rainStrength*0.3) + rainStrength*0.3;

float sunlightR=sunlightR0/(sunlightR0+sunlightG0+sunlightB0);
float sunlightG=sunlightG0/(sunlightR0+sunlightG0+sunlightB0);
float sunlightB=sunlightB0/(sunlightR0+sunlightG0+sunlightB0);
nsunColor=vec3(sunlightR,sunlightG,sunlightB);


///////////////////////////


}
