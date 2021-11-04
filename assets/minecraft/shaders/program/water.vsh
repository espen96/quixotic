#version 150

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 InSize;

uniform sampler2D DiffuseSampler;
uniform vec2 OutSize;

out vec2 texCoord;
out vec2 oneTexel;
out float skyIntensity;
out vec3 nsunColor;
out float skyIntensityNight;
out float near;
out float far;
out float rainStrength;
out vec3 sunDir;
out mat4 gbufferModelView;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out mat4 wgbufferModelViewInverse;
// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define PI 3.141592

vec2 getControl(int index, vec2 screenSize) {
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}
float facos(float inX) {

	const float C0 = 1.56467;
	const float C1 = -0.155972;

	float x = abs(inX);
	float res = C1 * x + C0;
	res *= sqrt(1.0f - x);

	return (inX >= 0) ? res : PI - res;
}
int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}
float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
void main() {
    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    //simply decoding all the control data and constructing the sunDir, ProjMat, ModelViewMat

    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);

    // ProjMat constructed assuming no translation or rotation matrices applied (aka no view bobbing).
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), 0.0, 0.0, decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0);

    gbufferModelView = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0, 0.0, 0.0, 0.0, 1.0);

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0, 0.0, 0.0, 0.0, 1.0);

    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), decodeFloat(texture(DiffuseSampler, start + inc).xyz), decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz), 1.0)).xyz);
	vec4 rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));

    bool time8 = sunDir.y > 0;
    float time4 = map(sunDir.x, -1, +1, 0, 1);
    float time5 = mix(12000, 0, time4);
    float time6 = mix(24000, 12000, 1 - time4);
    float time7 = mix(time6, time5, time8);

    float worldTime = time7;

    const float sunPathRotation = -35.0;
    const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994)); //radians() is not a const function on some drivers, so multiply by pi/180 manually.

//minecraft's native calculateCelestialAngle() function, ported to GLSL.
    float ang = fract(worldTime / 24000.0 - 0.25);
    ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) * 6.28318530717959; //0-2pi, rolls over from 2pi to 0 at noon.

    vec3 sunDirTemp = vec3(-sin(ang), cos(ang) * sunRotationData);
    sunDir = normalize(vec3(sunDirTemp.x, sunDir.y, sunDirTemp.z));

    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

    gbufferProjection = (ProjMat);

    wgbufferModelViewInverse = inverse(ProjMat * ModeViewMat);
    gbufferProjectionInverse = inverse(ProjMat);

    texCoord = outPos.xy * 0.5 + 0.5;

    oneTexel = 1.0 / InSize;





	vec3 sunDir2 = sunDir;

	vec3 sunPosition = sunDir2;
	const vec3 upPosition = vec3(0, 1, 0);
	 rainStrength = (1 - (rain.r)) * 0.25;

	float normSunVec = sqrt(sunPosition.x * sunPosition.x + sunPosition.y * sunPosition.y + sunPosition.z * sunPosition.z);
	float normUpVec = sqrt(upPosition.x * upPosition.x + upPosition.y * upPosition.y + upPosition.z * upPosition.z);

	float sunPosX = sunPosition.x / normSunVec;
	float sunPosY = sunPosition.y / normSunVec;
	float sunPosZ = sunPosition.z / normSunVec;

	float upPosX = upPosition.x / normUpVec;
	float upPosY = upPosition.y / normUpVec;
	float upPosZ = upPosition.z / normUpVec;

	float sunElevation = sunPosX * upPosX + sunPosY * upPosY + sunPosZ * upPosZ;

	float angSky = -((PI * 0.5128205128205128 - facos(sunElevation * 0.95 + 0.05)) / 1.5);
	float angSkyNight = -((PI * 0.5128205128205128 - facos(-sunElevation * 0.95 + 0.05)) / 1.5);
	float angMoon = -((PI * 0.5128205128205128 - facos(-sunElevation * 1.065 - 0.065)) / 1.5);
	float angSun = -((PI * 0.5128205128205128 - facos(sunElevation * 1.065 - 0.065)) / 1.5);

	float fading = clamp(sunElevation + 0.095, 0.0, 0.08) / 0.08;
	skyIntensity = max(0., 1.0 - exp(angSky)) * (1.0 - rainStrength * 0.4) * pow(fading, 5.0);
	float fading2 = clamp(-sunElevation + 0.095, 0.0, 0.08) / 0.08;
	skyIntensityNight = max(0., 1.0 - exp(angSkyNight)) * (1.0 - rainStrength * 0.4) * pow(fading2, 5.0);

	float skyIntensity = max(0., 1.0 - exp(angSky)) * (1.0 - rainStrength * 0.4) * pow(fading, 5.0);


	float sunElev = pow(clamp(1.0 - sunElevation, 0.0, 1.0), 4.0) * 1.8;
	const float sunlightR0 = 1.0;
	float sunlightG0 = (0.89 * exp(-sunElev * 0.57)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
	float sunlightB0 = (0.8 * exp(-sunElev * 1.4)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
	vec3 sunVec = vec3(sunPosX, sunPosY, sunPosZ);

	float sunlightR = sunlightR0 / (sunlightR0 + sunlightG0 + sunlightB0);
	float sunlightG = sunlightG0 / (sunlightR0 + sunlightG0 + sunlightB0);
	float sunlightB = sunlightB0 / (sunlightR0 + sunlightG0 + sunlightB0);
	nsunColor = vec3(sunlightR, sunlightG, sunlightB);



    gl_Position = vec4(outPos.xy, 0.2, 1.0);

}
