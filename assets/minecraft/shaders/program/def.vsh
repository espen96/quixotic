#version 150
const float sunPathRotation = -35.0;
in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform sampler2D DiffuseSampler;
uniform sampler2D pre;

uniform float Time;
out vec2 texCoord;
out vec3 sunDir;
out vec3 ds;
out vec3 ms;

out vec4 skycol;
out vec4 rain;

out float skyIntensity;
out vec3 nsunColor;
out float skyIntensityNight;
out float rainStrength;
out float sunIntensity;
out float moonIntensity;
out vec3 ambientUp;
out vec3 ambientLeft;
out vec3 ambientRight;
out vec3 ambientB;
out vec3 ambientF;
out vec3 ambientDown;
out vec3 avgSky;
out vec4 lightCol;

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
float map(float value, float min1, float max1, float min2, float max2) {
	return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

int decodeInt(vec3 ivec) {
	ivec *= 255.0;
	int s = ivec.b >= 128.0 ? -1 : 1;
	return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

float decodeFloat(vec3 ivec) {
	return decodeInt(ivec) / FPRECISION;
}
vec3 rodSample(vec2 Xi) {
	float r = sqrt(1.0f - Xi.x * Xi.y);
	float phi = 2 * 3.14159265359 * Xi.y;

	return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
//Low discrepancy 2D sequence, integration error is as low as sobol but easier to compute : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
vec2 R2_samples(int n) {
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}
vec4 textureGood(sampler2D sam, vec2 uv) {
	vec2 res = textureSize(sam, 0);

	vec2 st = uv * res - 0.5;

	vec2 iuv = floor(st);
	vec2 fuv = fract(st);

	vec4 a = textureLod(sam, (iuv + vec2(0.5, 0.5)) / res, 0);
	vec4 b = textureLod(sam, (iuv + vec2(1.5, 0.5)) / res, 0);
	vec4 c = textureLod(sam, (iuv + vec2(0.5, 1.5)) / res, 0);
	vec4 d = textureLod(sam, (iuv + vec2(1.5, 1.5)) / res, 0);

	return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
vec3 skyLut(vec3 sVector, vec3 sunVec, float cosT, sampler2D lut) {
	vec2 oneTexel = 1.0 / OutSize;
	float mCosT = clamp(cosT, 0.0, 1.);
	float cosY = dot(sunVec, sVector);
	float x = ((cosY * cosY) * (cosY * 0.5 * 256.) + 0.5 * 256. + 18. + 0.5) * oneTexel.x;
	float y = (mCosT * 256. + 1.0 + 0.5) * oneTexel.y;

	return textureGood(lut, vec2(x, y)).rgb;
}
void main() {
	vec2 oneTexel = 1.0 / OutSize;

	vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
	gl_Position = vec4(outPos.xy, 0.2, 1.0);
	gl_Position.xy = gl_Position.xy*vec2(18.+258*2,258.)*oneTexel;
	gl_Position.xy = gl_Position.xy*2.-1.0;


	texCoord = Position.xy / OutSize;

	vec2 start = getControl(0, OutSize);
	vec2 inc = vec2(2.0 / OutSize.x, 0.0);

	mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0, 0.0, 0.0, 0.0, 1.0);
	skycol = vec4((texture(DiffuseSampler, start + 26.0 * inc)));
	rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));

	sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), decodeFloat(texture(DiffuseSampler, start + inc).xyz), decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz), 1.0)).xyz);

////////////////////////////////////////////////
// 0     = +0.9765 +0.2154
// 6000  = +0.0 +1.0
// 12000 = -0.9765 +0.2154
// 18000 = -0.0 -1.0
// 24000 = +0.9765 +0.2154
	bool time8 = sunDir.y > 0;
	float time4 = map(sunDir.x, -1, +1, 0, 1);
	float time5 = mix(12000, 0, time4);
	float time6 = mix(24000, 12000, 1 - time4);
	float time7 = mix(time6, time5, time8);

	float worldTime = time7;

	const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994)); //radians() is not a const function on some drivers, so multiply by pi/180 manually.

//minecraft's native calculateCelestialAngle() function, ported to GLSL.
	float ang = fract(worldTime / 24000.0 - 0.25);
	ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) * 6.28318530717959; //0-2pi, rolls over from 2pi to 0 at noon.

	vec3 sunDirTemp = vec3(-sin(ang), cos(ang) * sunRotationData);
	sunDir = normalize(vec3(sunDirTemp.x, sunDir.y, sunDirTemp.z));

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
	moonIntensity = max(0., 1.0 - exp(angMoon));
	sunIntensity = max(0., 1.0 - exp(angSun));

	float sunElev = pow(clamp(1.0 - sunElevation, 0.0, 1.0), 4.0) * 1.8;
	const float sunlightR0 = 1.0;
	float sunlightG0 = (0.89 * exp(-sunElev * 0.57)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
	float sunlightB0 = (0.8 * exp(-sunElev * 1.4)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
	vec3 sunVec = vec3(sunPosX, sunPosY, sunPosZ);

	float sunlightR = sunlightR0 / (sunlightR0 + sunlightG0 + sunlightB0);
	float sunlightG = sunlightG0 / (sunlightR0 + sunlightG0 + sunlightB0);
	float sunlightB = sunlightB0 / (sunlightR0 + sunlightG0 + sunlightB0);
	nsunColor = vec3(sunlightR, sunlightG, sunlightB);
	float avgEyeIntensity = ((sunIntensity * 120. + moonIntensity * 4.) + skyIntensity * 230. + skyIntensityNight * 4.);
	float exposure = 0.18 / log(max(avgEyeIntensity * 0.16 + 1.0, 1.13)) * 0.3 * log(2.0);
	const float sunAmount = 27.0*2.0;
	float lightSign = clamp(sunIntensity * pow(10., 35.), 0., 1.);
	lightCol = vec4((sunlightR * 3. * sunAmount * sunIntensity + 0.16 / 5. - 0.16 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * 7.84 * exposure, 7.84 * (sunlightG * 3. * sunAmount * sunIntensity + 0.24 / 5. - 0.24 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * exposure, 7.84 * (sunlightB * 3. * sunAmount * sunIntensity + 0.36 / 5. - 0.36 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * exposure, lightSign * 2.0 - 1.0);
	rainStrength = (1 - (rain.r)) * 0.75;

	ambientUp = vec3(0.0);
	ambientDown = vec3(0.0);
	ambientLeft = vec3(0.0);
	ambientRight = vec3(0.0);
	ambientB = vec3(0.0);
	ambientF = vec3(0.0);
	avgSky = vec3(0.0);
	int maxIT = 20;
	for(int i = 0; i < maxIT; i++) {
		vec2 ij = R2_samples((int(Time) % 1000) * maxIT + i);
		vec3 pos = normalize(rodSample(ij));

		vec3 samplee = 2.2 * skyLut(pos.xyz, sunDir2, pos.y, pre) / maxIT;
		avgSky += samplee / 2.2;

		ambientUp += samplee * (pos.y + abs(pos.x) / 7. + abs(pos.z) / 7.);
		ambientLeft += samplee * (clamp(-pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
		ambientRight += samplee * (clamp(pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
		ambientB += samplee * (clamp(pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
		ambientF += samplee * (clamp(-pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
		ambientDown += samplee * (clamp(pos.y / 6., 0.0, 1.0) + abs(pos.x) / 7. + abs(pos.z) / 7.);

	}
	vec3 lightSourceColor = lightCol.rgb;
	float sunVis = clamp(sunElevation, 0.0, 0.05) / 0.05 * clamp(sunElevation, 0.0, 0.05) / 0.05;
	float lightDir = float(sunVis >= 1e-5) * 2.0 - 1.0;

	//Fake bounced sunlight
	vec3 bouncedSun = lightSourceColor / PI / PI / 4. * 0.5 * (abs(sunVec.x) * 0.2 + clamp(lightDir * sunVec.y, 0.0, 1.0) * 0.6 + abs(sunVec.z) * 0.2);
	vec3 fakegi = lightSourceColor * vec3(0.042, 0.046, 0.046) * (0.7 + 0.9) * 4.5 * (1.0 + rainStrength * 0.2);
	bouncedSun *= fakegi;
	ambientUp += bouncedSun * clamp(-lightDir * sunVec.y + 3., 0., 4.0);
	ambientLeft += bouncedSun * clamp(lightDir * sunVec.x + 3., 0.0, 4.);
	ambientRight += bouncedSun * clamp(-lightDir * sunVec.x + 3., 0.0, 4.);
	ambientB += bouncedSun * clamp(-lightDir * sunVec.z + 3., 0.0, 4.);
	ambientF += bouncedSun * clamp(lightDir * sunVec.z + 3., 0.0, 4.);
	ambientDown += bouncedSun * clamp(lightDir * sunVec.y + 3., 0.0, 4.) * 0.7;
	avgSky += bouncedSun * 0.6;

///////////////////////////
  //luminance (cie model)
	vec3 daySky = vec3(0.0);
	vec3 moonSky = vec3(0.0);
	// Day
	if(skyIntensity > 0.00001) {
		vec3 skyColor0 = mix(vec3(0.05, 0.5, 1.) / 1.5, vec3(0.4, 0.5, 0.6) / 1.5, rainStrength);
		vec3 skyColor = mix(skyColor0, nsunColor, 0.5);
		daySky = skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15. * 0.5;
	}
	// Night
	if(skyIntensityNight > 0.00001) {
		moonSky = skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) * 0.05;
	}
	ds = daySky;
	ms = moonSky;
}
