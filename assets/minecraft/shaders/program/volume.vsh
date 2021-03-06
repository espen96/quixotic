#version 150
const float sunPathRotation = -35.0;
in vec4 Position;
uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D temporals3Sampler;
uniform sampler2D DiffuseSampler;
uniform sampler2D clouds;
uniform float Time;
out vec2 texCoord;
out vec2 oneTexel;
out vec3 sunDir;
flat out vec4 fogcol;
flat out vec4 skycol;
out vec4 rain;
out mat4 gbufferModelViewInverse2;
out mat4 gbufferModelViewInverse;

out mat4 gbufferProjectionInverse;
out mat4 gbufferProjection;
out vec3 suncol;
out vec3 avgSky;
out vec3 ambientUp;
out vec3 ambientLeft;
out vec3 ambientRight;
out vec3 ambientB;
out vec3 ambientF;
out vec3 ambientDown;
flat out float near;
flat out float far;
flat out float end;
flat out float cloudy;
flat out float overworld;
flat out vec3 currChunkOffset;

flat out float sunElevation;
flat out vec3 sunVec;
flat out vec3 sunPosition;
flat out vec3 sunPosition3;
flat out float fogAmount;
flat out vec2 eyeBrightnessSmooth;
#define SUNBRIGHTNESS 20
float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
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

float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}

#define BASE_FOG_AMOUNT 10.0 
#define FOG_TOD_MULTIPLIER 0.15
#define FOG_RAIN_MULTIPLIER 0.15

const float pi = 3.141592653589793238462643383279502884197169;
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
vec3 skyLut(vec3 sVector, vec3 sunVec, float cosT, sampler2D lut) {
    const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

    float mCosT = clamp(cosT, 0.0, 1.);
    float cosY = dot(sunVec, sVector);
    float x = ((cosY * cosY) * (cosY * 0.5 * 256.) + 0.5 * 256. + 18. + 0.5) * oneTexel.x;
    float y = (mCosT * 256. + 1.0 + 0.5) * oneTexel.y;

    return texture(lut, vec2(x, y)).rgb;

}
float decodeFloat7_4(uint raw) {
    uint sign = raw >> 11u;
    uint exponent = (raw >> 7u) & 15u;
    uint mantissa = 128u | (raw & 127u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 14.0);
}

float decodeFloat6_4(uint raw) {
    uint sign = raw >> 10u;
    uint exponent = (raw >> 6u) & 15u;
    uint mantissa = 64u | (raw & 63u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 13.0);
}

vec3 decodeColor(vec4 raw) {
    uvec4 scaled = uvec4(round(raw * 255.0));
    uint encoded = (scaled.r << 24) | (scaled.g << 16) | (scaled.b << 8) | scaled.a;

    return vec3(decodeFloat7_4(encoded >> 21), decodeFloat7_4((encoded >> 10) & 2047u), decodeFloat6_4(encoded & 1023u));
}

uint encodeFloat7_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 14.0)) & 127u;
    return (sign << 11u) | (exponent << 7u) | mantissa;
}

uint encodeFloat6_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 13.0)) & 63u;
    return (sign << 10u) | (exponent << 6u) | mantissa;
}

vec4 encodeColor(vec3 color) {
    uint r = encodeFloat7_4(color.r);
    uint g = encodeFloat7_4(color.g);
    uint b = encodeFloat6_4(color.b);

    uint encoded = (r << 21) | (g << 10) | b;
    return vec4(encoded >> 24, (encoded >> 16) & 255u, (encoded >> 8) & 255u, encoded & 255u) / 255.0;
}

float facos(float inX) {

    const float C0 = 1.56467;
    const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}

vec3 skyLut2(vec3 sVector, vec3 sunVec, float cosT, float rainStrength, vec3 nsunColor, float skyIntensity, float skyIntensityNight) {
	#define SKY_BRIGHTNESS_DAY 1.0
	#define SKY_BRIGHTNESS_NIGHT 1.0;	
    float mCosT = clamp(cosT, 0.0, 1.0);
    float cosY = dot(sunVec, sVector);
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
    if(skyIntensity > 0.00001) {
        float L0 = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * Y) - exp(d * pi / 2.)) + e * cosY * cosY);
        vec3 skyColor0 = mix(vec3(0.05, 0.5, 1.) / 1.5, vec3(0.4, 0.5, 0.6) / 1.5, rainStrength);
        vec3 normalizedSunColor = nsunColor;
        vec3 skyColor = mix(skyColor0, normalizedSunColor, 1.0 - pow(1.0 + L0, -1.2)) * (1.0 - rainStrength);
        daySky = pow(L0, 1.0 - rainStrength) * skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15. * SKY_BRIGHTNESS_DAY;
    }
	// Night
    else if(skyIntensityNight > 0.00001) {
        float L0Moon = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * (pi - Y)) - exp(d * pi / 2.)) + e * cosY * cosY);
        moonSky = pow(L0Moon, 1.0 - rainStrength) * skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) * SKY_BRIGHTNESS_NIGHT;
    }
    return (daySky + moonSky);
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
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), 0.0, 0.0, decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0);

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0, 0.0, 0.0, 0.0, 1.0);
    currChunkOffset = vec3(decodeFloat(texture(DiffuseSampler, start + 100 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 101 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 102 * inc).xyz));

    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));
    skycol = vec4((texture(DiffuseSampler, start + 26.0 * inc)));

    overworld = vec4((texture(DiffuseSampler, start + 28.0 * inc))).r;
    end = vec4((texture(DiffuseSampler, start + 29.0 * inc))).r;
    rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));
    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), decodeFloat(texture(DiffuseSampler, start + inc).xyz), decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz), 1.0)).xyz);
    gbufferProjection = (ProjMat);
    gbufferProjectionInverse = inverse(ProjMat);
    gbufferModelViewInverse = inverse(ProjMat * ModeViewMat);
    gbufferModelViewInverse2 = inverse(ProjMat * ModeViewMat);



////////////////////////////////////////////////
    bool time8 = sunDir.y > 0;
    float time4 = map(sunDir.x, -1, +1, 0, 1);
    float time5 = mix(12000, 0, time4);
    float time6 = mix(24000, 12000, 1 - time4);
    float time7 = mix(time6, time5, time8);

    int worldTime = int(time7);

    const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994)); //radians() is not a const function on some drivers, so multiply by pi/180 manually.

//minecraft's native calculateCelestialAngle() function, ported to GLSL.
    float ang = fract(worldTime / 24000.0 - 0.25);
    ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) * 6.28318530717959; //0-2pi, rolls over from 2pi to 0 at noon.

    vec3 sunDirTemp = vec3(-sin(ang), cos(ang) * sunRotationData);
    sunDir = normalize(vec3(sunDirTemp.x, sunDir.y, sunDirTemp.z));

    float rainStrength = (1 - rain.r) * 0.5;
    vec3 sunDir2 = sunDir;
    sunPosition = sunDir2;

    sunPosition3 = mat3(ModeViewMat) * sunDir2;
    vec3 upPosition = vec3(0, 1, 0);
    sunVec = sunDir2;

    eyeBrightnessSmooth = vec2(240);

    float normSunVec = sqrt(sunPosition.x * sunPosition.x + sunPosition.y * sunPosition.y + sunPosition.z * sunPosition.z);
    float normUpVec = sqrt(upPosition.x * upPosition.x + upPosition.y * upPosition.y + upPosition.z * upPosition.z);

    float sunPosX = sunPosition.x / normSunVec;
    float sunPosY = sunPosition.y / normSunVec;
    float sunPosZ = sunPosition.z / normSunVec;

    float upPosX = upPosition.x / normUpVec;
    float upPosY = upPosition.y / normUpVec;
    float upPosZ = upPosition.z / normUpVec;

    sunElevation = sunPosX * upPosX + sunPosY * upPosY + sunPosZ * upPosZ;

    float modWT = (worldTime % 24000) * 1.0;
    float fogAmount0 = 1 / 3000. + FOG_TOD_MULTIPLIER * (1 / 180. * (clamp(modWT - 11000., 0., 2000.0) / 2000. + (1.0 - clamp(modWT, 0., 3000.0) / 3000.)) * (clamp(modWT - 11000., 0., 2000.0) / 2000. + (1.0 - clamp(modWT, 0., 3000.0) / 3000.)) + 1 / 200. * clamp(modWT - 13000., 0., 1000.0) / 1000. * (1.0 - clamp(modWT - 23000., 0., 1000.0) / 1000.));
    fogAmount = BASE_FOG_AMOUNT * (fogAmount0 + max(FOG_RAIN_MULTIPLIER * 1 / 20. * rainStrength, FOG_TOD_MULTIPLIER * 1 / 50. * clamp(modWT - 13000., 0., 1000.0) / 1000. * (1.0 - clamp(modWT - 23000., 0., 1000.0) / 1000.)));

    float angMoon = -((pi * 0.5128205128205128 - facos(-sunElevation * 1.065 - 0.065)) / 1.5);
    float angSun = -((pi * 0.5128205128205128 - facos(sunElevation * 1.065 - 0.065)) / 1.5);

    float sunElev = pow(clamp(1.0 - sunElevation, 0.0, 1.0), 4.0) * 1.8;
    const float sunlightR0 = 1.0;
    float sunlightG0 = (0.89 * exp(-sunElev * 0.57)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
    float sunlightB0 = (0.8 * exp(-sunElev * 1.4)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;

    float sunlightR = sunlightR0 / (sunlightR0 + sunlightG0 + sunlightB0);
    float sunlightG = sunlightG0 / (sunlightR0 + sunlightG0 + sunlightB0);
    float sunlightB = sunlightB0 / (sunlightR0 + sunlightG0 + sunlightB0);
    vec3 nsunColor = vec3(sunlightR, sunlightG, sunlightB);

///////////////////////////

    float angSkyNight = -((pi * 0.5128205128205128 - facos(-sunElevation * 0.95 + 0.05)) / 1.5);
    float angSky = -((pi * 0.5128205128205128 - facos(sunElevation * 0.95 + 0.05)) / 1.5);

    float fading = clamp(sunElevation + 0.095, 0.0, 0.08) / 0.08;
    float fading2 = clamp(-sunElevation + 0.095, 0.0, 0.08) / 0.08;
    float skyIntensity = max(0., 1.0 - exp(angSky)) * (1.0 - rainStrength * 0.4) * pow(fading, 5.0);
    float skyIntensityNight = max(0., 1.0 - exp(angSkyNight)) * (1.0 - rainStrength * 0.4) * pow(fading2, 5.0);

    float moonIntensity = max(0., 1.0 - exp(angMoon));
    float sunIntensity = max(0., 1.0 - exp(angSun));
    vec3 sunVec = vec3(sunPosX, sunPosY, sunPosZ);
    moonIntensity = max(0., 1.0 - exp(angMoon));

    nsunColor = vec3(sunlightR, sunlightG, sunlightB);
    float avgEyeIntensity = ((sunIntensity * 120. + moonIntensity * 4.) + skyIntensity * 230. + skyIntensityNight * 4.);
    float exposure = 0.18 / log(max(avgEyeIntensity * 0.16 + 1.0, 1.13)) * 0.3 * log(2.0);
    const float sunAmount = 27.0 * 1.5;
    float lightSign = clamp(sunIntensity * pow(10., 35.), 0., 1.);
    vec4 lightCol = vec4((sunlightR * 3. * sunAmount * sunIntensity + 0.16 / 5. - 0.16 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * 7.84 * exposure, 7.84 * (sunlightG * 3. * sunAmount * sunIntensity + 0.24 / 5. - 0.24 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * exposure, 7.84 * (sunlightB * 3. * sunAmount * sunIntensity + 0.36 / 5. - 0.36 / 5. * lightSign) * (1.0 - rainStrength * 0.95) * exposure, lightSign * 2.0 - 1.0);



    vec3 lightSourceColor = lightCol.rgb;

    
    float sunVis = clamp(sunElevation, 0.0, 0.05) / 0.05 * clamp(sunElevation, 0.0, 0.05) / 0.05;
    float lightDir = float(sunVis >= 1e-5) * 2.0 - 1.0;

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

        vec3 samplee = 2.2 * skyLut2(pos.xyz, sunDir2, pos.y, rainStrength*0.25, nsunColor, skyIntensity, skyIntensityNight) / maxIT;
        avgSky = samplee;
        ambientUp += samplee * (pos.y + abs(pos.x) / 7. + abs(pos.z) / 7.);
        ambientLeft += samplee * (clamp(-pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
        ambientRight += samplee * (clamp(pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
        ambientB += samplee * (clamp(pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
        ambientF += samplee * (clamp(-pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
        ambientDown += samplee * (clamp(pos.y / 6., 0.0, 1.0) + abs(pos.x) / 7. + abs(pos.z) / 7.);

    }

    moonIntensity = max(0., 1.0 - exp(angMoon));

    suncol = lightSourceColor;

	//Fake bounced sunlight
    vec3 bouncedSun = lightSourceColor / pi / pi / 4. * 0.5 * (abs(sunVec.x) * 0.2 + clamp(lightDir * sunVec.y, 0.0, 1.0) * 0.6 + abs(sunVec.z) * 0.2);
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

}
