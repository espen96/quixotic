#version 150
const float sunPathRotation = -35.0;

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D DiffuseSampler;
uniform sampler2D temporals3Sampler;

out vec3 ambientUp;
out vec3 ambientLeft;
out vec3 ambientRight;
out vec3 ambientB;
out vec3 ambientF;
out vec3 ambientDown;
out vec3 suncol;
out vec3 zMults;

out vec2 oneTexel;
out vec4 fogcol;

out vec2 texCoord;

out mat3 gbufferModelViewInverse;
out mat4 gbufferModelView;
out mat4 wgbufferModelView;
out mat4 gbufferProjection;
//out mat4 gbufferProjectionInverse;
out mat4 wgbufferModelViewInverse;

out float near;
out float far;
out float end;
out float overworld;

out float rainStrength;
out vec3 sunVec;

out vec3 sunPosition3;
out float skyIntensityNight;

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

void main() {

    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);

    texCoord = Position.xy / OutSize;
    oneTexel = 1.0 / OutSize;

    //simply decoding all the control data and constructing the sunDir, ProjMat, ModelViewMat

    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);

    // ProjMat constructed assuming no translation or rotation matrices applied (aka no view bobbing).
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), 0.0, 0.0, decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0);

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0, 0.0, 0.0, 0.0, 1.0);

    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));

    overworld = vec4((texture(DiffuseSampler, start + 28.0 * inc))).r;
    end = vec4((texture(DiffuseSampler, start + 29.0 * inc))).r;

    vec4 rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));

    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

    vec3 sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), decodeFloat(texture(DiffuseSampler, start + inc).xyz), decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz), 1.0)).xyz);

    gbufferModelViewInverse = inverse(mat3(ModeViewMat));
    wgbufferModelViewInverse = inverse(ProjMat * ModeViewMat);

    gbufferModelView = (ModeViewMat);
    wgbufferModelView = (ProjMat * ModeViewMat);

    gbufferProjection = ProjMat;
    //gbufferProjectionInverse = inverse(ProjMat);

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

    rainStrength = 1 - rain.r;
    vec3 sunDir2 = sunDir;
    vec3 sunPosition = mat3(gbufferModelView) * sunDir2;
    sunPosition3 = sunDir2;
    vec3 upPosition = vec3(gbufferModelView[1].xyz);
    const vec3 cameraPosition = vec3(0.0);
    zMults = vec3(1.0 / (far * near), far + near, far - near);

    float normSunVec = sqrt(sunPosition.x * sunPosition.x + sunPosition.y * sunPosition.y + sunPosition.z * sunPosition.z);
    float normUpVec = sqrt(upPosition.x * upPosition.x + upPosition.y * upPosition.y + upPosition.z * upPosition.z);

    float sunPosX = sunPosition.x / normSunVec;
    float sunPosY = sunPosition.y / normSunVec;
    float sunPosZ = sunPosition.z / normSunVec;
    vec3 sunVec2 = vec3(sunPosX, sunPosY, sunPosZ);

    float upPosX = upPosition.x / normUpVec;
    float upPosY = upPosition.y / normUpVec;
    float upPosZ = upPosition.z / normUpVec;

    float sunElevation = sunPosX * upPosX + sunPosY * upPosY + sunPosZ * upPosZ;

    float angSkyNight = -((pi * 0.5128205128205128 - facos(-sunElevation * 0.95 + 0.05)) / 1.5);

    float fading2 = clamp(-sunElevation + 0.095, 0.0, 0.08) / 0.08;
    skyIntensityNight = max(0., 1.0 - exp(angSkyNight)) * (1.0 - rainStrength * 0.4) * pow(fading2, 5.0);
    sunVec = mix(sunVec2, -sunVec2, clamp(skyIntensityNight * 3, 0, 1));

///////////////////////////
    suncol = decodeColor(texelFetch(temporals3Sampler, ivec2(8, 37), 0));
    ambientUp = texelFetch(temporals3Sampler, ivec2(0, 37), 0).rgb;
    ambientDown = texelFetch(temporals3Sampler, ivec2(1, 37), 0).rgb;
    ambientLeft = texelFetch(temporals3Sampler, ivec2(2, 37), 0).rgb;
    ambientRight = texelFetch(temporals3Sampler, ivec2(3, 37), 0).rgb;
    ambientB = texelFetch(temporals3Sampler, ivec2(4, 37), 0).rgb;
    ambientF = texelFetch(temporals3Sampler, ivec2(5, 37), 0).rgb;
    //avgSky = texelFetch(temporals3Sampler, ivec2(11, 37), 0).rgb;

    gl_Position = vec4(outPos.xy, 0.2, 1.0);

}
