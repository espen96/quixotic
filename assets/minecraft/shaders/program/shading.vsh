#version 150
const float sunPathRotation = -35.0;

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D noisetex;
uniform sampler2D DiffuseSampler;
uniform float Time;
out mat4 gbufferModelView;
out mat4 wgbufferModelView;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out float sunElevation;

out vec3 zenithColor;
out vec3 ambientUp;
out vec3 ambientLeft;
out vec3 ambientRight;
out vec3 ambientB;
out vec3 ambientF;
out vec3 ambientDown;
out vec3 suncol;
out vec3 nsunColor;
out float skys;
out vec2 oneTexel;
out vec4 fogcol;
out float cloudy;

out vec2 texCoord;

// out mat4 wgbufferModelViewInverse;

out float near;
out float far;
out float end;
out float overworld;

out float rainStrength;
out vec3 sunVec;

out vec3 sunPosition2;
out vec3 sunPosition3;
out vec3 sunPosition;
out float skyIntensityNight;
out float skyIntensity;

float map(float value, float min1, float max1, float min2, float max2)
{
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
const float pi = 3.141592653589793238462643383279502884197169;

float facos(float inX)
{

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

vec2 getControl(int index, vec2 screenSize)
{
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}

int decodeInt(vec3 ivec)
{
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

float decodeFloat(vec3 ivec)
{
    return decodeInt(ivec) / FPRECISION;
}
float decodeFloat7_4(uint raw)
{
    uint sign = raw >> 11u;
    uint exponent = (raw >> 7u) & 15u;
    uint mantissa = 128u | (raw & 127u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 14.0);
}
vec3 toLinear(vec3 sRGB)
{
    return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

float decodeFloat6_4(uint raw)
{
    uint sign = raw >> 10u;
    uint exponent = (raw >> 6u) & 15u;
    uint mantissa = 64u | (raw & 63u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 13.0);
}

vec3 decodeColor(vec4 raw)
{
    uvec4 scaled = uvec4(round(raw * 255.0));
    uint encoded = (scaled.r << 24) | (scaled.g << 16) | (scaled.b << 8) | scaled.a;

    return vec3(decodeFloat7_4(encoded >> 21), decodeFloat7_4((encoded >> 10) & 2047u),
                decodeFloat6_4(encoded & 1023u));
}

uint encodeFloat7_4(float val)
{
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 14.0)) & 127u;
    return (sign << 11u) | (exponent << 7u) | mantissa;
}

uint encodeFloat6_4(float val)
{
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 13.0)) & 63u;
    return (sign << 10u) | (exponent << 6u) | mantissa;
}

vec4 encodeColor(vec3 color)
{
    uint r = encodeFloat7_4(color.r);
    uint g = encodeFloat7_4(color.g);
    uint b = encodeFloat6_4(color.b);

    uint encoded = (r << 21) | (g << 10) | b;
    return vec4(encoded >> 24, (encoded >> 16) & 255u, (encoded >> 8) & 255u, encoded & 255u) / 255.0;
}
float decodeFloat24(vec3 raw)
{
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}
vec3 rodSample(vec2 Xi)
{
    float r = sqrt(1.0f - Xi.x * Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
// Low discrepancy 2D sequence, integration error is as low as sobol but easier
// to compute :
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
vec2 R2_samples(int n)
{
    vec2 alpha = vec2(0.75487765, 0.56984026);
    return fract(alpha * n);
}

vec2 start = getControl(0, OutSize);
vec2 inc = vec2(2.0 / OutSize.x, 0.0);
vec4 rain = vec4((texture(DiffuseSampler, start + 30.0 * inc)));

vec3 skyLut2(vec3 sVector, vec3 sunVec, float cosT, float rainStrength, vec3 nsunColor, float skyIntensity,
             float skyIntensityNight)
{
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

    // luminance (cie model)
    vec3 daySky = vec3(0.0);
    vec3 moonSky = vec3(0.0);
    // Day
    if (skyIntensity > 0.00001)
    {
        float L0 = (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * Y) - exp(d * pi / 2.)) + e * cosY * cosY);
        vec3 skyColor0 = mix(vec3(0.05, 0.5, 1.) / 1.5, vec3(0.4, 0.5, 0.6) / 1.5, rainStrength);
        vec3 normalizedSunColor = nsunColor;
        vec3 skyColor = mix(skyColor0, normalizedSunColor, 1.0 - pow(1.0 + L0, -1.2)) * (1.0 - rainStrength);
        daySky = pow(L0, 1.0 - rainStrength) * skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15. * SKY_BRIGHTNESS_DAY;
    }
    // Night
    else if (skyIntensityNight > 0.00001)
    {
        float L0Moon =
            (1.0 + a * exp(b / mCosT)) * (1.0 + c * (exp(d * (pi - Y)) - exp(d * pi / 2.)) + e * cosY * cosY);
        moonSky = pow(L0Moon, 1.0 - rainStrength) * skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) *
                  SKY_BRIGHTNESS_NIGHT;
    }
    return (daySky + moonSky);
}
#define PI 3.141592

////////////////////

const float M_PI = 3.1415926535;
const float DEGRAD = M_PI / 180.0;

float height = 500.0; // viewer height

// rendering quality
const int steps2 = 16; // 16 is fast, 128 or 256 is extreme high
const int stepss = 8;  // 8 is fast, 16 or 32 is high

float haze = 1 - rain.r; // 0.2

const float I = 10.0; // sun light power, 10.0 is normal
const float g = 0.76; // light concentration .76 //.45 //.6  .76 is normaL
const float g2 = g * g;

// Reyleigh scattering (sky color, atmospheric up to 8km)
// vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6); // normal earth
// vec3 bR = vec3(5.7e-6, 13.3e-6, 33.0e-6); // normal earth2
vec3 bR = vec3(3.8e-6f, 13.5e-6f, 33.1e-6f); // normal earth3
// vec3 bR = vec3(5.8e-6, 33.1e-6, 13.5e-6); //purple
// vec3 bR = vec3( 63.5e-6, 13.1e-6, 50.8e-6 ); //green
// vec3 bR = vec3( 13.5e-6, 23.1e-6, 115.8e-6 ); //yellow
// vec3 bR = vec3( 5.5e-6, 15.1e-6, 355.8e-6 ); //yeellow
// vec3 bR = vec3(3.5e-6, 333.1e-6, 235.8e-6 ); //red-purple

// Mie scattering (water particles up to 1km)
vec3 bM = vec3(21e-6); // normal mie
// vec3 bM = vec3(50e-6); //high mie

//-----
// positions

const float Hr = 7994.0; // Reyleight scattering top
const float Hm = 1200.0; // Mie scattering top

const float R0 = 6360e3;    // planet radius
const float Ra = 6420e3;    // atmosphere radius
vec3 C = vec3(0., -R0, 0.); // planet center

//--------------------------------------------------------------------------
// scattering

void densities(in vec3 pos, out float rayleigh, out float mie)
{
    float h = length(pos - C) - R0;
    rayleigh = exp(-h / Hr);
    vec3 d = pos;
    d.y = 0.0;
    float dist = length(d);
    mie = exp(-h / Hm) + haze;
}

float escape(in vec3 p, in vec3 d, in float R)
{
    vec3 v = p - C;
    float b = dot(v, d);
    float c = dot(v, v) - R * R;
    float det2 = b * b - c;
    if (det2 < 0.)
        return -1.;
    float det = sqrt(det2);
    float t1 = -b - det, t2 = -b + det;
    return (t1 >= 0.) ? t1 : t2;
}

// this can be explained:
// http://www.scratchapixel.com/lessons/3d-advanced-lessons/simulating-the-colors-of-the-sky/atmospheric-scattering/
void scatter(vec3 o, vec3 d, out vec3 col, out float scat, vec3 Ds)
{
    float L = escape(o, d, Ra);
    float mu = dot(d, Ds);
    float opmu2 = 1.0 + mu * mu;
    float phaseR = .0596831 * opmu2;
    float phaseM = .1193662 * (1. - g2) * opmu2 / ((2. + g2) * pow(1. + g2 - 2. * g * mu, 1.5));

    float depthR = 0., depthM = 0.;
    vec3 R = vec3(0.), M = vec3(0.);

    float dl = L / float(steps2);
    for (int i = 0; i < steps2; ++i)
    {
        float l = float(i) * dl;
        vec3 p = o + d * l;

        float dR, dM;
        densities(p, dR, dM);
        dR *= dl;
        dM *= dl;
        depthR += dR;
        depthM += dM;

        float Ls = escape(p, Ds, Ra);
        if (Ls > 0.)
        {
            float dls = Ls / float(stepss);
            float depthRs = 0., depthMs = 0.;
            for (int j = 0; j < stepss; ++j)
            {
                float ls = float(j) * dls;
                vec3 ps = p + Ds * ls;
                float dRs, dMs;
                densities(ps, dRs, dMs);
                depthRs += dRs * dls;
                depthMs += dMs * dls;
            }

            vec3 A = exp(-(bR * (depthRs + depthR) + bM * (depthMs + depthM)));
            R += A * dR;
            M += A * dM;
        }
    }

    col = I * (R * bR * phaseR + M * bM * phaseM);
    scat = 1.0 - clamp(depthM * 1e-5, 0., 1.);
}

//--------------------------------------------------------------------------
// ray casting

vec4 generate(in vec3 view, in vec3 sunpos)
{

    // moon
    float att = 1.0;
    float staratt = 0.0;
    if (sunpos.y < -0.20)
    {
        sunpos = -sunpos;
        att = 0.01;
        staratt = 1.0;
    }

    vec3 O = vec3(0., height, 0.);

    vec3 D = view;

    if (D.y <= -0.15)
    {
        D.y = -0.3 - D.y;
    }

    vec3 Ds = normalize(sunpos);
    float scat = 0.0;
    vec3 color = vec3(0.);
    scatter(O, clamp(D, 0, 1), color, scat, Ds);
    color *= att;

    float env = 1.0;
    return (vec4(env * pow(color, vec3(.7)), 1.0));
}

///////////////////////

void main()
{

    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);

    texCoord = Position.xy / OutSize;
    oneTexel = 1.0 / OutSize;

    // simply decoding all the control data and constructing the sunDir, ProjMat,
    // ModelViewMat

    // ProjMat constructed assuming no translation or rotation matrices applied
    // (aka no view bobbing).
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)),
                        decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), 0.0, 0.0,
                        decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz),
                        tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)),
                        decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0);

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz),
                            decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz),
                            decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz),
                            decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz),
                            decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz),
                            decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz),
                            decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0, 0.0, 0.0, 0.0, 1.0);

    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));

    overworld = vec4((texture(DiffuseSampler, start + 28.0 * inc))).r;
    end = vec4((texture(DiffuseSampler, start + 29.0 * inc))).r;

    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);
    if (overworld != 1.0)
    {
        near = 12;
        far = 256;
    }
    // zMults = vec3(1.0 / (far * near), far + near, far - near);

    vec3 sunDir =
        normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz),
                                               decodeFloat(texture(DiffuseSampler, start + inc).xyz),
                                               decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz), 1.0))
                      .xyz);

    mat4 gbufferModelViewInverse = inverse(mat4(ModeViewMat));
    // wgbufferModelViewInverse = inverse(ProjMat * ModeViewMat);

    gbufferModelView = (ModeViewMat);
    wgbufferModelView = (ProjMat * ModeViewMat);

    gbufferProjection = ProjMat;
    gbufferProjectionInverse = inverse(ProjMat);
    ///////////////////

    ////////////////////////////////////////////////

    bool time8 = sunDir.y > 0;
    float time4 = map(sunDir.x, -1, +1, 0, 1);
    float time5 = mix(12000, 0, time4);
    float time6 = mix(24000, 12000, 1 - time4);
    float time7 = mix(time6, time5, time8);

    float worldTime = time7;

    const vec2 sunRotationData =
        vec2(cos(sunPathRotation * 0.01745329251994),
             -sin(sunPathRotation * 0.01745329251994)); // radians() is not a const function on some
                                                        // drivers, so multiply by pi/180 manually.

    // minecraft's native calculateCelestialAngle() function, ported to GLSL.
    float ang = fract(worldTime / 24000.0 - 0.25);
    ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) *
          6.28318530717959; // 0-2pi, rolls over from 2pi to 0 at noon.

    vec3 sunDirTemp = vec3(-sin(ang), cos(ang) * sunRotationData);
    sunDir = normalize(vec3(sunDirTemp.x, sunDir.y, sunDirTemp.z));

    rainStrength = 1 - rain.r;
    vec3 sunDir2 = sunDir;
    sunPosition = mat3(gbufferModelView) * sunDir2;
    sunPosition3 = sunDir2;

    vec3 upPosition = vec3(gbufferModelView[1].xyz);
    const vec3 cameraPosition = vec3(0.0);

    float normSunVec =
        sqrt(sunPosition.x * sunPosition.x + sunPosition.y * sunPosition.y + sunPosition.z * sunPosition.z);
    float normUpVec = sqrt(upPosition.x * upPosition.x + upPosition.y * upPosition.y + upPosition.z * upPosition.z);

    float sunPosX = sunPosition.x / normSunVec;
    float sunPosY = sunPosition.y / normSunVec;
    float sunPosZ = sunPosition.z / normSunVec;
    vec3 sunVec2 = vec3(sunPosX, sunPosY, sunPosZ);

    float upPosX = upPosition.x / normUpVec;
    float upPosY = upPosition.y / normUpVec;
    float upPosZ = upPosition.z / normUpVec;

    sunElevation = sunPosX * upPosX + sunPosY * upPosY + sunPosZ * upPosZ;

    float angSkyNight = -((pi * 0.5128205128205128 - facos(-sunElevation * 0.95 + 0.05)) / 1.5);
    float angSky = -((pi * 0.5128205128205128 - facos(sunElevation * 0.95 + 0.05)) / 1.5);

    float fading = clamp(sunElevation + 0.095, 0.0, 0.08) / 0.08;
    float fading2 = clamp(-sunElevation + 0.095, 0.0, 0.08) / 0.08;
    skyIntensity = max(0., 1.0 - exp(angSky)) * (1.0 - rainStrength * 0.4) * pow(fading, 5.0);

    skyIntensityNight = max(0., 1.0 - exp(angSkyNight)) * (1.0 - rainStrength * 0.4) * pow(fading2, 5.0);
    sunVec = mix(sunVec2, -sunVec2, clamp(skyIntensityNight * 3, 0, 1));
    sunPosition2 = -sunPosition3 * clamp(skyIntensityNight, 0, 1);
    sunPosition2 += sunPosition3 * clamp(skyIntensity, 0, 1);
    sunPosition2 = normalize(sunPosition2);

    float angMoon = -((pi * 0.5128205128205128 - facos(-sunElevation * 1.065 - 0.065)) / 1.5);
    float angSun = -((pi * 0.5128205128205128 - facos(sunElevation * 1.065 - 0.065)) / 1.5);

    float sunElev = pow(clamp(1.0 - sunElevation, 0.0, 1.0), 4.0) * 1.8;
    const float sunlightR0 = 1.0;
    float sunlightG0 = (0.89 * exp(-sunElev * 0.57)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;
    float sunlightB0 = (0.8 * exp(-sunElev * 1.4)) * (1.0 - rainStrength * 0.3) + rainStrength * 0.3;

    float sunlightR = sunlightR0 / (sunlightR0 + sunlightG0 + sunlightB0);
    float sunlightG = sunlightG0 / (sunlightR0 + sunlightG0 + sunlightB0);
    float sunlightB = sunlightB0 / (sunlightR0 + sunlightG0 + sunlightB0);
    nsunColor = vec3(sunlightR, sunlightG, sunlightB);

    float skyIntensity = max(0., 1.0 - exp(angSky)) * (1.0 - rainStrength * 0.4) * pow(fading, 5.0);
    float moonIntensity = max(0., 1.0 - exp(angMoon));
    float sunIntensity = max(0., 1.0 - exp(angSun));
    vec3 sunVec = vec3(sunPosX, sunPosY, sunPosZ);
    moonIntensity = max(0., 1.0 - exp(angMoon));

    float avgEyeIntensity = ((sunIntensity * 120. + moonIntensity * 4.) + skyIntensity * 230. + skyIntensityNight * 4.);
    float exposure = 0.18 / log(max(avgEyeIntensity * 0.16 + 1.0, 1.13)) * 0.3 * log(2.0);
    const float sunAmount = 27.0 * 1.5;
    float lightSign = clamp(sunIntensity * pow(10., 35.), 0., 1.);
    vec4 lightCol = vec4((sunlightR * 3. * sunAmount * sunIntensity + 0.16 / 5. - 0.16 / 5. * lightSign) *
                             (1.0 - rainStrength * 0.95) * 7.84 * exposure,
                         7.84 * (sunlightG * 3. * sunAmount * sunIntensity + 0.24 / 5. - 0.24 / 5. * lightSign) *
                             (1.0 - rainStrength * 0.95) * exposure,
                         7.84 * (sunlightB * 3. * sunAmount * sunIntensity + 0.36 / 5. - 0.36 / 5. * lightSign) *
                             (1.0 - rainStrength * 0.95) * exposure,
                         lightSign * 2.0 - 1.0);

    lightCol.xyz = toLinear(generate(sunPosition3.xyz, sunPosition3).xyz);

    vec3 lightSourceColor = lightCol.rgb;

    float sunVis = clamp(sunElevation, 0.0, 0.05) / 0.05 * clamp(sunElevation, 0.0, 0.05) / 0.05;
    float lightDir = float(sunVis >= 1e-5) * 2.0 - 1.0;
    skys = 1.8 / log2(max(avgEyeIntensity * 0.16 + 1.0, 1.13)) * 0.3;
    cloudy = decodeFloat24((texture(noisetex, start + 51.0 * inc).rgb));
    vec2 planetSphere = vec2(0.0);
    vec3 skyAbsorb = vec3(0.0);
    vec3 absorb = vec3(0.0);
    vec2 tempOffsets = R2_samples(int(Time) % 10000);
    vec3 sunVec3 = normalize(mat3(gbufferModelViewInverse) * sunPosition);

    skyAbsorb = vec3(0.0);

    suncol = lightSourceColor.rgb;
    skyAbsorb = vec3(0.0);
    ///////////////////////////
    ambientUp = vec3(0.0);
    ambientDown = vec3(0.0);
    ambientLeft = vec3(0.0);
    ambientRight = vec3(0.0);
    ambientB = vec3(0.0);
    ambientF = vec3(0.0);

    int maxIT = 20;
    for (int i = 0; i < maxIT; i++)
    {
        vec2 ij = R2_samples((int(Time) % 1000) * maxIT + i);
        
        vec3 pos = normalize(rodSample(ij)+float(i/maxIT)+0.1);
    
        vec3 samplee =
            skyLut2(pos.xyz, sunDir2, pos.y, rainStrength * 0.25, nsunColor, skyIntensity, skyIntensityNight) / maxIT;
        samplee = ( 2.2*max(vec3(0.0), generate(pos.xyz, sunPosition3).xyz * 1.0)) / maxIT;
        ambientUp += samplee * (pos.y + abs(pos.x) / 7. + abs(pos.z) / 7.);
        ambientLeft += samplee * (clamp(-pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
        ambientRight += samplee * (clamp(pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
        ambientB += samplee * (clamp(pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
        ambientF += samplee * (clamp(-pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
        ambientDown += samplee * (clamp(pos.y / 6., 0.0, 1.0) + abs(pos.x) / 7. + abs(pos.z) / 7.);
    }
    	float dSun = 0.03;
    	vec3 modSunVec = sunPosition3*(1.0-dSun)+vec3(0.0,dSun,0.0);
	vec3 modSunVec2 = sunPosition3*(1.0-dSun)+vec3(0.0,dSun,0.0);
	if (modSunVec2.y > modSunVec.y) modSunVec = modSunVec2;
	vec3 sunColorCloud =  toLinear(generate(modSunVec.xyz, sunPosition3).xyz);

	//Fake bounced sunlight
	vec3 bouncedSun = lightSourceColor*1.0/5.0*0.5*clamp(lightDir*sunPosition3.y,0.0,1.0)*clamp(lightDir*sunPosition3.y,0.0,1.0);
	vec3 cloudAmbientSun = (sunColorCloud)*0.007;
	vec3 cloudAmbientMoon = (vec3(0.0))*0.007;
	ambientUp += bouncedSun*clamp(-lightDir*sunVec.y+4.,0.,4.0) + cloudAmbientSun*clamp(sunVec.y+2.,0.,4.0) + cloudAmbientMoon*clamp(-sunVec.y+2.,0.,4.0);
	ambientLeft += bouncedSun*clamp(lightDir*sunVec.x+4.,0.0,4.) + cloudAmbientSun*clamp(-sunVec.x+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(sunVec.x+2.,0.0,4.)*0.7;
	ambientRight += bouncedSun*clamp(-lightDir*sunVec.x+4.,0.0,4.) + cloudAmbientSun*clamp(sunVec.x+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(-sunVec.x+2.,0.0,4.)*0.7;
	ambientB += bouncedSun*clamp(-lightDir*sunVec.z+4.,0.0,4.) + cloudAmbientSun*clamp(sunVec.z+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(-sunVec.z+2.,0.0,4.)*0.7;
	ambientF += bouncedSun*clamp(lightDir*sunVec.z+4.,0.0,4.) + cloudAmbientSun*clamp(-sunVec.z+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(sunVec.z+2.,0.0,4.)*0.7;
	ambientDown += bouncedSun*clamp(lightDir*sunVec.y+4.,0.0,4.)*0.7 + cloudAmbientSun*clamp(-sunVec.y+2.,0.0,4.)*0.5 + cloudAmbientMoon*clamp(sunVec.y+2.,0.0,4.)*0.5;
	//avgSky += bouncedSun*5.;


    gl_Position = vec4(outPos.xy, 0.2, 1.0);
}
