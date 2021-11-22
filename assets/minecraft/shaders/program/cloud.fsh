#version 150

uniform sampler2D noisetex;
uniform sampler2D TranslucentDepthSampler;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;
in vec3 avgSky;
in vec3 sc;
in mat4 gbufferProjectionInverse;

in mat4 gbufferModelViewInverse;
in float sunElevation;
in float rainStrength;
in float cloudy;
in vec3 sunVec;

in float ambientMult;
in vec3 skyCol0;

in vec3 sunPosition3;
out vec4 fragColor;
#define CLOUDS_QUALITY 0.5
#define VOLUMETRIC_CLOUDS

float sqr(float x)
{
    return x * x;
}
float pow3(float x)
{
    return sqr(x) * x;
}
float pow4(float x)
{
    return sqr(x) * sqr(x);
}
float pow5(float x)
{
    return pow4(x) * x;
}
float pow6(float x)
{
    return pow5(x) * x;
}
float pow8(float x)
{
    return pow4(x) * pow4(x);
}
float pow16(float x)
{
    return pow8(x) * pow8(x);
}
float pow32(float x)
{
    return pow16(x) * pow16(x);
}

////////////////////////////////////////////////

const float sky_planetRadius = 6731e3;

const float PI = 3.141592;
vec3 cameraPosition = vec3(0, abs((cloudy)), 0);
const float cloud_height = 1500.;
const float maxHeight = 1650.;
int maxIT_clouds = 15;
const float cdensity = 0.2;

///////////////////////////

// Cloud without 3D noise, is used to exit early lighting calculations if there
// is no cloud
float cloudCov(in vec3 pos, vec3 samplePos)
{
    float mult = max(pos.y - 2000.0, 0.0) * 0.0005;
    float mult2 = max(-pos.y + 2000.0, 0.0) * 0.002;
    float coverage = clamp(texture(noisetex, fract(samplePos.xz * 0.00008)).x + 0.5 * rainStrength, 0.0, 1.0);
    float cloud = sqr(coverage) - pow3(mult) * 3.0 - sqr(mult2);
    return max(cloud, 0.0);
}
// Erode cloud with 3d Perlin-worley noise, actual cloud value
vec3 LinearTosRGB(in vec3 color)
{
    vec3 x = color * 12.92f;
    vec3 y = 1.055f * pow(clamp(color, 0.0, 1.0), vec3(1.0f / 2.4f)) - 0.055f;

    vec3 clr = color;
    clr.r = color.r < 0.0031308f ? x.r : y.r;
    clr.g = color.g < 0.0031308f ? x.g : y.g;
    clr.b = color.b < 0.0031308f ? x.b : y.b;

    return clr;
}
float cloudVol(in vec3 pos, in vec3 samplePos, in float cov)
{
    float mult2 = (pos.y - 1500) * 0.0004 + rainStrength * 0.4;

    float cloud = clamp(cov - 0.11 * (0.2 + mult2), 0.0, 1.0);
    return cloud;
}
const float pi = 3.141592653589793238462643383279502884197169;

const float pidiv = 0.31830988618; // 1/pi

// Mie phase function
float phaseg(float x, float g)
{
    float gg = sqr(g);
    return ((-0.25 * gg + 0.25) * pidiv) * pow(-2.0 * g * x + gg + 1.0, -1.5);
}

vec4 renderClouds(vec3 fragpositi, vec3 color, float dither, vec3 sunColor, vec3 moonColor, vec3 avgAmbient)
{
    vec4 fragposition = gbufferModelViewInverse * vec4(fragpositi, 1.0);

    vec3 worldV = normalize(fragposition.rgb);
    float VdotU = worldV.y;
    maxIT_clouds = int(clamp(maxIT_clouds / sqrt(VdotU), 0.0, maxIT_clouds));

    vec3 dV_view = worldV;

    vec3 progress_view = dV_view * dither + cameraPosition;

    float total_extinction = 1.0;

    worldV = normalize(worldV) * 300000. + cameraPosition; // makes max cloud distance not dependant of
                                                           // render distance

    dV_view = normalize(dV_view);

    // setup ray to start at the start of the cloud plane and end at the end of
    // the cloud plane
    dV_view *= max(maxHeight - cloud_height, 0.0) / dV_view.y / maxIT_clouds;

    vec3 startOffset = dV_view * clamp(dither, 0.0, 1.0);
    progress_view = startOffset + cameraPosition + dV_view * (cloud_height - cameraPosition.y) / (dV_view.y);

    float mult = length(dV_view);

    color = vec3(0.0);
    float SdotV = dot(sunVec, normalize(fragpositi));
    // fake multiple scattering approx 1 (from horizon zero down clouds)
    float mieDay = max(phaseg(SdotV, 0.22), phaseg(SdotV, 0.2));
    float mieNight = max(phaseg(-SdotV, 0.22), phaseg(-SdotV, 0.2));

    float shadowStep = 240.;
    vec3 sunContribution = mieDay * sunColor * pi;
    vec3 moonContribution = mieNight * moonColor * pi;
    float powderMulMoon = 0.5 + SdotV * 0.5;
    float powderMulSun = 1.0 - powderMulMoon;
    for (int i = 0; i < maxIT_clouds; i++)
    {
        vec3 curvedPos = progress_view;
        vec2 xz = progress_view.xz - cameraPosition.xz;

        curvedPos.y -= sqrt((sky_planetRadius * sky_planetRadius) - dot(xz, xz)) - sky_planetRadius;
        vec3 samplePos = curvedPos * vec3(1.0, 0.03125, 1.0) * 0.25 + (sunElevation * 1000);

        float coverageSP = cloudCov(curvedPos, samplePos);
        if (coverageSP > 0.00)
        {
            float cloud = cloudVol(curvedPos, samplePos, coverageSP);
            if (cloud > 0.0005)
            {
                float mu = cloud * cdensity;

                // fake multiple scattering approx 2 and 3 (from horizon zero down clouds)
                float sunShadow = max(0, exp(-0.25 * 0) * 0.7) * (1.0 - powderMulSun * exp(-mu * mult * 2.0));
                float moonShadow = max(0, exp(-0.25 * 0) * 0.7) * (1.0 - powderMulMoon * exp(-mu * mult * 2.0));
                float h = 0.35 - 0.35 * clamp((progress_view.y - 1500.0) * 0.00025, 0.0, 1.0);
                float ambientPowder = 1.0 - exp(-mu * mult);
                vec3 S = vec3(sunContribution * sunShadow + moonShadow * moonContribution + skyCol0 * ambientPowder);

                vec3 Sint = (S - S * exp2(-mult * mu)) / (mu);
                color += mu * Sint * total_extinction;
                total_extinction *= exp2(-mu * mult);

                if (total_extinction < 1 / 250.)
                    break;
            }
        }

        progress_view += dV_view;
    }

    float cosY = normalize(dV_view).y;

    color.rgb = mix(color.rgb * vec3(0.2, 0.21, 0.21), color.rgb, 1 - rainStrength);
    return mix(vec4(color, clamp(total_extinction, 0.0, 1.0)), vec4(0.0, 0.0, 0.0, 1.0),
               1 - smoothstep(0.02, 0.20, cosY));
}
float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * (Time*1000));
}

vec4 backProject(vec4 vec)
{
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
}

// simplified version of joeedh's https://www.shadertoy.com/view/Md3GWf
// see also https://www.shadertoy.com/view/MdtGD7

// --- checkerboard noise : to decorelate the pattern between size x size tiles

// simple x-y decorrelated noise seems enough
#define stepnoise0(p, size) rnd(floor(p / size) * size)
#define rnd(U) fract(sin(1e3 * (U)*mat2(1, -7.131, 12.9898, 1.233)) * 43758.5453)

//   joeedh's original noise (cleaned-up)
vec2 stepnoise(vec2 p, float size)
{
    p = floor((p + 10.) / size) * size; // is p+10. useful ?
    p = fract(p * .1) + 1. + p * vec2(2, 3) / 1e4;
    p = fract(1e5 / (.1 * p.x * (p.y + vec2(0, 1)) + 1.));
    p = fract(1e5 / (p * vec2(.1234, 2.35) + 1.));
    return p;
}

// --- stippling mask  : regular stippling + per-tile random offset +
// tone-mapping

#define SEED1 1.705
#define DMUL 8.12235325 // are exact DMUL and -.5 important ?

float mask(vec2 p)
{
    p += (stepnoise0(p, 5.5) - .5) * DMUL;                 // bias [-2,2] per tile otherwise too regular
    float f = fract(p.x * SEED1 + p.y / (SEED1 + .15555)); //  weights: 1.705 , 0.5375

    // return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the
    // image, see tests ) returned value in [0,37.2] , but < 0.57 with P=50%

    return (pow(f, 150.) + 1.3 * f) * 0.43478260869; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%
}
float dither5x3()
{
    const int ditherPattern[15] = int[15](9, 3, 7, 12, 0, 11, 5, 1, 14, 8, 2, 13, 10, 4, 6);

    vec2 position = floor(mod(vec2(texCoord.s * ScreenSize.x, texCoord.t * ScreenSize.y), vec2(5.0, 3.0)));

    int dither = ditherPattern[int(position.x) + int(position.y) * 5];

    return float(dither) * 0.0666666666666667f;
}
#define g(a) (-4. * a.x * a.y + 3. * a.x + a.y * 2.)

float bayer16x16(vec2 p)
{
    vec2 m0 = vec2(mod(floor(p * 0.125), 2.));
    vec2 m1 = vec2(mod(floor(p * 0.25), 2.));
    vec2 m2 = vec2(mod(floor(p * 0.5), 2.));
    vec2 m3 = vec2(mod(floor(p), 2.));

    return (g(m0) + g(m1) * 4.0 + g(m2) * 16.0 + g(m3) * 64.0) * 0.003921568627451;
}
// Dithering from Jodie
float bayer2(vec2 a)
{
    a = floor(a);
    return fract(dot(a, vec2(.5, a.y * .75)));
}

#define bayer4(a) (bayer2(.5 * (a)) * .25 + bayer2(a))
#define bayer8(a) (bayer4(.5 * (a)) * .25 + bayer2(a))
#define bayer16(a) (bayer8(.5 * (a)) * .25 + bayer2(a))
#define bayer32(a) (bayer16(.5 * (a)) * .25 + bayer2(a))
#define bayer64(a) (bayer32(.5 * (a)) * .25 + bayer2(a))
#define bayer128(a) (bayer64(.5 * (a)) * .25 + bayer2(a))

float dither64 = bayer64(gl_FragCoord.xy);

vec3 reinhard(vec3 x)
{
    x *= 1.66;
    return x / (1.0 + x);
}

#define PI 3.141592

vec3 normVec(vec3 vec)
{
    return vec * inversesqrt(dot(vec, vec));
}
vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    color = pow(color, vec3(0.45454545454));
    return color;
}

//

#define PI 3.141592

////////////////////

const float M_PI = 3.1415926535;
const float DEGRAD = M_PI / 180.0;

float height = 500.0; // viewer height

// rendering quality
const int steps2 = 16; // 16 is fast, 128 or 256 is extreme high
const int stepss = 8;  // 8 is fast, 16 or 32 is high

float haze = rainStrength; // 0.2

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

const float R0 = 6360e3;      // planet radius
const float Ra = 6420e3;      // atmosphere radius
vec3 C = vec3(0.0, -R0, 0.0); // planet center

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
        return -1.0;
    float det = sqrt(det2);
    float t1 = -b - det, t2 = -b + det;
    return (t1 >= 0.0) ? t1 : t2;
}
// No intersection if returned y component is < 0.0
vec2 rsi(vec3 position, vec3 direction, float radius)
{
    float PoD = dot(position, direction);
    float radiusSquared = radius * radius;

    float delta = PoD * PoD + radiusSquared - dot(position, position);
    if (delta < 0.0)
        return vec2(-1.0);
    delta = sqrt(delta);

    return -PoD + vec2(-delta, delta);
}
// this can be explained:
// http://www.scratchapixel.com/lessons/3d-advanced-lessons/simulating-the-colors-of-the-sky/atmospheric-scattering/
void scatter(vec3 o, vec3 d, out vec3 col, out float scat, vec3 Ds)
{
    float L = escape(o, d, Ra);
    float mu = dot(d, Ds);
    float opmu2 = 1.0 + mu * mu;
    float phaseR = 0.0596831 * opmu2;
    float phaseM = 0.1193662 * (1.0 - g2) * opmu2 / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));

    float depthR = 0.0, depthM = 0.0;
    vec3 R = vec3(0.0), M = vec3(0.0);

    float dl = L / float(steps2);
    for (int i = 0; i < steps2; ++i)
    {
        float l = float(i) * dl;
        vec3 p = d * l + o;

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
                vec3 ps = Ds * ls * p;
                float dRs, dMs;
                densities(ps, dRs, dMs);
                depthRs += dRs * dls;
                depthMs += dMs * dls;
            }

            vec3 A = exp(-(bR * (depthRs + depthR) + (depthMs + depthM) * bM));
            R += A * dR;
            M += A * dM;
        }
    }

    col = I * (R * bR * phaseR + M * bM * phaseM);
    scat = 1.0 - clamp(depthM * 1e-5, 0.0, 1.0);
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
    }

    vec3 O = vec3(0.0, height, 0.0);

    vec3 D = view;

    if (D.y <= -0.15)
    {
        D.y = -0.3 - D.y;
    }

    vec3 Ds = normalize(sunpos);
    float scat = 0.0;
    vec3 color = vec3(0.);
    scatter(O, clamp(D, 0.0, 1.0), color, scat, Ds);
    color *= att;

    float env = 1.0;
    return (vec4(env * pow(color, vec3(.7)), 1.0));
}
vec3 toLinear(vec3 sRGB)
{
    return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}
void main()
{
    // vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
    float noise = R2_dither();

    float depth = 1.0;

    vec2 halfResTC = vec2(gl_FragCoord.xy / CLOUDS_QUALITY);
#ifdef VOLUMETRIC_CLOUDS
    bool doClouds = false;
    for (int i = 0; i < floor(1.0 / CLOUDS_QUALITY) + 1.0; i++)
    {
        for (int j = 0; j < floor(1.0 / CLOUDS_QUALITY) + 1.0; j++)
        {
            if (texelFetch(TranslucentDepthSampler, ivec2(halfResTC) + ivec2(i, j), 0).x >= 1.0)
                doClouds = true;
        }
    }

    if (doClouds)
    {
        vec3 sc = sc * (1 - ((rainStrength)*0.5));
        vec3 screenPos = vec3(halfResTC * oneTexel, depth);
        vec3 clipPos = screenPos * 2.0 - 1.0;
        vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
        vec3 viewPos = tmp.xyz / tmp.w;

        vec3 p3 = mat3(gbufferModelViewInverse) * viewPos;
        vec3 view = normVec(p3);

        vec3 atmosphere = toLinear(generate(view.xyz, sunPosition3).xyz) + (noise / 255);

        vec4 cloud = vec4(0.0, 0.0, 0.0, 1.0);
        if (view.y > 0.)
        {
            cloud = renderClouds(viewPos, avgSky, noise, sc/3, sc/3, avgSky/3).rgba;
        }

        fragColor.rgb = atmosphere.xyz * cloud.a + (cloud.rgb);

        fragColor.rgb = reinhard(fragColor.rgb);

        fragColor.a = 1.0;
    }
    else
        fragColor = vec4(0, 0, 0, 1.0);

#else
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
#endif
}
