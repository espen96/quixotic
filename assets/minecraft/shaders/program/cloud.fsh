#version 150

uniform sampler2D noisetex;
uniform sampler2D DiffuseDepthSampler;
uniform vec2 ScreenSize;
uniform float Time;

in vec2 texCoord;
in vec2 oneTexel;
in vec3 avgSky;
in vec3 sc;
in mat4 gbufferProjectionInverse;
in float skyIntensityNight;

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

#define CloudSize                                                                                                      \
    12500 //(12500) When using low values make sure to also lower the cloud thickness aswell. The highest values will
          // create clouds the size of continents, if the cloud is not visible at these values it is because it is
          // currently too far away, increase the speed. Higher values will require a higher quality to minimise
          // aliasing.
#define CloudDensity 0.20 //(0.05)

#define cloudMieG                                                                                                      \
    0.55 // Values close to 1 will create a strong peak of luminance around the sun and weak elsewhere, values close to
         // 0 means uniform fog.
#define cloudMieG2                                                                                                     \
    0.2 // Multiple scattering approximation. Values close to 1 will create a strong peak of luminance around the sun
        // and weak elsewhere, values close to 0 means uniform fog.

float frameTimeCounter = (sunElevation * 1000);

float cdensity = CloudDensity;

// Cloud without 3D noise, is used to exit early lighting calculations if there is no cloud
float cloudCov(in vec3 pos, vec3 samplePos)
{
    float mult = max(pos.y - 2000.0, 0.0) / 2000.0;
    float mult2 = max(-pos.y + 2000.0, 0.0) / 500.0;
    float coverage = clamp(texture(noisetex, fract(samplePos.xz / CloudSize)).r + 0.2 * rainStrength - 0.2, 0.0, 1.0) /
                     (0.2 * rainStrength + 0.8);

    float cloud = (coverage * coverage) - 3.0 * (mult * mult * mult) - (mult2 * mult2);

    return max(cloud, 0.0);
}

// Mie phase function
float phaseg(float x, float g)
{
    float gg = g * g;

    return (gg * -0.25 / 3.14 + 0.25 / 3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}

vec4 renderClouds(vec3 fragpositi, vec3 color, float dither, vec3 sunColor, vec3 moonColor, vec3 avgAmbient)
{

    float SdotU = dot(normalize(fragpositi.xyz), sunVec);

    // project pixel position into projected shadowmap space
    vec4 fragposition = gbufferModelViewInverse * vec4(fragpositi, 1.0);

    vec3 worldV = normalize(fragposition.rgb);
    float VdotU = worldV.y;
    maxIT_clouds = int(clamp(maxIT_clouds / sqrt(VdotU), 0.0, maxIT_clouds * 1.0));
    // worldV.y -= -length(worldV.xz)/sqrt(-length(worldV.xz/6731e3)*length(worldV.xz/6731e3)+6731e3); // Simulates the
    // Earths curvature

    // project view origin into projected shadowmap space
    vec4 start = (gbufferModelViewInverse * vec4(0.0, 0.0, 0., 1.));
    vec3 dV_view = worldV;

    vec3 progress_view = dV_view * dither + cameraPosition;

    float total_extinction = 1.0;

    float distW = length(worldV);
    worldV = normalize(worldV) * 300000. + cameraPosition; // makes max cloud distance not dependant of render distance
    dV_view = normalize(dV_view);

    // setup ray to start at the start of the cloud plane and end at the end of the cloud plane
    dV_view *= max(maxHeight - cloud_height, 0.0) / dV_view.y / maxIT_clouds;
    vec3 startOffset = dV_view * dither;

    progress_view = startOffset + cameraPosition + dV_view * (cloud_height - cameraPosition.y) / (dV_view.y);

    if (worldV.y < cloud_height)
        return vec4(0., 0., 0., 1.); // don't trace if no intersection is possible

    float mult = length(dV_view);

    color = vec3(0.0);

    float SdotV = dot(sunVec, normalize(fragpositi));

    // fake multiple scattering approx 1 (from horizon zero down clouds)
    float mieDay = max(phaseg(SdotV, cloudMieG), phaseg(SdotV, cloudMieG2));
    float mieNight = max(phaseg(-SdotV, cloudMieG), phaseg(-SdotV, cloudMieG2));

    vec3 sunContribution = mieDay * sunColor * 3.14;
    vec3 moonContribution = mieNight * moonColor * 3.14;
    float ambientMult = exp(-(1 + 0.24 + 0.8 * rainStrength) * cdensity * 75.0);
    vec3 skyCol0 = avgAmbient * ambientMult;

    for (int i = 0; i < maxIT_clouds; i++)
    {
        vec3 curvedPos = progress_view;
        vec2 xz = progress_view.xz - cameraPosition.xz;
        curvedPos.y -= sqrt(pow(6731e3, 2.0) - dot(xz, xz)) - 6731e3;
        vec3 samplePos = curvedPos * vec3(1.0, 0.03125, 1.0) / 4 + frameTimeCounter * vec3(0.5, 0.0, 0.5);
        float coverageSP = cloudCov(curvedPos, samplePos);
        if (coverageSP > 0.00)
        {
            float cloud = coverageSP;

            float mu = cloud * cdensity;

            // fake multiple scattering approx 2  (from horizon zero down clouds)
            float h = 0.5 - 0.5 * clamp((progress_view.y - 1500) / 4000, 0.0, 1.0);
            float powder = 1.0 - exp(-mu * mult);
            float sunShadow = max(exp(-0.0), 0.7 * exp(-0.25 * 0.0)) * mix(1.0, powder, h);
            float moonShadow = max(exp2(-0.0), 0.7 * exp(-0.25 * 0.0)) * mix(1.0, powder, h);
            float ambientPowder = mix(1.0, powder, h * ambientMult);
            vec3 S = vec3(sunContribution * sunShadow + moonShadow * moonContribution + skyCol0 * ambientPowder);

            vec3 Sint = (S - S * exp(-mult * mu)) / (mu);
            color += mu * Sint * total_extinction;
            total_extinction *= exp(-mu * mult);

            if (total_extinction < 0.004)
                break;
        }

        progress_view += dV_view;
    }

    // high altitude clouds
    progress_view = progress_view + (5500.0 - progress_view.y) * dV_view / dV_view.y;
    mult = 400.0 * inversesqrt(abs(normalize(dV_view).y));

    float cosY = normalize(dV_view).y;

    return mix(vec4(color, clamp((251 * total_extinction - 1) / 250, 0.0, 1.0)), vec4(0.0, 0.0, 0.0, 1.0),
               1 - smoothstep(0.02, 0.50, cosY));
}

float R2_dither()
{
    vec2 alpha = vec2(0.75487765, 0.56984026);
    return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0 / 1.6180339887 * (Time * 1000));
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

////////////////////

#define HW_PERFORMANCE 0

// first, lets define some constants to use (planet radius, position, and scattering coefficients)
#define PLANET_POS vec3(0.0) /* the position of the planet */
#define PLANET_RADIUS 6371e3 /* radius of the planet */
#define ATMOS_RADIUS 6471e3 /* radius of the atmosphere */
// scattering coeffs
#define RAY_BETA vec3(5.5e-6, 13.0e-6, 22.4e-6) /* rayleigh, affects the color of the sky */
#define MIE_BETA vec3(21e-6) /* mie, affects the color of the blob around the sun */
#define AMBIENT_BETA vec3(0.0) /* ambient, affects the scattering color when there is no lighting from the sun */
#define ABSORPTION_BETA                                                                                                \
    vec3(2.04e-5, 4.97e-5, 1.95e-6) /* what color gets absorbed by the atmosphere (Due to things like ozone) */
#define G 0.7 /* mie scattering direction, or how big the blob around the sun is */
// and the heights (how far to go up before the scattering has no effect)
#define HEIGHT_RAY 8e3 /* rayleigh height */
#define HEIGHT_MIE 1.2e3 /* and mie */
#define HEIGHT_ABSORPTION 30e3 /* at what height the absorption is at it's maximum */
#define ABSORPTION_FALLOFF                                                                                             \
    4e3 /* how much the absorption decreases the further away it gets from the maximum height                          \
         */
// and the steps (more looks better, but is slower)
// the primary step has the most effect on looks
#if HW_PERFORMANCE == 0
// edit these if you are on mobile
#define PRIMARY_STEPS 12
#define LIGHT_STEPS 4
#else
// and these on desktop
#define PRIMARY_STEPS 32 /* primary steps, affects quality the most */
#define LIGHT_STEPS 8 /* light steps, how much steps in the light direction are taken */
#endif

// first off, we'll need to calculate the optical depth
// this is effectively how much air is in the way from the camera ray to the object
// we can calculate this as the integral over beta * exp(-scale_height * (sqrt(t^2 + 2bt + c) - planet_radius)), from t
// = 0 to infinity with t as the distance from the start position, b = dot(ray direction, ray start) and c = dot(ray
// start, ray start) - planet radius * planet_radius due to the multiplication by constant rule, we can keep beta
// outside of the integral we can do it to infinity, because if we calculate the same at the object pos and subtract it
// from the one at the camera pos, we get the same result this is also needed because we can't get the exact integral of
// this, so an approximation is needed

// TODO: maybe use approximation with height and zenith cos angle instead
// We first need to get the density of the atmosphere at a certain point
// we can calculate this along a point on the ray
// for that, we need the height along the ray first
// which is h(x, a, t) = sqrt((start_height + t * a)^2 + (t*sin(cos^-1(a)))^2)
// if we then put that inside an integral, we get exp(-g * (h(x, a, t) - p)) dt
// this has a scale factor of exp(0.7 * p), NO WRONG
// for either end it's ((2 - ) exp(-g*x)) / g
float get_optical_depth(float b, float c, float inv_scale_height, float planet_radius)
{

    // here's the derivation process:
    // plot the formula
    // look at what it looks like from
    // for the planet radius this is exp(x)

    // this means the integral can be reduced to 0-infinity exp(-sqrt(t^2+2bt+c)) dt * scaling factor
    // with the scaling factor being exp(scale_height) * ...?

    // if we graph this, it comes close to 1 / b + 2
    // this is obv wrong for now
    // TODO linear or symbolic regression
    return 1.0 / (b + 2.0);

    // OTHER DERIVATION
    // we can approximate exp(-x) with (1-x)^2, which results in x^2-2x+1
    // this then expands to x^2 + 2bx + c - 2sqrt(x^2 + 2bx + c) + 1
    // the integral of this is 1/3x^3 + bx^2 + cx - (c - b^2) * ln(sqrt(2bx + c + x^2) + b + x) + (b + x) * sqrt(2bx + c
    // + x^2) + x r\left(x,\ c,\
    // b\right)=\frac{1}{3}x^{3}+2bx^{2}+cx-\left(c-b^{2}\right)\ln\left(\left(\sqrt{2bx+c+x^{2}}\right)+b+x\right)+\left(b+x\right)\sqrt{2bx+c+x^{2}}+x
    // doesn't seem to work?
}

// now, we also want the full single scattering
// we're gonna use 3 channels, and this calculates scattering for one channel, and takes the absorption of the 3 into
// account single scattering happens when light hits a particle in the atmosphere, and changes direction to go straight
// to the camera the light is first attenuated by the air in the way from the light source (sun in this case) to the
// particle attenuation means that the light is blocked, which can be calculated with exp(-optical depth light to
// particle) after the light is scattered, it's attenuated again from the particle position to the light multiplying
// with the beta can be done after the total scattering and attenuation is not calculated in this function it's also
// possible for the amount of light that is scattered to differ depending on the angle of the camera view ray and the
// light direction this amount of light scattered is described as the phase function, which can also be done after the
// total scattering is done, so this is also not done in this function
vec3 get_single_scattering(float b, float c)
{

    return vec3(0.0);
}

// the total scattering function
vec4 total_scattering(
    vec3 start,             // start position of the ray
    vec3 dir,               // direction of the ray
    float max_dist,         // length of the ray, -1 if infinite
    float planet_radius,    // planet radius
    float scale_height_a,   // scale height for the first scattering type
    float scale_height_b,   // scale height for the second scattering type
    float scale_height_c,   // scale height for the third scattering type
    vec3 scattering_beta_a, // scattering beta (how much light it scatters) for the first scattering type
    vec3 scattering_beta_b, // scattering beta (how much light it scatters) for the second scattering type
    vec3 scattering_beta_c, // scattering beta (how much light it scatters) for the third scattering type
    vec3 absorption_beta_a, // absorption beta (how much light it takes away) for the first scattering type, added to
                            // the scattering
    vec3 absorption_beta_b, // absorption beta (how much light it takes away) for the second scattering type, added to
                            // the scattering
    vec3 absorption_beta_c // absorption beta (how much light it takes away) for the third scattering type, added to the
                           // scattering
)
{

    // calculate b and c for the start of the ray
    float start_b = dot(start, dir);
    float start_c = dot(start, start) - (planet_radius * planet_radius);

    // and for the end of the ray
    float end_b = dot(start + dir * max_dist, dir);
    float end_c = dot(start + dir * max_dist, start + dir * max_dist) - (planet_radius * planet_radius);

    // and calculate the halfway point, where the ray is closest to the planet
    float halfway = length(start) * dot(normalize(start), dir);

    // now, calculate the optical depth for the entire ray
    // we'll use the functions we made earlier for this
    vec3 optical_depth = (get_optical_depth(start_b, start_c, 1.0 / scale_height_a, planet_radius) *
                              (scattering_beta_a + absorption_beta_a) +
                          get_optical_depth(start_b, start_c, 1.0 / scale_height_b, planet_radius) *
                              (scattering_beta_b + absorption_beta_b) +
                          get_optical_depth(start_b, start_c, 1.0 / scale_height_c, planet_radius) *
                              (scattering_beta_c + absorption_beta_c)) -
                         (max_dist < 0.0 ? vec3(0.0)
                                         : ( // we don't need to subtract the rest of the ray from the end position, so
                                             // that we only get the segment we want
                                               get_optical_depth(end_b, end_c, 1.0 / scale_height_a, planet_radius) *
                                                   (scattering_beta_a + absorption_beta_a) +
                                               get_optical_depth(end_b, end_c, 1.0 / scale_height_b, planet_radius) *
                                                   (scattering_beta_b + absorption_beta_b) +
                                               get_optical_depth(end_b, end_c, 1.0 / scale_height_c, planet_radius) *
                                                   (scattering_beta_c + absorption_beta_c)));

    // next up, get the attenuation for the segment
    vec3 atmosphere_attn = exp(-optical_depth);

    // and return the final color
    return vec4(optical_depth, atmosphere_attn);
}

// Next we'll define the main scattering function.
// This traces a ray from start to end and takes a certain amount of samples along this ray, in order to calculate the
// color. For every sample, we'll also trace a ray in the direction of the light, because the color that reaches the
// sample also changes due to scattering
vec3 calculate_scattering(
    vec3 start,           // the start of the ray (the camera position)
    vec3 dir,             // the direction of the ray (the camera vector)
    float max_dist,       // the maximum distance the ray can travel (because something is in the way, like an object)
    vec3 light_dir,       // the direction of the light
    vec3 light_intensity, // how bright the light is, affects the brightness of the atmosphere
    vec3 planet_position, // the position of the planet
    float planet_radius,  // the radius of the planet
    float atmo_radius,    // the radius of the atmosphere
    vec3 beta_ray,        // the amount rayleigh scattering scatters the colors (for earth: causes the blue atmosphere)
    vec3 beta_mie,        // the amount mie scattering scatters colors
    vec3 beta_absorption, // how much air is absorbed
    vec3 beta_ambient, // the amount of scattering that always occurs, cna help make the back side of the atmosphere a
                       // bit brighter
    float
        g, // the direction mie scatters the light in (like a cone). closer to -1 means more towards a single direction
    float height_ray,         // how high do you have to go before there is no rayleigh scattering?
    float height_mie,         // the same, but for mie
    float height_absorption,  // the height at which the most absorption happens
    float absorption_falloff, // how fast the absorption falls off from the absorption height
    int steps_i,              // the amount of steps along the 'primary' ray, more looks better but slower
    int steps_l               // the amount of steps along the light ray, more looks better but slower
)
{
    // add an offset to the camera position, so that the atmosphere is in the correct position
    start -= planet_position;
    // calculate the start and end position of the ray, as a distance along the ray
    // we do this with a ray sphere intersect
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, start);
    float c = dot(start, start) - (atmo_radius * atmo_radius);
    float d = (b * b) - 4.0 * a * c;

    // stop early if there is no intersect
    if (d < 0.0)
        return vec3(0.0);

    // calculate the ray length
    vec2 ray_length = vec2(max((-b - sqrt(d)) / (2.0 * a), 0.0), min((-b + sqrt(d)) / (2.0 * a), max_dist));

    // if the ray did not hit the atmosphere, return a black color
    if (ray_length.x > ray_length.y)
        return vec3(0.0);
    // prevent the mie glow from appearing if there's an object in front of the camera
    bool allow_mie = max_dist > ray_length.y;
    // make sure the ray is no longer than allowed
    ray_length.y = min(ray_length.y, max_dist);
    ray_length.x = max(ray_length.x, 0.0);
    // get the step size of the ray
    float step_size_i = (ray_length.y - ray_length.x) / float(steps_i);

    // next, set how far we are along the ray, so we can calculate the position of the sample
    // if the camera is outside the atmosphere, the ray should start at the edge of the atmosphere
    // if it's inside, it should start at the position of the camera
    // the min statement makes sure of that
    float ray_pos_i = ray_length.x + step_size_i * 0.5;

    // these are the values we use to gather all the scattered light
    vec3 total_ray = vec3(0.0); // for rayleigh
    vec3 total_mie = vec3(0.0); // for mie

    // initialize the optical depth. This is used to calculate how much air was in the ray
    vec3 opt_i = vec3(0.0);

    // we define the density early, as this helps doing integration
    // usually we would do riemans summing, which is just the squares under the integral area
    // this is a bit innefficient, and we can make it better by also taking the extra triangle at the top of the square
    // into account the starting value is a bit inaccurate, but it should make it better overall
    vec3 prev_density = vec3(0.0);

    // also init the scale height, avoids some vec2's later on
    vec2 scale_height = vec2(height_ray, height_mie);

    // Calculate the Rayleigh and Mie phases.
    // This is the color that will be scattered for this ray
    // mu, mumu and gg are used quite a lot in the calculation, so to speed it up, precalculate them
    float mu = dot(dir, light_dir);
    float mumu = mu * mu;
    float gg = g * g;
    float phase_ray = 3.0 / (50.2654824574 /* (16 * pi) */) * (1.0 + mumu);
    float phase_mie = allow_mie ? 3.0 / (25.1327412287 /* (8 * pi) */) * ((1.0 - gg) * (mumu + 1.0)) /
                                      (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg))
                                : 0.0;

    // now we need to sample the 'primary' ray. this ray gathers the light that gets scattered onto it
    for (int i = 0; i < steps_i; ++i)
    {

        // calculate where we are along this ray
        vec3 pos_i = start + dir * ray_pos_i;

        // and how high we are above the surface
        float height_i = length(pos_i) - planet_radius;

        // now calculate the density of the particles (both for rayleigh and mie)
        vec3 density = vec3(exp(-height_i / scale_height), 0.0);

        // and the absorption density. this is for ozone, which scales together with the rayleigh,
        // but absorbs the most at a specific height, so use the sech function for a nice curve falloff for this height
        // clamp it to avoid it going out of bounds. This prevents weird black spheres on the night side
        float denom = (height_absorption - height_i) / absorption_falloff;
        density.z = (1.0 / (denom * denom + 1.0)) * density.x;

        // multiply it by the step size here
        // we are going to use the density later on as well
        density *= step_size_i;

        // Add these densities to the optical depth, so that we know how many particles are on this ray.
        // max here is needed to prevent opt_i from potentially becoming negative
        opt_i += density;

        // and update the previous density
        prev_density = density;

        // Calculate the step size of the light ray.
        // again with a ray sphere intersect
        // a, b, c and d are already defined
        a = dot(light_dir, light_dir);
        b = 2.0 * dot(light_dir, pos_i);
        c = dot(pos_i, pos_i) - (atmo_radius * atmo_radius);
        d = (b * b) - 4.0 * a * c;

        // no early stopping, this one should always be inside the atmosphere
        // calculate the ray length
        float step_size_l = (-b + sqrt(d)) / (2.0 * a * float(steps_l));

        // and the position along this ray
        // this time we are sure the ray is in the atmosphere, so set it to 0
        float ray_pos_l = step_size_l * 0.5;

        // and the optical depth of this ray
        vec3 opt_l = vec3(0.0);

        // again, use the prev density for better integration
        vec3 prev_density_l = vec3(0.0);

        // now sample the light ray
        // this is similar to what we did before
        for (int l = 0; l < steps_l; ++l)
        {

            // calculate where we are along this ray
            vec3 pos_l = pos_i + light_dir * ray_pos_l;

            // the heigth of the position
            float height_l = length(pos_l) - planet_radius;

            // calculate the particle density, and add it
            // this is a bit verbose
            // first, set the density for ray and mie
            vec3 density_l = vec3(exp(-height_l / scale_height), 0.0);

            // then, the absorption
            float denom = (height_absorption - height_l) / absorption_falloff;
            density_l.z = (1.0 / (denom * denom + 1.0)) * density_l.x;

            // multiply the density by the step size
            density_l *= step_size_l;

            // and add it to the total optical depth
            opt_l += density_l;

            // and update the previous density
            prev_density_l = density_l;

            // and increment where we are along the light ray.
            ray_pos_l += step_size_l;
        }

        // Now we need to calculate the attenuation
        // this is essentially how much light reaches the current sample point due to scattering
        vec3 attn = exp(-beta_ray * (opt_i.x + opt_l.x) - beta_mie * (opt_i.y + opt_l.y) -
                        beta_absorption * (opt_i.z + opt_l.z));

        // accumulate the scattered light (how much will be scattered towards the camera)
        total_ray += density.x * attn;
        total_mie += density.y * attn;

        // and increment the position on this ray
        ray_pos_i += step_size_i;
    }

    // calculate how much light can pass through the atmosphere
    // vec3 opacity = exp(-(beta_mie * opt_i.y + beta_ray * opt_i.x + beta_absorption * opt_i.z));

    // calculate and return the final color
    return (phase_ray * beta_ray * total_ray   // rayleigh color
            + phase_mie * beta_mie * total_mie // mie
            + opt_i.x * beta_ambient           // and ambient
            ) *
           light_intensity;
}

// A ray-sphere intersect
// This was previously used in the atmosphere as well, but it's only used for the planet intersect now, since the
// atmosphere has this ray sphere intersect built in
vec2 ray_sphere_intersect(vec3 start,  // starting position of the ray
                          vec3 dir,    // the direction of the ray
                          float radius // and the sphere radius
)
{
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    float a = dot(dir, dir);
    float b = 2.0 * dot(dir, start);
    float c = dot(start, start) - (radius * radius);
    float d = (b * b) - 4.0 * a * c;
    if (d < 0.0)
        return vec2(1e5, -1e5);
    return vec2((-b - sqrt(d)) / (2.0 * a), (-b + sqrt(d)) / (2.0 * a));
}

// To make the planet we're rendering look nicer, we implemented a skylight function here
// Essentially it just takes a sample of the atmosphere in the direction of the surface normal
vec3 skylight(vec3 sample_pos, vec3 surface_normal, vec3 light_dir, vec3 background_col)
{

    // slightly bend the surface normal towards the light direction
    surface_normal = normalize(mix(surface_normal, light_dir, 0.6));

    // and sample the atmosphere
    return calculate_scattering(
        sample_pos,         // the position of the camera
        surface_normal,     // the camera vector (ray direction of this pixel)
        3.0 * ATMOS_RADIUS, // max dist, since nothing will stop the ray here, just use some arbitrary value
        light_dir,          // light direction
        vec3(40.0),         // light intensity, 40 looks nice
        PLANET_POS,         // position of the planet
        PLANET_RADIUS,      // radius of the planet in meters
        ATMOS_RADIUS,       // radius of the atmosphere in meters
        RAY_BETA,           // Rayleigh scattering coefficient
        MIE_BETA,           // Mie scattering coefficient
        ABSORPTION_BETA,    // Absorbtion coefficient
        AMBIENT_BETA,      // ambient scattering, turned off. This causes the air to glow a bit when no light reaches it
        G,                 // Mie preferred scattering direction
        HEIGHT_RAY,        // Rayleigh scale height
        HEIGHT_MIE,        // Mie scale height
        HEIGHT_ABSORPTION, // the height at which the most absorption happens
        ABSORPTION_FALLOFF, // how fast the absorption falls off from the absorption height
        LIGHT_STEPS,        // steps in the ray direction
        LIGHT_STEPS         // steps in the light direction
    );
}

// The following function returns the scene color and depth
// (the color of the pixel without the atmosphere, and the distance to the surface that is visible on that pixel)
// in this case, the function renders a green sphere on the place where the planet should be
// color is in .xyz, distance in .w
// I won't explain too much about how this works, since that's not the aim of this shader
vec4 render_scene(vec3 pos, vec3 dir, vec3 light_dir)
{

    // the color to use, w is the scene depth
    vec4 color = vec4(0.0, 0.0, 0.0, 1e12);

    // add a sun, if the angle between the ray direction and the light direction is small enough, color the pixels white
    color.xyz = vec3(dot(dir, light_dir) > 0.9998 ? 40.0 : 0.0);

    // get where the ray intersects the planet
    vec2 planet_intersect = ray_sphere_intersect(pos - PLANET_POS, dir, PLANET_RADIUS);

    // if the ray hit the planet, set the max distance to that ray
    if (0.0 < planet_intersect.y)
    {

        color.w = max(planet_intersect.x, 0.0);
        /*
                // sample position, where the pixel is
                vec3 sample_pos = pos + (dir * planet_intersect.x) - PLANET_POS;

                // and the surface normal
                vec3 surface_normal = normalize(sample_pos);

                // get the color of the sphere
                color.xyz = vec3(1.0);

                // get wether this point is shadowed, + how much light scatters towards the camera according to the
                // lommel-seelinger law
                vec3 N = surface_normal;
                vec3 V = -dir;
                vec3 L = light_dir;
                float dotNV = max(1e-6, dot(N, V));
                float dotNL = max(1e-6, dot(N, L));
                float shadow = dotNL / (dotNL + dotNV);

                // apply the shadow
                color.xyz *= shadow;

                // apply skylight
                color.xyz +=
                    clamp(skylight(sample_pos, surface_normal, light_dir, vec3(0.0)) * vec3(0.0, 0.25, 0.05),
           0.0, 1.0);*/
    }

    return color;
}

// next, we need a way to do something with the scattering function
// to do something with it we need the camera vector (which is the ray direction) of the current pixel
// this function calculates it
vec3 get_camera_vector(vec2 resolution, vec2 coord)
{

    // convert the uv to -1 to 1
    vec2 uv = coord.xy / resolution - vec2(0.5);

    // scale for aspect ratio
    uv.x *= resolution.x / resolution.y;

    // and normalize to get the correct ray
    // the -1 here is for the angle of view
    // this can be calculated from an actual angle, but that's not needed here
    return normalize(vec3(uv.x, uv.y, -1.0));
}

// Finally, draw the atmosphere to screen
// we first get the camera vector and position, as well as the light dir
void mainImage(out vec3 atmosphere, in vec2 fragCoord, vec3 view)
{

    // get the camera vector
    vec3 camera_vector = vec3(view.x, clamp(view.y, 0.1, 1), view.z);

    // get the camera position, switch based on the defines
    vec3 camera_position = vec3(0.0, PLANET_RADIUS + 100.0, 0.0);

    // get the light direction
    vec3 light_dir = sunPosition3;
    // get the scene color and depth, color is in xyz, depth in w
    // replace this with something better if you are using this shader for something else
    vec4 scene = render_scene(camera_position, camera_vector, light_dir);

    // the color of this pixel
    vec3 col = vec3(0.0); // scene.xyz;

    // get the atmosphere color
    col += calculate_scattering(camera_position,    // the position of the camera
                                camera_vector,      // the camera vector (ray direction of this pixel)
                                scene.w,            // max dist, essentially the scene depth
                                light_dir,          // light direction
                                vec3(40.0),         // light intensity, 40 looks nice
                                PLANET_POS,         // position of the planet
                                PLANET_RADIUS,      // radius of the planet in meters
                                ATMOS_RADIUS,       // radius of the atmosphere in meters
                                RAY_BETA,           // Rayleigh scattering coefficient
                                MIE_BETA,           // Mie scattering coefficient
                                ABSORPTION_BETA,    // Absorbtion coefficient
                                AMBIENT_BETA,       // ambient scattering, turned off. This causes the air to glow a bit
                                                    // when no light reaches it
                                G,                  // Mie preferred scattering direction
                                HEIGHT_RAY,         // Rayleigh scale height
                                HEIGHT_MIE,         // Mie scale height
                                HEIGHT_ABSORPTION,  // the height at which the most absorption happens
                                ABSORPTION_FALLOFF, // how fast the absorption falls off from the absorption height
                                PRIMARY_STEPS,      // steps in the ray direction
                                LIGHT_STEPS         // steps in the light direction
    );

    // apply exposure, removing this makes the brighter colors look ugly
    // you can play around with removing this
    col = 1.0 - exp(-col);

    // Output to screen
    atmosphere.xyz = vec4(col, 1.0).xyz;
}
vec3 drawSun(float cosY, float sunInt, vec3 nsunlight, vec3 inColor)
{
    return inColor + nsunlight * 1133 * pow3(smoothstep(0.99955640208, 0.99985963575, cosY)) * 0.62;
}

// Return random noise in the range [0.0, 1.0], as a function of x.
float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float StableStarField(in vec2 vSamplePos, float fThreshhold)
{
    vec2 floorSample = floor(vSamplePos);
    float StarVal = hash12(floorSample);

    float v1 = clamp(StarVal / (1.0 - fThreshhold) - fThreshhold / (1.0 - fThreshhold), 0.0, 1.0);

    StarVal = v1 * 30.0 * skyIntensityNight;
    return StarVal;
}

float stars(vec3 fragpos)
{
    float elevation = clamp(fragpos.y, 0., 1.);
    vec2 uv = fragpos.xz / (1. + elevation);

    return StableStarField(uv * 700., 0.999) * 0.5 * 0.3;
}

//////////////////////////////////

// License (MIT) Copyright (C) 2017-2018 Rui. All rights reserved.

#define float2 vec2
#define float3 vec3
#define float4 vec4

#define PI_2 (3.1415926535f * 2.0)

#define EPSILON 1e-5

#define SAMPLES_NUMS 16

float saturate(float x)
{
    return clamp(x, 0.0, 1.0);
}

struct ScatteringParams
{
    float sunRadius;
    float sunRadiance;

    float mieG;
    float mieHeight;

    float rayleighHeight;

    float3 waveLambdaMie;
    float3 waveLambdaOzone;
    float3 waveLambdaRayleigh;

    float earthRadius;
    float earthAtmTopRadius;
    float3 earthCenter;
};

float3 ComputeSphereNormal(float2 coord, float phiStart, float phiLength, float thetaStart, float thetaLength)
{
    float3 normal;
    normal.x = -sin(thetaStart + coord.y * thetaLength) * sin(phiStart + coord.x * phiLength);
    normal.y = -cos(thetaStart + coord.y * thetaLength);
    normal.z = -sin(thetaStart + coord.y * thetaLength) * cos(phiStart + coord.x * phiLength);
    return normalize(normal);
}

float2 ComputeRaySphereIntersection(float3 position, float3 dir, float3 center, float radius)
{
    float3 origin = position - center;
    float B = dot(origin, dir);
    float C = dot(origin, origin) - radius * radius;
    float D = B * B - C;

    float2 minimaxIntersections;
    if (D < 0.0)
    {
        minimaxIntersections = float2(-1.0, -1.0);
    }
    else
    {
        D = sqrt(D);
        minimaxIntersections = float2(-B - D, -B + D);
    }

    return minimaxIntersections;
}

float3 ComputeWaveLambdaRayleigh(float3 lambda)
{
    const float n = 1.0003;
    const float N = 2.545E25;
    const float pn = 0.035;
    const float n2 = n * n;
    const float pi3 = PI * PI * PI;
    const float rayleighConst = (8.0 * pi3 * pow(n2 - 1.0, 2.0)) / (3.0 * N) * ((6.0 + 3.0 * pn) / (6.0 - 7.0 * pn));
    return rayleighConst / (lambda * lambda * lambda * lambda);
}

float ComputePhaseMie(float theta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * saturate(theta), 1.5) / (4.0 * PI);
}

float ComputePhaseRayleigh(float theta)
{
    float theta2 = theta * theta;
    return (theta2 * 0.75 + 0.75) / (4.0 * PI);
}

float ChapmanApproximation(float X, float h, float cosZenith)
{
    float c = sqrt(X + h);
    float c_exp_h = c * exp(-h);

    if (cosZenith >= 0.0)
    {
        return c_exp_h / (c * cosZenith + 1.0);
    }
    else
    {
        float x0 = sqrt(1.0 - cosZenith * cosZenith) * (X + h);
        float c0 = sqrt(x0);

        return 2.0 * c0 * exp(X - x0) - c_exp_h / (1.0 - c * cosZenith);
    }
}

float GetOpticalDepthSchueler(float h, float H, float earthRadius, float cosZenith)
{
    return H * ChapmanApproximation(earthRadius / H, h / H, cosZenith);
}

float3 GetTransmittance(ScatteringParams setting, float3 L, float3 V)
{
    float ch = GetOpticalDepthSchueler(L.y, setting.rayleighHeight, setting.earthRadius, V.y);
    return exp(-(setting.waveLambdaMie + setting.waveLambdaRayleigh) * ch);
}

float2 ComputeOpticalDepth(ScatteringParams setting, float3 samplePoint, float3 V, float3 L, float neg)
{
    float rl = length(samplePoint);
    float h = rl - setting.earthRadius;
    float3 r = samplePoint / rl;

    float cos_chi_sun = dot(r, L);
    float cos_chi_ray = dot(r, V * neg);

    float opticalDepthSun = GetOpticalDepthSchueler(h, setting.rayleighHeight, setting.earthRadius, cos_chi_sun);
    float opticalDepthCamera =
        GetOpticalDepthSchueler(h, setting.rayleighHeight, setting.earthRadius, cos_chi_ray) * neg;

    return float2(opticalDepthSun, opticalDepthCamera);
}

void AerialPerspective(ScatteringParams setting, float3 start, float3 end, float3 V, float3 L, bool infinite,
                       out float3 transmittance, out float3 insctrMie, out float3 insctrRayleigh)
{
    float inf_neg = infinite ? 1.0 : -1.0;

    float3 sampleStep = (end - start) / float(SAMPLES_NUMS);
    float3 samplePoint = end - sampleStep;
    float3 sampleLambda = setting.waveLambdaMie + setting.waveLambdaRayleigh + setting.waveLambdaOzone;

    float sampleLength = length(sampleStep);

    float3 scattering = float3(0.0);
    float2 lastOpticalDepth = ComputeOpticalDepth(setting, end, V, L, inf_neg);

    for (int i = 1; i < SAMPLES_NUMS; i++, samplePoint -= sampleStep)
    {
        float2 opticalDepth = ComputeOpticalDepth(setting, samplePoint, V, L, inf_neg);

        float3 segment_s = exp(-sampleLambda * (opticalDepth.x + lastOpticalDepth.x));
        float3 segment_t = exp(-sampleLambda * (opticalDepth.y - lastOpticalDepth.y));

        transmittance *= segment_t;

        scattering = scattering * segment_t;
        scattering += exp(-(length(samplePoint) - setting.earthRadius) / setting.rayleighHeight) * segment_s;

        lastOpticalDepth = opticalDepth;
    }

    insctrMie = scattering * setting.waveLambdaMie * sampleLength;
    insctrRayleigh = scattering * setting.waveLambdaRayleigh * sampleLength;
}

float ComputeSkyboxChapman(ScatteringParams setting, float3 eye, float3 V, float3 L, out float3 transmittance,
                           out float3 insctrMie, out float3 insctrRayleigh)
{
    bool neg = true;

    float2 outerIntersections = ComputeRaySphereIntersection(eye, V, setting.earthCenter, setting.earthAtmTopRadius);
    if (outerIntersections.y < 0.0)
        return 0.0;

    float2 innerIntersections = ComputeRaySphereIntersection(eye, V, setting.earthCenter, setting.earthRadius);
    if (innerIntersections.x > 0.0)
    {
        neg = false;
        outerIntersections.y = innerIntersections.x;
    }

    eye -= setting.earthCenter;

    float3 start = eye + V * max(0.0, outerIntersections.x);
    float3 end = eye + V * outerIntersections.y;

    AerialPerspective(setting, start, end, V, L, neg, transmittance, insctrMie, insctrRayleigh);

    bool intersectionTest = innerIntersections.x < 0.0 && innerIntersections.y < 0.0;
    return intersectionTest ? 1.0 : 0.0;
}

float4 ComputeSkyInscattering(ScatteringParams setting, float3 eye, float3 V, float3 L)
{
    float3 insctrMie = float3(0.0);
    float3 insctrRayleigh = float3(0.0);
    float3 insctrOpticalLength = float3(1.0);
    float intersectionTest = ComputeSkyboxChapman(setting, eye, V, L, insctrOpticalLength, insctrMie, insctrRayleigh);

    float phaseTheta = dot(V, L);
    float phaseMie = ComputePhaseMie(phaseTheta, setting.mieG);
    float phaseRayleigh = ComputePhaseRayleigh(phaseTheta);
    float phaseNight = 1.0 - saturate(insctrOpticalLength.x * EPSILON);

    float3 insctrTotalMie = insctrMie * phaseMie;
    float3 insctrTotalRayleigh = insctrRayleigh * phaseRayleigh;

    float3 sky = (insctrTotalMie + insctrTotalRayleigh) * setting.sunRadiance;

    float angle = saturate((1.0 - phaseTheta) * setting.sunRadius);
    float cosAngle = cos(angle * PI * 0.5);
    float edge = ((angle >= 0.9) ? smoothstep(0.9, 1.0, angle) : 0.0);

    float3 limbDarkening = GetTransmittance(setting, -L, V);
    limbDarkening *= pow(float3(cosAngle), float3(0.420, 0.503, 0.652)) * mix(vec3(1.0), float3(1.2, 0.9, 0.5), edge) *
                     intersectionTest;

    sky += limbDarkening;

    return float4(sky, phaseNight * intersectionTest);
}

float3 TonemapACES(float3 x)
{
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;
    return (x * (A * x + B)) / (x * (C * x + D) + E);
}

float noise(float2 uv)
{
    return fract(dot(sin(uv.xyx * uv.xyy * 1024.0), float3(341896.483, 891618.637, 602649.7031)));
}

/////////////////////////////////
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
            if (texelFetch(DiffuseDepthSampler, ivec2(halfResTC) + ivec2(i, j), 0).x >= 1.0)
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
        float vdots = dot(view, sunPosition3);
        vec3 atmosphere = vec3(0.0);
        // atmosphere = toLinear(generate(view.xyz, sunPosition3).xyz) + (noise / 255);

        mainImage(atmosphere, gl_FragCoord.xy, view);

        //////////////////////
        /*
                float3 V = vec3(view.x, clamp(view.y, 0.05, 1), view.z);
                float3 L = sunPosition3;

                ScatteringParams setting;
                setting.sunRadius = 2500.0;
                setting.sunRadiance = 20.0;
                setting.mieG = 0.76;
                setting.mieHeight = 1200.0;
                setting.rayleighHeight = 8000.0;
                setting.earthRadius = 6360000.0;
                setting.earthAtmTopRadius = 6420000.0;
                setting.earthCenter = float3(0, -setting.earthRadius, 0);
                setting.waveLambdaMie = float3(2e-7);

                // wavelength with 680nm, 550nm, 450nm
                setting.waveLambdaRayleigh = ComputeWaveLambdaRayleigh(float3(680e-9, 550e-9, 450e-9));

                // see https://www.shadertoy.com/view/MllBR2
                setting.waveLambdaOzone = float3(1.36820899679147, 3.31405330400124, 0.13601728252538) * 0.6e-6 * 2.504;

                float3 eye = float3(0, 1000.0, 0);
                float4 sky = ComputeSkyInscattering(setting, eye, V, L);
                sky.rgb = TonemapACES(sky.rgb * 2.0);
                 sky.rgb = pow(sky.rgb, float3(1.0 / 2.2)); // gamma
                // sky.rgb += noise(uv*iTime) / 255.0; // dither
                atmosphere = vec3(toLinear(sky.xyz));
        */

        /////////////////////
        vec4 cloud = vec4(0.0, 0.0, 0.0, 1.0);
        if (view.y > 0.)
        {
            cloud = renderClouds(viewPos, avgSky, noise, sc, sc, avgSky).rgba;

            atmosphere += ((stars(view) * 2.0) * clamp(1 - (rainStrength * 1), 0, 1)) * 0.05;
            atmosphere += drawSun(vdots, 0, sc.rgb * 0.006, vec3(0.0)) * clamp(1 - (rainStrength * 1), 0, 1);
            // atmosphere = atmosphere.xyz * cloud.a + (cloud.rgb);
        }

        fragColor.rgb = toLinear(atmosphere.xyz + (noise / 255)) * cloud.a + (cloud.rgb);

        fragColor.rgb = lumaBasedReinhardToneMapping(fragColor.rgb);

        fragColor.a = 1.0;
    }
    else
        fragColor = vec4(0, 0, 0, 1.0);

#else
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
#endif
}
