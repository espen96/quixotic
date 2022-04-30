#version 150
const float sunPathRotation = -35.0;

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D noisetex;
uniform sampler2D DiffuseSampler;
uniform sampler2D PreviousFrameSampler;
uniform float Time;
out mat4 gbufferModelView;
out mat4 wgbufferModelView;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out float sunElevation;
out vec4 exposure;
out vec2 rodExposureDepth;
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
#define SUNBRIGHTNESS 20
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
float luma(vec3 color)
{
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 reinhard_jodie(vec3 v)
{
    float l = luma(v);
    vec3 tv = v / (1.0f + v);
    tv = mix(v / (1.0f + l), tv, tv);
    return tv;
}
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
#if HW_PERFORMANCE == 1
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
        vec3(SUNBRIGHTNESS),         // light intensity, 40 looks nice
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
const vec3 camera_position = vec3(0.0, PLANET_RADIUS + 100.0, 0.0);
// Finally, draw the atmosphere to screen
// we first get the camera vector and position, as well as the light dir
void mainImage(out vec3 atmosphere, in vec3 view)
{

    // get the camera vector
    vec3 camera_vector = vec3(view.x, clamp(view.y, 0.05, 1), view.z);

    // get the camera position, switch based on the defines

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
                                vec3(SUNBRIGHTNESS),         // light intensity, 40 looks nice
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
    col = reinhard_jodie(col);

    // Output to screen
    atmosphere.xyz = vec4(col*2.0, 1.0).xyz;
}

void main()
{
    exposure=vec4(texelFetch(PreviousFrameSampler,ivec2(10,37),0).r*vec3(1.0),texelFetch(PreviousFrameSampler,ivec2(10,37),0).r)*10.1;
	rodExposureDepth = texelFetch(PreviousFrameSampler,ivec2(14,37),0).rg;
	rodExposureDepth.y = sqrt(rodExposureDepth.y/65000.0);
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

    lightCol.xyz = vec3(0.0);
    mainImage(lightCol.xyz, sunPosition3);
    vec3 lightSourceColor = toLinear(lightCol.rgb);

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

        vec3 pos = normalize(rodSample(ij) + float(i / maxIT) + 0.1);

        vec3 samplee =
            skyLut2(pos.xyz, sunDir2, pos.y, rainStrength * 0.25, nsunColor, skyIntensity, skyIntensityNight) / maxIT;
        samplee.xyz = vec3(0.0);
        mainImage(samplee.xyz, pos);
        samplee.xyz = (2.2*toLinear(samplee)) / maxIT;
        ambientUp += samplee * (pos.y + abs(pos.x) / 7. + abs(pos.z) / 7.);
        ambientLeft += samplee * (clamp(-pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
        ambientRight += samplee * (clamp(pos.x, 0.0, 1.0) + clamp(pos.y / 7., 0.0, 1.0) + abs(pos.z) / 7.);
        ambientB += samplee * (clamp(pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
        ambientF += samplee * (clamp(-pos.z, 0.0, 1.0) + abs(pos.x) / 7. + clamp(pos.y / 7., 0.0, 1.0));
        ambientDown += samplee * (clamp(pos.y / 6., 0.0, 1.0) + abs(pos.x) / 7. + abs(pos.z) / 7.);
    }
    float dSun = 0.03;
    vec3 modSunVec = sunPosition3 * (1.0 - dSun) + vec3(0.0, dSun, 0.0);
    vec3 modSunVec2 = sunPosition3 * (1.0 - dSun) + vec3(0.0, dSun, 0.0);
    if (modSunVec2.y > modSunVec.y)
        modSunVec = modSunVec2;
    vec3 sunColorCloud = vec3(0.0);
    mainImage(sunColorCloud.xyz, modSunVec);
    // Fake bounced sunlight
    vec3 bouncedSun = lightSourceColor * 1.0 / 5.0 * 0.5 * clamp(lightDir * sunPosition3.y, 0.0, 1.0) *
                      clamp(lightDir * sunPosition3.y, 0.0, 1.0);
    vec3 cloudAmbientSun = (sunColorCloud)*0.007;
    vec3 cloudAmbientMoon = (vec3(0.0)) * 0.007;
    ambientUp += bouncedSun * clamp(-lightDir * sunVec.y + 4., 0., 4.0) +
                 cloudAmbientSun * clamp(sunVec.y + 2., 0., 4.0) + cloudAmbientMoon * clamp(-sunVec.y + 2., 0., 4.0);
    ambientLeft += bouncedSun * clamp(lightDir * sunVec.x + 4., 0.0, 4.) +
                   cloudAmbientSun * clamp(-sunVec.x + 2., 0.0, 4.) * 0.7 +
                   cloudAmbientMoon * clamp(sunVec.x + 2., 0.0, 4.) * 0.7;
    ambientRight += bouncedSun * clamp(-lightDir * sunVec.x + 4., 0.0, 4.) +
                    cloudAmbientSun * clamp(sunVec.x + 2., 0.0, 4.) * 0.7 +
                    cloudAmbientMoon * clamp(-sunVec.x + 2., 0.0, 4.) * 0.7;
    ambientB += bouncedSun * clamp(-lightDir * sunVec.z + 4., 0.0, 4.) +
                cloudAmbientSun * clamp(sunVec.z + 2., 0.0, 4.) * 0.7 +
                cloudAmbientMoon * clamp(-sunVec.z + 2., 0.0, 4.) * 0.7;
    ambientF += bouncedSun * clamp(lightDir * sunVec.z + 4., 0.0, 4.) +
                cloudAmbientSun * clamp(-sunVec.z + 2., 0.0, 4.) * 0.7 +
                cloudAmbientMoon * clamp(sunVec.z + 2., 0.0, 4.) * 0.7;
    ambientDown += bouncedSun * clamp(lightDir * sunVec.y + 4., 0.0, 4.) * 0.7 +
                   cloudAmbientSun * clamp(-sunVec.y + 2., 0.0, 4.) * 0.5 +
                   cloudAmbientMoon * clamp(sunVec.y + 2., 0.0, 4.) * 0.5;
    // avgSky += bouncedSun*5.;

    gl_Position = vec4(outPos.xy, 0.2, 1.0);
}
