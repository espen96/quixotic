#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <settings.glsl>
#moj_import <mappings.glsl>

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
uniform vec2 ScreenSize;
uniform mat4 ModelViewMat;
uniform mat4 ProjMat;

in mat4 gbufferModelViewInverse;
in float isSky;
in float vertexDistance;
in vec3 test;
out vec4 fragColor;

// at this point, the entire sky is drawable: isSky for sky, stars and void plane for everything else.
// similar logic can be added in vsh to separate void plane from stars.

vec3 renderSky(vec3 reddishTint, vec3 horizonColor, vec3 zenithColor, float h)
{

    h = 1.0 - abs(h);

    float hsq = h * h;

    // gradient 1 = h^8
    float gradient1 = hsq * hsq;
    gradient1 *= gradient1;

    float gradient2 = 0.5 * (hsq + h * hsq);

    horizonColor = mix(horizonColor, reddishTint, gradient1);
    return mix(zenithColor, horizonColor, gradient2);
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
#define ABSORPTION_BETA vec3(2.04e-5, 4.97e-5, 1.95e-6)
#define G 0.7 /* mie scattering direction, or how big the blob around the sun is */
// and the heights (how far to go up before the scattering has no effect)
#define HEIGHT_RAY 8e3 /* rayleigh height */
#define HEIGHT_MIE 1.2e3 /* and mie */
#define HEIGHT_ABSORPTION 30e3 /* at what height the absorption is at it's maximum */
#define ABSORPTION_FALLOFF 4e3
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
// we can calculate this as the integral over beta * exp(-scale_height * (sqrt(t^2 + 2bt + c) - planet_radius)),
// from t = 0 to infinity with t as the distance from the start position, b = dot(ray direction, ray start) and c =
// dot(ray start, ray start) - planet radius * planet_radius due to the multiplication by constant rule, we can keep
// beta outside of the integral we can do it to infinity, because if we calculate the same at the object pos and
// subtract it from the one at the camera pos, we get the same result this is also needed because we can't get the
// exact integral of this, so an approximation is needed

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
        sample_pos,          // the position of the camera
        surface_normal,      // the camera vector (ray direction of this pixel)
        3.0 * ATMOS_RADIUS,  // max dist, since nothing will stop the ray here, just use some arbitrary value
        light_dir,           // light direction
        vec3(22), // light intensity, 40 looks nice
        PLANET_POS,          // position of the planet
        PLANET_RADIUS,       // radius of the planet in meters
        ATMOS_RADIUS,        // radius of the atmosphere in meters
        RAY_BETA,            // Rayleigh scattering coefficient
        MIE_BETA,            // Mie scattering coefficient
        ABSORPTION_BETA,     // Absorbtion coefficient
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
    vec3 light_dir = vec3(0,1,0);
    // get the scene color and depth, color is in xyz, depth in w
    // replace this with something better if you are using this shader for something else
    vec4 scene = render_scene(camera_position, camera_vector, light_dir);

    // the color of this pixel
    vec3 col = vec3(0.0); // scene.xyz;

    // get the atmosphere color
    col += calculate_scattering(camera_position,     // the position of the camera
                                camera_vector,       // the camera vector (ray direction of this pixel)
                                scene.w,             // max dist, essentially the scene depth
                                light_dir,           // light direction
                                vec3(22), // light intensity, 40 looks nice
                                PLANET_POS,          // position of the planet
                                PLANET_RADIUS,       // radius of the planet in meters
                                ATMOS_RADIUS,        // radius of the atmosphere in meters
                                RAY_BETA,            // Rayleigh scattering coefficient
                                MIE_BETA,            // Mie scattering coefficient
                                ABSORPTION_BETA,     // Absorbtion coefficient
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
void main()
{
    int index = inControl(gl_FragCoord.xy, ScreenSize.x);
    if (index != -1)
    {
        if (isSky > 0.5)
        {
            if (index >= 5 && index <= 15)
            {
                int c = (index - 5) / 4;
                int r = (index - 5) - c * 4;
                c = (c == 0 && r == 1) ? c : c + 1;
                fragColor = vec4(encodeFloat(ProjMat[c][r]), 1.0);
            }
            else if (index >= 16 && index <= 24)
            {
                int c = (index - 16) / 3;
                int r = (index - 16) - c * 3;
                fragColor = vec4(encodeFloat(ModelViewMat[c][r]), 1.0);
            }
            else if (index >= 3 && index <= 4)
            {
                fragColor = vec4(encodeFloat(atan(ProjMat[index - 3][index - 3])), 1.0);
            }
            // store FogColor in control pixels
            else if (index == 25)
            {
                fragColor = vec4(FogColor.rgb, clamp(abs(FogStart) * 0.01, 0, 1));
            }
            // store SkyColor? in control pixels
            else if (index == 26)
            {
                fragColor = vec4(ColorModulator.rgb, 1);
            }
            else if (index == 27)
            {
                fragColor = vec4(vec3(0.5), 1);
            }
            else if (index == 28)
            {
                fragColor = vec4(1.0);
            }
            else if (index == 52)
            {
                fragColor = vec4(test, 1);
                // mappings
            }
            else if (index == 100)
            {
                fragColor = vec4(float(sssMin) / 255, float(sssMax) / 255, 0, 1);
            }
            else if (index == 101)
            {
                fragColor = vec4(float(lightMin) / 255, float(lightMax) / 255, 0, 1);
            }
            else if (index == 102)
            {
                fragColor = vec4(float(roughMin) / 255, float(roughMax) / 255, 0, 1);
            }
            else if (index == 103)
            {
                fragColor = vec4(float(metalMin) / 255, float(metalMax) / 255, 0, 1);
            }
            // settings
            else if (index == 104)
            {
                fragColor = vec4(sExposure / 255, sWhiteCurve / 255, sLowerCurve / 255, 1);
            }
            else if (index == 105)
            {
                fragColor = vec4(sUpperCurve / 255, sCrossTalk / 255, sSaturation / 255, 1);
            }

            // blackout control pixels for sunDir so sun can write to them (by default, all pixels are FogColor)
            else
            {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        else
        {
            discard;
        }
    }

    // not a control pixel, draw sky like normal
    else if (isSky > 0.5)
    {
        discard;

        vec4 screenPos = gl_FragCoord;
        screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
        screenPos.zw = vec2(1.0);
        vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);
        float ndusq = clamp(dot(view, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
        ndusq = ndusq * ndusq;

        vec3 atmosphere = vec3(0.0);

        //mainImage(atmosphere, gl_FragCoord.xy, view);

        //fragColor = linear_fog(ColorModulator, pow(1.0 - ndusq, 8.0), 0.0, 1.0, FogColor);

        //fragColor.rgb = renderSky(ColorModulator.rgb, ColorModulator.rgb, FogColor.rgb, view.y);
        //fragColor.rgb = atmosphere.rgb;
        //fragColor.a = 1;
    }
}